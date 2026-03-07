"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Message,
    PhotoSize,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.sdk_integration import StreamUpdate
from ..claude.desktop_sessions import (
    scan_desktop_sessions,
    format_session_list,
    format_project_sessions,
    find_group_by_name,
    get_session_by_index,
    find_session_title_by_id,
    extract_recent_messages,
)
from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .utils.html_format import escape_html
from .utils.image_extractor import (
    ImageAttachment,
    should_send_as_photo,
    validate_image_path,
)

logger = structlog.get_logger()

# Patterns that look like secrets/credentials in CLI arguments
_SECRET_PATTERNS: List[re.Pattern[str]] = [
    # API keys / tokens (sk-ant-..., sk-..., ghp_..., gho_..., github_pat_..., xoxb-...)
    re.compile(
        r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"
        r"|(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"
        r"|(ghp_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(gho_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"
        r"|(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"
    ),
    # AWS access keys
    re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
    # Generic long hex/base64 tokens after common flags/env patterns
    re.compile(
        r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
        r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?"
    ),
    # Inline env assignments like KEY=value
    re.compile(
        r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
        r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
        r"=)['\"]?[^\s'\"]{8,}['\"]?"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}" r"|(Basic )[A-Za-z0-9+/=]{8,}"),
    # Connection strings with credentials  user:pass@host
    re.compile(r"://([^:]+:)[^@]{4,}(@)"),
]


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda m: next((g + "***" for g in m.groups() if g is not None), "***"),
            result,
        )
    return result


# Tool name -> friendly emoji mapping for verbose output
_TOOL_ICONS: Dict[str, str] = {
    "Read": "\U0001f4d6",
    "Write": "\u270f\ufe0f",
    "Edit": "\u270f\ufe0f",
    "MultiEdit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50d",
    "LS": "\U0001f4c2",
    "Task": "\U0001f9e0",
    "TaskOutput": "\U0001f9e0",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookRead": "\U0001f4d3",
    "NotebookEdit": "\U0001f4d3",
    "TodoRead": "\u2611\ufe0f",
    "TodoWrite": "\u2611\ufe0f",
}


def _tool_icon(name: str) -> str:
    """Return emoji for a tool, with a default wrench."""
    return _TOOL_ICONS.get(name, "\U0001f527")


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps
        self._photo_group_buffer_seconds = 1.2
        self._pending_photo_groups: Dict[str, Dict[str, Any]] = {}

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            context.user_data.pop("_thread_context", None)

            is_sync_bypass = handler.__name__ == "sync_threads"
            is_start_bypass = handler.__name__ in {"start_command", "agentic_start"}
            message_thread_id = self._extract_message_thread_id(update)
            should_enforce = self.settings.enable_project_threads

            if should_enforce:
                if self.settings.project_threads_mode == "private":
                    should_enforce = not is_sync_bypass and not (
                        is_start_bypass and message_thread_id is None
                    )
                else:
                    should_enforce = not is_sync_bypass

            if should_enforce:
                allowed = await self._apply_thread_routing_context(update, context)
                if not allowed:
                    return

            try:
                await handler(update, context)
            finally:
                if should_enforce:
                    self._persist_thread_state(context)

        return wrapped

    async def _apply_thread_routing_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enforce strict project-thread routing and load thread-local state."""
        manager = context.bot_data.get("project_threads_manager")
        if manager is None:
            await self._reject_for_thread_mode(
                update,
                "❌ <b>Project Thread Mode Misconfigured</b>\n\n"
                "Thread manager is not initialized.",
            )
            return False

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return False

        if self.settings.project_threads_mode == "group":
            if chat.id != self.settings.project_threads_chat_id:
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False
        else:
            if getattr(chat, "type", "") != "private":
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False

        message_thread_id = self._extract_message_thread_id(update)
        if not message_thread_id:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        project = await manager.resolve_project(chat.id, message_thread_id)
        if not project:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        state_key = f"{chat.id}:{message_thread_id}"
        thread_states = context.user_data.setdefault("thread_state", {})
        state = thread_states.get(state_key, {})

        project_root = project.absolute_path
        current_dir_raw = state.get("current_directory")
        current_dir = (
            Path(current_dir_raw).resolve() if current_dir_raw else project_root
        )
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        context.user_data["current_directory"] = current_dir
        context.user_data["claude_session_id"] = state.get("claude_session_id")
        context.user_data["_thread_context"] = {
            "chat_id": chat.id,
            "message_thread_id": message_thread_id,
            "state_key": state_key,
            "project_slug": project.slug,
            "project_root": str(project_root),
            "project_name": project.name,
        }
        return True

    def _persist_thread_state(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Persist compatibility keys back into per-thread state."""
        thread_context = context.user_data.get("_thread_context")
        if not thread_context:
            return

        project_root = Path(thread_context["project_root"])
        current_dir = context.user_data.get("current_directory", project_root)
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))
        current_dir = current_dir.resolve()
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        thread_states = context.user_data.setdefault("thread_state", {})
        thread_states[thread_context["state_key"]] = {
            "current_directory": str(current_dir),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "project_slug": thread_context["project_slug"],
        }

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return True if path is within root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_message_thread_id(update: Update) -> Optional[int]:
        """Extract topic/thread id from update message for forum/direct topics."""
        message = update.effective_message
        if not message:
            return None
        message_thread_id = getattr(message, "message_thread_id", None)
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            return message_thread_id
        dm_topic = getattr(message, "direct_messages_topic", None)
        topic_id = getattr(dm_topic, "topic_id", None) if dm_topic else None
        if isinstance(topic_id, int) and topic_id > 0:
            return topic_id
        # Telegram omits message_thread_id for the General topic in forum
        # supergroups; its canonical thread ID is 1.
        chat = update.effective_chat
        if chat and getattr(chat, "is_forum", False):
            return 1
        return None

    async def _reject_for_thread_mode(self, update: Update, message: str) -> None:
        """Send a guidance response when strict thread routing rejects an update."""
        query = update.callback_query
        if query:
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(message, parse_mode="HTML")
            return

        if update.effective_message:
            await update.effective_message.reply_text(message, parse_mode="HTML")

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register agentic handlers: commands + text/file/photo."""
        from .handlers import command

        # Commands
        handlers = [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            ("verbose", self.agentic_verbose),
            ("repo", self.agentic_repo),
            ("resume", self.agentic_resume),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Only cd: callbacks (for project selection), scoped by pattern
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        # new: callbacks (for /new project picker)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._new_project_callback),
                pattern=r"^new:",
            )
        )

        # resume: callbacks (for /resume project drill-down & session pick)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._resume_callback),
                pattern=r"^resume:",
            )
        )

        logger.info("Agentic handlers registered")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (13 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            commands = [
                BotCommand("start", "Start the bot"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("status", "Show session status"),
                BotCommand("verbose", "Set output verbosity (0/1/2)"),
                BotCommand("repo", "List repos / switch workspace"),
                BotCommand("resume", "Resume a desktop Claude Code session"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands
        else:
            commands = [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        sync_line = ""
        if (
            self.settings.enable_project_threads
            and self.settings.project_threads_mode == "private"
        ):
            if (
                not update.effective_chat
                or getattr(update.effective_chat, "type", "") != "private"
            ):
                await update.message.reply_text(
                    "🚫 <b>Private Topics Mode</b>\n\n"
                    "Use this bot in a private chat and run <code>/start</code> there.",
                    parse_mode="HTML",
                )
                return
            manager = context.bot_data.get("project_threads_manager")
            if manager:
                try:
                    result = await manager.sync_topics(
                        context.bot,
                        chat_id=update.effective_chat.id,
                    )
                    sync_line = (
                        "\n\n🧵 Topics synced"
                        f" (created {result.created}, reused {result.reused})."
                    )
                except PrivateTopicsUnavailableError:
                    await update.message.reply_text(
                        manager.private_topics_unavailable_message(),
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    sync_line = "\n\n🧵 Topic sync failed. Run /sync_threads to retry."
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        await update.message.reply_text(
            f"Hi {safe_name}! I'm your AI coding assistant.\n"
            f"Just tell me what you need — I can read, write, and run code.\n\n"
            f"Working in: {dir_display}\n"
            f"Commands: /new (reset) · /status"
            f"{sync_line}",
            parse_mode="HTML",
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session and let user pick a project directory.

        /new         — show project picker (inline buttons, grouped by activity)
        /new <name>  — reset and switch to named project directly
        """
        args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory

        # Direct switch: /new <project_name>
        if args:
            target_name = args[0]
            target_path = base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"❌ Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return
            self._reset_session(context, target_path)
            await update.message.reply_text(
                f"🔄 New session in <b>{escape_html(target_name)}</b>",
                parse_mode="HTML",
            )
            return

        # No args — show project picker grouped by activity
        try:
            entries = [
                d
                for d in base.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        # Classify projects by activity level
        active, recent, archived = self._classify_projects(entries)

        # Build message text with section headers (left-aligned)
        text_lines = ["🔄 <b>New Session</b>\n"]
        keyboard_rows: List[list] = []

        if active:
            text_lines.append(f"🟢 <b>活跃项目</b>（7天内）")
            self._add_project_rows(keyboard_rows, active)

        if recent:
            text_lines.append(f"⚪ 近期项目")
            self._add_project_rows(keyboard_rows, recent)

        # Archived — collapsed
        if archived:
            keyboard_rows.append(
                [
                    InlineKeyboardButton(
                        f"📂 更多项目 ({len(archived)})...",
                        callback_data="new:__archived__",
                    )
                ]
            )

        # Bottom row: root + new directory
        keyboard_rows.append(
            [
                InlineKeyboardButton(
                    "🏠 JoyaProjects", callback_data="new:__root__"
                ),
                InlineKeyboardButton(
                    "➕ 新建目录", callback_data="new:__create__"
                ),
            ]
        )

        reply_markup = InlineKeyboardMarkup(keyboard_rows)
        await update.message.reply_text(
            "\n".join(text_lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    def _classify_projects(
        self, entries: List[Path]
    ) -> tuple:
        """Classify projects into active/recent/archived by activity.

        Activity is determined by Claude session file modification time,
        falling back to directory modification time.

        Returns (active, recent, archived) lists of (path, sort_key) tuples.
        """
        import time as _time

        now = _time.time()
        seven_days = 7 * 86400
        thirty_days = 30 * 86400
        claude_projects = Path.home() / ".claude" / "projects"

        scored: List[tuple] = []
        for entry in entries:
            # Check Claude session activity
            session_dir_name = (
                f"-Users-joya-JoyaProjects-{entry.name}"
            )
            session_dir = claude_projects / session_dir_name
            latest_activity = entry.stat().st_mtime  # fallback: dir mtime

            if session_dir.is_dir():
                try:
                    jsonl_files = list(session_dir.glob("*.jsonl"))
                    if jsonl_files:
                        latest_session = max(
                            f.stat().st_mtime for f in jsonl_files
                        )
                        latest_activity = max(latest_activity, latest_session)
                except OSError:
                    pass

            age = now - latest_activity
            scored.append((entry, latest_activity, age))

        # Sort by most recent activity first
        scored.sort(key=lambda x: x[1], reverse=True)

        active = []
        recent = []
        archived = []
        for entry, activity, age in scored:
            if age <= seven_days:
                active.append(entry)
            elif age <= thirty_days:
                recent.append(entry)
            else:
                archived.append(entry)

        return active, recent, archived

    def _add_project_rows(
        self,
        keyboard_rows: List[list],
        projects: List[Path],
    ) -> None:
        """Add project buttons to keyboard (2 per row)."""
        for i in range(0, len(projects), 2):
            row = []
            for j in range(2):
                if i + j < len(projects):
                    name = projects[i + j].name
                    row.append(
                        InlineKeyboardButton(
                            name,
                            callback_data=f"new:{name}",
                        )
                    )
            keyboard_rows.append(row)

    def _reset_session(
        self, context: ContextTypes.DEFAULT_TYPE, project_path: Path
    ) -> None:
        """Reset session state and switch to project directory."""
        context.user_data["claude_session_id"] = None
        context.user_data["_session_title"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True
        context.user_data["current_directory"] = project_path
        context.user_data.pop("_resume_groups", None)
        context.user_data.pop("_strict_resume_once", None)
        context.user_data.pop("_awaiting_new_dir", None)

    async def _new_project_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle new: callbacks — select project for new session."""
        query = update.callback_query
        await query.answer()

        _, action = query.data.split(":", 1)
        base = self.settings.approved_directory

        # Section header — no-op
        if action == "__noop__":
            return

        # Show archived projects
        if action == "__archived__":
            try:
                entries = [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
            except OSError:
                return
            _, _, archived = self._classify_projects(entries)
            if not archived:
                await query.edit_message_text("📂 没有更多项目。")
                return
            keyboard_rows: List[list] = []
            self._add_project_rows(keyboard_rows, archived)
            keyboard_rows.append(
                [InlineKeyboardButton("← 返回", callback_data="new:__back__")]
            )
            await query.edit_message_text(
                "🔄 <b>New Session</b>\n\n📂 更多项目",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard_rows),
            )
            return

        # Back to main project list
        if action == "__back__":
            # Re-render the main project picker
            try:
                entries = [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
            except OSError:
                return
            active, recent, archived = self._classify_projects(entries)

            text_lines = ["🔄 <b>New Session</b>\n"]
            keyboard_rows = []
            if active:
                text_lines.append(f"🟢 <b>活跃项目</b>（7天内）")
                self._add_project_rows(keyboard_rows, active)
            if recent:
                text_lines.append(f"⚪ 近期项目")
                self._add_project_rows(keyboard_rows, recent)
            if archived:
                keyboard_rows.append(
                    [
                        InlineKeyboardButton(
                            f"📂 更多项目 ({len(archived)})...",
                            callback_data="new:__archived__",
                        )
                    ]
                )
            keyboard_rows.append(
                [
                    InlineKeyboardButton(
                        "🏠 JoyaProjects", callback_data="new:__root__"
                    ),
                    InlineKeyboardButton(
                        "➕ 新建目录", callback_data="new:__create__"
                    ),
                ]
            )
            await query.edit_message_text(
                "\n".join(text_lines),
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard_rows),
            )
            return

        if action == "__create__":
            # Ask user to type the new directory name
            context.user_data["_awaiting_new_dir"] = True
            await query.edit_message_text(
                "📝 请输入新目录名称（在 JoyaProjects 下创建）："
            )
            return

        if action == "__root__":
            project_path = base
            project_display = "JoyaProjects"
        else:
            project_path = base / action
            project_display = action

        if not project_path.is_dir():
            await query.edit_message_text(
                f"❌ Directory not found: <code>{escape_html(action)}</code>",
                parse_mode="HTML",
            )
            return

        self._reset_session(context, project_path)

        # Send self-intro prompt to Claude
        await query.edit_message_text(
            f"🔄 New session in <b>{escape_html(project_display)}</b>...",
            parse_mode="HTML",
        )

        # Trigger Claude self-introduction
        await self._send_intro(update, context, query.message.chat)

    async def _send_intro(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, chat: Any
    ) -> None:
        """Show a static intro message for the new session."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        project_name = Path(str(current_dir)).name if current_dir else "JoyaProjects"
        agent_name = self.settings.claude_default_agent or "claude"

        # Agent display name mapping
        agent_display = {
            "meta": "Alice 🎀",
            "dev": "Salomé 💃",
            "qa": "Rêve 🔍",
            "corp": "Zoey 📋",
            "creator": "Aria 🦋",
        }.get(agent_name, agent_name)

        await chat.send_message(
            f"✅ <b>{agent_display}</b> · {escape_html(project_name)}\n"
            f"Ready. What's next?",
            parse_mode="HTML",
        )

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show current project, session, and cost info."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )

        # Project name: use directory name for brevity
        project_name = Path(str(current_dir)).name if current_dir else "unknown"

        session_id = context.user_data.get("claude_session_id")
        session_title = context.user_data.get("_session_title")

        # If no cached title, try to look it up from desktop session files
        if session_id and not session_title:
            session_title = find_session_title_by_id(session_id)
            if session_title:
                context.user_data["_session_title"] = session_title

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(update.effective_user.id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f"\n💰 Cost: ${current_cost:.2f}"
            except Exception:
                pass

        # Build status message
        lines = [f"📂 <b>{escape_html(project_name)}</b>"]
        lines.append(f"   <code>{escape_html(str(current_dir))}</code>")

        if session_id:
            short_id = session_id[:8]
            lines.append(f"\n🔗 Session: <code>{short_id}</code>")
            if session_title:
                lines.append(f"💬 \"{escape_html(session_title)}\"")
        else:
            lines.append("\n⚪ No active session")

        lines.append(cost_str)

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.user_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (final response only)\n"
                "  1 = normal (tools + reasoning)\n"
                "  2 = detailed (tools with inputs + reasoning)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    def _format_verbose_progress(
        self,
        activity_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
    ) -> str:
        """Build the progress message text based on activity so far."""
        if not activity_log:
            return "Working..."

        elapsed = time.time() - start_time
        lines: List[str] = [f"Working... ({elapsed:.0f}s)\n"]

        for entry in activity_log[-15:]:  # Show last 15 entries max
            kind = entry.get("kind", "tool")
            if kind == "text":
                # Claude's intermediate reasoning/commentary
                snippet = entry.get("detail", "")
                if verbose_level >= 2:
                    lines.append(f"\U0001f4ac {snippet}")
                else:
                    # Level 1: one short line
                    lines.append(f"\U0001f4ac {snippet[:80]}")
            else:
                # Tool call
                icon = _tool_icon(entry["name"])
                if verbose_level >= 2 and entry.get("detail"):
                    lines.append(f"{icon} {entry['name']}: {entry['detail']}")
                else:
                    lines.append(f"{icon} {entry['name']}")

        if len(activity_log) > 15:
            lines.insert(1, f"... ({len(activity_log) - 15} earlier entries)\n")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short summary of tool input for verbose level 2."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                # Show just the filename, not the full path
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:60]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:100])[:80]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:60]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:60]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task.

        Sends typing every *interval* seconds, independently of
        stream events. Cancel the returned task in a ``finally``
        block.
        """

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
        mcp_images: Optional[List[ImageAttachment]] = None,
        approved_directory: Optional[Path] = None,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback for verbose progress updates.

        When *mcp_images* is provided, the callback also intercepts
        ``send_image_to_user`` tool calls and collects validated
        :class:`ImageAttachment` objects for later Telegram delivery.

        Returns None when verbose_level is 0 **and** no MCP image
        collection is requested.
        Typing indicators are handled by a separate heartbeat task.
        """
        need_mcp_intercept = mcp_images is not None and approved_directory is not None

        if verbose_level == 0 and not need_mcp_intercept:
            return None

        last_edit_time = [0.0]  # mutable container for closure

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Intercept send_image_to_user MCP tool calls.
            # The SDK namespaces MCP tools as "mcp__<server>__<tool>",
            # so match both the bare name and the namespaced variant.
            if update_obj.tool_calls and need_mcp_intercept:
                for tc in update_obj.tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name == "send_image_to_user" or tc_name.endswith(
                        "__send_image_to_user"
                    ):
                        tc_input = tc.get("input", {})
                        file_path = tc_input.get("file_path", "")
                        caption = tc_input.get("caption", "")
                        img = validate_image_path(
                            file_path, approved_directory, caption
                        )
                        if img:
                            mcp_images.append(img)

            # Capture tool calls for verbose log
            if update_obj.tool_calls and verbose_level >= 1:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    tool_log.append({"kind": "tool", "name": name, "detail": detail})

            # Capture assistant text (reasoning / commentary)
            if update_obj.type == "assistant" and update_obj.content:
                text = update_obj.content.strip()
                if text and verbose_level >= 1:
                    # Collapse to first meaningful line, cap length
                    first_line = text.split("\n", 1)[0].strip()
                    if first_line:
                        tool_log.append({"kind": "text", "detail": first_line[:120]})

            # Throttle progress message edits to avoid Telegram rate limits
            if verbose_level >= 1:
                now = time.time()
                if (now - last_edit_time[0]) >= 2.0 and tool_log:
                    last_edit_time[0] = now
                    new_text = self._format_verbose_progress(
                        tool_log, verbose_level, start_time
                    )
                    try:
                        await progress_msg.edit_text(new_text)
                    except Exception:
                        pass

        return _on_stream

    async def _send_images(
        self,
        update: Update,
        images: List[ImageAttachment],
        reply_to_message_id: Optional[int] = None,
        caption: Optional[str] = None,
        caption_parse_mode: Optional[str] = None,
    ) -> bool:
        """Send extracted images as a media group (album) or documents.

        If *caption* is provided and fits (≤1024 chars), it is attached to the
        photo / first album item so text + images appear as one message.

        Returns True if the caption was successfully embedded in the photo message.
        """
        photos: List[ImageAttachment] = []
        documents: List[ImageAttachment] = []
        for img in images:
            if should_send_as_photo(img.path):
                photos.append(img)
            else:
                documents.append(img)

        # Telegram caption limit
        use_caption = bool(
            caption and len(caption) <= 1024 and photos and not documents
        )
        caption_sent = False

        # Send raster photos as a single album (Telegram groups 2-10 items)
        if photos:
            try:
                if len(photos) == 1:
                    with open(photos[0].path, "rb") as f:
                        await update.message.reply_photo(
                            photo=f,
                            reply_to_message_id=reply_to_message_id,
                            caption=caption if use_caption else None,
                            parse_mode=caption_parse_mode if use_caption else None,
                        )
                    caption_sent = use_caption
                else:
                    media = []
                    file_handles = []
                    for idx, img in enumerate(photos[:10]):
                        fh = open(img.path, "rb")  # noqa: SIM115
                        file_handles.append(fh)
                        media.append(
                            InputMediaPhoto(
                                media=fh,
                                caption=caption if use_caption and idx == 0 else None,
                                parse_mode=(
                                    caption_parse_mode
                                    if use_caption and idx == 0
                                    else None
                                ),
                            )
                        )
                    try:
                        await update.message.chat.send_media_group(
                            media=media,
                            reply_to_message_id=reply_to_message_id,
                        )
                        caption_sent = use_caption
                    finally:
                        for fh in file_handles:
                            fh.close()
            except Exception as e:
                logger.warning("Failed to send photo album", error=str(e))

        # Send SVGs / large files as documents (one by one — can't mix in album)
        for img in documents:
            try:
                with open(img.path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=img.path.name,
                        reply_to_message_id=reply_to_message_id,
                    )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send document image",
                    path=str(img.path),
                    error=str(e),
                )

        return caption_sent

    @staticmethod
    def _parse_resume_index(text: str) -> Optional[int]:
        """Parse a resume selection index from common numeric reply formats."""
        match = re.fullmatch(
            r"\s*(?:第\s*)?(\d+)\s*(?:个|号)?\s*[.。、\)）]?\s*",
            text,
        )
        if not match:
            return None

        return int(match.group(1))

    async def agentic_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Direct Claude passthrough. Simple progress. No suggestions."""
        user_id = update.effective_user.id
        message_text = update.message.text

        # If user sends an index after /resume list, treat it as a resume selection.
        resume_groups = context.user_data.get("_resume_groups")
        resume_index = (
            self._parse_resume_index(message_text) if resume_groups else None
        )
        if resume_groups and resume_index is not None:
            context.user_data["_resume_selection"] = resume_index
            await self._do_resume_session(update, context, resume_index)
            return

        # If waiting for new directory name input
        if context.user_data.get("_awaiting_new_dir"):
            context.user_data.pop("_awaiting_new_dir", None)
            dir_name = message_text.strip()
            if not dir_name or "/" in dir_name or dir_name.startswith("."):
                await update.message.reply_text("❌ 无效目录名，请重新 /new")
                return
            base = self.settings.approved_directory
            new_path = base / dir_name
            try:
                new_path.mkdir(parents=False, exist_ok=True)
            except OSError as e:
                await update.message.reply_text(f"❌ 创建失败: {e}")
                return
            self._reset_session(context, new_path)
            await update.message.reply_text(
                f"📁 Created <b>{escape_html(dir_name)}</b>",
                parse_mode="HTML",
            )
            await self._send_intro(update, context, update.message.chat)
            return

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
        )

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"⏱️ {limit_message}")
                return

        chat = update.message.chat
        await chat.send_action("typing")

        verbose_level = self._get_verbose_level(context)
        progress_msg = await update.message.reply_text("Working...")

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))
        strict_resume = bool(context.user_data.get("_strict_resume_once"))

        # --- Verbose progress tracking via stream callback ---
        tool_log: List[Dict[str, Any]] = []
        start_time = time.time()
        mcp_images: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            start_time,
            mcp_images=mcp_images,
            approved_directory=self.settings.approved_directory,
        )

        # Independent typing heartbeat — stays alive even with no stream events
        heartbeat = self._start_typing_heartbeat(chat)

        success = True
        try:
            claude_response = await claude_integration.run_command(
                prompt=message_text,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                strict_resume=strict_resume,
            )

            # New session created successfully — clear the one-shot flag
            if force_new:
                context.user_data["force_new_session"] = False

            # Only update session_id when Claude returned a valid one.
            # An empty string (e.g. from a failed silent retry) would clobber
            # the previous session_id and cause every subsequent message to
            # start yet another new session.
            if claude_response.session_id:
                context.user_data["claude_session_id"] = claude_response.session_id
            context.user_data.pop("_strict_resume_once", None)

            # Set session title from first user message if not already set
            if not context.user_data.get("_session_title"):
                first_line = message_text.strip().split("\n")[0][:80]
                context.user_data["_session_title"] = first_line

            # Track directory changes
            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            # Format response (no reply_markup — strip keyboards)
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(e), parse_mode="HTML")
            ]
        finally:
            heartbeat.cancel()

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls)
        images: List[ImageAttachment] = mcp_images

        # Try to combine text + images in one message when possible
        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        # Send text messages (skip if caption was already embedded in photos)
        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                try:
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,  # No keyboards in agentic mode
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)
                except Exception as send_err:
                    logger.warning(
                        "Failed to send HTML response, retrying as plain text",
                        error=str(send_err),
                        message_index=i,
                    )
                    try:
                        await update.message.reply_text(
                            message.text,
                            reply_markup=None,
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )
                    except Exception as plain_err:
                        await update.message.reply_text(
                            f"Failed to deliver response "
                            f"(Telegram error: {str(plain_err)[:150]}). "
                            f"Please try again.",
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )

            # Send images separately if caption wasn't used
            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))
        strict_resume = bool(context.user_data.get("_strict_resume_once"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_doc: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_doc,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                strict_resume=strict_resume,
            )

            if force_new:
                context.user_data["force_new_session"] = False

            if claude_response.session_id:
                context.user_data["claude_session_id"] = claude_response.session_id
            context.user_data.pop("_strict_resume_once", None)

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            try:
                await progress_msg.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

            # Use MCP-collected images (from send_image_to_user tool calls)
            images: List[ImageAttachment] = mcp_images_doc

            caption_sent = False
            if images and len(formatted_messages) == 1:
                msg = formatted_messages[0]
                if msg.text and len(msg.text) <= 1024:
                    try:
                        caption_sent = await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                            caption=msg.text,
                            caption_parse_mode=msg.parse_mode,
                        )
                    except Exception as img_err:
                        logger.warning("Image+caption send failed", error=str(img_err))

            if not caption_sent:
                for i, message in enumerate(formatted_messages):
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)

                if images:
                    try:
                        await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                        )
                    except Exception as img_err:
                        logger.warning("Image send failed", error=str(img_err))

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo(s) -> Claude, with media-group aggregation support."""
        if not update.message or not update.message.photo:
            return

        media_group_id = update.message.media_group_id
        if media_group_id:
            await self._enqueue_photo_group(update, context, media_group_id)
            return

        await self._process_photo_batch(
            update=update,
            context=context,
            photos=[update.message.photo[-1]],
            caption=update.message.caption,
        )

    async def _enqueue_photo_group(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        media_group_id: str,
    ) -> None:
        """Buffer Telegram media-group photos and process them as one request."""
        if not update.message or not update.effective_chat or not update.effective_user:
            return

        key = f"{update.effective_chat.id}:{update.effective_user.id}:{media_group_id}"
        state = self._pending_photo_groups.get(key)
        photo = update.message.photo[-1]

        if state is None:
            progress_msg = await update.message.reply_text("📷 已收到多图，正在合并分析…")
            state = {
                "update": update,
                "context": context,
                "photos": [photo],
                "caption": update.message.caption,
                "progress_msg": progress_msg,
                "task": None,
            }
            self._pending_photo_groups[key] = state
        else:
            state["photos"].append(photo)
            if not state.get("caption") and update.message.caption:
                state["caption"] = update.message.caption

        prev_task = state.get("task")
        if prev_task:
            prev_task.cancel()
        state["task"] = asyncio.create_task(self._flush_photo_group(key))

    async def _flush_photo_group(self, key: str) -> None:
        """Wait for media-group completion, then run one combined analysis."""
        try:
            await asyncio.sleep(self._photo_group_buffer_seconds)
        except asyncio.CancelledError:
            return

        state = self._pending_photo_groups.pop(key, None)
        if not state:
            return

        update: Update = state["update"]
        context: ContextTypes.DEFAULT_TYPE = state["context"]
        progress_msg: Message = state["progress_msg"]
        photos: List[PhotoSize] = state["photos"]
        caption: Optional[str] = state.get("caption")

        await self._process_photo_batch(
            update=update,
            context=context,
            photos=photos,
            caption=caption,
            progress_msg=progress_msg,
        )

    def _build_photo_prompt(self, image_paths: List[Path], caption: Optional[str]) -> str:
        """Build a prompt that references uploaded image file paths."""
        count = len(image_paths)
        header = f"我上传了{count}张图片，请基于图片内容回答问题。"
        if caption:
            header += f"\n用户补充说明：{caption}"

        refs = "\n".join(
            f"- 图片{i + 1}: @{path}" for i, path in enumerate(image_paths)
        )
        if count > 1:
            tail = "\n\n请综合比较这些图片后再回答。"
        else:
            tail = "\n\n请直接根据图片内容回答。"
        return f"{header}\n\n图片文件：\n{refs}{tail}"

    async def _download_photos_for_prompt(
        self,
        photos: List[PhotoSize],
        working_directory: Path,
    ) -> List[Path]:
        """Download Telegram photos to working dir and return local file paths."""
        upload_dir = (working_directory / ".telegram_uploads").resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)

        downloaded: List[Path] = []
        for idx, photo in enumerate(photos):
            tg_file = await photo.get_file()
            remote_path = getattr(tg_file, "file_path", "") or ""
            suffix = Path(remote_path).suffix.lower() if remote_path else ""
            if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                suffix = ".jpg"
            local_path = upload_dir / f"photo_{int(time.time() * 1000)}_{idx}_{uuid.uuid4().hex[:8]}{suffix}"
            await tg_file.download_to_drive(str(local_path))
            downloaded.append(local_path)
        return downloaded

    async def _process_photo_batch(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        photos: List[PhotoSize],
        caption: Optional[str] = None,
        progress_msg: Optional[Message] = None,
    ) -> None:
        """Run Claude analysis for one or many uploaded photos."""
        if not update.message:
            return
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        image_handler = features.get_image_handler() if features else None
        if not image_handler:
            await update.message.reply_text("Photo processing is not available.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress = progress_msg or await update.message.reply_text("Working...")

        local_images: List[Path] = []
        try:
            claude_integration = context.bot_data.get("claude_integration")
            if not claude_integration:
                await progress.edit_text(
                    "Claude integration not available. Check configuration."
                )
                return

            current_dir = Path(
                context.user_data.get(
                    "current_directory", self.settings.approved_directory
                )
            )
            local_images = await self._download_photos_for_prompt(photos, current_dir)
            prompt = self._build_photo_prompt(local_images, caption)

            session_id = context.user_data.get("claude_session_id")
            force_new = bool(context.user_data.get("force_new_session"))
            strict_resume = bool(context.user_data.get("_strict_resume_once"))

            verbose_level = self._get_verbose_level(context)
            tool_log: List[Dict[str, Any]] = []
            mcp_images_photo: List[ImageAttachment] = []
            on_stream = self._make_stream_callback(
                verbose_level,
                progress,
                tool_log,
                time.time(),
                mcp_images=mcp_images_photo,
                approved_directory=self.settings.approved_directory,
            )

            heartbeat = self._start_typing_heartbeat(chat)
            try:
                claude_response = await claude_integration.run_command(
                    prompt=prompt,
                    working_directory=current_dir,
                    user_id=user_id,
                    session_id=session_id,
                    on_stream=on_stream,
                    force_new=force_new,
                    strict_resume=strict_resume,
                )
            finally:
                heartbeat.cancel()

            if force_new:
                context.user_data["force_new_session"] = False
            if claude_response.session_id:
                context.user_data["claude_session_id"] = claude_response.session_id
            context.user_data.pop("_strict_resume_once", None)

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            try:
                await progress.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

            images: List[ImageAttachment] = mcp_images_photo
            caption_sent = False
            if images and len(formatted_messages) == 1:
                msg = formatted_messages[0]
                if msg.text and len(msg.text) <= 1024:
                    try:
                        caption_sent = await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                            caption=msg.text,
                            caption_parse_mode=msg.parse_mode,
                        )
                    except Exception as img_err:
                        logger.warning("Image+caption send failed", error=str(img_err))

            if not caption_sent:
                for i, message in enumerate(formatted_messages):
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)

                if images:
                    try:
                        await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                        )
                    except Exception as img_err:
                        logger.warning("Image send failed", error=str(img_err))

        except Exception as e:
            from .handlers.message import _format_error_message

            try:
                await progress.edit_text(_format_error_message(e), parse_mode="HTML")
            except Exception:
                pass
            logger.error(
                "Claude photo processing failed",
                error=str(e),
                user_id=user_id,
                photo_count=len(photos),
            )
        finally:
            for path in local_images:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    logger.debug("Failed to cleanup temp photo", path=str(path))

    async def agentic_resume(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Resume a desktop Claude Code session.

        /resume              — compact view (recent sessions) + project buttons
        /resume all          — full list of all sessions
        /resume <project>    — filter sessions by project name
        /resume <number>     — resume the session at that index
        """
        args = update.message.text.split()[1:] if update.message.text else []
        await update.message.chat.send_action("typing")
        groups = scan_desktop_sessions()
        context.user_data["_resume_groups"] = groups

        if not args:
            # Default: compact view + project inline buttons
            context.user_data["_resume_compact"] = True
            text = format_session_list(groups, compact=True)
            keyboard = self._build_project_keyboard(groups)
            await update.message.reply_text(
                text,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
            return

        arg = args[0]

        # /resume all — full list
        if arg.lower() == "all":
            context.user_data["_resume_compact"] = False
            text = format_session_list(groups, compact=False)
            await update.message.reply_text(text, parse_mode="HTML")
            return

        # /resume <number> — resume by index
        try:
            index = int(arg)
            await self._do_resume_session(update, context, index)
            return
        except ValueError:
            pass

        # /resume <project_name> — filter by project
        group = find_group_by_name(groups, arg)
        if group:
            text = format_project_sessions(group)
            keyboard = self._build_session_keyboard(groups, group)
            await update.message.reply_text(
                text,
                parse_mode="HTML",
                reply_markup=keyboard,
            )
        else:
            await update.message.reply_text(
                f"❌ 未找到项目 \"{escape_html(arg)}\"。\n"
                f"发送 <code>/resume</code> 查看所有项目。",
                parse_mode="HTML",
            )

    def _build_project_keyboard(
        self, groups: List,
    ) -> InlineKeyboardMarkup:
        """Build inline keyboard with project buttons for /resume."""
        buttons = []
        row = []
        for group in groups[:8]:
            name = group.project_name
            if len(name) > 15:
                name = name[:13] + ".."
            icon = "🟢" if group.sessions[0].is_recent else "📂"
            row.append(
                InlineKeyboardButton(
                    f"{icon} {name}",
                    callback_data=f"resume:proj:{group.project_name}",
                )
            )
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        # "All sessions" button
        buttons.append([
            InlineKeyboardButton(
                "📋 全部会话", callback_data="resume:all"
            )
        ])
        return InlineKeyboardMarkup(buttons)

    def _build_session_keyboard(
        self, groups: List, group: "ProjectGroup",
    ) -> InlineKeyboardMarkup:
        """Build inline keyboard for sessions within a project."""
        # Find global index offset for this group
        global_idx = 1
        for g in groups:
            if g.project_name == group.project_name:
                break
            global_idx += len(g.sessions)

        buttons = []
        row = []
        for i, session in enumerate(group.sessions):
            status = "🟢" if session.is_recent else "⚪"
            idx = global_idx + i
            row.append(
                InlineKeyboardButton(
                    f"{status} {idx}",
                    callback_data=f"resume:pick:{idx}",
                )
            )
            if len(row) == 3:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)

        buttons.append([
            InlineKeyboardButton("← 返回", callback_data="resume:back")
        ])
        return InlineKeyboardMarkup(buttons)

    async def _resume_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle resume: inline keyboard callbacks."""
        query = update.callback_query
        await query.answer()
        data = query.data or ""

        groups = context.user_data.get("_resume_groups")
        if not groups:
            groups = scan_desktop_sessions()
            context.user_data["_resume_groups"] = groups

        if data == "resume:all":
            # Show full list
            context.user_data["_resume_compact"] = False
            text = format_session_list(groups, compact=False)
            await query.edit_message_text(text, parse_mode="HTML")

        elif data == "resume:back":
            # Back to compact view with project buttons
            context.user_data["_resume_compact"] = True
            text = format_session_list(groups, compact=True)
            keyboard = self._build_project_keyboard(groups)
            await query.edit_message_text(
                text, parse_mode="HTML", reply_markup=keyboard
            )

        elif data.startswith("resume:proj:"):
            # Drill down into a project
            project_name = data[len("resume:proj:"):]
            group = find_group_by_name(groups, project_name)
            if group:
                text = format_project_sessions(group)
                keyboard = self._build_session_keyboard(groups, group)
                await query.edit_message_text(
                    text, parse_mode="HTML", reply_markup=keyboard
                )

        elif data.startswith("resume:pick:"):
            # Pick a session by global index (buttons always use full-list indices)
            try:
                index = int(data[len("resume:pick:"):])
            except ValueError:
                return
            session = get_session_by_index(groups, index, compact=False)
            if not session:
                await query.edit_message_text("❌ 会话不存在，请重试 /resume")
                return
            # Apply the session and show preview
            result_text = await self._apply_session_resume(
                context, session, index
            )
            await query.edit_message_text(result_text, parse_mode="HTML")

    async def _do_resume_session(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE, index: int
    ) -> None:
        """Resume a desktop session by its 1-based global index (text command)."""
        groups = context.user_data.get("_resume_groups")
        if not groups:
            groups = scan_desktop_sessions()

        compact = context.user_data.get("_resume_compact", False)
        session = get_session_by_index(groups, index, compact=compact)
        if not session:
            await update.message.reply_text(
                f"❌ 编号 {index} 不存在。发送 <code>/resume</code> 查看最新列表。",
                parse_mode="HTML",
            )
            return

        result_text = await self._apply_session_resume(context, session, index)
        await update.message.reply_text(result_text, parse_mode="HTML")

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=update.effective_user.id,
                command="resume",
                args=[str(index), session.session_id[:8]],
                success=True,
            )

    async def _apply_session_resume(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        session: "DesktopSession",
        index: int,
    ) -> str:
        """Apply session resume to context and return formatted result text."""
        from pathlib import Path

        project_path = Path(session.project_path)
        if not project_path.is_dir():
            return (
                f"❌ 项目目录不存在: "
                f"<code>{escape_html(session.project_path)}</code>"
            )

        context.user_data["current_directory"] = project_path
        context.user_data["claude_session_id"] = session.session_id
        context.user_data["_session_title"] = session.title
        context.user_data["force_new_session"] = False
        context.user_data["_strict_resume_once"] = True
        context.user_data.pop("_resume_groups", None)

        title_display = escape_html(session.title)
        project_display = escape_html(session.project_name)

        # Extract recent messages for context preview
        recent_msgs = extract_recent_messages(session.jsonl_path, max_pairs=3)
        preview_lines = []
        if recent_msgs:
            preview_lines.append("\n📝 <b>最近对话：</b>")
            for msg in recent_msgs:
                icon = "👤" if msg.role == "user" else "🤖"
                text = msg.text
                if len(text) > 200:
                    text = text[:200] + "…"
                lines = escape_html(text).split("\n")[:3]
                if len(msg.text.split("\n")) > 3:
                    lines.append("…")
                text_display = "\n".join(lines)
                preview_lines.append(f"{icon} {text_display}")

        preview_section = "\n".join(preview_lines) if preview_lines else ""

        return (
            f"🔗 <b>已接续桌面会话</b>\n\n"
            f"📂 {project_display}\n"
            f"💬 \"{title_display}\"\n"
            f"🕐 {session.age_display}"
            f"{preview_section}\n\n"
            f"现在发消息即可继续这个会话。"
        )

    async def agentic_repo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List repos in workspace or switch to one.

        /repo          — list subdirectories with git indicators
        /repo <name>   — switch to that directory, resume session if available
        """
        args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory
        current_dir = context.user_data.get("current_directory", base)

        if args:
            # Switch to named repo
            target_name = args[0]
            target_path = base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return

            context.user_data["current_directory"] = target_path

            # Try to find a resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id
            context.user_data.pop("_strict_resume_once", None)

            is_git = (target_path / ".git").is_dir()
            git_badge = " (git)" if is_git else ""
            session_badge = " · session resumed" if session_id else ""

            await update.message.reply_text(
                f"Switched to <code>{escape_html(target_name)}/</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
            )
            return

        # No args — list repos
        try:
            entries = sorted(
                [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name,
            )
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        if not entries:
            await update.message.reply_text(
                f"No repos in <code>{escape_html(str(base))}</code>.\n"
                'Clone one by telling me, e.g. <i>"clone org/repo"</i>.',
                parse_mode="HTML",
            )
            return

        lines: List[str] = []
        keyboard_rows: List[list] = []  # type: ignore[type-arg]
        current_name = current_dir.name if current_dir != base else None

        for d in entries:
            is_git = (d / ".git").is_dir()
            icon = "\U0001f4e6" if is_git else "\U0001f4c1"
            marker = " \u25c0" if d.name == current_name else ""
            lines.append(f"{icon} <code>{escape_html(d.name)}/</code>{marker}")

        # Build inline keyboard (2 per row)
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    name = entries[i + j].name
                    row.append(InlineKeyboardButton(name, callback_data=f"cd:{name}"))
            keyboard_rows.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard_rows)

        await update.message.reply_text(
            "<b>Repos</b>\n\n" + "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        new_path = base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Directory not found: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

        context.user_data["current_directory"] = new_path

        # Look for a resumable session instead of always clearing
        claude_integration = context.bot_data.get("claude_integration")
        session_id = None
        if claude_integration:
            existing = await claude_integration._find_resumable_session(
                query.from_user.id, new_path
            )
            if existing:
                session_id = existing.session_id
        context.user_data["claude_session_id"] = session_id
        context.user_data.pop("_strict_resume_once", None)

        is_git = (new_path / ".git").is_dir()
        git_badge = " (git)" if is_git else ""
        session_badge = " · session resumed" if session_id else ""

        await query.edit_message_text(
            f"Switched to <code>{escape_html(project_name)}/</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
        )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="cd",
                args=[project_name],
                success=True,
            )

"""Scanner for desktop Claude Code sessions stored in ~/.claude/projects/."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger()

# Directories to skip (not real coding projects)
_SKIP_DIRS = {
    "-Users-joya--claude-mem-observer-sessions",
}

# User message prefixes to skip when extracting title
_SKIP_TITLE_PREFIXES = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<command-name>",
    "[Request interrupted",
    "You are a",
    "Implement the following plan:",
    "<system-reminder>",
)


@dataclass
class DesktopSession:
    """A Claude Code desktop session."""

    session_id: str
    project_dir: str  # raw directory name in ~/.claude/projects/
    project_name: str  # human-readable project name
    project_path: str  # actual filesystem path (from cwd in jsonl)
    title: str  # first user message (session topic)
    modified_at: datetime
    file_size: int  # bytes
    jsonl_path: str  # full path to .jsonl file

    @property
    def age_display(self) -> str:
        """Human-readable age string."""
        now = datetime.now(timezone.utc)
        modified_utc = (
            self.modified_at.replace(tzinfo=timezone.utc)
            if self.modified_at.tzinfo is None
            else self.modified_at
        )
        delta = now - modified_utc
        seconds = int(delta.total_seconds())

        if seconds < 60:
            return "刚刚"
        elif seconds < 3600:
            return f"{seconds // 60}分钟前"
        elif seconds < 86400:
            return f"{seconds // 3600}小时前"
        else:
            days = seconds // 86400
            return f"{days}天前"

    @property
    def is_recent(self) -> bool:
        """Whether session was active in the last hour."""
        now = datetime.now(timezone.utc)
        modified_utc = (
            self.modified_at.replace(tzinfo=timezone.utc)
            if self.modified_at.tzinfo is None
            else self.modified_at
        )
        delta = now - modified_utc
        return delta.total_seconds() < 3600


@dataclass
class ProjectGroup:
    """A group of sessions under one project."""

    project_name: str
    project_path: str
    sessions: List[DesktopSession] = field(default_factory=list)

    @property
    def latest_modified(self) -> datetime:
        """Most recent session modification time."""
        if not self.sessions:
            return datetime.min.replace(tzinfo=timezone.utc)
        return max(s.modified_at for s in self.sessions)

    @property
    def latest_age_display(self) -> str:
        """Human-readable age of the most recent session."""
        if not self.sessions:
            return "无活动"
        return self.sessions[0].age_display  # sessions are sorted by time


def _extract_session_metadata(jsonl_path: str, max_lines: int = 80) -> Dict:
    """Extract cwd and first user message from a session .jsonl file.

    Returns dict with keys: 'cwd', 'title'
    """
    result = {"cwd": None, "title": "(无标题)"}

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            found_title = False
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Extract cwd from any message that has it
                if result["cwd"] is None and "cwd" in data:
                    result["cwd"] = data["cwd"]

                # Extract title from first meaningful user message
                if not found_title and data.get("type") == "user":
                    title = _extract_text_from_user_message(data)
                    if title:
                        result["title"] = title
                        found_title = True

                # Early exit if we have both
                if result["cwd"] is not None and found_title:
                    break

    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Failed to read session file", path=jsonl_path, error=str(e))

    return result


def _extract_text_from_user_message(data: dict) -> Optional[str]:
    """Extract displayable text from a user-type message.

    Returns None if the message should be skipped (system-injected, etc.)
    """
    message = data.get("message", {})
    content = (
        message.get("content", "") if isinstance(message, dict) else str(message)
    )

    texts_to_check = []

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts_to_check.append(block["text"].strip())
    elif isinstance(content, str) and content.strip():
        texts_to_check.append(content.strip())

    for text in texts_to_check:
        # Skip system-injected / non-meaningful messages
        if any(text.startswith(prefix) for prefix in _SKIP_TITLE_PREFIXES):
            continue
        # Take first line, truncate
        first_line = text.split("\n")[0].strip()
        if first_line:
            return first_line[:100]

    return None


def _cwd_to_project_name(cwd: str) -> str:
    """Convert a cwd path to a human-readable project name.

    Examples:
        /Users/joya/JoyaProjects/Laicai -> Laicai
        /Users/joya/JoyaProjects/vendor-kit -> vendor-kit
        /Users/joya -> ~
        /Users/joya/JoyaProjects -> JoyaProjects
    """
    home = os.path.expanduser("~")
    projects_base = os.path.join(home, "JoyaProjects")

    if cwd.startswith(projects_base + "/"):
        return cwd[len(projects_base) + 1:]
    elif cwd == projects_base:
        return "JoyaProjects"
    elif cwd.startswith(home + "/"):
        return "~/" + cwd[len(home) + 1:]
    elif cwd == home:
        return "~"
    return cwd


def _dir_name_to_fallback_path(dir_name: str) -> str:
    """Fallback: convert directory name to a path when cwd is unavailable."""
    if dir_name.startswith("-"):
        return "/" + dir_name[1:].replace("-", "/")
    return dir_name


def scan_desktop_sessions(
    claude_home: Optional[str] = None,
    max_sessions_per_project: int = 5,
    skip_dirs: Optional[set] = None,
) -> List[ProjectGroup]:
    """Scan ~/.claude/projects/ for desktop Claude Code sessions.

    Returns a list of ProjectGroups sorted by most recent activity (newest first).
    Each group's sessions are also sorted by modification time (newest first).
    """
    if claude_home is None:
        claude_home = os.path.expanduser("~/.claude")

    projects_dir = os.path.join(claude_home, "projects")

    if not os.path.isdir(projects_dir):
        logger.warning("Claude projects directory not found", path=projects_dir)
        return []

    if skip_dirs is None:
        skip_dirs = _SKIP_DIRS

    groups: Dict[str, ProjectGroup] = {}

    for dir_name in os.listdir(projects_dir):
        dir_path = os.path.join(projects_dir, dir_name)

        if not os.path.isdir(dir_path):
            continue
        if dir_name in skip_dirs:
            continue

        # Find all .jsonl session files
        jsonl_files = []
        for fname in os.listdir(dir_path):
            if fname.endswith(".jsonl"):
                full_path = os.path.join(dir_path, fname)
                size = os.path.getsize(full_path)
                if size == 0:
                    continue  # skip empty sessions
                jsonl_files.append((full_path, fname, size))

        if not jsonl_files:
            continue

        # Sort by modification time (newest first)
        jsonl_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)

        # Limit sessions per project
        jsonl_files = jsonl_files[:max_sessions_per_project]

        # Determine project name and path from the most recent session's cwd
        project_path = None
        project_name = None

        sessions = []
        for full_path, fname, size in jsonl_files:
            session_id = fname.replace(".jsonl", "")
            mtime = datetime.fromtimestamp(
                os.path.getmtime(full_path), tz=timezone.utc
            )
            metadata = _extract_session_metadata(full_path)

            # Use cwd from most recent session for the project group
            if project_path is None and metadata["cwd"]:
                project_path = metadata["cwd"]
                project_name = _cwd_to_project_name(project_path)

            sessions.append(
                DesktopSession(
                    session_id=session_id,
                    project_dir=dir_name,
                    project_name=project_name or dir_name,
                    project_path=project_path
                    or _dir_name_to_fallback_path(dir_name),
                    title=metadata["title"],
                    modified_at=mtime,
                    file_size=size,
                    jsonl_path=full_path,
                )
            )

        # Fallback if no cwd found in any session
        if project_path is None:
            project_path = _dir_name_to_fallback_path(dir_name)
            project_name = dir_name

        if sessions:
            group = ProjectGroup(
                project_name=project_name or dir_name,
                project_path=project_path,
                sessions=sessions,
            )
            groups[dir_name] = group

    # Sort groups by most recent session activity (newest first)
    sorted_groups = sorted(
        groups.values(), key=lambda g: g.latest_modified, reverse=True
    )

    return sorted_groups


def format_session_list(
    groups: List[ProjectGroup], compact: bool = False
) -> str:
    """Format session groups into a Telegram-friendly message.

    Args:
        groups: List of ProjectGroups to display.
        compact: If True, show only the most recent session per project
                 (for the default /resume view). If False, show all sessions.

    Returns a numbered list where each session has a global index
    that can be used with /resume <number>.
    """
    if not groups:
        return "🔍 未找到桌面 Claude Code 会话。"

    if compact:
        return _format_compact_list(groups)
    return _format_full_list(groups)


def _format_compact_list(groups: List[ProjectGroup]) -> str:
    """Compact view: one line per most-recent session, max 8 entries."""
    lines = ["🖥 <b>最近会话</b>\n"]
    global_index = 1

    # Flatten: take the most recent session from each project, sort globally
    recent_sessions = []
    for group in groups:
        if group.sessions:
            recent_sessions.append((group, group.sessions[0]))

    # Already sorted by group.latest_modified (newest first), take top 8
    recent_sessions = recent_sessions[:8]

    for group, session in recent_sessions:
        status = "🟢" if session.is_recent else "⚪"
        project = _escape_html(group.project_name)
        title = _escape_html(session.title)
        if len(title) > 40:
            title = title[:40] + "…"
        lines.append(
            f"{global_index}. {status} <b>{project}</b>"
            f"\n    \"{title}\" · {session.age_display}"
        )
        global_index += 1

    lines.append("")
    lines.append("💡 输入编号接续 · 点击下方按钮查看更多")

    return "\n".join(lines)


def _format_full_list(groups: List[ProjectGroup]) -> str:
    """Full view: all sessions grouped by project."""
    lines = ["🖥 <b>全部桌面会话</b>\n"]
    global_index = 1

    for group in groups:
        icon = "🟢" if group.sessions[0].is_recent else "📂"
        lines.append(
            f"{icon} <b>{_escape_html(group.project_name)}</b>"
            f"（{group.latest_age_display}）"
        )

        for session in group.sessions:
            status = "🟢" if session.is_recent else "⚪"
            size_str = _format_size(session.file_size)
            title_display = _escape_html(session.title)
            lines.append(
                f"  {global_index}. {status} \"{title_display}\""
                f"\n      {session.age_display} · {size_str}"
            )
            global_index += 1

        lines.append("")  # blank line between groups

    lines.append("💡 发送 <code>/resume 编号</code> 接续对应会话")

    return "\n".join(lines)


def format_project_sessions(group: ProjectGroup) -> str:
    """Format a single project's sessions for drill-down view."""
    lines = [
        f"📂 <b>{_escape_html(group.project_name)}</b> 的会话\n"
    ]
    for i, session in enumerate(group.sessions, 1):
        status = "🟢" if session.is_recent else "⚪"
        title_display = _escape_html(session.title)
        size_str = _format_size(session.file_size)
        lines.append(
            f"{i}. {status} \"{title_display}\""
            f"\n    {session.age_display} · {size_str}"
        )
    lines.append("")
    lines.append("💡 点击下方按钮接续对应会话")
    return "\n".join(lines)


def find_group_by_name(
    groups: List[ProjectGroup], name: str
) -> Optional[ProjectGroup]:
    """Find a project group by name (case-insensitive partial match)."""
    name_lower = name.lower()
    # Exact match first
    for group in groups:
        if group.project_name.lower() == name_lower:
            return group
    # Partial match
    for group in groups:
        if name_lower in group.project_name.lower():
            return group
    return None


def get_session_by_index(
    groups: List[ProjectGroup], index: int
) -> Optional[DesktopSession]:
    """Get a session by its 1-based global index in the formatted list."""
    current = 1
    for group in groups:
        for session in group.sessions:
            if current == index:
                return session
            current += 1
    return None


@dataclass
class SessionMessage:
    """A single message extracted from a session .jsonl file."""

    role: str  # "user" or "assistant"
    text: str


def extract_recent_messages(
    jsonl_path: str, max_pairs: int = 3
) -> List[SessionMessage]:
    """Extract the last N user-assistant text exchanges from a .jsonl session file.

    Skips tool_result/tool_use messages, only extracts human-readable text.
    Returns a list of SessionMessage ordered chronologically (oldest first).
    """
    messages: List[SessionMessage] = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                message = data.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    continue

                # --- User text message ---
                if msg_type == "user" and message.get("role") == "user":
                    text = _extract_plain_text(content)
                    if text:
                        messages.append(SessionMessage(role="user", text=text))

                # --- Assistant text message ---
                elif msg_type != "user" and msg_type not in (
                    "progress",
                    "queue-operation",
                    "agent-setting",
                ):
                    text = _extract_plain_text(content)
                    if text:
                        messages.append(
                            SessionMessage(role="assistant", text=text)
                        )

    except (OSError, UnicodeDecodeError) as e:
        logger.debug(
            "Failed to extract recent messages",
            path=jsonl_path,
            error=str(e),
        )

    # Return last N*2 messages (N pairs of user+assistant)
    return messages[-(max_pairs * 2) :]


def _extract_plain_text(content: list) -> Optional[str]:
    """Extract human-readable text from a message content list.

    Skips tool_result and tool_use blocks. Returns None if no text found.
    """
    texts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        # Skip tool_result and tool_use blocks
        if block.get("type") in ("tool_result", "tool_use"):
            continue
        # Extract text blocks
        text = block.get("text", "")
        if isinstance(text, str) and text.strip():
            # Skip system-injected content
            stripped = text.strip()
            if any(stripped.startswith(p) for p in _SKIP_TITLE_PREFIXES):
                continue
            texts.append(stripped)

    if not texts:
        return None

    combined = "\n".join(texts)
    return combined if combined.strip() else None


def find_session_title_by_id(session_id: str) -> Optional[str]:
    """Look up a session title from desktop JSONL files by session ID.

    Useful for populating title when session was not resumed via /resume.
    """
    claude_home = os.path.expanduser("~/.claude")
    projects_dir = os.path.join(claude_home, "projects")

    if not os.path.isdir(projects_dir):
        return None

    target_fname = f"{session_id}.jsonl"

    for dir_name in os.listdir(projects_dir):
        dir_path = os.path.join(projects_dir, dir_name)
        if not os.path.isdir(dir_path):
            continue
        candidate = os.path.join(dir_path, target_fname)
        if os.path.isfile(candidate):
            metadata = _extract_session_metadata(candidate)
            title = metadata.get("title")
            return title if title and title != "(无标题)" else None

    return None


def _format_size(size_bytes: int) -> str:
    """Format file size as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

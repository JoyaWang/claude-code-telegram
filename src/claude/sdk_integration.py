"""Claude Code Python SDK integration."""

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import json

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    Message,
    PermissionResultAllow,
    PermissionResultDeny,
    ProcessError,
    ResultMessage,
    ToolPermissionContext,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk._errors import MessageParseError
from claude_agent_sdk._internal.message_parser import parse_message

from ..config.settings import Settings
from ..security.validators import SecurityValidator
from .exceptions import (
    ClaudeMCPError,
    ClaudeParsingError,
    ClaudeProcessError,
    ClaudeTimeoutError,
)
from .monitor import _is_claude_internal_path, check_bash_directory_boundary

logger = structlog.get_logger()


@dataclass
class ClaudeResponse:
    """Response from Claude Code SDK."""

    content: str
    session_id: str
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool = False
    error_type: Optional[str] = None
    tools_used: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamUpdate:
    """Streaming update from Claude SDK."""

    type: str  # 'assistant', 'user', 'system', 'result'
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


def _strip_empty_cli_flag(command: List[str], flag: str) -> List[str]:
    """Remove CLI flag/value pairs where value is the empty string."""
    sanitized: List[str] = []
    i = 0
    while i < len(command):
        if (
            command[i] == flag
            and i + 1 < len(command)
            and command[i + 1] == ""
        ):
            i += 2
            continue
        sanitized.append(command[i])
        i += 1
    return sanitized


def _sanitize_sdk_cli_command(command: List[str]) -> List[str]:
    """Strip SDK-emitted empty CLI args that break upstream validation."""
    command = _strip_empty_cli_flag(command, "--system-prompt")
    command = _strip_empty_cli_flag(command, "--setting-sources")
    return command


def _make_can_use_tool_callback(
    security_validator: SecurityValidator,
    working_directory: Path,
    approved_directory: Path,
) -> Any:
    """Create a can_use_tool callback for SDK-level tool permission validation.

    The callback validates file path boundaries and bash directory boundaries
    *before* the SDK executes the tool, providing preventive security enforcement.
    """
    _FILE_TOOLS = {"Write", "Edit", "Read", "create_file", "edit_file", "read_file"}
    _BASH_TOOLS = {"Bash", "bash", "shell"}

    async def can_use_tool(
        tool_name: str,
        tool_input: Dict[str, Any],
        context: ToolPermissionContext,
    ) -> Any:
        # File path validation
        if tool_name in _FILE_TOOLS:
            file_path = tool_input.get("file_path") or tool_input.get("path")
            if file_path:
                # Allow Claude Code internal paths (~/.claude/plans/, etc.)
                if _is_claude_internal_path(file_path):
                    return PermissionResultAllow()

                valid, _resolved, error = security_validator.validate_path(
                    file_path, working_directory
                )
                if not valid:
                    logger.warning(
                        "can_use_tool denied file operation",
                        tool_name=tool_name,
                        file_path=file_path,
                        error=error,
                    )
                    return PermissionResultDeny(message=error or "Invalid file path")

        # Bash directory boundary validation
        if tool_name in _BASH_TOOLS:
            command = tool_input.get("command", "")
            if command:
                valid, error = check_bash_directory_boundary(
                    command, working_directory, approved_directory
                )
                if not valid:
                    logger.warning(
                        "can_use_tool denied bash command",
                        tool_name=tool_name,
                        command=command,
                        error=error,
                    )
                    return PermissionResultDeny(
                        message=error or "Bash directory boundary violation"
                    )

        return PermissionResultAllow()

    return can_use_tool


class ClaudeSDKManager:
    """Manage Claude Code SDK integration."""

    # Env vars from ~/.claude/settings.json that should be synced
    _CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")
    _SDK_CLI_PATCH_APPLIED = False

    def __init__(
        self,
        config: Settings,
        security_validator: Optional[SecurityValidator] = None,
    ):
        """Initialize SDK manager with configuration."""
        self.config = config
        self.security_validator = security_validator
        self._patch_sdk_empty_cli_args()

        # Set up environment for Claude Code SDK
        # Priority: .env ANTHROPIC_API_KEY > ~/.claude/settings.json env section
        if config.anthropic_api_key_str:
            os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key_str
            logger.info("Using provided API key for Claude SDK authentication")
        else:
            # Sync env vars from ~/.claude/settings.json (cc-switch compatible)
            self._sync_claude_settings_env()

    @classmethod
    def _patch_sdk_empty_cli_args(cls) -> None:
        """Patch SDK CLI command builder to avoid empty-string flags.

        claude-agent-sdk 0.1.44 renders None as:
          --system-prompt ""
          --setting-sources ""
        which can trigger upstream "invalid claude code request" on resume.
        """
        if cls._SDK_CLI_PATCH_APPLIED:
            return

        try:
            from claude_agent_sdk._internal.transport.subprocess_cli import (
                SubprocessCLITransport,
            )
        except Exception as e:
            logger.warning(
                "Failed to import SDK subprocess transport for CLI patch",
                error=str(e),
            )
            return

        original_build_command = SubprocessCLITransport._build_command

        def _patched_build_command(self: Any) -> List[str]:
            command = original_build_command(self)
            return _sanitize_sdk_cli_command(command)

        SubprocessCLITransport._build_command = _patched_build_command  # type: ignore[assignment]
        cls._SDK_CLI_PATCH_APPLIED = True
        logger.info("Applied SDK CLI empty-argument patch")

    @classmethod
    def _sync_claude_settings_env(cls) -> None:
        """Read ~/.claude/settings.json and apply its env section to os.environ.

        This ensures cc-switch configuration changes are picked up dynamically.
        Called before each CLI invocation for live sync.
        """
        # Prevent "nested session" error when bot is launched from Claude Code
        os.environ.pop("CLAUDECODE", None)

        try:
            with open(cls._CLAUDE_SETTINGS_PATH, "r", encoding="utf-8") as f:
                settings = json.load(f)
            env_section = settings.get("env", {})
            if env_section:
                for key, value in env_section.items():
                    os.environ[key] = str(value)
                logger.info(
                    "Synced env from ~/.claude/settings.json",
                    keys=list(env_section.keys()),
                )
            else:
                logger.info(
                    "No env section in ~/.claude/settings.json, "
                    "using existing CLI authentication"
                )
        except FileNotFoundError:
            logger.warning("~/.claude/settings.json not found")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Failed to read ~/.claude/settings.json",
                error=str(e),
            )

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable[[StreamUpdate], None]] = None,
    ) -> ClaudeResponse:
        """Execute Claude Code command via SDK."""
        start_time = asyncio.get_event_loop().time()

        # Re-sync env from settings.json before each call (cc-switch live reload)
        if not self.config.anthropic_api_key_str:
            self._sync_claude_settings_env()

        logger.info(
            "Starting Claude SDK command",
            working_directory=str(working_directory),
            session_id=session_id,
            continue_session=continue_session,
        )

        try:
            # Resume requests can hang for many minutes on upstream/API issues.
            # Cap the effective timeout so Telegram users don't wait excessively.
            timeout_seconds = self.config.claude_timeout_seconds
            if continue_session:
                timeout_seconds = min(timeout_seconds, 180)

            # Capture stderr from Claude CLI for better error diagnostics
            stderr_lines: List[str] = []

            def _stderr_callback(line: str) -> None:
                stderr_lines.append(line)
                logger.debug("Claude CLI stderr", line=line)

            # Build system prompt, loading CLAUDE.md from working directory if present
            base_prompt = (
                f"All file operations must stay within {working_directory}. "
                "Use relative paths."
            )
            claude_md_path = Path(working_directory) / "CLAUDE.md"
            if claude_md_path.exists():
                base_prompt += "\n\n" + claude_md_path.read_text(encoding="utf-8")
                logger.info(
                    "Loaded CLAUDE.md into system prompt",
                    path=str(claude_md_path),
                )

            is_resuming = bool(session_id and continue_session)

            # Resume flows are unstable with empty CLI values
            # (--system-prompt "" / --setting-sources ""). Prefer explicit
            # prompt and setting sources for deterministic initialization.
            use_agent = bool(self.config.claude_default_agent and not is_resuming)
            if use_agent:
                system_prompt_value = {"type": "preset", "append": base_prompt}
            else:
                system_prompt_value = base_prompt

            # When DISABLE_TOOL_VALIDATION=true, pass None for allowed/disallowed
            # tools so the SDK does not restrict tool usage (e.g. MCP tools).
            if self.config.disable_tool_validation:
                sdk_allowed_tools = None
                sdk_disallowed_tools = None
            else:
                sdk_allowed_tools = self.config.claude_allowed_tools
                sdk_disallowed_tools = self.config.claude_disallowed_tools

            # Build extra CLI args
            # Skip --agent when resuming: the resumed session already has its
            # agent/system-prompt context embedded.  Passing --agent on top of
            # --resume causes "Control request timeout: initialize" for desktop
            # sessions that were created without an agent flag.
            extra_args: Dict[str, str | None] = {}
            if self.config.claude_default_agent and not is_resuming:
                extra_args["agent"] = self.config.claude_default_agent

            # Keep setting sources explicit for both new and resumed sessions.
            setting_sources = ["user", "project"]

            logger.info(
                "SDK options",
                extra_args=extra_args,
                system_prompt_type=type(system_prompt_value).__name__,
                default_agent=self.config.claude_default_agent,
                setting_sources=setting_sources,
            )

            # Build Claude Agent options
            options = ClaudeAgentOptions(
                max_turns=self.config.claude_max_turns,
                max_budget_usd=self.config.claude_max_cost_per_request,
                cwd=str(working_directory),
                allowed_tools=sdk_allowed_tools,
                disallowed_tools=sdk_disallowed_tools,
                cli_path=self.config.claude_cli_path or None,
                sandbox={
                    "enabled": self.config.sandbox_enabled,
                    "autoAllowBashIfSandboxed": True,
                    "excludedCommands": self.config.sandbox_excluded_commands or [],
                },
                system_prompt=system_prompt_value,
                setting_sources=setting_sources,
                stderr=_stderr_callback,
                extra_args=extra_args,
            )

            # Pass MCP server configuration if enabled
            if self.config.enable_mcp and self.config.mcp_config_path:
                options.mcp_servers = self._load_mcp_config(self.config.mcp_config_path)
                logger.info(
                    "MCP servers configured",
                    mcp_config_path=str(self.config.mcp_config_path),
                )

            # Wire can_use_tool callback for preventive tool validation
            if self.security_validator:
                options.can_use_tool = _make_can_use_tool_callback(
                    security_validator=self.security_validator,
                    working_directory=working_directory,
                    approved_directory=self.config.approved_directory,
                )

            # Resume previous session if we have a session_id
            if session_id and continue_session:
                options.resume = session_id
                logger.info(
                    "Resuming previous session",
                    session_id=session_id,
                )

            # Collect messages via ClaudeSDKClient
            messages: List[Message] = []

            async def _run_client() -> None:
                # Use connect(None) + query(prompt) pattern because
                # can_use_tool requires the prompt as AsyncIterable, not
                # a plain string. connect(None) uses an empty async
                # iterable internally, satisfying the requirement.
                client = ClaudeSDKClient(options)
                try:
                    await client.connect()
                    await client.query(prompt)

                    # Iterate over raw messages and parse them ourselves
                    # so that MessageParseError (e.g. from rate_limit_event)
                    # doesn't kill the underlying async generator. When
                    # parse_message raises inside the SDK's receive_messages()
                    # generator, Python terminates that generator permanently,
                    # causing us to lose all subsequent messages including
                    # the ResultMessage.
                    async for raw_data in client._query.receive_messages():
                        try:
                            message = parse_message(raw_data)
                        except MessageParseError as e:
                            logger.debug(
                                "Skipping unparseable message",
                                error=str(e),
                            )
                            continue

                        messages.append(message)

                        if isinstance(message, ResultMessage):
                            break

                        # Handle streaming callback
                        if stream_callback:
                            try:
                                await self._handle_stream_message(
                                    message, stream_callback
                                )
                            except Exception as callback_error:
                                logger.warning(
                                    "Stream callback failed",
                                    error=str(callback_error),
                                    error_type=type(callback_error).__name__,
                                )
                finally:
                    await client.disconnect()

            # Execute with timeout
            await asyncio.wait_for(
                _run_client(),
                timeout=timeout_seconds,
            )

            # Extract cost, tools, and session_id from result message
            cost = 0.0
            tools_used: List[Dict[str, Any]] = []
            claude_session_id = None
            result_content = None
            result_is_error = False
            result_error_type = None
            for message in messages:
                if isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", 0.0) or 0.0
                    claude_session_id = getattr(message, "session_id", None)
                    result_content = getattr(message, "result", None)
                    result_is_error = bool(getattr(message, "is_error", False))
                    result_error_type = getattr(message, "subtype", None)
                    current_time = asyncio.get_event_loop().time()
                    for msg in messages:
                        if isinstance(msg, AssistantMessage):
                            msg_content = getattr(msg, "content", [])
                            if msg_content and isinstance(msg_content, list):
                                for block in msg_content:
                                    if isinstance(block, ToolUseBlock):
                                        tools_used.append(
                                            {
                                                "name": getattr(
                                                    block, "name", "unknown"
                                                ),
                                                "timestamp": current_time,
                                                "input": getattr(block, "input", {}),
                                            }
                                        )
                    break

            # Fallback: extract session_id from StreamEvent messages if
            # ResultMessage didn't provide one (can happen with some CLI versions)
            if not claude_session_id:
                for message in messages:
                    msg_session_id = getattr(message, "session_id", None)
                    if msg_session_id and not isinstance(message, ResultMessage):
                        claude_session_id = msg_session_id
                        logger.info(
                            "Got session ID from stream event (fallback)",
                            session_id=claude_session_id,
                        )
                        break

            # Calculate duration
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Use Claude's session_id if available, otherwise fall back
            final_session_id = claude_session_id or session_id or ""

            if claude_session_id and claude_session_id != session_id:
                logger.info(
                    "Got session ID from Claude",
                    claude_session_id=claude_session_id,
                    previous_session_id=session_id,
                )

            # Use ResultMessage.result if available, fall back to message extraction
            if result_content is not None:
                content = result_content
            else:
                content_parts = []
                for msg in messages:
                    if isinstance(msg, AssistantMessage):
                        msg_content = getattr(msg, "content", [])
                        if msg_content and isinstance(msg_content, list):
                            for block in msg_content:
                                if hasattr(block, "text"):
                                    content_parts.append(block.text)
                        elif msg_content:
                            content_parts.append(str(msg_content))
                content = "\n".join(content_parts)

            # Some Claude backends return transport/API failures as plain text in
            # ResultMessage.result while is_error stays false. Treat these as
            # execution failures so callers can retry instead of displaying them
            # as a normal assistant reply.
            content_text = content.strip() if isinstance(content, str) else str(content)
            looks_like_api_error = bool(
                re.match(r"^(API Error|Error):", content_text, flags=re.IGNORECASE)
            ) and '"type":"error"' in content_text
            if result_is_error or looks_like_api_error:
                detail = content_text[:320]
                logger.error(
                    "Claude returned error result",
                    session_id=final_session_id or session_id,
                    error_type=result_error_type,
                    is_error=result_is_error,
                    detail=detail,
                )
                raise ClaudeProcessError(
                    f"Claude returned an error response: {detail}"
                )

            return ClaudeResponse(
                content=content,
                session_id=final_session_id,
                cost=cost,
                duration_ms=duration_ms,
                num_turns=len(
                    [
                        m
                        for m in messages
                        if isinstance(m, (UserMessage, AssistantMessage))
                    ]
                ),
                is_error=result_is_error,
                error_type=result_error_type,
                tools_used=tools_used,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Claude SDK command timed out",
                timeout_seconds=timeout_seconds,
            )
            raise ClaudeTimeoutError(
                f"Claude SDK timed out after {timeout_seconds}s"
            )

        except CLINotFoundError as e:
            logger.error("Claude CLI not found", error=str(e))
            error_msg = (
                "Claude Code not found. Please ensure Claude is installed:\n"
                "  npm install -g @anthropic-ai/claude-code\n\n"
                "If already installed, try one of these:\n"
                "  1. Add Claude to your PATH\n"
                "  2. Create a symlink: ln -s $(which claude) /usr/local/bin/claude\n"
                "  3. Set CLAUDE_CLI_PATH environment variable"
            )
            raise ClaudeProcessError(error_msg)

        except ProcessError as e:
            error_str = str(e)
            # Include captured stderr for better diagnostics
            captured_stderr = "\n".join(stderr_lines[-20:]) if stderr_lines else ""
            if captured_stderr:
                error_str = f"{error_str}\nStderr: {captured_stderr}"
            logger.error(
                "Claude process failed",
                error=error_str,
                exit_code=getattr(e, "exit_code", None),
                stderr=captured_stderr or None,
            )
            # Check if the process error is MCP-related
            if "mcp" in error_str.lower():
                raise ClaudeMCPError(f"MCP server error: {error_str}")
            raise ClaudeProcessError(f"Claude process error: {error_str}")

        except CLIConnectionError as e:
            error_str = str(e)
            logger.error("Claude connection error", error=error_str)
            # Check if the connection error is MCP-related
            if "mcp" in error_str.lower() or "server" in error_str.lower():
                raise ClaudeMCPError(f"MCP server connection failed: {error_str}")
            raise ClaudeProcessError(f"Failed to connect to Claude: {error_str}")

        except CLIJSONDecodeError as e:
            logger.error("Claude SDK JSON decode error", error=str(e))
            raise ClaudeParsingError(f"Failed to decode Claude response: {str(e)}")

        except ClaudeSDKError as e:
            logger.error("Claude SDK error", error=str(e))
            raise ClaudeProcessError(f"Claude SDK error: {str(e)}")

        except ClaudeProcessError:
            # Preserve already-classified process errors without double wrapping.
            raise

        except Exception as e:
            exceptions = getattr(e, "exceptions", None)
            if exceptions is not None:
                # ExceptionGroup from TaskGroup operations (Python 3.11+)
                logger.error(
                    "Task group error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                    exception_count=len(exceptions),
                    exceptions=[str(ex) for ex in exceptions[:3]],
                )
                raise ClaudeProcessError(
                    f"Claude SDK task error: {exceptions[0] if exceptions else e}"
                )

            logger.error(
                "Unexpected error in Claude SDK",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ClaudeProcessError(f"Unexpected error: {str(e)}")

    async def _handle_stream_message(
        self, message: Message, stream_callback: Callable[[StreamUpdate], None]
    ) -> None:
        """Handle streaming message from claude-agent-sdk."""
        try:
            if isinstance(message, AssistantMessage):
                # Extract content from assistant message
                content = getattr(message, "content", [])
                text_parts = []
                tool_calls = []

                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "name": getattr(block, "name", "unknown"),
                                    "input": getattr(block, "input", {}),
                                    "id": getattr(block, "id", None),
                                }
                            )
                        elif hasattr(block, "text"):
                            text_parts.append(block.text)

                if text_parts or tool_calls:
                    update = StreamUpdate(
                        type="assistant",
                        content=("\n".join(text_parts) if text_parts else None),
                        tool_calls=tool_calls if tool_calls else None,
                    )
                    await stream_callback(update)
                elif content:
                    # Fallback for non-list content
                    update = StreamUpdate(
                        type="assistant",
                        content=str(content),
                    )
                    await stream_callback(update)

            elif isinstance(message, UserMessage):
                content = getattr(message, "content", "")
                if content:
                    update = StreamUpdate(
                        type="user",
                        content=content,
                    )
                    await stream_callback(update)

        except Exception as e:
            logger.warning("Stream callback failed", error=str(e))

    def _load_mcp_config(self, config_path: Path) -> Dict[str, Any]:
        """Load MCP server configuration from a JSON file.

        The new claude-agent-sdk expects mcp_servers as a dict, not a file path.
        """
        import json

        try:
            with open(config_path) as f:
                config_data = json.load(f)
            return config_data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "Failed to load MCP config", path=str(config_path), error=str(e)
            )
            return {}

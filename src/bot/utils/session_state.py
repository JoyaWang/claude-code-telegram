"""Helpers for resetting Claude session state consistently."""

from typing import Any, MutableMapping


def reset_claude_session_state(
    user_data: MutableMapping[str, Any],
    *,
    session_started: bool,
    clear_last_message: bool = False,
) -> None:
    """Clear the active session and force the next message to start fresh."""
    user_data["claude_session_id"] = None
    user_data["session_started"] = session_started
    user_data["force_new_session"] = True

    if clear_last_message:
        user_data["last_message"] = None

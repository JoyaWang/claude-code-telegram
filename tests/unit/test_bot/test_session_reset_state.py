"""Regression tests for explicit session reset flows."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bot.handlers import callback, command
from src.config import create_test_config


@pytest.fixture
def settings(tmp_path: Path):
    approved = tmp_path / "projects"
    approved.mkdir()
    return create_test_config(approved_directory=str(approved))


async def test_new_session_command_forces_fresh_session(settings):
    """/new should prevent the next message from auto-resuming."""
    update = MagicMock()
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"settings": settings}
    context.user_data = {
        "current_directory": settings.approved_directory,
        "claude_session_id": "old-session",
        "force_new_session": False,
    }

    await command.new_session(update, context)

    assert context.user_data["claude_session_id"] is None
    assert context.user_data["session_started"] is True
    assert context.user_data["force_new_session"] is True


async def test_end_session_command_forces_fresh_session(settings):
    """/end should not allow the next message to auto-resume the old session."""
    update = MagicMock()
    update.effective_user.id = 1
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"settings": settings}
    context.user_data = {
        "current_directory": settings.approved_directory,
        "claude_session_id": "old-session",
        "force_new_session": False,
        "last_message": "stale",
    }

    await command.end_session(update, context)

    assert context.user_data["claude_session_id"] is None
    assert context.user_data["session_started"] is False
    assert context.user_data["force_new_session"] is True
    assert context.user_data["last_message"] is None


async def test_new_session_callback_forces_fresh_session(settings):
    """New Session button should behave like /new, not auto-resume."""
    query = MagicMock()
    query.edit_message_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"settings": settings}
    context.user_data = {
        "current_directory": settings.approved_directory,
        "claude_session_id": "old-session",
        "force_new_session": False,
    }

    await callback.handle_action_callback(query, "new_session", context)

    assert context.user_data["claude_session_id"] is None
    assert context.user_data["session_started"] is True
    assert context.user_data["force_new_session"] is True


async def test_end_session_callback_forces_fresh_session(settings):
    """End Session button should clear state and force a fresh next turn."""
    query = MagicMock()
    query.edit_message_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"settings": settings}
    context.user_data = {
        "current_directory": settings.approved_directory,
        "claude_session_id": "old-session",
        "force_new_session": False,
        "last_message": "stale",
    }

    await callback.handle_action_callback(query, "end_session", context)

    assert context.user_data["claude_session_id"] is None
    assert context.user_data["session_started"] is False
    assert context.user_data["force_new_session"] is True
    assert context.user_data["last_message"] is None


async def test_conversation_end_callback_forces_fresh_session(settings):
    """Conversation end flow should also block implicit auto-resume."""
    query = MagicMock()
    query.from_user.id = 1
    query.edit_message_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"settings": settings}
    context.user_data = {
        "current_directory": settings.approved_directory,
        "claude_session_id": "old-session",
        "force_new_session": False,
    }

    await callback.handle_conversation_callback(query, "end", context)

    assert context.user_data["claude_session_id"] is None
    assert context.user_data["session_started"] is False
    assert context.user_data["force_new_session"] is True

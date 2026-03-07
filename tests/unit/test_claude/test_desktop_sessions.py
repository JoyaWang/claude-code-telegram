"""Tests for desktop session scanning."""

import json
import os
from pathlib import Path

from src.claude.desktop_sessions import scan_desktop_sessions


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_scan_desktop_sessions_uses_older_cwd_when_newest_missing(tmp_path: Path):
    """Newest file without cwd should still inherit project path from older files."""
    claude_home = tmp_path / "claude_home"
    project_dir = (
        claude_home
        / "projects"
        / "-Users-joya-JoyaProjects-claude-code-telegram"
    )
    project_dir.mkdir(parents=True)

    newest = project_dir / "9700566c-7ffa-4f24-ae19-88708c43cff7.jsonl"
    older = project_dir / "ea2c1a92-3e3d-4f6b-b37c-ddd40c8aa618.jsonl"

    _write_jsonl(
        newest,
        [
            {
                "type": "user",
                "message": {"content": "继续"},
            }
        ],
    )
    _write_jsonl(
        older,
        [
            {"cwd": "/Users/joya/JoyaProjects/claude-code-telegram"},
            {
                "type": "user",
                "message": {"content": "修复 bug"},
            },
        ],
    )

    # Ensure newest is sorted first by modification time.
    now = int(os.path.getmtime(newest))
    os.utime(older, (now - 100, now - 100))
    os.utime(newest, (now, now))

    groups = scan_desktop_sessions(
        claude_home=str(claude_home),
        max_sessions_per_project=5,
        skip_dirs=set(),
    )

    assert len(groups) == 1
    group = groups[0]
    expected_path = "/Users/joya/JoyaProjects/claude-code-telegram"

    assert group.project_path == expected_path
    assert group.sessions[0].session_id == "9700566c-7ffa-4f24-ae19-88708c43cff7"
    assert all(s.project_path == expected_path for s in group.sessions)


def test_scan_desktop_sessions_skips_agent_only_empty_sessions(tmp_path: Path):
    """Session files with only agent-setting rows should not be shown in /resume."""
    claude_home = tmp_path / "claude_home"
    project_dir = (
        claude_home
        / "projects"
        / "-Users-joya-JoyaProjects-claude-code-telegram"
    )
    project_dir.mkdir(parents=True)

    empty_newest = project_dir / "9700566c-7ffa-4f24-ae19-88708c43cff7.jsonl"
    valid_older = project_dir / "ea2c1a92-3e3d-4f6b-b37c-ddd40c8aa618.jsonl"

    _write_jsonl(
        empty_newest,
        [
            {
                "type": "agent-setting",
                "agentSetting": "meta",
                "sessionId": "9700566c-7ffa-4f24-ae19-88708c43cff7",
            }
        ],
    )
    _write_jsonl(
        valid_older,
        [
            {"cwd": "/Users/joya/JoyaProjects/claude-code-telegram"},
            {"type": "user", "message": {"content": "继续排查"}},
        ],
    )

    now = int(os.path.getmtime(valid_older))
    os.utime(valid_older, (now - 100, now - 100))
    os.utime(empty_newest, (now, now))

    groups = scan_desktop_sessions(
        claude_home=str(claude_home),
        max_sessions_per_project=5,
        skip_dirs=set(),
    )

    assert len(groups) == 1
    sessions = groups[0].sessions
    assert [s.session_id for s in sessions] == ["ea2c1a92-3e3d-4f6b-b37c-ddd40c8aa618"]

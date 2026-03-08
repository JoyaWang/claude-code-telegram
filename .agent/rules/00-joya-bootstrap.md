---
trigger: always_on
---

# Joya Project Bootstrap

This file is a thin adapter only.

Before doing any work in this project:

1. Self-check the project doc structure.
2. If required files or shared links are missing, repair the structure from `$JOYA_ROOT/joya-lib/Coding/doc-template/`, then tell the user what was fixed.
3. Read `GEMINI.md` in the project root.
4. Read `steering/AI_RULES_BASE.md`.
5. Read `steering/PROJECT_RULES.md`.
6. Read `progress.md`.
7. If a task requires a shared role persona, enable one of the `joya-*` role rules in this folder.

Rules:

- Do not duplicate shared agent memory into this project.
- The canonical shared memory repo is:
  - Windows: `D:/JoyaProjects/joya-lib/Coding`
  - macOS: `/Users/joya/JoyaProjects/joya-lib/Coding`
- If a shared role needs to change, edit only the canonical files under `joya-lib/Coding/agents/<id>/`.

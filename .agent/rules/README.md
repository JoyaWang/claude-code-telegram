# Joya Antigravity Rules

These files are thin Antigravity adapters for project workspaces.

- `00-joya-bootstrap.md` should be always-on.
- `10-joya-meta.md`, `20-joya-dev.md`, `30-joya-qa.md`, `40-joya-corp.md`, and `50-joya-creator.md` are role rules.
- `05-joya-active-role.md` is generated per project and acts as the default active role. In the current workflow it defaults to `meta` / Alice.
- The bulk sync scripts do not overwrite an existing `05-joya-active-role.md`. Use `scripts/antigravity-role.ps1` or `scripts/antigravity-role.sh` when you intentionally want to switch that default role.
- The role files do not contain canonical memory. They only point Antigravity back to `joya-lib/Coding/agents/<id>/...`.

On macOS, these files can be symlinked into project `.agent/rules/` directories because they do not depend on project-specific relative includes.

On this Windows setup, the sync script uses hardlinks instead of file symlinks because file symlinks require administrator privilege.

Before syncing these files, the bulk sync scripts repair the target project's base doc contract so missing `AGENTS.md` / `CLAUDE.md` / `GEMINI.md` or `steering/PROJECT_RULES.md` do not cause the project to be skipped.

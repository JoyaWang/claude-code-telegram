# 项目进度

## 已完成
- 修复显式新会话状态重置不完整的问题：经典模式下 `/new`、`/end` 和相关按钮入口现在统一设置 `force_new_session`，避免下一条消息误自动恢复旧会话
- 修复 Claude SDK 新会话初始化 60 秒超时问题：`setting_sources` 从 `["user", "project"]` 调整为仅 `["project"]`
- 完成根因验证：通过 bot 日志、CLI 最小复现、SDK 参数矩阵和真实 `ClaudeSDKManager.execute_command()` 调用，确认超时根因是显式传入 `user` setting source，而不是 `/new` 会话状态机本身

## 进行中
- 项目初始化和文档体系搭建
- 补充本次 session / SDK 故障的项目内经验沉淀

## 接下来
- 继续观察运行日志，确认新会话在真实 Telegram 流量下不再出现 `Control request timeout: initialize`
- 补齐测试环境依赖后，运行 `tests/unit/test_claude/test_sdk_integration.py` 与新增的 session reset 回归测试

## 已知问题
- 当前本地 `.venv` 缺少 `pytest`，因此单元测试未在项目虚拟环境内完整执行；本次修复已通过真实 SDK 调用链验证

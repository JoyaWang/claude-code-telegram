# 经验教训

> AI 每次会话应读取此文件，避免重复犯错。
> 项目特定经验写在这里；跨项目通用经验写入 `shared_memories/经验教训登记册.md`。

## 记录规则

- 每条记录只沉淀一个明确教训，标题尽量短。
- 重点写清问题、根因、解法和最终沉淀出的规则。
- 如果这条经验已经提升为共享规则或项目规则，要在“规则”字段标明去向。

## 标准记录格式

### YYYY-MM-DD: [简短标题]
- **问题**：遇到了什么问题
- **根因**：根本原因是什么
- **解法**：最终如何解决
- **规则**：由此沉淀的规则；如已写入其他文档，注明位置

## 示例条目

### 2025-01-15: [简短标题]
- **问题**：描述遇到的问题
- **根因**：分析根本原因
- **解法**：最终的解决方案
- **规则**：由此新增的规则 → 已写入 `steering/PROJECT_RULES.md` 的 [章节名]

---

## 记录

### 2026-03-09: `/new` 失效不一定是会话状态机问题
- **问题**：用户反馈 `/new`、`New Session`、`/end` 后首条消息仍然表现得像旧会话没有被清掉
- **根因**：经典模式里存在多条显式重置入口，只清除了 `claude_session_id`，但没有统一设置 `force_new_session=True`，导致下一条消息仍可能触发 auto-resume
- **解法**：抽出统一的 session reset 逻辑，所有显式新会话/结束会话入口统一走同一函数，同时补回归测试覆盖这些入口
- **规则**：凡是语义上“明确开始新会话”或“明确结束当前会话”的入口，必须同时清 `claude_session_id` 并阻断下一条消息自动恢复旧 session

### 2026-03-09: Claude SDK 显式传 `user` setting source 会卡死初始化
- **问题**：即使 `/new` 后 `force_new=true`、`session_id=null`、`continue_session=false`，新会话仍在约 60 秒后报 `Control request timeout: initialize`
- **根因**：问题不在 `/new` 状态机，而在 `ClaudeAgentOptions(setting_sources=["user", "project"])`。在当前环境中，只要 SDK 显式传入 `user` setting source，Claude CLI 初始化就会超时；不显式传入时，CLI 仍会按默认机制读取用户级配置
- **解法**：将 SDK 显式传入的 `setting_sources` 改为仅 `["project"]`，保留项目级 settings / skills，同时让 CLI 按默认行为隐式处理用户级配置；并通过 CLI 最小复现、SDK 参数矩阵、真实 `ClaudeSDKManager.execute_command()` 调用三层验证修复
- **规则**：以后遇到 Claude 初始化类故障，先区分是 session/resume 问题还是 SDK 选项问题；对 `ClaudeAgentOptions` 的新增或修改参数必须做最小复现验证，尤其是 `setting_sources`、`system_prompt`、`extra_args.agent`、`sandbox`

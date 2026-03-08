# AI 通用开发规则

> 本文件包含所有项目共享的通用开发规则。此文件通过符号链接共享到所有项目，修改即同步生效。
> 项目特定的规则请写在各项目的 `steering/PROJECT_RULES.md` 中。

## 核心开发原则（强制执行）

### 术语定义

- **体系**：指 `joya-lib/Coding` 目录下的共享文档、agents、skills、shared memories 和工具适配层。

### 验证与证据协议（硬性要求）

**所有任务完成后，必须经过验证并向用户展示成功证据，才能汇报“完成”。**

规则：
1. 任何代码、配置、脚本、流程修改完成后，都必须实际验证结果。
2. 验证方式包括但不限于：运行测试、执行命令、触发构建、检查输出、访问页面、核对产物。
3. 向用户汇报时，必须附带成功证据，而不是只说“已经改了”。
4. 禁止在未验证的情况下声称完成。
5. 如果验证失败，必须继续修复直到通过，或如实汇报失败原因和阻塞点。

### 需求确认协议（强制）

**在执行任何非平凡任务之前，必须先复述对需求的理解，等用户确认后再动手。**

规则：
1. 先复述理解，再执行。
2. 用户纠正后，必须先更新理解，再继续。
3. 简单查询和闲聊除外。

### 文档先行与实现对齐（强制）

**写文档记录需求、方案和计划，然后实现，根据实现过程中的实际情况不断更新文档，确保最终实现始终和文档是对应的。**

规则：
1. 在执行非平凡任务前，先明确需求、方案、计划分别落在哪些文档里。
2. 文档确认后再开始实现，不允许跳过文档直接进入代码、配置、测试或发布动作。
3. 实现过程中如果真实情况和原方案不一致，必须立即同步更新对应文档。
4. 汇报完成前，必须检查最终实现、最终配置、最终流程和最终文档一致。

### 文档路由与禁止随意新建（强制）

**必须优先写入现有文档体系，不允许为了省事或图方便随意新建文档。**

文档路由：
- 产品需求、范围、验收标准、非功能需求 → `steering/PRD.md`
- 用户流程、页面流程、交互路径、信息架构 → `steering/APP_FLOW.md`
- 技术栈、依赖、基础设施、环境约束、第三方服务选型 → `steering/TECH_STACK.md`
- 前端界面规范、组件约定、视觉/交互约束 → `steering/FRONTEND_GUIDELINES.md`
- 后端模块划分、接口设计、数据流、存储结构、服务边界 → `steering/BACKEND_STRUCTURE.md`
- 实施阶段、任务拆解、里程碑、验收计划 → `steering/IMPLEMENTATION_PLAN.md`
- 当前已完成、进行中、下一步、阻塞问题 → `progress.md`
- 新会话恢复入口、当前阶段摘要、立即行动项 → `steering/SESSION_CONTEXT.md`
- 项目特定操作规则、编码规则、项目禁止事项 → `steering/PROJECT_RULES.md`
- 复盘、踩坑、经验教训、后续避免方式 → `steering/LESSONS_LEARNED.md`

补充规则：
1. 同一个任务涉及多个维度时，分别更新对应文档，不要新建“临时总文档”替代既有体系。
2. 只有在现有文档体系确实没有合适归属时，才允许提议新建文档。
3. 提议新建文档时，必须先说明为什么现有核心文档都不合适。
4. 未经用户明确确认，禁止新建新的 `steering/` 文档、根目录文档或临时方案文档。

### 项目文档体系自检（强制）

进入任何项目工作前，默认先自检项目文档体系是否健全：
- 根目录：`AGENTS.md`、`CLAUDE.md`、`GEMINI.md`、`progress.md`
- `steering/`：`AI_RULES_BASE.md`、`PROJECT_RULES.md`、`PRD.md`、`APP_FLOW.md`、`TECH_STACK.md`、`FRONTEND_GUIDELINES.md`、`BACKEND_STRUCTURE.md`、`IMPLEMENTATION_PLAN.md`、`SESSION_CONTEXT.md`、`LESSONS_LEARNED.md`

如果发现缺失、断链或结构不完整：
1. 默认补齐模板结构并修复共享链接。
2. 只修结构，不编造缺失的业务内容。
3. 修复后必须明确告知用户修了什么。
4. 文档结构自检与结构修复属于 bootstrap 行为，不算业务内容修改，可先执行。

### 工程红线

- 不要在未读取相关代码和相关文档的情况下提议修改。
- 不要引入文档未列出的依赖，需先讨论。
- 不要硬编码敏感信息。
- 不要跳过测试或 CI 检查来制造“通过”。
- 不要在测试失败后用 mock、fallback 或删断言的方式绕过问题。
- 同一问题连续 2 次尝试未修好时，必须增加调试证据并排查根因，不要死磕 workaround。
- 单文件超过 500 行时，优先评估拆分。
- `CLAUDE.md`、`AGENTS.md`、`GEMINI.md` 是入口指针文件，不承载项目规则正文。

## 文档治理

### 文档分层

- **共享层**：`AGENTS.md`、`CLAUDE.md`、`GEMINI.md`、`steering/AI_RULES_BASE.md`
- **项目层**：`steering/PROJECT_RULES.md`、`steering/PRD.md`、`steering/APP_FLOW.md`、`steering/TECH_STACK.md`、`steering/FRONTEND_GUIDELINES.md`、`steering/BACKEND_STRUCTURE.md`、`steering/IMPLEMENTATION_PLAN.md`、`steering/SESSION_CONTEXT.md`、`steering/LESSONS_LEARNED.md`、`progress.md`

### 文档修改规则

1. 结构性修复可以自动执行，但修完必须通知用户。
2. 业务内容、项目规则、需求方案、经验总结的实质修改，必须先经用户确认。
3. 项目特定规则留在 `PROJECT_RULES.md`；跨项目共用规则才提升到本文件。
4. 项目经验留在 `LESSONS_LEARNED.md`；跨项目通用经验留在 `shared_memories/经验教训登记册.md`。

### 经验教训规则

1. 每次进入项目应读取 `steering/LESSONS_LEARNED.md`，避免重复犯错。
2. `LESSONS_LEARNED.md` 自带标准记录格式，新增经验时按该格式写入。
3. 调试某个问题超过 2 轮才解决时，默认应补记经验教训。
4. 用户反复纠正同一类问题时，应提议把规则沉淀到正确文档中。

## 参考文档

- `steering/PROJECT_RULES.md` - 项目特定规则
- `steering/PRD.md` - 产品需求
- `steering/APP_FLOW.md` - 用户流程
- `steering/TECH_STACK.md` - 技术栈
- `steering/FRONTEND_GUIDELINES.md` - 前端规范
- `steering/BACKEND_STRUCTURE.md` - 后端结构
- `steering/IMPLEMENTATION_PLAN.md` - 实施计划
- `steering/SESSION_CONTEXT.md` - 会话恢复入口
- `steering/LESSONS_LEARNED.md` - 项目经验教训
- `progress.md` - 当前进度

# 结构化故障条目规范

每个故障一个 YAML 文件：`catalog/entries/<ID>.yaml`。**只能通过 `tools/catalog_tool.py` 写入或修改**，不要手改条目，也不要手改 `views/` 下的生成文件。
受控词表见 [vocab.yaml](vocab.yaml)。迁移来历见 RAG 仓库 `normalize/`（2026-10 由 Markdown 编目迁移而来）。

## 编号与状态

- 编号 `INF-0001` / `TRN-0001`，序号只表示分配顺序，**不含分类语义**；由 `catalog_tool.py new` 分配，不复用。
- 旧 Markdown 编号（如 `1.44`、`HW.01`）记在 `legacy_ids`；`catalog/id_registry.tsv` 是编号登记表。
- `status`：`pending_audit`（扫描刚写入，待每日自动审计）→ `active`；`merged`（并入 `merged_into`）；`retired`。

## 占位符

任何字段都可写 `待定`（应有值、尚未填写或核实）或 `N/A`（对本条不适用）。

## 字段

| 键 | 说明 |
|---|---|
| `slug` / `title` | snake_case 名；一句话中文标题，≤40 字 |
| `kind` | `故障` / `结论` / `统计` / `方法` / `指针` |
| `cause.objects[]` | 原因对象：`subsystem`（严格词表）、`stack`（半受控）、`component`（≤25 字）。只有修复必须同时改两个不同 subsystem 时才写多项 |
| `cause.behavior` | 原因行为：缺陷本身做错了什么，≤120 字，只写原因 |
| `effect.objects[]` / `effect.detail` / `effect.visibility` | 影响对象（词表，1–3 项，第一项最直接）、影响详情（≤150 字，只写外部表现与触发条件）、可见性（词表） |
| `mitigation.detect` / `mitigation.fix` | 改进措施。**扫描时看到修复 PR / 规避方法就写进 `fix`**（写明 PR 链接、状态与是否已发布，未核实就说未核实）；没有就写 `待定` |
| `labels` | `family`、`trigger`、`trigger_review`、`trigger_note`（推理编目）；`stage`（训练编目）；`confidence`、`coverage`、`discovered` |
| `evidence[]` | **必填**，由工具从 `original.source` 中的链接提取：`url` + `kind`；无链接的历史条目为 `{kind: citation, text}` |
| `relations[]` | `{kind: var, target: <已有条目编号>}`；`coverage=var` 时必填。target 可写新编号或旧编号（如 `4.28`），工具换算成新编号并回显目标标题 |
| `original` | 扫描写的原始说明：`mechanism`（机制）、`source`（来源，含可点击链接）、`behavior`（行为效应）、`trigger_surface`（触发面，推理编目）、可选 `signature` / `repro`，`discovery`（发现来源）、`notes`（回访记录，由 update 追加） |
| `migration` | 来历：`scan`（扫描写入）/ 迁移时的抽取与审计记录 / `updates`（修改历史） |

## 扫描写入新条目（JSON 输入）

```bash
python3 tools/catalog_tool.py find "vllm-project/vllm#12345"     # 先查重：编号、旧号、slug、链接、仓库#编号都可查
python3 tools/catalog_tool.py new --file /tmp/entry.json         # 返回 {"ok": false, "errors": [...]} 就改 JSON 重交
```

`entry.json` 是一个 JSON 对象，字段同上，外加 `original`；不要写 `id`、`evidence`、`status`、`migration`。必须用 JSON（不要手写 YAML：`#` 会被当成注释）。

```json
{
  "slug": "...", "title": "...", "kind": "故障",
  "cause": {"objects": [{"subsystem": "...", "stack": "...", "component": "..."}], "behavior": "..."},
  "effect": {"objects": ["..."], "detail": "...", "visibility": "..."},
  "mitigation": {"detect": "待定", "fix": "修复 PR https://github.com/... （open，未发布）"},
  "labels": {"family": "infra_off", "trigger": "partial", "trigger_review": false, "trigger_note": "...",
             "stage": "N/A", "confidence": "documented", "coverage": "NEW", "discovered": "2026-10-13"},
  "relations": [], "remarks": "",
  "original": {"mechanism": "...", "source": "... https://github.com/...", "behavior": "...",
               "trigger_surface": "...", "discovery": "2026-10-13 每日扫描"}
}
```

**`original` 是完整的原始说明，结构化字段是它的压缩**：`original.mechanism` 要像以往 Markdown 条目一样写全——代码/配置层面的具体原因、涉及的版本与配置、证据（复现、源码定位、维护者结论）和条件；`original.behavior` 写清观察到的现象与触发条件。`title`、`component`、`cause.behavior`、`effect.detail` 里的每个事实都必须能在 `original` 里找到——每日审计以 `original` 为准，原文没写的事实会被判为“写了原文没有的事实”并改写掉。工具要求 `original.mechanism` 不少于 80 字、`original.behavior` 不少于 40 字，且机制+行为效应+触发面的总长不短于 `cause.behavior` 与 `effect.detail` 之和。

工具会：分配编号、提取来源链接、检查来源是否已被其他条目引用（同一机制请改用 update 补证据）、把关系目标的旧编号换算成新编号并检查目标存在（结果里回显每个目标的标题，务必核对）、按词表和长度上限校验，然后写入 `status: pending_audit`。

## 修改已有条目

```bash
python3 tools/catalog_tool.py update INF-0044 --add-evidence https://github.com/.../pull/123
python3 tools/catalog_tool.py update INF-0044 --note "回访：报告者撤回根因，改为 ..."
python3 tools/catalog_tool.py update INF-0044 --set labels.confidence=verified --set 'mitigation.fix="PR #123 已合入 v0.9"'
```

## 生成视图与校验

```bash
python3 tools/catalog_tool.py build      # views/CATALOG.zh.md、fault_catalog.csv、fault_catalog.xlsx
python3 tools/catalog_tool.py validate   # 失败（exit 1）就不要提交
```

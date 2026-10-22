# 推理故障编目（LLM 推理期静默故障）

对照仓库：[`../training_error_review`](../training_error_review)（训练期故障）。检索工具：`/workspace/RAG/fault_review`（读本仓库的只读快照）。

本仓库收集 LLM 推理期「进程正常、结果已坏」的故障模式。每晚 23:00 UTC 由 hermes cron 扫描写入（状态 `pending_audit`），02:00 UTC 审计定稿。

## 先读

| 文件 | 用途 |
|---|---|
| [`catalog/entries/`](catalog/entries) | **唯一编目真源**：每条一个 YAML，只能用 `tools/catalog_tool.py` 读写；字段见 [`catalog/SCHEMA.md`](catalog/SCHEMA.md)，受控词表见 [`catalog/vocab.yaml`](catalog/vocab.yaml) |
| [`catalog/patterns/`](catalog/patterns) | 故障模式层，由条目的 `var` 关系派生（`catalog_tool.py pattern-sync`，审计任务自动运行） |
| [`views/`](views) | 由条目生成的浏览视图：`CATALOG.zh.md`、`fault_catalog.csv`、`fault_catalog.xlsx`（`catalog_tool.py build`，不要手改） |
| [`daily/reports/`](daily/reports) | 每日扫描日报 `daily_scan_YYYY-MM-DD.md` |
| [`daily_scan.prompt.md`](daily_scan.prompt.md) | 每日扫描执行规范 |
| [`NPU_SCAN.md`](NPU_SCAN.md) | NPU 来源扫描规范 |
| [`FAULT_MASTER_REFERENCE.zh.md`](FAULT_MASTER_REFERENCE.zh.md) | 2026-10-13 起冻结的 Markdown 历史档案 |

## 目录

```
catalog/          编目真源（entries/、patterns/）、SCHEMA.md、vocab.yaml、id_registry.tsv、audit/（每日审计记录）
tools/            catalog_tool.py（写入/查询/生成/校验）、catalog_audit.py + daily_audit.sh（每日审计）、scan/npu/（NPU 采集）
views/            由 catalog/entries 生成的视图
daily/reports/    每日扫描日报
docs/             分析短文，不是编目
```

## 自动化依赖（改动前先看）

- **按路径读取、不要移动**：`daily_scan.prompt.md`、`NPU_SCAN.md`（扫描 cron 读取）；`FAULT_MASTER_REFERENCE.zh.md`（扫描规范引用；RAG 同步时仍从其附录 C/D 读取负结果）；`survey_internal_signal_hallu_ood_quality_2026.md`（hermes 若干 skill 按绝对路径引用）。
- **前置检查**：每晚扫描和审计开始前都要求 `main` 分支、`git status --porcelain -uall` 为空、fetch 后 `HEAD == origin/main`。手工改动必须提交并推送，否则当晚任务会停在前置检查。
- **根目录没有 `AGENTS.md`，是有意的**：hermes cron 以本目录为工作目录时，会把根目录的 `AGENTS.md` 注入扫描 agent 的系统提示。扫描规则统一写在 `daily_scan.prompt.md`。

## 约定

- 根目录只放上面列出的文件。日报写在 `daily/reports/`；扫描与采集的临时文件写到 `$TMPDIR`（hermes profile 的 cache），不写进仓库。
- `.gitignore` 兜底忽略历史上用过的临时目录与备份（`.scan_*/`、`scan_raw*/`、`scan_cache_*/`、`scan_queries_*.sh`、`*.bak*`）。

## 2026-10-22 整理

日报从根目录移入 `daily/reports/`（文件名不变：有 3 个条目的 `original.discovery` 按文件名引用 `daily_scan_2026-07-29.md`）。以下历史遗留已从工作树移除，均可在 git 历史中找回：

- 早期提交的扫描临时目录（`.scan_*`、`scan_raw*`、`scan_cache_*`、`scan_queries_*.sh`）；
- `FAULT_MASTER_REFERENCE.zh.md.bak`、`.bak2`；
- `agent-act-ms/`（10-21 NPU 回扫时误提交的 agent 输出）；
- 根目录的 `fault_catalog.csv` / `fault_catalog.xlsx`（停在 10-12 的旧导出，最新版在 `views/`）；
- Markdown 时代的工具 `export_faults.py`、`verify_catalog.py`、`test_verify_catalog.py`、`category_map.py`（已由 `tools/catalog_tool.py` 取代）。

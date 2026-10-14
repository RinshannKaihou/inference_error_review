# NPU 来源接入：真实采集验收

日期：2026-10-14。本文件记录只读数据采集和测试，不代表所有候选已被认定为故障，也不代表已完成一轮自动写编目的模型任务。

## 结果

- 注册表独立新增来源：22；实际完整采集：22；失败：0。
- 初次从 2026-10-07 00:00 UTC 起收集更新，完整成功批次截止：2026-10-14T03:14:21Z；21 个先成功来源在第二轮使用水位减一小时补增量，vLLM-Ascend 从初次窗口完整补抓。
- 归一唯一记录：7455（issue 2151、PR 5304），两视图条数：推理 7346、训练 7377；同一源可用于两种判定，不把重叠条数相加当独立发现。
- 原来源保留后：推理 26 仓，训练 27 仓，另有 arXiv。
- 完整原始证据 batch：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261014T031421311014-kpets6ra`。
- 采集器 SHA-256：`80affe96ba4039ba3c500ec92eb018eab7eeea5c2457e6de9586a95b9f1ee30b`。部署与版本化副本逐字节一致，两个库的注册表逐字节一致。
- 跨 scope 缓存复用实际返回 cached=true、同一 batch；检查全部 body_path 存在、全部 key 唯一、scope 路由正确。
- 版本化包执行 26 项测试（21 collector + 5 Git entry），含 7 个真实网络测试：全部 OK，未跳过。
- 两库各 7 项 catalog_tool 回归测试通过；存量测试有 ResourceWarning，不是新增代码失败。
- 两库 catalog_tool.validate 均 ok=true：推理 1370 条、训练 269 条；本次没有新增/修改编目条目或生成视图。

## 全部来源覆盖

| 平台 | 仓库 | 本轮完整 | 本轮取回条数 | 完成列表页 |
|---|---|---|---:|---:|
| gitcode | `Ascend/pytorch` | True | 28 | 2 |
| gitcode | `Ascend/op-plugin` | True | 1 | 2 |
| gitcode | `Ascend/torchair` | True | 49 | 2 |
| gitcode | `cann/ops-transformer` | True | 88 | 2 |
| gitcode | `cann/ops-nn` | True | 56 | 2 |
| gitcode | `cann/ops-math` | True | 24 | 2 |
| gitcode | `cann/ge` | True | 9 | 2 |
| gitcode | `cann/runtime` | True | 7 | 2 |
| gitcode | `cann/hccl` | True | 21 | 2 |
| gitcode | `cann/hcomm` | True | 43 | 2 |
| gitcode | `mindspore/mindspore` | True | 3 | 2 |
| gitcode | `mindspore/mindformers` | True | 7 | 2 |
| gitcode | `Ascend/msprobe` | True | 6 | 2 |
| gitcode | `Ascend/mssanitizer` | True | 3 | 2 |
| gitcode | `Ascend/MindSpeed` | True | 4 | 2 |
| gitcode | `Ascend/MindSpeed-LLM` | True | 0 | 2 |
| gitcode | `Ascend/MindSpeed-RL` | True | 0 | 2 |
| gitcode | `Ascend/MindIE-LLM` | True | 2 | 2 |
| gitcode | `Ascend/msmodelslim` | True | 12 | 2 |
| github | `vllm-project/vllm-ascend` | True | 1199 | 12 |
| github | `sgl-project/sgl-kernel-npu` | True | 0 | 1 |
| github | `triton-lang/triton-ascend` | True | 6 | 1 |

上表是第二次采集的实际增量/补抓条数，不是所有历史候选总数；空结果来源也经有效 API 和分页校验，不能把 fetched=0 当作未访问。

## 验收发现与修复

1. 初次窗口按 UTC 日期零点计算，与已有七日扫描规则一致；新增回归测试曾出现 RED（12:30 而非 00:00），修正后通过。
2. vLLM-Ascend 大数据集的 GitHub 第 11 页只用 page 参数返回 422，服务端明确要求 cursor。保存的真实 Link 含 after/before；新增回归测试先 RED，再改为原样跟随同源 HTTPS next。真实补抓完成 12 页、1199 条；未重试掩盖错误或改用别的平台。
3. 另一轮维护在本轮准备过程中修订了凭据、分段等待、关系目标回显护栏。NPU 规则在最新版本上定向合并，并留下护栏保留检查；未覆盖那些并发修订。

4. 独立审查发现 GitHub `/issues/{n}` 可能返回 PR 对象。增加真实网络 RED→GREEN 回归：归一化前拒绝该别名并要求规范 `/pull/{n}`，不再漏读 review 后误报 issue 完整；真正的 PR 链路仍完整采集普通/行内评论与 reviews。

## 正式 Cron 接入核验（2026-10-14）

两条现有任务已通过 `hermes -p third cron edit` 更新，只修改各自的 `prompt` 和 `script`；保存更新前后定义逐字段核对，其他任务未改动。

| 任务 | 原任务 ID | 时间（UTC） | 前置脚本 |
|---|---|---|---|
| 推理 | `8645356b40ae` | 每天 23:00 | `fault_scan_npu_inference.sh` |
| 训练 | `1b8481fa4e2f` | 每天 00:30 | `fault_scan_npu_training.sh` |

- 两任务仍启用；`zai / glm-5.3`、模型 pin、`reasoning_effort=max`、投递、workdir、工具集、下一次运行时间和历史记录均保持。脚本字段使用当前 profile 的 `scripts/` 内相对文件名，不是绝对路径；仍为 LLM 任务，未切为 `no_agent`。
- 已使用 Hermes venv 和 `HERMES_HOME=/root/.hermes/profiles/third` 调用实际 `cron.scheduler_script._run_job_script_with_claim_heartbeat`，两条实际 job 的脚本与 workdir 均成功，返回完整 22 来源、同一缓存批次。
- 实际 `cron.scheduler_prompt._build_job_prompt` 通过提示安全扫描；两任务均注入 `Script Output`、真实 batch 与 NPU 扫描规则，wake gate 为 true。验证时现有脚本超时值为 3600 秒，未改该设置。
- 再次 `catalog_tool.py validate`：推理 1370 条、训练 269 条，均 `ok=true`、0 errors；存量 warnings 分别为 43 和 4。本次未写入新编目条目。
- 实现提交已推送，并以 `git ls-remote origin refs/heads/main` 核对：推理 `a3726252337bbb69e919bf99335a82552d1e332e`，训练 `fcd71adc861afeaef063fc902250ee29d57b0c8b`。本文件随后可有文档补充提交，不改变实现提交身份。

这里验证的是正式配置、实际前置脚本执行和提示组装，**没有提前触发完整 LLM 扫描、深读全部候选或自动编目**；后续按原定时执行，不能把采集/接入验证说成未来定时 tick 已运行。

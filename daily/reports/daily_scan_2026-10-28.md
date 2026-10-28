# 每日扫描日报 2026-10-28（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（852d240a22696db22d6f11e126333f13f114d5d9）。通过，开始扫描。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261028T230043096320-82q4n1up`，采集窗口 2026-10-27T22:00:43Z → 2026-10-28T23:00:43Z（registry_hash 2f6422d0…，22 源全 complete、failures 空）。
- inference scope 候选 19426 条（normalized.jsonl 19669 行，含 training）；窗口内新建（created_at ≥ since）760 条，标题强信号命中 62 条（ops-nn 22、ops-transformer 12、ops-math 11、vllm-ascend 9、pytorch 3、msprobe 2、mindformers 2、sgl-kernel-npu 1）。
- 深读正文约 15 条；完整 URL 查重（近 14 天日报 + catalog_tool find）后无既往判定冲突。

### 新增条目（NPU 来源）

1. **INF-2241** [NEW] [ops-transformer#6822](https://gitcode.com/cann/ops-transformer/issues/6822) — aclnnQuantCompressor stateBlockTable 元素越界（==block_num）静默越界写 state_cache：两段 rc=0，1MiB 守护区被写 65536B，mssanitizer memcheck 报 illegal_write 256 次；对照组合法值 0B。Ascend950PR / CANN 9.2.0-beta.2，完整 C++ 复现。
2. **INF-2236** [NEW] [ops-nn#7569](https://gitcode.com/cann/ops-nn/issues/7569) — aclnnDequantSwigluQuant 出参 yOut/scaleOut 尺寸未校验：声明为推导尺寸一半时 rc=0 且边界外被写 128B（行数 1/4/16 → 32/128/512B 线性）；scaleOut 同型。910B2 / CANN 9.2.0 / 711a4f5d。另：**INF-2242** [NEW] [ops-nn#7533](https://gitcode.com/cann/ops-nn/issues/7533) — aclnnDynamicQuantV2 bf16 极值（|x|>1.70e38）行极差在 fp32 计算溢出 inf，yOut 全饱和 +127 符号全丢、rc=0 无告警；fp16 量程不可达不触发；含阈值扫描与 C++ 复现。
3. **INF-2243** [NEW] [ops-nn#7527](https://gitcode.com/cann/ops-nn/issues/7527)（closed）+ 修复 [!12728](https://gitcode.com/cann/ops-nn/pull/12728)（merged）— MultiAddRmsNormDynamicQuant arch35 四缺陷：smooth_scale 次正规致 int8 饱和、累加序与 A2 差 1-2 ULP、零行 scale 随 D 变 1.0/0、inf 输入补偿 NaN。修复后泛化 2495 例全 PASS。
4. **INF-2246** [NEW] [ops-nn#7542](https://gitcode.com/cann/ops-nn/issues/7542) — aclnnMseLossOut 空输入早退分支无填充动作：reduction=1/2 时 rc=0 但 out 整块未写（金丝雀 -777.0 全保留）；同仓 aclnnMseLoss 正确填 NaN/0。var → INF-2187（同「空输入早退成功+输出未写」机制，算子不同）。
6. **INF-2249** [NEW] [ops-transformer !14799](https://gitcode.com/cann/ops-transformer/merge_requests/14799)（open）— KvQuantSparseFlashAttention TND 布局尾部 [actual,T) 不清零残留 NaN/Inf（arch22）；修复 MR 为 TND 启用初始化按 actual_seq_lengths_query 清零，6048 个预填 NaN 图输出全过；社区 CI 待运行。
7. **INF-2240** [NEW] [vllm-ascend#20115](https://github.com/vllm-project/vllm-ascend/issues/20115) — GLM-5.3-Flash-w8a8 图像理解在 nightly-main（vLLM 0.32.0）全错（红→Black、蓝→Yellow），0.28.0/v0.30.0rc2 同 checkpoint 正确；文本/工具/1M 长文本正常。视觉路径回归，commit 未 bisect。
8. **INF-2250** [NEW] [vllm-ascend#20146](https://github.com/vllm-project/vllm-ascend/pull/20146) — V1 runner 构造 AscendSampler 不转发 logprobs_mode：配置 processed_logprobs 仍按 raw 计算，RL train/rollout 一致性度量引入偏差（历史诊断修正后 prob diff 0.0145→0.0066）。修复 PR open。
9. **INF-2252** [NEW] [vllm-ascend#20113](https://github.com/vllm-project/vllm-ascend/pull/20113) — embedding 细粒度 TP>1 + DP>1：空闲 DP 引擎 dummy batch 喂 inputs_embeds 跳过查表致跨 DP 交换无对端（首请求 hang 至 HCCL 超时）或 HCCL 惰性创建竞态（EI0019/首请求 prefill 首词起乱码）；draft embedding 误路由 FINE_GRAINED 使交换组失步。修复 PR open（dummy join+draft 排除+预热）。

### NPU 拒收/留观（窗口内代表）

- **校验缺失/契约大族增量**（ops-nn#7530/#7532/#7535/#7537/#7550/#7555/#7592/#7599/#7626/#7630、ops-math#4164/#4178/#4179/#4187/#4188、ops-transformer#6844/#6845/#6849、pytorch#5686 等）— 与 10-22~10-27 已判「非法入参静默接受/错误码契约/文档声明」大族同型，按族留观维持。其中数值面真实且有完整实测的 #7533 已单列为 INF-2242；#7550（FakeQuant int32 回绕）证据完整但属 PTQ 前处理路径，端到端影响未证，留观。
- **InplaceUpdate 系列**（ops-nn#7592/#7599/#7626/#7630）— 纯文档/样例/校验域静态分析，未见静默数值后果，留观。
- **pytorch#5694 / PR#48610**（native BN 保存量契约、推理未初始化输出）— 任务帖属性（硬件解耦任务），方向为「确认语义后配套修复」，非已确认缺陷；留观，待后续修复结论。
- **msprobe#658/#659、mindformers#2619/PR!8888** — RFC/设计文档类；#2619 DeepSeekV3 PyNative 静默忽略 scaling_factor（实际 attention scale 40 vs 1.0）证据链完整但为训练侧配置转换问题且关联训练库范围，本轮留观（如需跨库收录由训练侧判定）。
- **PR 修复流**（ops-nn !12950/!12961/!12968/!12974/!12997、ops-transformer !14764/!14778、pytorch !48610、ops-math !6754/!6755/!6771/!6791、hcomm !6101）— 白盒修复流按惯例不逐条开号；!6791 为 INF-2160（Addcmul NaN-as-1）的修复 PR，已补证据+回访注记。
- **msmodelslim、msprobe、mssanitizer、hccl、runtime、ge、mindspore** 窗口内关键词命中经索引筛查无新的静默数值故障候选。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境 GITHUB_TOKEN 为无效占位值（未认证）；search 走代理出口共享配额。21 组查询（7 仓 × correctness/wrong output/NaN，created:>=2026-10-21）：13 组首轮 200；8 组 403（search 配额耗尽）等待 60s 后重试全部 200。total：vllm 36/25/5、sglang 20/6/1、transformers 0/4/1、TRT-LLM 1/0/2、triton 2/1/2、tt-metal 5/44/20、flashinfer 27/12/10。
- 深读：新候选 6 个（search item 自带 body，core API 因出口 IP 限流不可用，已按索引+正文判定）；既往条目回访 11 个。

### 新增条目（GitHub）

1. **INF-2237** [NEW] [vllm#64106](https://github.com/vllm-project/vllm/issues/64106) — Kimi-K3 MI355X `--kv-cache-dtype fp8` GSM8K 0.975→0.305 静默塌陷；单变量扫描排除 prefix caching/SITUV2/DSpark/load-format，锁定 ROCM_AITER FP8 MLA prefill 路径（#51040 12→16 头 padding）；官方 recipes 配方直接触发；SGLang 同卡 FP8 KV 0.950 证明非 checkpoint/硬件问题。规避：KV auto。
2. **INF-2244** [NEW var] [vllm#62781](https://github.com/vllm-project/vllm/issues/62781) — GDN fp16 溢出族在 Intel XPU 复现且修复不达：core_attn_out 5 处按 activation dtype 分配（含 :974 forward_xpu）、两修复门 float32；Arc Pro B70 + Holo4-27B needle 4.7K-37.4K 全 0/9。var → INF-1446。
3. **INF-2245** [NEW] [vllm#63811](https://github.com/vllm-project/vllm/issues/63811)（closed）— V2 Gumbel sampler 把全 NaN logits 确定性映射为 token 1023 并持续重复，掩盖首个 NaN 来源（HiSparse/NIXL P/D）；issue 内建议默认 NaN guard 未实施。closed 但根因未闭环。
5. **INF-2251** [NEW] [tt-metal#62605+#62465+#62613](https://github.com/tenstorrent/tt-metal/issues/62605)（quasar 三连，#58407 item 18/22/23）— transpose 查 tile 倍数而非 shard 行 16B 对齐（PCC 近 0）；1D matmul 借用 in0 缓冲多块输出行块只读一次（PCC≈0.3）；slice rm_only 降级后 TILE 预分配输出交 RM writer（错值+越界写）。均静默。
6. **INF-2253** [NEW] [sglang#46314](https://github.com/sgl-project/sglang/issues/46314) — 扩散量化层 11 处正确性缺陷（代码审查）：ModelSlim 双 scale 键名错必崩、smooth-quant scale 静默丢弃、ROCm FP8 漏 fnuz、MXFP4 恒 bf16、4D 不展平等。

### tt-metal 其余新候选（均收录）

- **INF-2238** [NEW] [tt-metal#62901](https://github.com/tenstorrent/tt-metal/issues/62901) — paged sdpa_decode 复用块 cur_pos 后残留 inf/NaN 被读入 attention：Llama-3.2-1B 8/8 在 14-27 token 发散，无报错；有限残值不影响。var → INF-2113（vLLM ROCM_ATTN 滑窗不 mask V，同「窗口外 KV 残值 inf/NaN 未掩蔽」跨平台机制）。
- **INF-2239** [NEW] [tt-metal#62628](https://github.com/tenstorrent/tt-metal/issues/62628) — ttnn.softmax float32 TILE padding 残留 +inf/NaN 污染整行（dim24 +inf 错 21.8%、NaN 100%；dim8200 NaN 100%），输出无 NaN，行和≠1 可检。
- **INF-2248 前一编号冲突说明**：tt-metal#62902（BINARY embedding 非 32B 行宽索引 1 全错）为独立条目 **INF-2248**。

编号按提交顺序连续分配（INF-2236 DequantSwigluQuant 为本批第一条），最终清单见 §4 表。

- **vllm#63276**（DS V4.1 数字复述错）— 无环境归因（报告者自身输出漂移、无配置细节），留观；**vllm#64239**（Timed-out overflow dequeue 错消息）— 详情正文为 collect_env 无故障描述，留观；**vllm#62969/INF-2150、#62945/INF-2117、#63759/INF-2233** 等 11 个既往来源本轮回访见 §3。
- **arXiv**：2 组查询（silent data corruption+inference、bit flip+LLM+inference）HTTP 200 有效。新论文 1 篇收录：**INF-2247** [2610.30798](https://arxiv.org/abs/2610.30798) LLVMhammer——Rowhammer 翻转 Triton JIT 优化 pass 生成功能等价但次优 kernel，磁盘缓存持久化，推理延迟 +4-16%，输出正确（生存性攻击）；与既有 INF-0294/0295/0313（权重/显存 rowhammer 正确性攻击）机制不同（编译管线 vs 权重）。其余为已收录或范围外（BARE-AI 10-06 已见、Cauchy-Schwarz 2610.18557 已在库）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- GitHub search API 状态快照（本批缓存）：INF-2113 源 vllm#62828 **closed**（2026-10-27）——已 update 注记「closed 不自动证明已修复，关闭原因与修复 commit 未核实」；INF-2223 源 vllm#63653 **closed**（2026-10-26）——同上注记。INF-2233/#63759、INF-2117/#62945、INF-2114/#62811、INF-2232/#64048、INF-2150/#62969、INF-2156/#62052、INF-2227/#62719、INF-2134/transformers#49873、INF-2234/sglang#46572、INF-2166/flashinfer#7179、INF-2125/#6994 均仍 open。
- NPU 近 7 日条目来源（normalized.jsonl 快照）：ops-math#3974 维持 closed（10-26 已记）；ops-transformer#6655/#6656/#6675/#6766 已 closed（其余 open）——closed 均无修复结论可引用，不改判定，待对应修复 MR 佐证后复核。
- **INF-2160**（Addcmul NaN-as-1，ops-math#3894）：修复 PR [!6791](https://gitcode.com/cann/ops-math/pull/6791)（open）出现，已 `--add-evidence` + 回访注记（修复方向与报告一致，未合入）。
- 轮巡：mindspore/mindformers、Ascend/msmodelslim、Ascend/MindSpeed-LLM 窗口内 0/17/11 条，无遗留 open 旧源需改判（MindSpeed-RL、MindIE-LLM 本窗口 0 条）。

## 4. 编目与视图

- 新写入 **18 条**（全部 pending_audit）：INF-2236 ~ INF-2253，编号连续，以 §4 表为准。

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2236 | ops-nn#7569 | DequantSwigluQuant 出参偏小越界写 |
| INF-2237 | vllm#64106 | Kimi-K3 FP8 KV GSM8K 静默塌陷 |
| INF-2238 | tt-metal#62901 | paged sdpa_decode 复用块残值错输出 |
| INF-2239 | tt-metal#62628 | ttnn.softmax tile padding inf/NaN 污染 |
| INF-2240 | vllm-ascend#20115 | GLM-5.3-Flash 视觉全错回归 |
| INF-2241 | ops-transformer#6822 | QuantCompressor blockTable 越界写 |
| INF-2242 | ops-nn#7533 | DynamicQuantV2 bf16 极值量化全饱和 |
| INF-2243 | ops-nn#7527+!12728 | MultiAddRmsNormDynQuant arch35 量化饱和 |
| INF-2244 | vllm#62781 | GDN fp16 溢出 XPU 未修复复现（var→INF-1446） |
| INF-2245 | vllm#63811 | Gumbel 采样器全 NaN logits→固定 token |
| INF-2246 | ops-nn#7542 | MseLossOut 空输入 out 未写（var→INF-2187） |
| INF-2247 | arXiv:2610.30798 | Rowhammer 翻转 Triton JIT pass 持续减速 |
| INF-2248 | tt-metal#62902 | embedding BINARY 非 32B 行宽索引 1 全错 |
| INF-2249 | ops-transformer!14799 | KvQuantSparseFA TND 尾部残留 NaN/Inf |
| INF-2250 | vllm-ascend#20146 | V1 sampler 不透传 logprobs 模式 |
| INF-2251 | tt-metal#62605/62465/62613 | quasar 分片布局守卫失配三连 |
| INF-2252 | vllm-ascend#20113 | embedding 细粒度 TP 空闲秩失效 |
| INF-2253 | sglang#46314 | 扩散量化层 11 处正确性缺陷 |

- 更新 3 条：INF-2160（+证据 !6791 + 回访注记）、INF-2113（closed 回访注记）、INF-2223（closed 回访注记）。
- `catalog_tool.py stats`：2253 条（active 2235 + pending_audit 18）。`build` 与 `validate` 见提交记录（本轮 validate ok:true）。
- var 关系回显核对：INF-2238→INF-2113（标题「vLLM ROCM_ATTN 滑窗不 mask V」✓）、INF-2244→INF-1446（「GDN chunk prefill fp32 状态被降 fp16 溢出」✓）、INF-2246→INF-2187（「CANN GatherPaKvCache 空输入静默 no-op」✓）。

## 5. 未覆盖范围与 API 账本

- GitHub core API（issue 详情/评论）经代理出口 IP 未认证 60/h 限流耗尽（多组 403 rate limit），6 个新候选按 search item 自带 body 判定（含完整环境/复现/数据），未拉取评论线程——若报告者后续在评论中撤回或维护者定位根因，本轮判定可能需改判；已按「documented」置信度收录。
- search 配额等待：1 次 60s 前台 sleep 后配额恢复，累计等待 60s（上限 5 分钟内）。
- NPU PR 修复流正文未逐条读（!12950/!12961/!12968/!12974/!12997/!14764/!14778 等）——标题与描述一致性未核实，可能低估修复覆盖。
- arXiv 仅 2 组查询（SDC/bit-flip 主题），KV 损坏/数值正确性主题词未单独跑（前两组结果覆盖面内无新增，缺口小）。
- tt-metal#62720（Eltwise RM→WIDTH/BLOCK sharded 错值/挂起）与 #62719/INF-2227 系列同域：#62720 标题+正文摘要已读（PIL 无关），未单独开号（与已收录 quasar/eltwise 分片族机制相近但独立 issue，留观待深度复核）。
- GitCode issue 评论线程未拉取（API 未提供低成本批量入口），仅按正文判定。

## 6. 结论

18 条新机制入库（待审计），3 条既有条目回访/补证。本轮覆盖：NPU 22 源完整批次窗口 + GitHub 7 仓 3 关键词 + arXiv 2 组；缺口集中在 GitHub core API 评论与 NPU PR 修复流正文。

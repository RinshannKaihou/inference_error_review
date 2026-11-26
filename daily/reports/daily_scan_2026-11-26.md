# 每日扫描日报 2026-11-26（推理库）

## 0. Git 前置检查与共享采集批次

- 工作目录 `/workspace/inference_error_review`；分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 `origin/main` 一致（`5568770492aaa319b7b31fe0378cbcea7223af4e`）。通过，开始扫描。
- 前置脚本 `fault_scan_npu_inference.sh` 的 Git 检查通过，但采集器以退出码 2 结束：`Ascend/pytorch` pulls 第 4 页、`cann/ops-transformer` pulls 第 12 页在遍历中分页总数变化（`pagination totals changed during traversal`），批次 `complete=false`。本轮按 NPU_SCAN.md「采集失败不等于无发现」使用该批次，并如实记录缺口；没有把部分 API 失败写成「无新发现」。
- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261126T045512667016-kgio4u46`，`generated_at=2026-11-26T04:55:12Z`，`registry_hash=2f6422d0…8d94d`，22 个注册来源，`cached=false`。
- 窗口：`2026-11-16T21:00:12Z → 2026-11-26T04:55:12Z`（约 10 天；上次消费水位 `2026-11-16T22:00:12Z`，含 1 小时重叠）。上一份 NPU 日报为 2026-11-16，11-17～11-25 没有成功日报，本轮覆盖这 10 天的积压。
- 候选索引 `inference.jsonl` 45,500 条；`inference_delta.jsonl` **16,773 条**（issue 7,251 + PR 9,522）。按仓库：ops-nn 4,786、ops-transformer 2,505、vllm-ascend 2,216、ops-math 1,662、Ascend/pytorch 1,338、hcomm 862、runtime 719、ge 595、hccl 531、triton-ascend 403、op-plugin 277、torchair 255、msmodelslim 138、mindspore 130、msprobe 111、mindformers 95、sgl-kernel-npu 72、mssanitizer 63、MindIE-LLM 15。

### 逐仓覆盖（fetched 为窗口内 issue+PR 计数）

| 平台 | 仓库 | fetched | complete | 缺口 |
|---|---|---:|---|---|
| gitcode | Ascend/pytorch | 1,283 | 否 | pulls 第 4 页起未验证；total_page 11、validated_pages 3 |
| gitcode | Ascend/op-plugin | 277 | 是 | — |
| gitcode | Ascend/torchair | 249 | 是 | — |
| gitcode | cann/ops-transformer | 2,455 | 否 | pulls 第 12 页起未验证；total_page 18、validated_pages 11 |
| gitcode | cann/ops-nn | 4,734 | 是 | — |
| gitcode | cann/ops-math | 1,623 | 是 | — |
| gitcode | cann/ge | 588 | 是 | — |
| gitcode | cann/runtime | 713 | 是 | — |
| gitcode | cann/hccl | 528 | 是 | — |
| gitcode | cann/hcomm | 845 | 是 | — |
| gitcode | mindspore/mindspore | 116 | 是 | — |
| gitcode | mindspore/mindformers | 93 | 是 | — |
| gitcode | Ascend/msprobe | 107 | 是 | — |
| gitcode | Ascend/mssanitizer | 62 | 是 | — |
| gitcode | Ascend/MindSpeed | 86 | 是 | — |
| gitcode | Ascend/MindSpeed-LLM | 88 | 是 | — |
| gitcode | Ascend/MindSpeed-RL | 2 | 是 | — |
| gitcode | Ascend/MindIE-LLM | 15 | 是 | — |
| gitcode | Ascend/msmodelslim | 135 | 是 | — |
| github | vllm-project/vllm-ascend | 2,196 | 是 | — |
| github | sgl-project/sgl-kernel-npu | 71 | 是 | — |
| github | triton-lang/triton-ascend | 402 | 是 | — |

20/22 源完整；2 个源的 issue 列表完整、pulls 列表在窗口尾部缺页。未覆盖范围见 §4。

## 1. NPU 来源判定

- 低成本筛选：7,251 条 issue 中按精度/静默/错值/NaN/溢出/不一致/未初始化/越界/量化/KV 等关键词与标题筛出 **1,918 条**候选；其中 455 条进入高优先级标题级复核，约 25 条读批次正文（含最小复现、SELFPROOF、源码定位）。9,522 条 PR 只做标题级分类，未逐条读正文。
- 本轮新增 **18 条**，均 `status: pending_audit`、`confidence: documented`（报告者自述/自带复现/源码定位；本轮未在本地复现，不写 `verified`）。

| 编号 | 来源 | 机制摘要 | 关系 |
|---|---|---|---|
| INF-2583 | [ops-nn#9827](https://gitcode.com/cann/ops-nn/issues/9827) | ACLGraph 捕获/重放的 BF16 aclnnMatmul 在 M=8 verify block 最后一行与 eager 结构性不一致（行 sum −17576 vs −5.55），输入与调用参数逐位相同。 | — |
| INF-2584 | [ops-nn#8997](https://gitcode.com/cann/ops-nn/issues/8997) | `aclnnDynamicBlockQuant` minScale=+Inf 被放行，scaleOut=0、y 全饱和到 INT8 127；NaN 静默退化为不启用上界；-1/-Inf 正确报 561002。 | — |
| INF-2585 | [ops-nn#8965](https://gitcode.com/cann/ops-nn/issues/8965) | `aclnnEmbedding` 负索引未拦截，两段 rc=0，对应行整行 NaN（NaN 数=负索引数×16）；正向越界行为不同。 | var INF-1983 |
| INF-2586 | [ops-math#5050](https://gitcode.com/cann/ops-math/issues/5050) | A3 上 `aclnnExpm1` 实际按 exp(x)−1 计算，|x|<2^-24 时 99.5% 输出 0，相对误差 100%，输出被 2^-23 格点锁死。 | — |
| INF-2587 | [ops-transformer#8413](https://gitcode.com/cann/ops-transformer/issues/8413) | 950PR fp16、scaleValue=0、headNum≥16、Skv%128≠0 时 FlashAttentionScore 输出整块偏离 mean(V)（Skv=129 偏差 0.095，192000/262144 元素超 1e-2）。 | var INF-2192 |
| INF-2588 | [ops-transformer#8709](https://gitcode.com/cann/ops-transformer/issues/8709) | `aclnnKvRmsNormRopeCacheV2` 不校验 index rank；3/4 维 index 两段 rc=0，kCache/ckvCache 与 2 维基线错位（512/4088、448/3580 元素）。 | var INF-1573 |
| INF-2589 | [ops-math#5253](https://gitcode.com/cann/ops-math/issues/5253) | 950 RegBase `aclnnMaximum/Minimum` 对同 dtype 输入先 Cast 到 out dtype 再比较（200→−56 后 max 得 100；INT64 min 得 −1 而非 0），与 max-then-cast 文档语义相反。 | — |
| INF-2590 | [ops-transformer#8343](https://gitcode.com/cann/ops-transformer/issues/8343) | A2 membase 路径的 MoeFinalizeRoutingV3/V4 完全不读 zero/copy/constantExpertRange，输出与纯 V2 逐位一致（[[4,4],[6,6]] vs [[30,30],[200,200]]）。 | cross TRN-0378 |
| INF-2591 | [ops-transformer#8666](https://gitcode.com/cann/ops-transformer/issues/8666) | MoeFinalizeRoutingV4 fp16 路径用 half 顺序累加，K=96 的 1024/1/−1024 输入数学结果 32.0、实测 0.0；大 K 抵消场景被静默算错。 | var INF-1582 |
| INF-2592 | [ops-transformer#7825](https://gitcode.com/cann/ops-transformer/issues/7825) | 950PR `aclnnSparseFlashAttentionV2` attentionMode=2 不读非空 value，V 恒取 key；value 全 0/全 1/等于 key 输出逐字节相同。 | — |
| INF-2593 | [flashinfer#9981](https://github.com/flashinfer-ai/flashinfer/issues/9981) | FlashInfer top_k/top_p 的 probs/logits 两条 kernel 把 k=0/p≤0 当空候选，静默返回 token 0；`top_k_renorm_probs(k=0)` 返回全零分布。 | var INF-1045 |
| INF-2594 | [msmodelslim#728](https://gitcode.com/Ascend/msmodelslim/issues/728) | HiF4 W4A4/W4A8 导出 `weight_scale` 的 Lv1/Lv2/Lv3 打包顺序与设备解码相反（应为 C0 AA 01 04，实测 80 20 55 C0）。 | — |
| INF-2595 | [transformers#51079](https://github.com/huggingface/transformers/issues/51079) | Qwen2 eager fp16 左 padding：加性 mask 的 −65504 与低于约 −16 的 score 相加舍入为 −inf，全掩码行 softmax 变 NaN 并污染整条序列（pad 22–40 触发，generate 返回全 0）。 | var INF-2158 |
| INF-2596 | [vllm#69663](https://github.com/vllm-project/vllm/issues/69663) | vLLM CUDA 激活/门控族 8 个 launcher 不做 stride 校验：同值 strided 输入 rel error 1.82；strided 输出只写一半单元并改写 view 外父 buffer。 | — |
| INF-2597 | [vllm-ascend#22199](https://github.com/vllm-project/vllm-ascend/issues/22199) | async EPLB commit 的 non_blocking 权重 copy 未与在途前向做流序同步；16 冗余专家 + step=30 时 gsm8k 95.6→89.3，加 synchronize 恢复 95.3。 | — |
| INF-2598 | [vllm-ascend#23084](https://github.com/vllm-project/vllm-ascend/issues/23084) | `causal_conv1d_update_npu` 从未实现，import 失败回退到只支持 dim-first 的 upstream 包装；GDN/KDA 的 dim-last 状态被按 mamba 布局读，输出 maxdiff=239。 | var INF-0194 |
| INF-2599 | [vllm-ascend#22944](https://github.com/vllm-project/vllm-ascend/issues/22944) | 910C PD 分离、P/D 都开 prefix caching 时，P→D MooncakeConnectorV1 传输的 KV 被错误复用；25.9K prompt 2-3/5 次空内容/回显/乱码，直连 D 5/5 正确。 | — |
| INF-2600 | [vllm-ascend#22798](https://github.com/vllm-project/vllm-ascend/issues/22798) | StagedWriteTensor 的非阻塞 H2D 元数据拷贝与 apply-write kernel 在 Ascend 上无序，KV 池命中请求间歇答错（9/9 engine 运行至少一次；阻塞拷贝 5/5 正确）。 | — |

### 重点留观 / 不收（NPU）

- [vllm-ascend#22659](https://github.com/vllm-project/vllm-ascend/issues/22659)：MooncakeHybridConnector 按位置配对 KV 地址、未校验条目数；源码分析未上机，影响仅投机解码接受率/TPOT（纯性能），留观。
- [vllm-ascend#22213](https://github.com/vllm-project/vllm-ascend/issues/22213)：首 prefill 产出坏 KV；0.23.0.post1 复现，0.30.0 单次测试正常，版本相关且证据未闭合，留观。
- [ops-transformer#7800](https://gitcode.com/cann/ops-transformer/issues/7800)：`aclnnApplyRotaryPosEmbV2` A2 不读 rotaryMode，与 INF-2111（`aclnnRotaryPositionEmbedding` 同机制）重复，已用 `--add-evidence` 归并，不另立条目。
- [ops-nn#10087](https://gitcode.com/cann/ops-nn/issues/10087)：`inplace_index_add` bf16 高重复索引累加错值/丢更新；属优化器/embedding 更新核，归训练域，不新建 `INF-*`。
- [ops-math#4989](https://gitcode.com/cann/ops-math/issues/4989)：`aclnnAddLora` 两段 rc=0 但 out 完全未写；属 LoRA 微调累加，归训练域，不新建 `INF-*`。
- [ops-transformer#8693](https://gitcode.com/cann/ops-transformer/issues/8693)、[#8746](https://gitcode.com/cann/ops-transformer/issues/8746)、[#8756](https://gitcode.com/cann/ops-transformer/issues/8756)、[ops-nn#9859](https://gitcode.com/cann/ops-nn/issues/9859)、[#9885](https://gitcode.com/cann/ops-nn/issues/9885)、[#10176](https://gitcode.com/cann/ops-nn/issues/10176)、[ops-math#5209](https://gitcode.com/cann/ops-math/issues/5209)、[#5230](https://gitcode.com/cann/ops-math/issues/5230)、[#5238](https://gitcode.com/cann/ops-math/issues/5238)：本轮只做标题级筛选，未深读正文；保留为下一轮补查候选，不预判为无问题。
- 其余 1,918 条关键词候选中的大多数只做标题级筛选，未逐条判定；9,522 条 PR 只做标题分类。未覆盖范围见 §4，不宣称完整扫描。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：`gh_api.py` 从 profile `.env` 读取 `GITHUB_TOKEN`，所有请求 `authenticated: true`；Search 间隔 ≥2 秒。
- **API 账本**：7 仓 × 3 关键词组（correctness / silent / NaN）= **21/21 HTTP 200**，`created:>=2026-11-19`（`cutoff=$(date -u -d '7 days ago' +%F)`），`per_page=100`、`--max-pages 3`，各 total 均 <100，无分页缺口。计数（correctness/silent/NaN）：vllm 22/34/3，sglang 7/15/1，transformers 3/13/1，TRT-LLM 1/0/1，triton 1/5/2，tt-metal 3/18/9，flashinfer 5/5/5；去重后 **135 个 issue**。
- **新增条目（3 条）**：INF-2595（transformers #51079，var INF-2158）、INF-2596（vllm #69663）、INF-2593（flashinfer #9981，var INF-1045）。
- **补充证据（2 条）**：INF-1766 + [triton#12860](https://github.com/triton-lang/triton/issues/12860)（同一 `_compare_and_swap` 比较器，单个 NaN 级联覆盖全部有限元素）；INF-0532 + [vllm#69632](https://github.com/vllm-project/vllm/issues/69632)（block-level KV load failure 后 worker 侧 num_computed_tokens 未回退）。
- **留观/不收**：
  - [vllm#68671](https://github.com/vllm-project/vllm/issues/68671)：NIXL push + UCX SGL 可能让 D 端在 KV 未写完时解码；当前 vLLM 固定 NIXL 1.5.2 不受影响，属升级 NIXL 的条件性风险，留观。
  - [vllm#68779](https://github.com/vllm-project/vllm/issues/68779)：Hybrid Mamba + DFlash 的 block split 用错 block_size，默认 0% prefix-cache 命中；纯性能，不收。
  - [vllm#69154](https://github.com/vllm-project/vllm/issues/69154)：GLM-5.3-Flash + MTP + prefix caching 静默重复 token，根因未定位，留观。
  - [sglang#51990](https://github.com/sgl-project/sglang/issues/51990)：双流 MoE 的共享专家读与原地 routed 写存在跨流竞争；报告者实测 3.09M 次服务期 MoE 调用 0 次命中，仅 capture warm-up 约 20%，当前无服务期影响证据，留观。
  - [triton#12858](https://github.com/triton-lang/triton/issues/12858)：`tl.dot_scaled` FP16 E8M0 lowering 静默产生 NaN/Inf/0；机制独立、证据强，但本轮未写入（优先归并 #12860），标记待补。
  - [flashinfer#9887](https://github.com/flashinfer-ai/flashinfer/issues/9887)：min_p fallback 可返回零概率 token；与 INF-1045/#9981 同属采样退化族，未另立条目，留观。
  - [tt-metal#66564](https://github.com/tenstorrent/tt-metal/issues/66564)：`ttnn.maximum/minimum/bias_gelu` 标量重载静默丢弃 activations（closed）；本轮未深读，留观。
  - [transformers#51114](https://github.com/huggingface/transformers/issues/51114)、[#51120](https://github.com/huggingface/transformers/issues/51120)、[#51216](https://github.com/huggingface/transformers/issues/51216)：异步连续批处理输入未初始化、多模态 position_ids 轴错、混合 decoder-start-token；标题级候选，保留补查。
  - 其余 RFC/feature/tracking、纯性能、fail-loud、训练域候选按标题级分类不收，未逐条列全。
- **arXiv**：3 组查询（`"silent data corruption"` + LLM、`"bit flip"` + inference、`"KV cache"` + corruption）均 HTTP 200 有效，返回 8 + 25 + 20 篇。
  - 新出现且与推理相关：[2611.25945](https://arxiv.org/abs/2611.25945) BlockJack（跨租户 paged KV block-table 重定向故障攻击）、[2611.34684](https://arxiv.org/abs/2611.34684) BridgeHammer（CPU-iGPU 共享内存 Rowhammer 后门）。两者均属外部攻击，非自然发生的静默故障；本轮未入编目，留观/待安全范围评估。[2611.34630](https://arxiv.org/abs/2611.34630) 为 bit-slice 加速器，非故障。
  - 其余 ID（2610.37033/2610.30798/2610.27621/2610.08739/2610.06816/2610.03117/2608.30769/2608.11034/2608.23774/2604.10390/2604.00726/2502.12340/2609.19743/2610.18557/2611.06138 等）已在既往日报或编目中；本轮没有新的自然静默故障论文。

## 3. 回访

- **INF-1534**：cann/ops-nn#6232 仍 open（updated 2026-11-21T16:23:45+08:00）；最新评论仍为 2026-11-16 的进展说明，关联 PR #11647 仍 open、责任人 KaranocaVe 2026-10-23 指派后无新进展。结论不变。
- **INF-1749**：cann/ops-nn#6256 仍 open（updated 2026-11-21T16:23:44+08:00）；最新评论仍为 2026-11-16 的进展说明，关联 PR #11271 仍 open、指派后无新进展。结论不变。
- **INF-2564**：vllm-ascend#21902 仍 open（updated 2026-11-16T11:59:43Z）；仅有 2026-11-16 的 @Tflowers-0129 PTAL 评论，无修复 PR/发布说明。结论不变。
- 本轮新增 18 条均来自本批次正文，尚无后续更新可回访；其余窗口内被更新触发的旧项只做标题级复核，未逐条重读评论。

## 4. 未覆盖范围与缺口

- NPU 采集：22 源中 20 源完整；`Ascend/pytorch` pulls 第 4–11 页、`cann/ops-transformer` pulls 第 12–18 页因遍历中总数变化未取全/未验证。两仓 issue 列表完整，pulls 的窗口尾部候选可能遗漏。
- NPU 评估：7,251 条 issue → 1,918 条关键词候选 → 455 条标题级高优先级 → 约 25 条正文深读；9,522 条 PR 仅标题分类。未逐条评估的候选保留在批次 `inference_delta.jsonl`，下一轮可续查；不能据此宣称「无新发现」或「完整扫描」。
- 新增条目多为 A2/A3/950PR/910C + CANN 9.2/9.3 实测；910B/A5 的等价触发面未逐一验证，条目里写的是报告者实测机型，不外推。全部为本轮未本地复现，`confidence: documented`。
- GitHub：21 组 Search 均 200 且 total<100，无分页缺口；135 个唯一 issue 只深读约 6 条正文，其余按标题级分类，未逐条拉评论线程。core API 只用于 2 条 issue 详情核验。
- arXiv：仅 3 组短语查询，未覆盖全库；BlockJack/BridgeHammer 只读摘要，未读全文。
- 未对 msprobe/mssanitizer 工具误报做新实验，相关结论仍为「未评估/无实测」。

## 5. 写入与校验

- 新条目 18 条：**INF-2583～INF-2600**，均 `status: pending_audit`，由 2026-11-27 02:00 UTC 审计复核。
- 关系核对：INF-2585→INF-1983（EmbeddingBag）、INF-2587→INF-2192（FIA V5）、INF-2588→INF-1573（KvRmsNormRopeCache regbase）、INF-2590→TRN-0378（训练库 MoeFinalizeRoutingV3 A2）、INF-2591→INF-1582（MoeFinalizeRouting 精度口径）、INF-2593→INF-1045（SGLang seeded 空分布）、INF-2595→INF-2158（Qwen4-Exp eager 左 padding）、INF-2598→INF-0194（SGLang causal_conv1d_update 回退核）；工具回显标题均与目标一致。
- 补充证据：INF-1766 + triton#12860；INF-0532 + vllm#69632；INF-2111 + ops-transformer#7800。
- 回访修改：INF-1534、INF-1749、INF-2564（各追加 notes）。
- `catalog_tool.py build` 生成 `views/` 成功；`validate` 返回 `ok: true`，entries **2600**、patterns **202**、errors 0、warnings 46、pending_audit 18。
- 编目统计：故障 2583、结论 8、方法 8、指针 1；active 2508、merged 15、retired 59、pending_audit 18。

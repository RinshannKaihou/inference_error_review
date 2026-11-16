# 每日扫描日报 2026-11-16（推理库）

## 0. Git 前置检查

- 工作目录 `/workspace/inference_error_review`；分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 `origin/main` 一致（`7732ed57f881692701d4669f4b58dab9fdff12ad`），ahead/behind 0/0。通过，开始扫描。
- 本库无脏文件、无暂存变更、无本地待推提交，远程未领先。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261116T230012756346-r17zx2_p`，`generated_at=2026-11-16T23:00:12Z`，`cached: false`，`complete: true`；`failures` 为空；22/22 源 `complete`；registry_hash `2f6422d0…8d94d`。
- 窗口 `2026-11-15T22:00:12Z → 2026-11-16T23:00:12Z`（约 25 小时）。inference 候选索引 **35,132 条**（19 个推理适用仓库，含历史滚动缓存）；`created_at` 落在窗口内的新候选 **1,002 条**（issue 332 + PR 670；GitCode 801 + GitHub 201），窗口内被更新触发的旧项 **2,498 条**（只做标题级复核）。
- 判定方法：332 条新 issue 标题全部过目；排除 Documentation/Requirement/Question/RFC 与纯 CI 后 **196 条**逐条判定，其中 **17 条**读批次缓存正文（含 #7637/#7646/#7674/#7678/#7689/#7699/#7648/#7712/#8881/#8907/#8925/#4791/#4793/#4786/#4810/#4797/#712）+ 2 条 GitHub NPU 仓正文；其余按标题+摘要判定。670 条 PR 只做标题级分类，未逐条读正文。

### 逐仓覆盖（issue/PR 总数与页数，均 complete）

| 平台 | 仓库 | fetched | 分页明细 | complete |
|---|---|---|---|---|
| gitcode | Ascend/pytorch | 355 | issues 78(1p) pulls 277(3p) | 是 |
| gitcode | Ascend/op-plugin | 41 | issues 7(1p) pulls 34(1p) | 是 |
| gitcode | Ascend/torchair | 98 | issues 9(1p) pulls 89(1p) | 是 |
| gitcode | cann/ops-transformer | 494 | issues 190(2p) pulls 304(4p) | 是 |
| gitcode | cann/ops-nn | 1055 | issues 706(8p) pulls 349(4p) | 是 |
| gitcode | cann/ops-math | 297 | issues 99(1p) pulls 198(2p) | 是 |
| gitcode | cann/ge | 55 | issues 4(1p) pulls 51(1p) | 是 |
| gitcode | cann/runtime | 104 | issues 19(1p) pulls 85(1p) | 是 |
| gitcode | cann/hccl | 97 | issues 21(1p) pulls 76(1p) | 是 |
| gitcode | cann/hcomm | 159 | issues 48(1p) pulls 111(2p) | 是 |
| gitcode | mindspore/mindspore | 35 | issues 3(1p) pulls 32(1p) | 是 |
| gitcode | mindspore/mindformers | 13 | issues 2(1p) pulls 11(1p) | 是 |
| gitcode | Ascend/msprobe | 21 | issues 14(1p) pulls 7(1p) | 是 |
| gitcode | Ascend/mssanitizer | 13 | issues 1(1p) pulls 12(1p) | 是 |
| gitcode | Ascend/MindSpeed | 12 | 训练域，不进本视图 | 是 |
| gitcode | Ascend/MindSpeed-LLM | 14 | 训练域，不进本视图 | 是 |
| gitcode | Ascend/MindSpeed-RL | 0 | 训练域，无更新 | 是 |
| gitcode | Ascend/MindIE-LLM | 0 | 无更新 | 是 |
| gitcode | Ascend/msmodelslim | 36 | issues 19(1p) pulls 17(1p) | 是 |
| github | vllm-project/vllm-ascend | 555 | issues 6 页 | 是 |
| github | sgl-project/sgl-kernel-npu | 15 | issues 1 页 | 是 |
| github | triton-lang/triton-ascend | 56 | issues 1 页 | 是 |

合计 fetched 3,525。三个 MindSpeed 仓属训练 scope，不进入本视图。

### 新增条目（15 条，pending_audit）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2568 | [ops-transformer#7674](https://gitcode.com/cann/ops-transformer/issues/7674) | `aclnnMsaIndexScore` sparse_mode=3 下 kernel `(void)attenMask` 直接丢弃 mask：全 0/全 1/上三角三组输出 FNV 哈希逐字节相同（blk1=12416），不传 mask 为 16384；bf16/fp16、两组 Hq/D 一致，两段 rc=0。 |
| INF-2569 | [ops-transformer#7689](https://gitcode.com/cann/ops-transformer/issues/7689) | `aclnnBlockSparseAttention/V2/V3` 在 blockSparseMask 全零（空选择）时 kernel `curSelectNum==0 → continue`，attentionOut/lse 不写；V3 脏值 0.977~1.997e11（输入值域 [-1,1]），同输入复跑 74→64、6 次 5 种签名；部分选择 40/40 全对。 |
| INF-2570 | [ops-transformer#7678](https://gitcode.com/cann/ops-transformer/issues/7678) | `aclnnNsaCompressAttentionInfer` 多 token 场景每进程**首次**调用输出 0.03%~0.35% 元素恒为 0.0/128.0（落在 DV=128 边界），20 个新进程 14 次首调错；单 token 20/20 正确；2/20 伴随 507015。 |
| INF-2571 | [ops-transformer#7712](https://gitcode.com/cann/ops-transformer/issues/7712) | `aclnnCausalConv1dFn` 的 `InferIsFnMode` 3D 分支把长度 1 判为 update，派发 update kernel 无条件读缓存历史；prefill 未传 initialStateMode 时应零历史，输出与 prefill 语义不符（源码定位，无实测数值）。 |
| INF-2572 | [ops-nn#8881](https://gitcode.com/cann/ops-nn/issues/8881) | `npu_quant_matmul` w8a8 动态量化 GEMM 跨 M 档不逐位一致（M≤256 vs M≥257，NZ 档界 257）；27B 模型冷/温 prefix-cache 同 token logprob band mean 4.8e-3、max 1.9e-1，可翻转工具调用。 |
| INF-2573 | [ops-nn#8925](https://gitcode.com/cann/ops-nn/issues/8925) | `aclnnAscendQuantV3` NZ 合法形态（x fp32 NZ、dstType=3、y INT32、round）两段 rc=0/0 但 y 零字节写出：NZ tiling 只写 E/K/N/needCoreNum，其余 22 字段全 0，40 block 空转；K=4..64 六点全复现，ND 对照正确。 |
| INF-2574 | [ops-math#4791](https://gitcode.com/cann/ops-math/issues/4791) / [#4793](https://gitcode.com/cann/ops-math/issues/4793) | `aclnnReduceSumV2` axes=1 且 out≥2304(fp32)/≥2050(fp16) 时 rc=0 但 out 完全不写；Mean/Min/MaxV2 同阈值、ProdV2 4096 起，四份 kernel 独立；R=2049 及以下正常。 |
| INF-2575 | [ops-math#4786](https://gitcode.com/cann/ops-math/issues/4786) | `aclnnLinspace` 整数输出按 float32 计算后截断：`linspace(1e8,1e8+6,4,int64)` 得 [1e8,1e8,1e8+8,1e8+8]；INT64 被 OutPromoteType 降为 FLOAT、INT32 kernel 模板也是 `<int32_t,float>`；float 输出正常。 |
| INF-2576 | [ops-math#4797](https://gitcode.com/cann/ops-math/issues/4797) | `TopKV2D`（AICPU）排序/分组/选择/堆路径用数值容差 `IsValueEqual` 判并列：`[1,1+2e-6,1+4e-6]` k=1 返回 index=0，k=2 时 values 与 indices 不对应；FLOAT/DOUBLE 均受影响；修复提交 f13883ba（未核实发布）。 |
| INF-2577 | [tt-metal#65954](https://github.com/tenstorrent/tt-metal/issues/65954) | `ttnn.max/min` INT32：padding 哨兵以 float 位模式存放，#61611 改用 `pack_scalar_into_uint32` 数值转换后哨兵变 0；dim0/1 非 32 倍数时全负组 max、全正组 min 返回 0；单测 4/4 失败（ATOL 38712）。 |
| INF-2578 | [flashinfer#9508](https://github.com/flashinfer-ai/flashinfer/issues/9508) | `cutlass_fused_moe` 不检查 input stride，列切片视图按 packed 行步长读：仅 row0 正确，其余行读到 padding/邻行，无告警；`.contiguous()` 后正确。 |
| INF-2579 | [vllm#67923](https://github.com/vllm-project/vllm/issues/67923) | vLLM Rust 前端 `qwen3_coder` 在缺 `</parameter>` 时把下一个 `<parameter=state>` 并入 city 值并丢掉 state；Python parser 返回三个正确键，qwen3_xml 同样；无 error。 |
| INF-2580 | [ops-transformer#7699](https://gitcode.com/cann/ops-transformer/issues/7699) | `aclnnGenericBlockSparseAttentionMetadata` 返回 SUCCESS 但整段输出未写（header 全缺失）：round-1 24/200、随机探针 12/41（29%）、独立容器 14/90；同进程两次调用可一次写一次不写；主算子对全零 metadata 仍 SUCCESS。 |
| INF-2581 | [ops-transformer#7637](https://gitcode.com/cann/ops-transformer/issues/7637) / [#7646](https://gitcode.com/cann/ops-transformer/issues/7646) | `aclnnQkvRmsNormRopeCacheWithKScale` M-RoPE MX 极小 amax 时 E8M0 scale 静默为 0（码字应为 max(-127,ceil(log2(a/448)))+127），qOut/kCache 全 0x7E/0xFE（FP8 NaN）；两提交实测边界不同（e_b=-125/-126 vs a_b≤2^-118.19），根因是否同源待确认。 |
| INF-2582 | [ops-transformer#7648](https://gitcode.com/cann/ops-transformer/issues/7648) | `aclnnQkvRmsNormRopeCacheWithKScale` slotMapping 为负/越界时不校验，按 slot/BlockSize 直接取址：slot=-1 写 kCache slot31、slot=32 写进 vCache slot0；同输入 6 次 kCache 得 5 种签名，rc=0/0。 |

均 `status: pending_audit`、`confidence: documented`（报告者自述+自带复现；本轮未在本地复现）。来源勘误：无。

### 重点留观 / 不收（本轮判定）

- **数值/静默但未入编（证据或范围不足）**：ops-transformer **#7685**（四个整型属性按 char 读取，越界值被截断后执行；输入超文档上界，属非法参数域）、**#7688**（TND ragged-KV 按 max-seqlen 迭代"幻影"块，online softmax 状态污染；未给数值影响量化）、**#7690**（innerPrecise=1 时 softmaxLse fp16 先量化，相对误差 6.6e-4~7.8e-4，属该精度档预期量级，未证任务影响）、**#7676**（PoolKeyIndexer returnValue=false 不校验 sparseValuesOut 形状、不写该输出；调用方选择不取值，影响未实测）、**#7632**（MhcPreSinkhorn needBackward=false 五个出参不写；输出属反向辅助）、**#7671**（moe_init_routing_v3/v4 空 tensor 输出未初始化 + golden 崩溃；空输入边界，未证主路径）、**#7641/#7642**（FusedFloydAttention 形状不交叉校验 / fp32 输入 361001；文档与实例化不符，后者 fail-loud）。
- **训练反向**：ops-transformer **#7694/#7695/#7697/#7698/#7700**（CompressorGrad / BlockSparseAttentionGrad 精度）、**#7696**（GenericBlockSparseAttentionGrad bf16 超差）、**#7655**（BlockSparseAttentionGrad blockShapeX 非 16 倍数未拦截）——梯度输出，归训练域，留观。
- **错误码/校验/文档类不收**：ops-transformer **#7675/#7719**（161002 与 561002 归位）、#7682/#7686/#7687（dtype/属性域校验缺口，无静默错值证据）、#7713（UT 失败）、#7720/#7683（output validation 方法学）、#7704/#7705（metadata doc-镜像不一致）；ops-nn **#8910/#8911/#8900/#8903**（DequantSwigluQuant 非法入参/表外码）、#8886（错误码归位）、#8887（ONNX 插件越界读，构图期 fail-loud）、#8923/#8906（无正文）；ops-math **#4826–#4829**（UT 缺失门禁）、#4816（OpDef 冗余声明）、#4814/#4804（文档/空指针校验）。
- **纯性能/特性/CI**：vllm-ascend #22007/#22051/#22062/#22063/#22084/#22085/#22087、#21976；Ascend/pytorch #6119/#6120/#6130/#6131/#6133/#6143/#6145/#6146/#6147/#6148；triton-ascend #3203/#3207；cann/runtime #1382/#1384；hccl/hcomm 检视意见跟踪单（#977/#979/#1347–#1357）。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 `GITHUB_TOKEN`（token=no），全部请求未认证；search 间隔 7 秒。
- **API 账本**：7 仓 × 3 关键词组（correctness / silent wrong output / NaN）= **21/21 HTTP 200**，`created:>=2026-11-09`（`cutoff=$(date -u -d '7 days ago' +%F)`），`per_page=100`，各 total 均 <100，无分页缺口。计数（correctness/silent/NaN）：vllm 25/15/5，sglang 20/5/9，transformers 3/5/1，TRT-LLM 0/0/1，triton 0/2/1，tt-metal 5/3/5，flashinfer 3/5/7。去重后 **100 个 issue**，其中 71 个未在既往日报/编目出现。
- **新增条目（3 条）**：INF-2577（tt-metal #65954）、INF-2578（flashinfer #9508）、INF-2579（vllm #67923）。
- **留观/不收**：
  - [vllm#67603](https://github.com/vllm-project/vllm/issues/67603)：Quark W4A4 gpt-oss MoE 在 SM100 加载失败（fail-loud），加分支后 `CutlassExpertsMxfp4` 静默丢 w1_bias/w2_bias；目前需自行打补丁才进入静默路径，作为条件性缺陷留观。
  - [sglang#50521](https://github.com/sgl-project/sglang/issues/50521)：FP16 Stream-K GEMM 精确 0 结果产出全 NaN（H20 3/3），NaN 可被上层直接看到，且仅单卡未扩散验证，留观。
  - [sglang#50715](https://github.com/sgl-project/sglang/issues/50715)：NPU RMSNorm 忽略显式 `cast_x_before_out_mul=True`，属舍入顺序差异，无任务影响实测，留观。
  - [transformers#50669](https://github.com/huggingface/transformers/issues/50669)：并发 chat template 丢 assistant mask（closed），影响 SFT 预处理（训练域），不收。
  - [transformers#50581/#50545/#50612/#50622](https://github.com/huggingface/transformers/issues/50581)：bf16 AdamW / MoE 负载均衡 / TimesFM / LeRoPE，训练域，不收。
  - [flashinfer#9529](https://github.com/flashinfer-ai/flashinfer/issues/9529)（no kernel image，fail-loud）、[#9501](https://github.com/flashinfer-ai/flashinfer/issues/9501)（head_dim=256 smem 不足，报错）不收；[#9486](https://github.com/flashinfer-ai/flashinfer/issues/9486) 沿用 11-14 留观。
  - [tt-metal#65680/#65634](https://github.com/tenstorrent/tt-metal/issues/65680)（悬赏聚合单）、#65873（tt-train lerp 溢出，训练）、#65888（特性）不收。
  - [triton#12757](https://github.com/triton-lang/triton/issues/12757)：沿用 11-13/11-15 留观，状态未变。
  - vllm #67922/#67865/#67769/#67357/#67334、sglang #50756/#50405/#50241 等 RFC/feature/tracking 不收；#67183（启动崩溃）、#67575（cudaErrorIllegalAddress）fail-loud 不收。
- **arXiv**：2 组短语查询（`"silent data corruption" AND LLM`、`"bit flip" AND inference`，sortBy=submittedDate）HTTP 200 有效，返回 8 + 25 篇；逐 ID 核对 2610.37033/2610.30798/2610.27621/2610.08739/2610.06816/2610.03117/2608.30769/2608.11034/2608.23774/2604.10390/2604.00726/2502.12340，均已在既往日报或编目（INF-0006/0097/0313/2247 等），**无新论文**。

## 3. 回访

- **INF-1534**：cann/ops-nn#6232 仍 open（updated 2026-11-16T16:21:42+08:00）；报告者补充评论称关联 PR #11647 仍 open、责任人 KaranocaVe（2026-10-23 指派）、指派后评论静止。结论不变。
- **INF-2059**：cann/ops-nn#6227 仍 open（updated 2026-11-16T16:21:31+08:00）；关联 PR #11232 仍 open，2026-09-27 指派后无进展。结论不变。
- **INF-1749**：cann/ops-nn#6256 仍 open（updated 2026-11-16T16:21:28+08:00）；关联 PR #11271 仍 open，2026-09-28 指派后无进展。结论不变。
- **INF-2564**：vllm-ascend#21902 仍 open（updated 2026-11-16T11:59:43Z），未见修复 PR/发布说明，结论不变。
- 其余 2,498 条窗口内被更新的旧项只做标题级复核，未逐条重读评论。

## 4. 未覆盖范围与缺口

- NPU：22 源 issue/PR 列表均 `complete`、无 API 失败；但 1,002 条新候选只深读 17 条正文，179 条 issue 按标题+摘要判定、670 条 PR 只做标题分类，存在漏判可能。
- 新增 NPU 坏例多为 A2/A3/950PR + CANN 9.2/9.3 实测；910B/A2 的等价触发面未逐一验证，条目里已写实测机型，不外推。
- GitHub：21 组 search 均 200 且 total<100，无分页缺口；只深读 7 条正文，未逐条拉评论线程；core API 未用于状态核对（无 token）。
- arXiv：仅 2 组短语查询，未覆盖全库。
- 未对 msprobe/mssanitizer 工具误报做新实验，相关结论仍为「未评估/无实测」。

## 5. 写入与校验

- 新条目 15 条：INF-2568～INF-2582，均 `status: pending_audit`，由 2026-11-17 02:00 UTC 审计复核。
- 回访修改：INF-1534、INF-2059、INF-1749、INF-2564（各追加 notes）。
- `catalog_tool.py build` 生成 `views/` 成功；`validate` 返回 `ok: true`，entries 2582、patterns 205、errors 0、warnings 71、pending_audit 15。

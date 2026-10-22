# LLM 推理故障每日扫描 — 2026-09-15

- **扫描日期**：2026-09-15（UTC 23:00 执行）
- **扫描窗口**：GitHub `created:>=2026-09-13`（上次扫描 09-14 覆盖 `>=2026-09-12`，本窗口重叠 09-13/09-14 两天，已逐条对主文档去重；arxiv 按提交日期取最新）
- **基线**：FAULT_MASTER_REFERENCE.zh.md 共 **1082 条**（1.1–15.10 共 968 + K1.1–K10.31 共 114；无 A* 前缀条目）

## 结果概览

- GitHub Search API：10 组查询全部一次成功（无限流 403、无 Unicorn 错误页）
- core REST 详情：20 次调用全部成功（`--http1.1`）
- **arxiv：6 组查询全部成功**——09-13/09-14 连续两天 IP 级 429 后今日恢复（20s 间隔 + UA）；全部命中已编目或为方法论文，无新编目
- 合并：**18 条新增**（第 1 类 +3、第 4 类 +1、第 5 类 +1、第 6 类 +4、第 7 类 +2、第 8 类 +1、第 10 类 +2、第 11 类 +2、第 12 类 +1、第 13 类 +1 = 9 个新子签名 + 9 个已知类别变体），1082 → **1100**；另有 1 条判定升级（vLLM #56889 09-14 留观 → 经 #57019 最小复现确认，随 6.239 合并记述，不占新编号）

## 🆕 新故障模式 / 新子签名（本日合并）

| # | 条目 | 归类 | 为什么是新签名 |
|---|---|---|---|
| 1 | [vLLM #57014](https://github.com/vllm-project/vllm/issues/57014) `hadacore_transform` 尾部 warp `cp.async.wait_group` 经 `switch(k+diff)` 下标 −1 无 case → 不发 wait，`rows%8∈1..4` 的尾行读未落地的 shared memory；读到的是**上一个内核**留在该 SM 的共享内存残留（117/188 行整行恢复出前次 launch 的输入）——int4_per_token_head KV 路径 Q/KV 双中招，损坏 KV 落 cache 持久化 | 第 1 类 1.218 | **`cp.async` wait-group 负下标越 case** + **跨 launch 共享内存残留数据泄漏**：竞态窗口读到的不只是未定义值而是前一内核数据；sm_80+ 全架构、A100/H100/4090 三代复现 |
| 2 | [tt-metal #56659](https://github.com/tenstorrent/tt-metal/issues/56659) sfpi ≥7.75.0 把 `vFloat ==/!=` 下推为**整数减 + SFPSETCC LREG_EQ0**（位等同判断）：`NaN==NaN`=true、`x!=x`（isnan 惯用法）=false、`+0.0==-0.0`=false；7.70–7.74 是浮点减（NaN 对/inf 错）——两种下推都不符合 IEEE 754 | 第 1 类 1.219 | **编译器 lowering 语义反转随版本摆动**：isnan 惯用法静默失效且 7.75.0 起**方向翻转**（旧版 NaN 对、新版 NaN 错）；1.109/1.151（比较器 NaN/±0 dtype 分裂）族的编译器维度 |
| 3 | [tt-metal #56626](https://github.com/tenstorrent/tt-metal/issues/56626) `ttnn.topk` 按原始位模式的 (sign, magnitude) 排序、无 NaN case：`-NaN` 排在 `-inf` 之下 → `largest=True` 时 **-NaN 整条静默掉出 top-k**（16/16 值 + 16/16 索引全错）；+NaN 恰与 torch 一致纯属巧合 | 第 1 类 1.220 | **NaN 按 mantissa 排序导致元素集错误**：不是数值 re-encode 而是返回**不同的元素集合**；golden 就是 `torch.topk` 仍然静默分歧 |
| 4 | [vLLM #56655](https://github.com/vllm-project/vllm/issues/56655) 混合 prompt embedding（`prompt_is_token_ids` 逐位置选择 learned embedding vs 供给张量行）的**选择 mask 不进 prefix-cache 键**：embedding 位归一化为 token id 0（本身是合法词表 id）；缓存 A 后同样的 B 命中 32 个 token 生成完全不同输出 | 第 6 类 6.236 | **输入条件维度（per-position 源选择）缺失于缓存键**：与 6.169（外部键丢 cache_salt）、6.180（prompt_cache_key 丢弃）、5.37（名字排序契约）同族但缺的维度是**同 token id 的双源语义** |
| 5 | [vLLM #57087](https://github.com/vllm-project/vllm/issues/57087) GLM-5.3 hybrid nvfp4 GB10/sm121：并发 batching 下非 ASCII 生成损坏——近失 CJK 字形、byte-fragment token 在损坏槽位**霸占整个 top-3**（分布级 scramble 非 near-tie）、自回归级联（一次 fragment 后后续 fragment 超高置信度）→ 80 FFFD/492 字符；NaN 探测 0 命中、顺序请求干净 | 第 6 类 6.237 | **并发诱导的 fragment-token 分布毒化 + 自回归级联**：损坏在 sampled token ids（state poisoning 类），ASCII 不触发、非 ASCII 密度是灵敏度旋钮 |
| 6 | [vLLM #57064](https://github.com/vllm-project/vllm/issues/57064) ROCm MI300X PD 分离（MoRIIO WRITE）：**remote-prefilled** 请求的 decode 在**任何** CUDA graph 模式下输出 `locklocklock…` 垃圾；`DECODE_CUDAGRAPH_MODE=NONE`（eager）同栈解码连贯；KV 传输逐层 bit-exact 已排除 | 第 6 类 6.238 | **PD 远端 KV + CUDA graph replay 交互**：传输内容完好、colocated 不触发、v0.27.1 不触发——故障在 graph 捕获的元数据/buffer 假设而非数据面 |
| 7 | [vLLM #57019](https://github.com/vllm-project/vllm/issues/57019) HiSparse host-resident KV：**decode 从第 2 个 token 起全噪声**（首个生成 token 与对照完全一致、`max_tokens:1` 链式 8 次完全复现对照）→ prefill 完好、host-mirror 读路径损坏；GSM8K 5-shot 0.0000 vs 0.9409；单节点 TP2 最小复现 | 第 6 类 6.239 | **host-mirror decode 读路径**损坏且**首 token 正确**的确定性签名；#56889（09-14 留观）经此最小复现确认同根因，升级为编目 |
| 8 | [tt-metal #56680](https://github.com/tenstorrent/tt-metal/issues/56680) Gemma4 dFlash spec 路径：**批式 baseline 突发后**的 solo spec 请求回答**上一个请求的 prompt**（400-600 token 条件于错误上下文）；`_replay_on_first` 守卫在 `reseed()`/`capture()` 时序下失效——`_out` 仍持有前一请求的 replay | 第 11 类 11.78 | **spec session 复用时 replay 跳过 → 前请求 token 注入**：6.163（跨请求上下文污染）族但根因在 spec 状态机的 replay-on-first 守卫时序，触发是"批式→solo"的正常流量转换 |
| 9 | [TRT-LLM #19172](https://github.com/NVIDIA/TensorRT-LLM/issues/19172) prompt-tuning embedding 表的异步 H2D copy **outlive 其 pooled-pinned host 源**：pinned pool 复用把缓冲区交还时新分配者可在 copy 进行中覆写——高并发 + 同 GPU 竞争 CUDA context 才 widening 到可观察 | 第 12 类 12.37 | **异步拷贝生命周期 vs pinned pool 复用竞态**：host 侧对象生命周期缺陷（非内核/架构），静默把 prompt embedding 表复制成半旧半新 |

## 🔄 已知类别的新案例（本日合并）

| # | 条目 | 归类 | var 关系 |
|---|---|---|---|
| 10 | [vLLM #57016](https://github.com/vllm-project/vllm/issues/57016) `VLLM_BATCH_INVARIANT=1` 对 Triton fused-MoE 专家静默无效：在线 FP8 无 `per_act_token_quant` → **全批单动态 scale**（谁共享 batch 谁定标）；bf16 专家按 token 数选 kernel 配置无固定；flag 既不重定向也不警告 | 第 4 类 4.22 | var:4.17（batch-invariant matmul 自测违约 + TF32 泄漏）+ 4.13（确定性 flag 静默无效）族 |
| 11 | [vLLM #57017](https://github.com/vllm-project/vllm/issues/57017) DeepSeek-V2-Lite 在线 FP8 + TRITON_MLA（sm89 唯一 MLA 后端）：**同批 ≥3 序列** decode 的 CUDA-graph replay 每请求 NaN logits；compile-only 无 graph 干净、eager 干净、`--max-num-seqs 2` 绕过 | 第 7 类 7.90 | var:6.160（FlashInfer sm120 NVFP4+fp8KV 16-token IMA）族：CUDA graph × 批尺寸 × 量化交互 |
| 12 | [vLLM #56949](https://github.com/vllm-project/vllm/issues/56949) Dense DP 权重传输：`reconfigure_for_independent_dp_rank()` 把每个 dense DP engine 的 `data_parallel_rank` 重置为 0 → 非 0 DP replica 的 worker 用 `dp_rank*world_size+rank` 选 payload **永远选中 DP0 的 IPC 句柄**；`/update_weights` HTTP 500 | 第 5 类 5.38 | var:5.22（PD Mamba state dtype/布局失配）族相邻——分布式权重路由身份错位 |
| 13 | [SGLang #39626](https://github.com/sgl-project/sglang/issues/39626) `w8a8_block_fp8_matmul_triton` 对 `block_size=[32,32]` 静默接受 `BLOCK_SIZE_K=64` 配置：512 个输出元素全部 **64 而非 288**（BK32 正确）；官方 tuner 过滤该组合但手动配置无守卫 | 第 7 类 7.91 | var:7.58（mxfp4 runner else 吞 9 值 fail-late）族：量化内核配置守卫缺失 |
| 14 | [tt-metal #56577](https://github.com/tenstorrent/tt-metal/issues/56577) Gemma4-12B TP=8 warmup trace 捕获后 decode 从第 2 步起 PCC≈0.06 完全发散（31B 正常）：nkv_local=1 时 sliding block **按 2 倍计入**（头数整除 8 的 31B 不触发）；伴随缺陷 (A) warmup 全零 page table 绑定所有层到 block 0 | 第 8 类 8.87 | var:8.60（Gemma4 WH 长度门控 garble）族：Gemma4 滑窗 decode 几何 + trace 捕获 |
| 15 | [vLLM #57038](https://github.com/vllm-project/vllm/issues/57038) `truncate_prompt_tokens` 在多模态占位符**扩展之前**截断 → 最终 prompt 超限（请求 96 token 得 147）；音频 span 本可完整放入 | 第 10 类 10.67 | var:9.19（pretokenized 重分词 mask 长度漂移）族相邻：截断作用点在流水线错误阶段 |
| 16 | [vLLM #56981](https://github.com/vllm-project/vllm/issues/56981) Qwen3-ASR 音频预处理 vs HF native：16001 样本 → vLLM 14 token vs HF 13；321 样本 → 1 vs 7——继承 Omni 的 hop 倍数 pad + 硬编码 `WhisperFeatureExtractor` 忽略 checkpoint 的 `Qwen3ASRFeatureExtractor` | 第 10 类 10.68 | var:10.56（TRT-LLM tokenizer 别名表漂移）族：预处理实现与 checkpoint 原生路径分歧 |
| 17 | [transformers #48818](https://github.com/huggingface/transformers/issues/48818) Gemma4 audio tower eager 路径：bool 语义 `logical_not()` 作用在**加性 float mask** 上（0.0=允许 → True → 被填 invalid logits）→ 全序列 (近)均匀注意力、内容无关；sdpa（bool mask）正确 | 第 13 类 13.38 | var:8.86（非因果滑窗硬编码因果）族镜像：mask 语义约定（bool vs additive）在 eager/sdpa 间翻转；Gemma4 音频塔为 13.27/8.60 Gemma4 族新 locus |
| 18 | [vLLM #57032](https://github.com/vllm-project/vllm/issues/57032) DFlash drafter KV group 永不被识别 → 消费者把**所有** group（含 Mamba 组）当 draft 组处理，Mamba 组前缀缓存复用被静默禁用（每次启动都打警告但复用确实没了：子请求 36.4s vs 命中应快得多） | 第 11 类 11.79 | var:6.171（spec decode 静默禁用 hybrid prefix cache）族：dflash 方法不满足 mtp 专属的组识别规则 |

## 补充来源 / 判定升级（不占新编号）

- **vLLM #56889**（09-14 因"组合变量过多无根因"留观）→ **#57019 提供单节点 TP2、BF16 KV、无 offload、确定性、prefill/decode 分离的最小复现**，确认为同根因（HiSparse decode host-mirror 路径），两 issue 已在 6.239 中合并记述。

## 本窗口内其余命中——已编目（跳过）

- vLLM #56792 → 11.77；#56868 → 6.235；#56830/#56815/#56797/#56790 → 09-14 不收录维持；SGLang #39311 → 7.86；#39300 → 8.86；#39299 → 09-14 不收录维持；flashinfer #5221 → 4.21 补充来源（09-14 已并引）；#5192 → 09-14 不收录维持；#5190 → 7.87；triton #11747 → 09-13 不收录维持（fail-loud LLVM abort）；tt-metal #56550 → 1.214；#56532/#56521 → 09-14 不收录维持；#56522/#56520 → 1.217；#56518 → 1.216；#56517 → 1.215；SGLang #39400 → 09-14 不收录维持（安全面）

## 本窗口内其余命中——本日判定不收录（附理由）

| 条目 | 理由 |
|---|---|
| vLLM #57034 max_model_len auto-fit 后 ModelState 残留旧值 → CUDA graph 捕获 assert | fail-loud 启动 abort（AssertionError），无静默损坏；治理类 |
| vLLM #57029 AiterExperts + DeepEPHT world_size=2 IMA (gfx942) | 无根因隔离的 IMA crash 报告；fail-loud；留观 |
| vLLM #57008 XPU cpu-offload-gb (UVA) 不降峰值显存（WNA16 MoE） | 内存治理/容量核算，非性能正确性 |
| vLLM #56992 ROCm gfx1100 compressed-tensors 静默启用 fp8 KV 比 bf16 慢得多 | 性能回退（无错值）；与 7.52（KV scale 丢弃）触发面相邻但无损坏 |
| vLLM #56980 MiniMax-M3 MSA quack-kernels 0.6.5 SM100 crash | fail-loud crash，无静默损坏 |
| vLLM #56975 GLM-5.3-Flash GB200 p99 ITL 15-20× SGLang | 性能（ITL 尾延迟），非性能正确性 |
| vLLM #56973 GLM-5.3-Flash DP4 fused_recurrent_kda grid.z 超 65535 capture 失败 | fail-loud 启动 crash（grid 维度上限） |
| SGLang #39515 NEXTN B200 静默切 triton 后端 2.6× 慢 | 性能（后端自动选择无日志），无正确性损坏；与 6.164（envs 声明/运行时分裂）观察面相邻 |
| SGLang #39624 Qwen4-Exp PD 分离 PLE pool 未 sizing（closed: main 已修） | 已修复追踪（RESOLVED on main），fail-loud AssertionError；无静默形态 |
| SGLang #39634/#39499/#39521/#39393/#39429 | tracking/RFC/feature/roadmap 类 |
| flashinfer #5238 SM121 moe_utils 预构建 .so 无 SM12x image | fail-loud "no kernel image" 启动/调用错误（PR #5237 修 JIT 过滤器）；无静默损坏 |
| flashinfer #5233 GB300 FP8 paged prefill cuDNN vs TRT-LLM 持续 mismatch | bench refcheck 容差争议（报告自述"不确立哪个后端错"）；数值族留观（与 7.75/4.17 触发面相邻） |
| triton #11795 gfx1250 MLIR 常量折叠 0.0/0.0 segfault | 编译器 crash（fail-loud），无错值输出；留观 |
| tt-metal #56702 Quasar cached-program re-enqueue 不刷新 DFB configs | 与 1.201/#56281 同触发面（cached command sequence 失配）——作为 1.201 族语境补充记录于本报告，不另立条目 |
| tt-metal #56688 sdpa_decode bring-up (Quasar) | bring-up/feature 追踪 |
| tt-metal #56673 copy_dest_value | 报告者自撤（caller-side DataFormat mismatch，非 tt-metal 缺陷） |
| tt-metal #56474 Qwen3.6 prefill state/decode bucket 准备 | 任务追踪 |
| transformers #48807 serge integration triage | 集成 triage 汇总帖 |
| q09 全域 "silent data corruption" 搜索 | 全部无关仓库（agent 工具/PR 噪声），无 LLM 推理命中 |
| arxiv 2609.10264 Maverick（可验证私密推理） | 方法论文（MPC 委托矩阵乘），非故障案例，不收录 |
| arxiv 2606.04238 Recover-LoRA / 2509.10790 GoldenTransformer | 前者为 2-bit 恢复方法论文、后者为 FI 框架方法论文，不收录 |
| 其余 arxiv 命中（TrainSDC/SCOUT/LLM-PRISM/JITterFlip/GBFA/BLINK/BitFlipScope/RangeGuard/2604.17249/2601.19912/2608.05863/MetaKV/Nexus/Moment-KV/2608.29934 等） | 均已编目（逐条 grep 主文档确认） |

## API 调用记录

| # | 查询 | 结果 |
|---|---|---|
| q01 | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-13` | 21 命中 |
| q02 | `repo:vllm-project/vllm is:issue created:>=2026-09-13 wrong OR incorrect OR corruption in:title` | 3 命中 |
| q03 | `repo:sgl-project/sglang is:issue created:>=2026-09-13 correctness OR wrong OR corruption OR NaN` | 13 命中 |
| q04 | `repo:huggingface/transformers is:issue created:>=2026-09-13 wrong output OR incorrect` | 2 命中 |
| q05 | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-13 correctness OR wrong OR NaN` | 1 命中 |
| q06 | `repo:triton-lang/triton is:issue created:>=2026-09-13 NaN OR miscompile OR wrong` | 2 命中 |
| q07 | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-13 incorrect OR NaN OR numerics OR corruption` | 25 命中 |
| q08 | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-13` | 8 命中 |
| q09 | `"silent data corruption" LLM created:>=2026-09-13` | 24（全部无关噪声） |
| q10 | `repo:vllm-project/vllm is:issue created:>=2026-09-13 quantization OR "KV cache" OR NaN` | 50 命中 |
| core REST ×20 | vLLM #57087/#57064/#57059/#57038/#57034/#57032/#57019/#57017/#57016/#57014/#56981/#56949/#56655 + SGLang #39626/#39624/#39515 + TRT-LLM #19172 + transformers #48818 + tt-metal #56702/#56659/#56626/#56680/#56577（部分经 jq 截取） | 全部成功 |
| arxiv ×6 | SDC/bit-flip/KV-cache/correctness/GPU-soft-error 组合（20s 间隔 + UA） | 全部成功（09-13/09-14 连续 429 后恢复）；无新编目 |

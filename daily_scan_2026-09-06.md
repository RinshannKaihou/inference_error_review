# 每日扫描报告 — 2026-09-06

- 扫描日期：2026-09-06（UTC）
- 扫描窗口：GitHub `created:>2026-09-05`；arxiv 最近提交排序
- 目标仓库：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal
- 主文档基线：937 条（2026-09-05 合并后）

## 结果总览

- 共执行 10 组 GitHub Search API 查询（首批因脚本 `q` 参数传递错误整体失败，修正为 `--data-urlencode "q=..."` 后 10 组全部一次成功，无限流 403）+ 5 组 arxiv API 查询（HTTPS，`curl -sL`）
- 去重核实后：**39 条新条目合并进主文档**（7 个新子签名 `[新类别/新签名]` + 32 个已知类别变体 `[已知类别]`），937 → **976**
- 另有 2 条根因定位/修复 PR 作为已编目条目的补充来源引用（不占新编号）：SGLang #38224 → 8.65（#38143 MiniMax-M3 全 NUL token 的根因修正：量化 embedding 被静默忽略，**非注意力路径**）、SGLang #38212 → 6.193（#38031 GLM-5.3-Flash HiCache 损坏的修复：DSA index buffers 不迁移 + 压缩组分裂共享 index 行）
- 8 条修复 PR 直接并入对应新条目的来源字段（#55601→6.204、#55593→8.70、#55591→8.71、#11611→1.174、#11605→1.175、#55572→1.176、#38203→5.33、#55577→1.180）

## 新合并条目（39 条）

### 第 1 类 · 硬件/SDC（+8）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 1.173 | [FuseNestedLoops] a flattened loop with one inner scf.for whose carried value is used afterwards crashes the pass | triton #11612 | `[已知类别]` var:1.127 | 同一 FuseNestedLoops pass 第三形态：外层 flatten 循环体含 `scf.while` 边界 +携带值的 `scf.for`——pass 在 MLIR 断言死亡或产出 yield 不匹配的 `scf.if`；3.7.1/3.8.0 sm_120 复现 |
| 1.174 | [interpreter] create_fma performs double rounding for float32 (survives the fix in #11424) | triton #11610（修复 PR #11611） | `[新类别/新签名]` | interpreter 对 fp32 FMA 经 fp64 中间两步舍入：精确值落在 fp32 半途点时 fp64 RN 先选错边（4097²+2⁻³⁰ 例：compiled 16785410 vs interpreter 16785408）——#11424 只修了 fp16 |
| 1.175 | [interpreter] tl.sum silently loses NaN for bfloat16/float8 dtypes (integer bit-pattern addition) | triton #11604（修复 PR #11605） | `[已知类别]` var:1.151 | interpreter 把 bf16/fp8 存为原始整数位模式，`ReduceOps.sum` 直接 `np.sum` 整数位模式相加——NaN 位模式被整数求和再重解释为有限值，compiled 正确传播 NaN |
| 1.176 | [Bug fix] Fix Blackhole bitwise_right_shift out-of-range shift guard | tt-metal PR #55572（closes #55332） | `[已知类别]` var:1.103 | BH 越界移位量守卫把寄存器选择符（`LCONST_0`）传到指令修饰符位置且缺 `TTI_SFPCOMPC`：负移位量变左移（`8 >> -1 = 16`），应得 0；80 组合 25 错 |
| 1.177 | clip_grad_norm silently mishandles norm_type = inf / -inf / 0 | tt-metal #55587（修复 PR #55593） | `[已知类别]` var:1.164 | p-十进制分解（整数/小数/符号）对 inf/0 未定义：`floor(inf)-inf=NaN`、`1/inf=0` 使 step2 全体 x⁰=1——total_norm 恒 1.0 或非有限，梯度按 max_norm 无差别缩放 |
| 1.178 | moreh_norm_backward returns NaN for all-zero reduced slices | tt-metal #55585（修复 PR #55593 同族） | `[已知类别]` var:1.171 | `dx = sign·\|x\|^(p-1)·y·dy·(1/y^p)` 无 `y==0` 守卫：全零切片 `1/0^p` 非有限 × 零因子 = NaN；PyTorch 每分支都有 `masked_fill_`——被 tt-metal 省略 |
| 1.179 | transpose/permute WH on RM + HEIGHT-sharded in → WIDTH-sharded out: silent all-zero output | tt-metal #55582（修复 PR #55591） | `[已知类别]` var:1.104 | WH 转置在 HEIGHT→WIDTH sharding 路由上静默输出全零（PCC≈0.01，无 error 无 fatal）；`num_hw_blocks_per_core=0`；permute 经同一 transpose 复现 |
| 1.180 | Fix is_real and is_imag goldens and make their sweeps exercise the predicate | tt-metal PR #55577 | `[已知类别]` var:1.135 | 两个 op 的 golden 调用不存在的 `torch.is_imag/is_real`（正确拼写 `torch.isreal`）→ 从未被验证过；sweep 输入分布恒非零使谓词恒 false——all-false vs all-false 恒过 |

### 第 4 类 · 数值/精度（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 4.18 | [diffusion] fix: stabilize H3 reference audio across repeated requests | SGLang PR #38225 | `[已知类别]` var:4.12 | audio encoder 的 scripted `snake` 激活在 profiling 后切换到融合 `TensorExprGroup` 图——不同浮点舍入扰动参考音频 conditioning，被联合去噪放大：同一请求首次/后续输出不同（MP4 字节不同） |

### 第 5 类 · 分布式/多GPU（+3）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 5.32 | [Bugfix][DBO] Handle a microbatch that leaves the step before its peers | vLLM PR #55586 | `[已知类别]` var:6.195 | 微批 ring handoff 中提前退出者清空 `_CURRENT_CONTEXTS` 槽位：(a) 幸存者解引用 None 槽 `AttributeError`；(b) parked 幸存者等永不到来的 handoff——step 挂起而非失败（V1/V2 共享 ubatching.py） |
| 5.33 | DFLASH/DSPARK draft KV pool budget uses tp_size instead of attn_tp_size, OOM under DP attention | SGLang #38202（修复 PR #38203） | `[已知类别]` var:6.136 | DP attention 下 draft 池按 attn_tp_size 分片但预留按原始 tp_size 计——差 attn_dp_size 倍少预留；Kimi-K3 TP16/DP16 解码节点 draft worker `torch.OutOfMemoryError` 3.50 GiB |
| 5.34 | [Fix] Normalize MiniCPM Lightning output across TP ranks | SGLang PR #38230 | `[已知类别]` var:5.26 | Lightning attention 的 RMS 归一化作用于全头组合输出，TP 下每 rank 只用自己子集算分母——激活量级不同的 rank 结果偏离全宽归一化（prefill 最大绝对误差 2.3125 → 修复后 0.0039） |

### 第 6 类 · 服务栈/缓存正确性（+8）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 6.204 | Hybrid mamba prefix-cache hit reads OOB: state index seeded with cache_config.block_size after engine lowered it below mamba_block_size (Xid 31) | vLLM #55600（修复 PR #55601） | `[已知类别]` var:6.131 | 6.131 同族第二形态：非用户 CLI 覆盖，而是 `_initialize_kv_caches` 把 `cache_config.block_size` 重赋为 min(可缓存组 block size)（drafter SW 组 64 < mamba 7168）后 add_request 仍用它播种——≥8 mamba 块前缀 Xid 31 crash，短前缀静默恢复错误 recurrent state |
| 6.205 | fix: wait for HiCache load-back before hybrid MLA KV reads | SGLang PR #38217 | `[已知类别]` var:6.193 | `HybridLinearKVPool.get_mla_kv_buffer()` 禁用了内层 wait（hybrid wrapper 拥有同步）却不做外层 wait——异步 load-back 未完成即读：20/20 次读回错误数据；Q/后缀投影先行执行时掩盖 |
| 6.206 | [mem_cache] Skip the empty-key insert in UnifiedRadixCache.cache_finished_req | SGLang PR #38191 | `[新类别/新签名]` | 短 mamba 请求未达 track boundary 时 cache_len=0 仍走空键 insert：`last_node` 被 repoint 到请求未持锁的 root + 注册零长 session ref；ReplaySSM 变体 cache_len 可为负 → `token_ids[:negative]` 尾部错切插入错键 |
| 6.207 | fix: legacy SGL_ env vars silently dropped when the rewritten name is itself deprecated | SGLang PR #38219 | `[已知类别]` var:6.164 | `_handle_deprecated_envs()` 先跑 deprecation registry 后做 SGL_→SGLANG_ 前缀重写——重写名本身在 registry 中的 legacy 变量被静默丢弃（转发永不发生），设置失效且告警指向另一个 deprecated 名 |
| 6.208 | fix(b12x): derive scratch capacity from scheduler contract | SGLang PR #38235 | `[已知类别]` var:6.136 | B12X Vision W4A8 arena 从 launcher 拥有的 token 值 + 幂次阶梯定容，但聚合 prefill 可超 per-request chunk、DSV4 Vision 切点可移到不可分图像 span 末尾——实测 412,183,756 B 需求 vs 400,008,864 B 预留，双 rank 终止 |
| 6.209 | fix(inkling): translate unified-memory checkpoint destinations to physical slots | SGLang PR #38229 | `[新类别/新签名]` | 统一内存 checkpoint 恢复把逻辑目的地当物理槽位写入（未翻译）——"Before = broken checkpoint implementation"，未修复基线产出不同输出；修复后逐字节验证 |
| 6.210 | Xid 13 "Out Of Range Address" / CUDA IMA on RTX PRO 5000 (SM120) with FP8 model under sustained load | vLLM #55571 | `[已知类别]` var:6.160 | Ministral-3-14B FP8（官方 0.28.0 镜像）持续负载下 EngineCore 死于 illegal memory access + 同秒 Xid 13（Graphics SM Warp Exception）；`VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` 或 `--enforce-eager` 均消失——指向 FlashInfer FP8 scaled MM 线性内核在 SM120 的负载相关 IMA |
| 6.211 | [Bugfix][Multimodal] Bind client-provided UUIDs to item content size in cache keys | vLLM PR #55549（fixes #55547） | `[已知类别]` var:lmcache_cache_salt_isolation_missing | 客户端 mm uuid 被逐字用作唯一缓存键（双层）：同 uuid 不同 payload——长度匹配时静默服错内容；长度不匹配时 `_merge_multimodal_embeddings` 抛异常 **tear down 整个引擎树**（DP8 内部 LB 部署 8 引擎全杀） |

### 第 7 类 · 量化（+4）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 7.74 | [CPU] AWQ (and GPTQ) WNA16 int4 GEMM silently unavailable on AVX2-only CPUs | vLLM #55613 | `[已知类别]` var:7.53 | `cpu_gemm_wna16` 只在 `__AVX512F__/riscv/s390x` 下编译注册，但 `can_implement()` 只查平台/dtype/形状不查 AVX-512——AVX2-only CPU 深入执行期才 `AttributeError`；发行 wheel 在此类 CPU 上仅剩 2 个自定义 op |
| 7.75 | CUTLASS 3.x scaled_mm ignores leading strides of sliced tensors | vLLM #55534 | `[新类别/新签名]` | op 入口接受 padded leading dim（只要求最内维 stride 1），C3x caller 却用 `make_cute_packed_stride` 从 shape 重建 packed strides 丢弃实际 leading strides——A/B/输出任一 padded 即 ~8.1% 元素错（max abs err 3.04），全 packed 为 0；静默错值 |
| 7.76 | [None][fix] MoE: keep separate NVFP4 gate/up global scales where supported | TensorRT-LLM PR #18772 | `[已知类别]` var:7.69 | NVFP4 checkpoint 的 w1/w3 `weight_scale_2` 可不同：加载器一律 `max()` 调和 + warn——较小 scale 一侧静默损失精度；trtllm-gen FC1 内核本可吃双标量（单 alpha 的 Cutlass 保持 max） |
| 7.77 | [None][fix] Re-resolve the MoE op provider once the per-layer quant config is known | TensorRT-LLM PR #18770 | `[已知类别]` var:6.164 | `TRTLLMGenFusedMoE` 在 `__init__` 解析 op provider，而 per-layer 量化配置到 `create_weights()` 前才安装——layerwise 量化下模型级 NVFP4 + 层级 BF16 的层保持 native provider（只有 FlashInfer 实现 BF16 MoE）→ warmup `run_bf16_moe` 失败 |

### 第 8 类 · KV缓存/注意力（+8）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 8.69 | [ROCm] test_mla_backends surfaces two MLA decode-path issues | vLLM #55609 | `[已知类别]` var:8.62 | (1) `ROCM_AITER_MLA` 无 qseqlen>4 的 gfx942 persistent kernel，启发式查找失败曾是 `std::abort()` 杀死 pytest worker（aiter#5297 后才可捕获）；(2) chunked context v_head_dim=256 bf16 的 atol/rtol=1e-2 低于 1 ULP——2 ULP 差被误判 fail（测试假阳性） |
| 8.70 | [Bug]: FlashInfer get_cudagraph_support() divides the target model's head count by the draft group's num_kv_heads | vLLM #55581（修复 PR #55593） | `[新类别/新签名]` | CUDA-graph 支持判定用 **target 模型**查询头数除以 **draft 组** num_kv_heads——两模型头比不同即算错：TP=3 时全引擎 CUDA graph 静默全关（实际 XQA 路径可用），TP=2 正常；spec 性能静默塌方 |
| 8.71 | [Bugfix][ROCm] Take the AITER MLA KV dtype from the group, not cache_config | vLLM PR #55591 | `[已知类别]` var:8.70 | AITER MLA builder 从全局 `cache_config.cache_dtype` 推 KV dtype，draft 组 `auto`（bf16）在 `--kv-cache-dtype fp8` 服务下被当 fp8——错误元素大小读 bf16 缓冲 → persistent asm decode memory access fault；eager 走 Gluon 掩盖、仅 cudagraph capture 显形 |
| 8.72 | [Bugfix] Propagate quantized-KV scales to KV-sharing layers | vLLM PR #55559 | `[已知类别]` var:8.59 | KV-sharing 层（Gemma-4-E2B 20/35 层）读 target 层写入的量化 KV 但自身 k/v_scale 停留 1.0 默认（无 K/V 投影故 checkpoint 无 scale）——错误 scale 反量化：needle 2/9 → 修复后 7/9（bf16 基线 5/9） |
| 8.73 | [Bugfix][GLM-5.3] Fit the kpool-widened sparse top-k buffer to the 2048-wide SM120 index | vLLM PR #55563 | `[已知类别]` var:8.56 | GLM-5.3-Flash `index_topk=2048, kpool=4`：kpool 尾部加宽 + 128 对齐使 buffer 2176 宽，SM120 sparse 后端按 2048 实例化——stock config 被启动检查拒绝（此前需手改 2044）；无检查时内核收超宽 page table |
| 8.74 | [Fix] Correct MiniCPM fused top-k synchronization and selection | SGLang PR #38232 | `[已知类别]` var:8.55 | fused top-k 选择器三缺陷：跨 warp MMA 读未同步的共享 Q/K tile、重叠压缩窗给未来块得分（causal 候选被挤掉）、先按 ID 排序再截断幂次缓冲丢高分布（含 forced 块）——错误块被选作 sparse attention |
| 8.75 | [Kernel] Fix the row stride passed by the InfLLM-v2 pooling wrapper | SGLang PR #38231 | `[已知类别]` var:7.75 | 压缩分池化 wrapper 行 stride 从逻辑 context 长度推导（8500//16=531），score 张量实际 padded 到 640——后续行/头读错 offset（39 个有限条目错，max err 0.136）；逻辑宽度 vs 物理填充宽度域混淆 |
| 8.76 | [NPU] ascend attention backend silently corrupts output with page_size 16 (128 OK, 1 rejected) | SGLang #38196 | `[已知类别]` var:8.52 | Ascend 910C `ascend` 后端 page_size=16：通过 kernel tiling 检查但每个请求输出连贯却语义垃圾（"iterate!!!"、"pectionpection..."）；page 1 被硬拒、128 正确；首 decode 步即损坏、无 spec 也复现 |

### 第 10 类 · 分词器/模板/结构化输出（+4）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 10.53 | BarthezTokenizer cannot load moussaKam/barthez: v4.57 → v5 regression | transformers #48567 | `[新类别/新签名]` | 类硬编码 `Unigram(...)` 但 `moussaKam/barthez` 是 BPE 模型（`mbarthez` 才是 UNIGRAM）——v5 `convert_to_native_format`（#42563）把 checkpoint 自带 dict vocab 喂给 Unigram → `TypeError`，两条加载路径全崩且无 workaround；测试只 pin 了 mbarthez |
| 10.54 | tool_choice="required" not enforced with Qwen3.8-Flash-Next when enable_thinking=false (streaming) | vLLM #55552 | `[已知类别]` var:6.181 | 流式 + thinking off 时 2/3 响应纯文本 `finish_reason=stop`（required 被丢）；非流式与 thinking on 正常、工具数少时正常——TC-45 场景 0/8 通过（SGLang 同客户端 8/8）；thinking on + MTP 时 xgrammar "Failed to advance FSM" 反复出现 |
| 10.55 | fix(function-call): publish DeepSeek V4 calls atomically | SGLang PR #38233 | `[已知类别]` var:10.42 | V4 继承 V3.2 增量 parser：opening invoke tag 一出现即发布函数名、闭合 DSML 信封前可发布临时参数——截断时客户端收到永不完整的事务；异常路径可把 malformed DSML 当普通 assistant content 返回 |
| 10.56 | [None][fix] Make llm_args use the canonical custom-tokenizer alias table | TensorRT-LLM PR #18768 | `[已知类别]` var:1.107 | `llm_args.py` 携带手维护的 `TOKENIZER_ALIASES` 副本，canonical 表新增别名永不镜像——新别名在 `TorchLlmArgs` 校验器落到裸字符串 `rsplit` 崩溃，错误信息指向格式而非缺失注册；同别名在其它三个入口正常 |

### 第 11 类 · 解码/采样（+2）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 11.67 | [Bug][Spec Decode] GLM-5.3 (non-Flash) MTP: draft acceptance ~1-2% pos-0 even on repetitive text | vLLM #55605 | `[已知类别]` var:11.63 | GLM-5.3（非 Flash，FP8）+ MTP：重复文本 pos-0 接受率 ≤2.4%（健康 60-80%）——**重复文本比随机文本还差**（反向签名）；eager+MTP-1 排除 graph/多次重跑效应：损坏在 proposer 自身；吞吐 -29~41%，非 MTP 输出正常；疑似 DSA sparse-indexer draft 状态或 FP8 block-scale draft 权重处理 |
| 11.68 | [Bug][Spec Decode] Hybrid GDN + MTP: scheduler runs only ~3 concurrent sequences at batch >= 4 | vLLM #55533 | `[已知类别]` var:6.136 | 27B hybrid GDN（48 GDN + 16 full）+ MTP：batch=8 每步只调 3 序列（3 宽 MTP 并发窗轮转），其余 5 个无限等待轮换——MTP 慢 2-3×；bs≤3 健康；9B 兄弟（24 GDN 层）无此现象——指向 mamba cache 预算在 target 状态与 MTP draft slot 间的共享记账 |

### 第 12 类 · 权重/适配器/检查点（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 12.32 | rsLoRA scaling factor ignored for MoE expert LoRA adapters (pack_moe recompute alpha/rank) | vLLM #55545 | `[新类别/新签名]` | dense 路径 honor `PEFTHelper` 存储的 `alpha/sqrt(r)`，`pack_moe`/`pack_moe_stacked` 却从原始字段重算 `alpha/rank`——r=16/alpha=32 时正确因子 8、实际 2，adapter delta 弱 √r 倍（4×）；无报错、输出静默错；CPU 纯 Python 复现 |

## 补充来源引用（不占新编号，2 条）

| PR | 并入条目 | 说明 |
|---|---|---|
| SGLang PR #38224 | 8.65（#38143） | **根因修正**：MiniMax-M3 W4A16 全 NUL token 的根因是 compressed-tensors 量化 embedding 被 `get_quant_method` 四分支落穿 → `UnquantizedEmbeddingMethod` + bf16 空参数（`weight_packed/scale/shape` 三条 "not found" 警告后被无视）→ `torch.empty` 零填充 → 每个 logit 为 0 → argmax 恒 id 0——**非 Triton 注意力路径**；修复改为显式 `NotImplementedError` 拒绝 |
| SGLang PR #38212 | 6.193（#38031） | 修复：HiCache 往返不备份/恢复 DSA target+draft index buffers（KV 与 recurrent state 恢复但选 token 的 index 不恢复）+ 前缀分裂落在压缩组内时不同后缀共享 index 行；报告者 8×H100 生产验证通过 |

## 跳过条目（核实后不编目）

- vLLM #55608（amd64 zstd 发布）/#55607（Pareto 图）/#55594（metrics 并行采样）/#55575（DSV4 fused post residual 性能复用）/#55584（R3 RFC）：CI/性能/功能类，非正确性
- vLLM #55556（Phi-4 openai content format str+list crash）、#55555（mm LoRA fp16 tower connector `torch.empty(None)`）、#55554（Mistral pooling weight-key KeyError）：fail-loud 启动/请求期 crash，正文为模板级信息、无根因分析（"should be validated"级），信息不足
- SGLang #38207（GLM-5.3 DPC>1 + triton_mla crash）：仅栈顶片段，无根因；#38236（DSv4 NEXTN alias 在 model hook 后解析 → 启动 assert）：合法配置被拒但 EAGLE 等价可用、无正确性/静默风险；#38210（OTel span 丢失）：可观测性而非推理正确性
- transformers #48570/#48563（shift_labels 训练 loss）/ #48569（TP checkpoint）/ #48559（find_spec torchvision）：训练域/工具链
- TRT-LLM #18775（cute::make_int_sequence 依赖模板上下文编译失败）：自定义内核工具链、信息薄；#18766（perf gating infra）：非故障
- tt-metal #55575（repeat_interleave 4D DRAM-sharded crash）、#55580（pow Veltkamp split 指数溢出守卫）、#55573（torch_random_with_zeros dtype）：标题级 fail-loud/工具函数修复，1.108/1.179 同域已覆盖（pow 族/ sharding 族），证据深度不足独立成条
- vLLM #55572（DSPARK indexer logits 预算下调）：集成 GPU 的资源适配（性能型），非静默正确性缺陷
- vLLM #55563 中 SM120 kpool 情形已编目（8.73），其中"无检查时内核收超宽 page table"为该条目行为效应的一部分

## arxiv 结果（无新编目论文）

- `silent data corruption LLM`：TrainSDC（已为 §3.1 补充来源）、SCOUT（已编目）、LLM-PRISM/Exploring SDC/Understanding SDC（均已编目）
- `bit flip LLM weights`：JITterFlip（已编目 3.18）、Decision-Level Hijacking（已编目）、Bit-Flip Vulnerability of Shared KV-Cache（已编目）、Rotated Robustness/TFL（防御方法论文或已编目）
- `KV cache corruption`：Addressable Memory（已编目 14.7）、Runtime Observability（系统方法论文）、Nexus/Moment-KV（方法论文不收录）、When Latent Agents Lie（已编目）
- `LLM inference correctness`：全部为评测/方法论文（Medical Hallucination、Semantic Correctness Framework、Cheap Verifiers 等），无新故障签名
- `GPU soft error inference`：Instruction-Level FI（已编目 1.98）、2021 Arithmetic-Intensity-Guided FT（早期容错方法论文，无新故障案例）

## API 调用记录

| # | 查询 | 结果数 | 状态 |
|---|---|---|---|
| Q1 | vllm `wrong/incorrect/corruption/NaN/silent created:>2026-09-05` | 38 | ✅（重试后） |
| Q2 | sglang `bug/wrong/corrupt/NaN/garbage created:>2026-09-05` | 23 | ✅ |
| Q3 | transformers `wrong/incorrect/corrupt/NaN created:>2026-09-05` | 4 | ✅ |
| Q4 | TensorRT-LLM `bug/wrong/corrupt/NaN/correctness created:>2026-09-05` | 6 | ✅ |
| Q5 | triton `wrong/incorrect/miscompile/corrupt created:>2026-09-05` | 5 | ✅ |
| Q6 | tt-metal `wrong/incorrect/corrupt/NaN/mismatch created:>2026-09-05` | 12 | ✅ |
| Q7 | vllm `quant/FP8/NVFP4/KV cache created:>2026-09-05 label:bug` | 24 | ✅ |
| Q8 | transformers `tokenizer/rope/config created:>2026-09-05 label:bug` | 1 | ✅ |
| Q9 | vllm is:issue `correctness/accuracy/"wrong result" created:>2026-09-05` | 7 | ✅ |
| Q10 | sglang `correctness/accuracy/silent/deterministic created:>2026-09-05` | 57 | ✅ |
| A1–A5 | arxiv 5 组关键词（HTTPS） | — | ✅ 无新编目论文 |

> 注：首批 10 组查询因脚本把 `q` 作为裸位置参数传给 curl（未进 `-G` 查询串）全部返回 422 `Validation Failed (missing q)`；修正为 `--data-urlencode "q=..."` 后全部成功。条目核实直接使用 search API 附带正文（dump 脚本剥离环境 details 块），未额外调用 core REST。

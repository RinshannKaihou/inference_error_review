# LLM 推理故障每日扫描报告 — 2026-09-18

- **扫描日期**：2026-09-18（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-16`（与 09-17 扫描窗口重叠两日，已逐条去重）；arxiv 按提交时间倒序取最近 5 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1125 条（2026-09-17 更新后）；编号编目 1011 条 + K* 扩展 114 条

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-16 wrong OR incorrect OR corruption OR NaN` | 20 命中 |
| 2 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-16 wrong OR incorrect OR corruption OR NaN` | 12 命中 |
| 3 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-16` | 20 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-16 wrong output OR incorrect OR NaN` | 2 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-16 wrong OR incorrect OR corruption` | 1 命中 |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-16 wrong OR miscompile OR NaN` | 1 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-16 numerical OR wrong OR pcc` | 33 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-16 wrong OR incorrect OR NaN` | 3 命中 |
| 9 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-16 silent OR incorrect OR garbage` | 14 命中 |
| 10–13 | arxiv | `"silent data corruption" AND LLM` / `"bit flip" AND "LLM inference"` / `"KV cache" AND corruption` / `"weight corruption" AND inference` | 5/1/5/2 命中 |

GitHub Search API 9 组查询全部一次成功，无限流 403、无 Validation Failed；条目正文直接使用 search API 附带 body，未额外调用 core REST。arxiv 4 组查询全部成功（无 429）。

## 已编目/已判条目（本次跳过）

以下命中条目经编号核对已在主文档中（或 09-17 扫描已明确判定），本次不重复添加：

- vLLM #57429（→7.93）、#57423（→5.40）、#57267（→4.23）、#57266（→6.241）、#57324（→10.71）、#57156（→8.89）、#57136（→11.80）、#57223（09-16 已判：训练侧 mask 消费者留观）、#57373（RFC）、#57346（Feature）、#57342（09-17 已判：性能）、#57350/#57555（性能）、#57303（资源治理）、#57259/#57276（fail-loud/功能）、#57449（MLA workspace OOM fail-loud）、#57440（内存治理）、#57376（API 字段兼容观察）、#57508（spec decode host-context hook——Feature/RFC 类）、#57521（性能）、#57515（无堆栈不可定位留观）
- SGLang #39836（→10.69）、#39831（→12.39）、#39830（→6.240）、#39782（→12.38）、#39752（→10.70）、#39708（→13.39）、#39692（09-16 已判留观）、#39684（→7.92）
- flashinfer #5260（→8.88）、triton #11836（→1.229）、TRT-LLM #19319（09-17 已判：设计澄清留观）
- tt-metal #56880/#56876/#56827/#56853（→1.224/1.225/1.226/1.227）、#56908（09-17 已判：bounty 招标帖留观）、#56847（09-17 已判：CI PCC 回归留观）、#56856（09-17 已判：放置语义当日 closed）、#56824（fail-loud assert + batching 设计）、#56808/#56823（测试基建）、#56791（perf counter 语义）、#56872/#56863/#56770（CI/机器人）、#56751（SFPLOADMACRO coverage instrumentation——LLK 测试基建下的错误结果，已 closed、bounty 已修）、#56769（`reduce_scatter_minimal_direct` 多环 hang——hang 类留观）、#56768（性能/重构）、#56745（CI 超时）、#56976（L1-interleaved matmul 4× 慢——性能）、#56873（agentic-workflows 机器人）
- transformers #48842（09-16 已判：训练侧）
- vLLM #56949 为已编目条目（DP vision shard-local grid 索引，10.34）来源——注意与 tt-metal #56949（topk_router_gpt DRAM cores）**撞号**，本日 tt-metal 侧目标条目经核验确认未编目后正常合并

## 🆕 新发现（本次合并，共 12 条）

### A. 全新子签名（7 条）

1. **[新类别/新签名] SGLang #40156 — EAGLE draft 批 `num_token_non_padded=0` 全体 topk id=-1 → negative-index wrap 路由到末位专家 OOB 写** — `#39574` 把 `num_token_non_padded` 穿入 MoE 路径后，EAGLE draft 批在 `ForwardBatch.init_new` 时 `input_ids is None` → token 计数推导为 **0**（而非 None）→ padded-region mask 以活计数 0 武装，把**每个** draft topk id 置 -1；`StandardDispatcher.dispatch` 在 drop 检查**之前**做 `local_expert_mapping[topk_ids]`，PyTorch 负索引回绕取表尾（末 EP rank 上 = 真实本地专家 71）→ 整批行灌入单专家：`masked_m[71]=384 > m_max=256`，`src2dst=[18176,18559]` 越 `72*256=18432` 区 OOB 写。Eager：`cudaErrorIllegalAddress`（coredump 指 `fill_gateup_input_triton_kernel`）；**CUDA graph：无 crash——部分在途请求 `spec_accept_rate=0.00` + `!!!!!` 垃圾洪泛至 token 上限**，corruption 为每请求持久状态；关 overlap scheduler / `CUDA_LAUNCH_BLOCKING=1` 即消失。H20×8、GLM-5.3-Flash（288 专家 top-8、ep_size=4）。→ 归入第 11 类（推测解码），Cov NEW。
2. **[新类别/新签名] SGLang #40144 — DFLASH anchor-first checkpoint 布局无校验：全体 draft token 静默位移一位、接受率坍塌至 ~1.0** — DFLASH draft checkpoint 有两种布局约定（anchor-as-first-prediction vs 1+N），SGLang 假设后者且不校验。anchor-first 训练的 checkpoint 被静默错位服务：服务器健康、无警告，接受率坍至 1.0621（同 checkpoint 在 vLLM 下 2.7562/2.9837）。布局其实**有声明但用了 SGLang 不读的字段名**（上游 override config `dflash_config.query_zero_predicts_next: true`；speculators 的 `sample_from_anchor` 字段在该官方 checkpoint 中根本不存在，marker 检查也救不了）。→ 归入第 11 类，Cov NEW（与 11.77 vLLM DSpark fill-in 布局缺旗标构成**跨栈姊妹对**：同一布局歧义在两个栈各自静默选错方向）。
3. **[新类别/新签名] SGLang #40127 — MiniMax-H3 INT8 ConvRot checkpoint 从量化格式推断 QKV 布局，跳过 head-interleaved reorder 静默花屏** — `KitchenInt8Config` 用 `checkpoint_uses_native_qkv_layout = is_checkpoint_int8_serialized` 从**量化格式**推断 QKV **布局**：任何 int8 序列化 checkpoint 被假定为已存 `[q_all,k_all,v_all]`、跳过 MiniMax 官方 head-interleaved reorder。`RunningHubAI/MiniMax-H3-INT8-CONVROT` repack 保留 interleaved 布局 → 每个 attention block 从错误行读 Q/K/V。形状合法、运行完成、无错误无 NaN 无警告，产出**结构合法但内容全错的 MP4**：视频塌成平坦低对比场（mean 149.6→79.4、std 50.78→2.80）、音频变噪声（-37.9dB→-5.0dB）。非 FSDP 路径（与 5.10 #34227 同缺陷不同触发面，issue 自述 "same defect as #34227, non-FSDP path"）。→ 归入第 12 类（权重/检查点加载），Cov NEW。
4. **[新类别/新签名] SGLang #40084 — QuickReduce Q8 低幅值 reciprocal-scale 悬崖：BF16 块最大值 <~2e-3 全块衰减** — `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` + `CAST_BF16_TO_FP16=1` 时 `CodecQ8<half>` 每 32 值块存 `decoding_scale≈block_max/128`（FP16）、算 `encoding_scale≈reciprocal`（FP16）：`block_max < 128/65504 ≈ 0.001954` 时 reciprocal 饱和于 65504、编解码不再互逆 → **整块衰减**，two-shot 集体通信施加两次。MI350X TP4/EP4 GLM-5.3 74,584-token prefill 实测：layer-4 MLP 归约相对 L2 误差 **0.6797**（RCCL 对照 0.0026，266.5×）；99.35% 块低于悬崖；CPU 复刻 codec **bit-for-bit** 复现 GPU 输出。是 7.68（高端 FP16 溢出 inf 毒化）的**低幅值对偶**——既有 BF16 range test 的 "low" 用例 2^-8 恰在悬崖上方 2×，从未测过 sub-cliff。→ 归入第 7 类（量化），Cov var:7.68（同为 QuickReduce 值域假设崩塌，方向相反）。
5. **[新类别/新签名] flashinfer #5334 — Blackwell GDN chunked TMEM stage 在 `tcgen05.ld` 在途时释放（cg1 双站点缺 load fence）** — `gated_delta_net_chunked.py` 两处 compute-group-1 消费者（`:4528` ks_handle、`:4569` nv_handle，后者**每 chunk 无条件**）在 `cute.copy` 发出异步 `LDTM` 后立即 `SYNCS.ARRIVE` 释放 TMEM stage，**不等在途读完成**——生产者下一次 `tcgen05.mma` 可中途覆写，消费者寄存器收到本 chunk 与下一 chunk 的混合值。cg0 的两处同构 handoff 正确（有 `fence_view_async_tmem_load()`）。**SASS 级证据**：fence 降为裸 NOP、真正 ordering 在后续 SYNCS.ARRIVE 的 wait-mask 控制位里——正确站点 wait=000010、缺陷站点 wait=000000（等Nothing）。另指出两个**形似但正确**的站点（qs_handle/o_qs_handle 有中间寄存器消费者已等 scoreboard）——"源码级 fence 审计在两个方向都不成立"。vLLM Qwen3-Next 首 prefill 路径（batch 1–16 非 CP 分支）即经过。→ 归入第 1 类（硬件/SDC，异步内存管线竞态族 1.9/1.121/1.223），Cov NEW。
6. **[新类别/新签名] tt-metal #56986 — `hf_rope` decode batch 32 cos/sin shard grid 与 Q 所在 grid 不重合读错角度（PCC 0.80–0.90）** — Blackhole 上 `test_attention.py` 仅 hf_rope decode batch 32 两例失败（0.897/0.825/0.796/0.797/0.767/0.774 vs 0.99 门槛；batch 1 与 mllama_rope 全过、0.9994+）。疑似根因：decode `HfRotarySetup` 把 cos/sin shard 在 11 宽 Blackhole grid 的 rows 0-1 + row 2 十核，而 decode Q 钉在 8x4 block——`rotary_embedding_hf` decode 模式在输入 bounding box 上跑 sharded factory 并从 resident shard 读 cos/sin，**某 batch 行的 cos/sin shard 在另一核心上即读错角度**；Wormhole 两 grid 均 8 宽恰好重合故不触发。→ 归入第 9 类（位置编码/RoPE），Cov NEW。
7. **[新类别/新签名] tt-metal #56955 — groupnorm sharded Welford 内核批次步进用 `block_hw` 而非 `block_ht*per_core_Nt`：第二批起归一化错误 tile** — Welford kernel 从不读 compile arg 18（`num_tiles_per_batch`），用 `tile_id = b * block_hw`（arg 9 = `block_ht*block_wt`，`block_wt` 是单 group tile 跨度）步进批次；`block_wt < per_core_Nt` 且 `num_batches_per_core≥2` 时 batch b 起点**提前 `b*block_ht*(per_core_Nt-block_wt)` 个 tile、落进前一批数据**。首批正确、后续全错；legacy（use_welford=False）同输入正确——两变体同 tensor 上分歧。静态分析+公式级触发条件（`[2,1,32,64]` 单核 shard、num_groups=2 即中招），报告者自述未上设备运行。→ 归入第 1 类（硬件/SDC），Cov NEW（与 1.31/1.42 Welford 精度族不同根因：本条是**索引步进错**非同步/精度）。

### B. 已知类别新变体（5 条）

8. **[已知类别] vLLM #57562 — AsyncScheduler chunked prefill 并发下 `num_output_placeholders` 下溢：未完成 prefill 的请求收到伪采样 token** — 模型 runner 在 `computed(8448) < num_tokens(9119)` 的 prefill chunk 步上**仍返回一个采样 token**（`discard_request_mask` 未抑制），scheduler 未预留 placeholder → 交付计数 -1 下溢 assert 杀 EngineCore；伪 token 被追加进响应（output 1→7 增长仍在 prefill 中）。0.24.0 同负载 0 下溢（runner 侧行为分歧、回归）；`--no-async-scheduling` 即 24/24 完成；#50692 一行修复无效。无 spec decode、无 preemption。44 次下溢签名全同。→ 归入第 6 类（服务栈，async scheduler 交付契约族），Cov NEW（06 条目族：placeholder 交付契约下溢为新子签名）。
9. **[已知类别] vLLM #57614 — ROCm gfx1151 Navi paged attention 静默接受不支持的 ALiBi 与 block size 32（无声错值）** — `torch.ops._rocm_C.paged_attention` 对 ALiBi 输入按 no-ALiBi 参考匹配（error vs ALiBi=0.0758 ≈ 全量输出幅值，即**ALiBi 被完全忽略**）；block size 32 从 seq 17 起错值与输出同量级（0.010–0.029 vs reference_scale 0.017–0.05，块 16 全过）。无 assert、无 fallback、静默。→ 归入第 8 类（KV/注意力：后端能力矩阵静默吞不支持的配置），Cov NEW。
10. **[已知类别] vLLM #57493 — gfx1151 ROCM_ATTN 同一 greedy 请求的 decode 输出依赖此前请求（跨请求非确定）** — `max_num_seqs=1`、enforce_eager、prefix caching off 下，固定 probe 请求在 20 次 filler 之后 logprob[5] 有 4 个不同值、completions 2 种（'...Spain is Madrid' vs '...Germany is Berlin'）；prefill 驱动的 logprob[0] 恒定（20/20 同）——**decode 路径状态跨请求泄漏**，非 prefill 非确定。TURBOQUANT 被.override 为 ROCM_ATTN。→ 归入第 4 类（数值/非确定，4.12 族），Cov var。
11. **[已知类别] vLLM #57486 — SM12x fp8 block linear 在 E8M0 关闭时仍选 DeepGEMM：无 float32 scale 路径 → NaN 0.78%–38.7% / 新版 RuntimeError** — `should_use_deepgemm_for_fp8_linear()` 不看 scale 格式与设备族：`VLLM_USE_DEEP_GEMM_E8M0=0` 时 oracle 返 FLOAT32 而路由不变，DeepGEMM 在 SM12x 无可用 float32 scale 路径——旧 commit `ad1f1726` 下 **NaN 占 0.78%–38.7% 输出**（M×N×K 扫描表），`a6bbb80` pin 后改 host assert RuntimeError。GB10 sm_121 实测、oracle 单测可复现。→ 归入第 7 类（量化：DeepGEMM 路由 oracle 维度），Cov NEW。
12. **[已知类别] vLLM #57490 — internlm tool parser 起始标记跨 chunk 分裂即丢 tool call** — 流式解码中 tool call 开标记（如 `<tool_call>`）被 chunk 边界劈开时 parser 丢弃整个 tool call。10.42 族（parser 通道终止符假设）的新触发面：**跨 chunk 边界**。→ 归入第 10 类，Cov var:10.42。

### C. 已有条目补充来源（不占新编号）

- **vLLM #57532**（GLM-5.3-Flash NVFP4+MTP `parallel_lm_head` weight_scale NaN 启动失败）→ **8.84 补充来源**：同为 GLM-5.x NoPE MLA 服务链路；#57532 是 fail-loud（assert + 明确修复指引），印证 8.84 的"GLM NoPE MLA 几何在量化 checkpoint 上 MTP 装配缺口"语境。
- **vLLM #57578**（GLM-5.3-Flash NoPE MLA `concat_and_cache_mla` 硬编码 `pe_dim==64` warmup 崩溃，报告者论证放宽 assert 将致 RoPE 线程读不存在的 `k_pe` → 静默损坏）→ **8.84 补充来源**：vLLM 侧 GLM NoPE MLA（8.84 记录的是 SGLang 侧 SM120 后端矩阵；本条是 vLLM 侧 FLASHINFER_MLA_SPARSE_SM120 的 fp8_ds_mla/nvfp4_ds_mla 内核 DeepSeek 几何硬编码）。
- **vLLM #57354**（DeepSeek-V4-Flash MI355X sparse-MLA mixed warmup 间歇 segfault，4/5 启动失败）→ **8.62 族语境引用**：ROCm aiter sparse-MLA warmup 非确定崩溃，无根因留观，与 8.69/8.62 的 ROCm sparse-MLA 路径同触发面。

### D. 评估后不收录（记录备查）

- **vLLM #57580**（Mamba align checkpoint 抢占后可见性）：报告者自述"scoped reproduction tracker, not a new root-cause claim"、**synthetic seeded metadata**（synthetic 初始 checkpoint + CPU 模拟 worker 完成）+ 明示 "does not demonstrate model-output corruption"、未对最新 main/#43569/#56525 验证。作为 #43569/#56525 回归覆盖追踪留观，不编目。
- **tt-metal #56938**（unary elementwise 28 op 类型对 int32 输入返回 float32 位型）：#56853（已编目 1.227）的**系统性扩展**（43 op 类型普查：28 float 位型、5 已由 1.227 描述、7 整数正确、3 拒绝）——同一根因（dtype policy 绕过 + 位型重解释）的覆盖面数据，作为 1.227 语境补充记录，不占新编号。
- **tt-metal #57004**（binary goldens 用 Torch promotion 而设备用 per-op scalar packing）：**测试 golden 基准与设备语义脱节**——影响的是"错误参考"（dtype 回归会被错误 golden 掩护）而非推理路径本身。属测试基建正确性，留观。
- **tt-metal #56945**（LLK 32-bit unpack-to-dest broadcast 因 stale DEST_TARGET_REG_CFG_MATH_Offset 读写错误 DST tile）：真实静默错值 + WH/BH 硬件复现 + 修复 PR #56932 在途，但**当前所有 in-tree 调用者恰好都 dst_index==0**（潜伏未触发）。LLK 原语缺陷收录门槛参考 1.218/1.219（sfpi/LLK 位级语义）；本条已有修复 PR 且无生产触发路径，作为 LLK 潜伏缺陷留观，待实际触发报告出现再编目。
- **tt-metal #57066**（Quasar TRISC3 mailbox incorrect results）：bring-up 阶段（DPRINT 调试中、"reading them at the wrong times maybe?"）、无最小复现结论。留观。
- **tt-metal #57023**（quasar conv2d DFB credit underflow semget assert）：functional simulator 上 fail-loud assert（credit 计账非性能数据错误），且 `SPLIT_PROGRAM=1` 为现行依赖路径。留观。
- **tt-metal #57022**（borrowed-memory DFB stateful constraint query TT_FATAL）：mock device 上 fail-loud、影响 tt-mlir 优化器静默丢 config（非性能正确性）。留观。
- **tt-metal #56993**（GLM chunked prefill indexer-K PCC 0.059 CI 失败）：仅 CI 链接 + "Related: #56259"，无根因分析（与 09-17 #56847 同判：CI 事件类留观）。
- **tt-metal #56958**（binary_ng 混合 FP32/BF16 sharded add/sub + 融合 ABS 激活 12 例失败）：与 1.221（#56738 混合 dtype 位域旋转）同族同触发面，issue 自述是在验证 #56850 时发现的 pre-existing 失败、"not necessarily identical erroneous values"。作为 1.221 语境补充（该族在融合 ABS 激活维度亦有实例），不占新编号。
- **tt-metal #56949**（GPT-OSS-120B BH Galaxy topk_router_gpt 需 12 DRAM 对齐核而 BH 只有 8 → decode warmup TT_FATAL）：fail-loud 启动失败（模型层未尊重 @skip_for_blackhole），非性能正确性。不收录。
- **tt-metal #56954**（`conj` memory_config 只作用于虚半边）：issue 本身结论是"documentation defect more than a runtime one"（决策 B：文档化不对称）。不收录。
- **flashinfer #5318**（v0.7.0 分支缺 #5097 sparse-MLA cpb L2 guard-rail halving）：release 分支回移缺失（集成/发布治理），非性能正确性。不收录。
- **transformers #48886**（四个 MoE 模型 `aux_loss` 输出类型从未赋值恒 None）：API 契约/输出字段缺陷，静默但非推理正确性（loss 恒 None 由调用方显式消费）。留观。
- **vLLM #57562 之外**的 vLLM 窗口其余命中（#57555/#57521/#57515/#57508 等）：性能/不可定位/RFC 类。

### E. arxiv

- **2609.19743 "Syndrome Decoding for Silent Data Corruption in Quantized Integer GPU Arithmetic"（SProbe，2026-09-17 提交）** — INT8 量化推理 GEMM INT32 累加器无 parity/ECC，瞬态故障返回合法错整数不触发中断；SProbe 追随验证内核：三素数域 Freivalds 随机化门（漏检概率 ≤2^-141）+ per-row 幂和 syndrome Reed-Solomon 解码（Berlekamp-Massey/Chien/Forney）定位并**原位修复**最多 4 个碰撞错误/行；H100 上检出七类注入故障全部、INT8 医疗 LLM 消除全部观察到的静默损坏；开销 49%（N=16384）→11%（N=65536）。**检测/修复方法论文**——但与第 3 类（BFA/SDC 检测）和第 1 类（SDC）直接相关的**推理侧 SDC**（此前 TrainSDC/SCOUT/LLM-PRISM 均为训练侧）。作为 §3.1 补充来源编入（不占新编号）。
- 其余命中（TrainSDC/SCOUT/LLM-PRISM/JITterFlip/BLINK/2604.00726/2608.05863/2606.28958/Nexus/2609.17983/2608.07408）均已编目或为已判方法论文。

## 重点突出的新发现总结

1. **SGLang #40156（MoE negative-index wrap OOB）**：一个 `0`（本应 None）+ PyTorch 负索引语义 + dispatch 顺序（映射先于 drop 检查）三层小错叠加成"整批灌入单专家 + OOB 写"，CUDA graph 下静默垃圾洪泛——典型的复合根因静默损坏。
2. **SGLang #40144 × 11.77（DFLASH/DSpark 布局歧义跨栈对）**：同一 anchor-first vs 1+N 布局歧义在 vLLM（11.77，09-14）与 SGLang（本日）各自静默选错方向，且两侧的正确元数据字段名互不兼容（`sample_from_anchor` 缺失 vs `query_zero_predicts_next` 不被读）——生态级元数据契约缺口。
3. **SGLang #40084 × 7.68（QuickReduce 悬崖对偶）**：7.68 是高端溢出（bf16→fp16 inf），本条是低幅值 reciprocal 饱和衰减——同一 codec 的两个值域方向现在都有实测端到端因果证据。
4. **flashinfer #5334（TMEM 时序）**：1.223（09-16）之后第二个 TMEM 消费者时序缺陷，且本条给出 SASS 控制位级证据 + "源码级 fence 审计不成立"的方法论警示。
5. **vLLM #57562（伪 token 注入）**：runner 在未完成 prefill 的步上返回采样 token 并被追加进用户可见响应——assert 挡住的是崩溃，真正危险的是移除 assert 后的静默输出污染。

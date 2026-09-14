# LLM 推理故障每日扫描报告 — 2026-08-31

- **扫描日期**：2026-08-31（滚动窗口 created:>=2026-08-29，与昨日报告重叠部分已按主文档 issue 编号查重剔除）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 805 条（第 0 层 17 + 第 1 层 674 + 第 2 层 114）
- **结果**：新发现并合并 **10 条**（6 个新子签名 + 4 个已知类别变体）；另有 3 条修复型 PR 作为补充来源引用不占新编号（PR #54643→新 6.175、PR #54634→已知 Kimi-K3 启动族、PR #54638→已知 #51035 FlashMLA wedge 族）。arxiv 命中两篇强相关新论文（Groundhog GBFA 2608.25276、ROBBIN 2608.23774）已编目为 3.16/3.17；其余命中均为已编目论文（SCOUT、LLM-PRISM、BitFlipScope、When Latent Agents Lie、Hollow-LLM、Addressable Memory for Video World Models 等）或不收录的方法论文（Moment-KV、Runtime Observability）。
- **API 事件记录**：GitHub Search API 当日前两次调用被 IP 级 403 rate-limit 拦截（`rate limit exceeded for 172.236.135.218`，共享出口 IP 的窗口配额被占满）；等待窗口重置后重试全部成功，8 组搜索全部完成，无丢弃。core REST API 配额全程充足。

---

## 一、新编目条目（10 条，已合并进主文档）

### 第 1 类 · 硬件 / SDC（数值内核原语）（+1）

1. **tt-metal #54970**（修复 PR，closed #5305）— Fix mul reduce scalar zero-flag initialization — https://github.com/tenstorrent/tt-metal/pull/54970
   `[新类别/新签名]`（→ 1.129）`mul_reduce_scalar` 初始化不恢复格式默认的 `ALU_ACC_CTRL_Zero_Flag` 状态：前序 `copy_init()` 配置的 PRESERVE（保留 SrcB denormal）模式泄漏进 GAPOOL/MVMUL FPU 归约——**硅上改变 denormal 处理语义**（最大有限 BF16 × 最小 subnormal scaler 结果非零）；TTSim 以 `UnsupportedFunctionality ... not modeled` 拒绝该泄漏状态。修复在 WH/BH 标量归约初始化路径调 `_configure_default_zero_flag_state_()`，并加了 BF16-denormal 定向回归测试。这是「**LLK 算子间隐藏架构状态位的跨算子泄漏**」——同一模式的第二个实例（第一个是 2026-08-29 的 1.122 softplus 修复未跨 arch 传播）。

### 第 3 类 · 比特翻转攻击（+2，均 arxiv）

2. **arXiv:2608.25276** — Groundhog Bit-Flip Attack: Seeding Infinite Generation Loops in Mixture-of-Experts LLMs through Bit Flips — https://arxiv.org/abs/2608.25276
   `[新类别/新签名]`（→ 3.16）首个针对 MoE LLM 的 **bit-flip Denial-of-Wallet 可用性攻击**：定位 routing 层中与 EOS 等终止 token 强相关的 expert 激活位并翻转/等效停用——平均只需停用 **<4 个 expert**，四个真实 MoE LLM 上平均输出膨胀 **5912%**，多数样本打到 max tokens，而语义保真度大体保留（对话/推理/agentic 三种模式均验证）。与既有 BFA（追求 ASR/语义操纵）不同，本条目标函数是**经济性可用性**（token 计费放大），监控面表现为"完全正常的推理在烧钱"。
3. **arXiv:2608.23774** — ROBBIN: Rowhammer-Based Backdoor Injection during Inference — https://arxiv.org/abs/2608.23774
   `[新类别/新签名]`（→ 3.17）硬件感知的 Rowhammer 后门注入：先刻画目标 DRAM 的**设备级 bit-flip 指纹**，再把模型权重页映射迭代安排到能最大化 ASR 同时保留 TA 的物理页——把"附带翻转"从副作用升级为攻击设计的组成部分，跨三种 DRAM 设备保持鲁棒。此前算法层 BFA 假设（任意位可翻）与硬件现实（翻转位置设备特定）脱节导致攻击不可靠，本条补上这一环。评估在 ResNet-20/VGG-16 + CIFAR-10（DNN 推理域，机制可迁移到 LLM 权重页映射）。

### 第 5 类 · 分布式 / 多 GPU 并行（+2）

4. **SGLang #37187 + PR #37199** — GPT-OSS with DP attention enabled produces garbage output (GSM8K 0/128 vs 124/128) — https://github.com/sgl-project/sglang/issues/37187
   `[已知类别]`（→ 5.26，var:5.24）`moe-a2a-backend none` + `TP==attention DP==EP` 时 `GptOssSparseMoeBlock.forward_normal()` 先做 TP all-reduce（每 rank 得完整 s），DP-attention 后处理再对同一 TP 组 `reduce_scatterv`（再次求和）→ 输出近似 **TP×s（TP4 即 ×4）**：GSM8K 124/128→**0/128**、MMLU 110/128→4/128，响应充满重复 Harmony channel marker、空答案、runaway 生成打满 8192 上限。系统性排除 chunk size / MoE 后端 / CUDA graph / overlap / radix cache / hybrid SWA KV pool 均无法恢复。与 5.24（MHC communicator 缺陷对）同一"all-reduce 与 reduce-scatter 双重归约"签名，本条在 GPT-OSS 官方 forward 路径且以准确率塌到 0% 显形。
5. **SGLang PR #37250** — fix(kda): preserve split-grid program ids — https://github.com/sgl-project/sglang/pull/37250
   `[已知类别]`（→ 5.27，var:6.159 `vllm_glm53flash_multi_kernel_random_ima_single_corrupter`）KDA Triton kernel 的 `SPLIT_N_HV_GRID` 选定 `(NV, N, HV)` program-id 映射后，残留的扁平 grid 赋值立即覆写 `i_k/i_n/i_hv`——`NV > 1` 时 axis 0 被重新解释为 `i_k`，产出**越界 K tile 并损坏 recurrent state**：GLM-5.3-Flash 在 MI355X 服务上显形为 BF16 NaN 与非法 KPool page-table 值。修复删除陈旧赋值（state max abs diff 1.19e-07）。按"多 GPU 服务下 state 损坏"归第 5 类并 var 引用 6.159（GLM-5.3-Flash KDA 家族）。

### 第 6 类 · 服务栈 / 缓存正确性（+2）

6. **vLLM PR #54643** — [Bugfix][MooncakeStore] Fix finish-time save crash on hybrid models — https://github.com/vllm-project/vllm/pull/54643
   `[新类别/新签名]`（→ 6.175）hybrid（多 KV-group）模型（Kimi-K3 MLA+GDN）finish-time save 路径：`MultiConnector` 下 `update_state_after_alloc` 只把真实 `num_external_tokens` 转发给第一个 connector，`KVCacheStoreSendingThread → _handle_request` 用全局 g_idx 索引 per-group `block_ids_per_group[g_idx]` 抛 IndexError。**为 offload/迁移引入的 saver 线程在 hybrid 多组模型上 crash**——与 6.157（KV offload 完成请求未写末槽跨请求投毒）同 Mooncake/offload 域，本条是调度器→connector 契约在多组泛化时的缺口。
7. **vLLM #54638（修复 PR，关联 #51035）** — Add worker-side watchdog for wedged device kernels — https://github.com/vllm-project/vllm/pull/54638
   `[已知类别]`（→ 补充来源，编入 6.176 新签名）FlashMLA sparse-decode kernel 可**静默 wedge**：ptxas 把 `tcgen05.alloc.cta_group::2` 降低为 CTA-pair TMEM 分配自旋循环（无超时），停滞 rank 以 100% SM 利用率空转，唯一错误 300s 后才在不相关 collective 中浮现。自旋循环是 ptxas 生成的、源级无法加界——PR 只能武装 worker 级 watchdog（复用 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`）+ 给 FlashMLA 编 `-lineinfo`。**编译器生成的无限自旋 = 源级不可修的挂起面**，作为第 6 类新子签名编目（6.176），并作为 #51035 的补充修复来源。

### 第 7 类 · 量化（+1）

8. **vLLM #54641** — Identity entry in packed_modules_mapping silently disables layer_quant_config (Quark) — https://github.com/vllm-project/vllm/issues/54641
   `[新类别/新签名]`（→ 7.65）`packed_modules_mapping` 的**恒等条目**（模块映射到自身，如 `"gate_up_proj": ["gate_up_proj"]`——plamo3/glm4_1v/molmo 等 in-tree 模型的既有合法模式）被两条查找路径一致解释为"无 shard 信息"：`shard_name == layer_name` → config=None → **静默回退 global_quant_config**。混合精度 Quark checkpoint 的 per-layer scheme 被全局 scheme 替换，直到 weight load 才以 shape mismatch 显形；无警告无日志。`layer_quant_config` 语义把"名字相同"与"无信息"混为一谈——对真融合（q_a/kv_a_proj_with_mqa）正确、对恒等条目错误。

### 第 8 类 · KV 缓存 / 注意力（+1）

9. **vLLM #54623 + PR #54624** — Qwen3.5 family never loads checkpoint KV-cache scales — fp8 KV silently runs at scale 1.0 even for calibrated checkpoints — https://github.com/vllm-project/vllm/issues/54623
   `[已知类别]`（→ 8.59，var:7.52）Qwen3.5 家族所有 `load_weights` 用裸 `AutoWeightsLoader` 且从不调 `maybe_remap_kv_scale_name`（gemma4/gpt_oss/hunyuan_v1 均调）；checkpoint 的 `...self_attn.{k,v}_scale` 无路由到运行时 `...self_attn.attn.{k,v}_scale`。unsloth/Qwen3.8-27B-NVFP4（**2.4M 下载**）携带 32 个校准 scale 张量 + 显式 `kv_cache_scheme: {dynamic: false}`，`--kv-cache-dtype fp8` 下仍以 scale 1.0 服务，仅一条 WARNING 提示。手工重命名 32 个张量即恢复加载（对照实验证明）。与 7.52（SGLang compressed-tensors KV scale 丢弃）完全同构——vLLM 侧、Qwen3.5 家族、以"校准白费 + unit-scale 量化精度劣化"显形。

### 第 9 类 · 位置编码 / RoPE（+1）

10. **transformers #48428** — RoPE corruption due to TF32 on B200 when using `torch.set_float32_matmul_precision("high")` — https://github.com/huggingface/transformers/issues/48428
    `[已知类别]`（→ 9.20，var:4.2）`Qwen3RotaryEmbedding.forward`（Llama/Qwen2/3/Mistral 共用路径）的 `freqs = inv_freq.float() @ position_ids.float()` 是 fp32 matmul：外层 `torch.autocast(enabled=False)` 只强制 dtype、**不控制后端**。`set_float32_matmul_precision("high")` 下 B200（batch>1）dispatch 到 TF32 GEMM——TF32 尾数 10 bit 只能精确表示到 2^11=2048，**position_ids 2049 被截断为 2048**，RoPE 角度大错。A100 不复现（不同 dispatch）。修复二选一：局部 `allow_tf32_ctx(False)` 或改逐元素乘法避开 matmul。同源报告：thinking-minds-lab/batch_invariant_ops#23、ByteDance-Seed/VeOmni#959。4.2（tf32_matmul_truncation）已知 TF32 截断，本条是它在**位置编码维度的新显形面**——整数 position_id 的离散截断（非连续精度损失）且 arch 依赖（B200 触发 A100 不触发）。

---

## 二、查重后跳过/仅记录不编目的条目

- **triton #11519**（FuseNestedLoops llvm.assume miscompile）— 昨日已编目为 1.127，今日再现搜索结果，跳过。
- **vLLM PR #54642**（compile-cache hash 加入 multimodal limits）— 性能/缓存键加固型 PR，修复的碰撞未以正确性故障报告显形，暂不编目；与 6.141（hash 漏 spec tokens）同族，留作线索。
- **vLLM #54607 / PR #54608**（0% prefix cache hit rate 可能只是共享前缀短于一个 KV block）— 文档澄清型（行为正确、语义易误读），非故障，不编目。
- **vLLM PR #54633**（MiniCPM-V `video_embeds` 键名 rename 后被 vision parser 丢弃，4.6 上视频静默丢失无错）— 有静默丢输入成分，但属模型 enablement 边缘路径（`--enable-mm-embeds`），本日未编目；若后续再现有真实用户命中可补。
- **vLLM PR #54622**（chat template content format mismatch 警告自 #19735 起不可达）— 检测器失效（warning 死代码）而非直接故障，不单独编目；已作为 10.43 的关联线索记录。
- **SGLang PR #37211**（deepep_v2 `ep_scatter_from_psum` 缺参 → 每 rank 首个 forward 必 crash）— fail-loud 启动期 crash（无静默成分），按惯例不占新编号。
- **SGLang PR #37276**（PD prefill worker 多收 10.3 GiB 幻影 speculative Mamba state 预留）— 内存超预留型（性能/容量面），未编目。
- **TRT-LLM PR #18448**（tool parser 匹配 marker 但零提取时静默返回空 tool_calls + HTTP 200，本 PR 补 warning）— 与 10.42（Kimi-K3 parser 静默丢 tool call）高度同族的**检测面增强**；因本 PR 是日志增强而非正确性修复、且 10.42 已覆盖该签名，不另立条目。
- **TRT-LLM PR #18421**（absorbed forward step 返回 None 被 executor 解引用 crash）— fail-loud crash 修复，不编目。
- **TRT-LLM PR #18443/#18436**（FlashInfer CUDA graph plan 共享 workspace 隔离，nvbugs 6566726）— nvbugs 内部引用、公开信息不足以判定故障形态，暂不编目。
- **tt-metal #54975**（LLK_ASSERT 宏在 -Wtemplate-body 下破坏 trisc 构建）— CI 构建失败，非数值正确性，不编目。
- **arxiv**: SCOUT（2608.11034）、LLM-PRISM（2604.10390）、BitFlipScope（2512.22174）、When Latent Agents Lie、Hollow-LLM、Addressable Memory for Video World Models（14.7 已引）均已编目；Moment-KV（压缩方法）、Runtime Observability for Heterogeneous Attention Memory（观测性框架，含 DeepSeek-V4 packed compressed-KV silent 定位案例但主体为方法论文）不收录。

---

## 三、重点新发现摘要

1. **Groundhog GBFA（3.16）**——首个 MoE 上的 bit-flip **Denial-of-Wallet** 攻击：停用 <4 个 expert 即平均放大输出 5912%，语义大体保留。为第 3 类引入"经济性可用性"新目标函数，监控面完全伪装成健康流量。
2. **ROBBIN（3.17）**——把 Rowhammer 后门从"算法假设"升级为"硬件感知"：先指纹化 DRAM 翻转位再安排权重页映射，跨设备鲁棒。补全了 BFA→Rowhammer 现实化的最后一环。
3. **HF transformers #48428（9.20）**——`set_float32_matmul_precision("high")` + B200 使 RoPE 角度计算走 TF32：**整数 position_id > 2048 被离散截断**。已知 TF32 机制（4.2）在位置编码上的全新显形面，且影响 Llama/Qwen/Mistral 全家（共用代码路径）。
4. **vLLM #54623（8.59）**——Qwen3.5 全家族 fp8 KV 静默丢校准 scale（240 万下载的 unsloth checkpoint 全体中招）；与 SGLang 7.52 构成跨栈同构对，说明"KV scale 名字重映射缺失"是行业级模式。
5. **vLLM #54641（7.65）**——量化配置查找把"恒等映射"当"无信息"：静默回退全局 scheme，混合精度 checkpoint 加载期 shape mismatch 才显形。
6. **SGLang #37187（5.26）**——GPT-OSS DP attention 双重归约把 GSM8K 打到 0/128，是 5.24 双重归约签名在官方 forward 路径的最严重显形（准确率塌零而非 schema 退化）。

---

## 四、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GH Search: vllm label:bug created:>=2026-08-29 | 126 命中（首 2 次 403 rate-limit，重置后成功） |
| 2 | GH Search: sglang correctness/wrong result/NaN | 43 命中 |
| 3 | GH Search: TensorRT-LLM correctness/wrong output/NaN | 20 命中 |
| 4 | GH Search: triton miscompile/wrong result/numerical/NaN | 1 命中（已编目 1.127） |
| 5 | GH Search: tt-metal numerical/PCC/correctness | 69 命中 |
| 6 | GH Search: vllm KV cache/quantization/bit flip/corruption/NaN | 114 命中 |
| 7 | GH Search: transformers RoPE/tokenizer/wrong output | 10 命中 |
| 8 | GH Search: 全局 "silent data corruption" GPU/LLM | 358 命中（多为无关仓库噪声） |
| 9-13 | arxiv: SDC+LLM / bit flip / KV cache corruption / LLM inference correctness / GPU soft error | 见上文编目与排除 |
| 14-25 | GH core API 单条抓取（#48428/#37187/#54623/#54641/#54638/#54633/#54622/#37250/#54970/#54597/#54555/#54636 详情） | 全部成功 |

原始响应存于 `scan_raw_2026-08-31/`。

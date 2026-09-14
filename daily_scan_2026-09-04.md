# 每日扫描报告 — 2026-09-04

- 扫描日期：2026-09-04（UTC）
- 扫描窗口：GitHub `created:>2026-08-28`（7 天滚动）；arxiv 最近提交排序
- 目标仓库：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal
- 主文档基线：882 条（2026-09-03 合并后）

## 结果总览

- 共执行 10 组 GitHub Search API 查询（全部一次成功，无限流 403、无 Validation Failed）+ 6 组 arxiv API 查询
- 去重核实后：**30 条新条目合并进主文档**（22 个新子签名 `[新类别/新签名]` + 8 个已知类别变体 `[已知类别]`），882 → **912**
- 另有 1 条疑似重复未合并（见"跳过条目"），4 条已编目确认不重复添加

## 新合并条目（30 条）

### 第 1 类 · 硬件/SDC（+7）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 1.162 | ttnn.log_sigmoid diverges to -inf for large positive bf16 inputs (x > ~172) | tt-metal #55457 | `[已知类别]` var:1.102 | logsigmoid 大正分支对已取负 x 再取负求 exp，等价 exp(+\|x\|) 指数增长至 −inf（x≥~266）；golden 应趋 0 |
| 1.163 | ttnn.batch_norm updates running_var with biased variance (missing Bessel's correction) | tt-metal #55438 | `[新类别/新签名]` | training 模式用有偏方差（1/m）更新 running_var，从不乘 m/(m−1)；统计量系统性偏低 |
| 1.164 | ttnn.polygamma returns exactly 0 for n ≥ 7 over a wide band of ordinary positive x | tt-metal #55105 | `[新类别/新签名]` | n! 尺度因子在累加后施放，中间值先行溢出 → 带状精确 0（最坏 1.678e7 ULP）；k≤11 是公开 API |
| 1.165 | ttnn.atan2 returns exactly 0 (or pi/2) for max(\|x\|,\|y\|) > 2^126 | tt-metal #55124 | `[已知类别]` var:1.154 | reciprocal 下溢致象限坍缩，最大误差 45°；#52340 修复之外的第三个未守卫调用点 |
| 1.166 | ttnn.conv1d: prepare_conv_weights vs internal prep disagree silently by up to 1e37 (L∈[8193,8704], Wormhole) | tt-metal #55545 | `[新类别/新签名]` | 同一卷积两条"可互换"的权重准备路径静默分歧；trace capture 只能用错的那条 |
| 1.167 | [Bounty $5000] INT_MIN correctness in int32 div/remainder/fmod/scalar promotion | tt-metal #55502 | `[新类别/新签名]` | INT32_MIN 流经残差校正路径产生静默、有限、貌似合理的错误（WH/BH），复现文档 #51476 |
| 1.168 | ptxas 12.9.86 drops the immediate of a second add.s16x2: wrong results on H100 | triton #11581 | `[新类别/新签名]` | **编译器工具链级 silent miscompile**：第二个 packed add 立即数被丢弃（+1 丢失）；TRITON_INTERPRET=1 与 ptxas-blackwell 13.3.33 均正确——数值正确性随 ptxas 版本翻转 |

### 第 4 类 · 数值/精度（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 4.17 | Batch-invariant matmul is not actually batch-invariant, and TF32 causes precision degradation | vLLM #55131 | `[已知类别]` var:4.2 | 官方 batch-invariant 内核自身测试就容许跨 batch 差异；且该路径仍放行 TF32——不变式与精度双失守 |

### 第 5 类 · 分布式/多 GPU（+2）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 5.30 | Heterogeneous TP can corrupt Mamba2 conv state in disaggregated serving | vLLM #55208 | `[新类别/新签名]` | 异构 TP（P/D 实例 TP 度不同）下 Mamba2 conv state 传输被损坏，decode 侧静默劣化 |
| 5.31 | [Kimi-K3][Hopper] Low-M TP8 KDA projection fails CUTLASS DSL compilation for sm_90a | vLLM #55350 | `[新类别/新签名]` | #54697 引入的 KDA projection-overlap 路径在 CUDA-graph profiling 期 NVVM_ERROR_COMPILATION，32×H100 双次独立复现 |

### 第 6 类 · 服务栈/缓存正确性（+8）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 6.189 | GLM-5.3 accuracy collapses on ROCm (GSM8K 91.6% → 14.9%) after #53155 forces MRV1 | vLLM #54924 | `[已知类别]` var:6.176 | 架构字符串过包含：GLM-5.2/5.3 共享 `GlmMoeDsaForCausalLM`，MRV1 对 5.3 产出损坏生成；main 仍在 |
| 6.190 | Model Runner V2 never locks the workspace; post-capture growth silently invalidates captured CUDA graphs | vLLM #55336 | `[新类别/新签名]` | arena 替换后旧地址烤入 graph 的内核 replay 写入已易主内存——**无 crash 的跨请求内存损坏**；V1 的锁在 V2 重写中丢失 |
| 6.191 | Scheduler batch defaults silently fall back to smallest-GPU values inside MIG containers | vLLM #55381 | `[新类别/新签名]` | NVML 对父设备权限错被 catch-all 吞掉 → device_memory=0 → H200 MIG 拿 2048/256 默认值，吞吐 -28~53%，仅 debug 日志留痕 |
| 6.192 | A single malformed multimodal request takes down the whole engine; client retries spread it node-by-node | SGLang #38022 | `[新类别/新签名]` | 单请求异常升级为引擎 SIGQUIT + 重试×粘性路由巡回击穿：2 小时 5 节点集群全灭 |
| 6.193 | GLM-5.3-Flash (DSA): HiCache host-tier load-back corrupts generation even without spec decoding | SGLang #38031 | `[已知类别]` var:6.184 | DSA 打包 KV 经 host 层往返即损坏（掉 tool call、重复退化）；EAGLE/MTP 全关仍复现——HiCache 单独即坏 |
| 6.194 | Mapped-layer courier can reuse CUDA storage before compute finishes | SGLang #37963 | `[新类别/新签名]` | offload courier 缺 `record_stream(compute_stream)`：allocator 只见 copy stream 即复用 storage，compute 仍在读 |
| 6.195 | [PDMux] scheduler event loop deadlocks in decode_stream.synchronize() on mixed batch after ≥4K prefill (hybrid GDN) | SGLang #37904 | `[新类别/新签名]` | 长 prefill 后混合 decode+split-prefill batch 死锁；时序敏感（30–70% 触发率）；PR #37905 含自包含复现 |
| 6.196 | custom_mask grows unbounded via torch.cat in EAGLE/DFlash speculative decoding | SGLang #37553 | `[新类别/新签名]` | 只增不缩的 cat 掩码：GPU 内存无界增长 + 陈旧掩码数据继续喂注意力内核 |

### 第 7 类 · 量化（+4）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 7.68 | [ROCm] QuickReduce converts bf16 to fp16 without saturation: >65504 silently becomes inf | vLLM #55345 | `[新类别/新签名]` | 通信库 dtype 收窄当无损：单 rank 单个 1e30 经 all-reduce 毒化全张量；位于所有 codec 公共路径 |
| 7.69 | modelopt NVFP4 MoE: mismatched w1/w3 global scales detected, warned about, then used anyway | vLLM #54974 | `[新类别/新签名]` | 检测→warning→照用错值三连；实测 69.1% 的 gate/up scale 对不匹配（max ratio 10.0），全部 43 层 MoE 受影响 |
| 7.70 | nvidia/Qwen3.8-2.4T-A95B-NVFP4 serving on Ampere | vLLM #55010 | `[新类别/新签名]` | 32×A100 TP16×PP2 启动不可达：tracking issue 记录两处独立缺陷（其一即 #55008） |
| 7.71 | CUTLASS FP8 linear kernel selected on Ampere while SM80 dispatch is INT8-only | vLLM #55008 | `[已知类别]` var:7.53 | 7.53「is_supported 不查算力」的生产级复现（官方 NVFP4 checkpoint + nightly 实测命中）；修复 PR #53376 未及受影响版本 |

### 第 8 类 · KV 缓存/注意力（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 8.64 | [ROCm] GLM-5.3-Flash kpool 32-page split (hybrid block_size=1152) GPU Memory access fault on MTP decode | vLLM #55280 | `[已知类别]` var:8.56 | 同一 kpool 页粒度错配：8.56 显形为静默自覆写，本条在 MTP decode 显形为硬件级 fault——同根因两端显形谱 |

### 第 9 类 · 位置编码/RoPE（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 9.21 | FourierRotaryEmbedding pads sin with 1 and corrupts unrotated channels | vLLM #55204 | `[新类别/新签名]` | padding 常量语义错误：sin pad 1 而非 0 → 未旋转通道被施加虚假旋转，位置编码系统性损坏 |

### 第 10 类 · 分词器/模板/结构化输出（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 10.48 | guidance backend rejects tool-calling structural tags (KeyError: 'triggers') | vLLM #55152 | `[新类别/新签名]` | 服务栈与 llguidance 的 schema 版本失配：strict/required/named tool_choice 全部 400；pydantic-ai 默认 strict → 实际全坏 |

### 第 11 类 · 解码/采样（+3）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 11.61 | thinking_token_budget ignored by Model Runner V2 with Qwen3.8 NVFP4 + MTP | vLLM #54906 | `[新类别/新签名]` | V2+MTP 下预算失联：单请求烧满全部 completion 配额不产 `</think>`；去掉 spec 即正常 |
| 11.62 | Qwen3.5/3.6 MTP resolves n_predict=None for multimodal-wrapper checkpoints | vLLM #55322 | `[新类别/新签名]` | mtp_num_hidden_layers 从 wrapper 顶层而非 text_config 读取 → None → MTP 静默失效 |
| 11.63 | Qwen3.8-Flash-Next + MTP: episodic 0% draft acceptance + repetition collapse | vLLM #55357 | `[已知类别]` var:6.178 | 偶发 spec 状态发散 episode（~2000 drafted/0 accepted + thinking block 无尽重复，一请求烧 1M token）；**跨模型同签名**（Gemma 4 同日同坏）→ 指向引擎构建层 |

### 第 12 类 · 权重/适配器/检查点（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 12.31 | DFlash2 draft gets 0% acceptance with --dtype float16 on XPU (bf16 works) | vLLM #55250 | `[新类别/新签名]` | fp16 draft 与目标 token 级永不一致：输出正确（验证环掩盖）但加速为零——「安全侧正确 + 性能侧静默失效」 |

### 第 13 类 · 模型图/架构（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 13.27 | Spark2.5 attention output gate hardcodes sigmoid, ignores config gate_attn_act_mode | SGLang #37639 | `[新类别/新签名]` | 实现只覆盖默认配置：config 键存在、实现不消费——非 sigmoid checkpoint 输出静默偏差 |

## 重点新发现（为什么值得编目）

1. **1.168（triton #11581）— 工具链级 silent miscompile 的新轴**：缺陷在 ptxas 汇编器而非内核源码，同一 Triton 源码在 ptxas 12.9.86（错）与 13.3.33（对）下数值不同。这是编目中首个「数值正确性随汇编器版本翻转」的条目，为故障编目新增**工具链版本维度**。
2. **6.190（vLLM #55336）— 最危险形态**：CUDA graph 烤入的 workspace 地址在 arena 替换后失效，replay 写入已被 allocator 转发的内存——无 crash、无警告的**跨请求内存损坏**，症状与根因完全解耦。
3. **6.192（SGLang #38022）— 故障放大链新形态**：「单请求异常 → 引擎自杀 → 客户端重试 × 粘性路由 → 巡回击穿全集群」，2 小时 5 节点全灭。输入校验、异常粒度、基础设施正反馈三段连环。
4. **7.69（vLLM #54974）— warn-and-continue 反模式**：量化 scale 失配被检测、被警告、然后照用，69.1% 的层带着最大 10× 的错缩放静默服务。
5. **11.63（vLLM #55357）— 跨模型指纹归因**：两个架构无关 checkpoint 在同一引擎构建上同日同签名失败（0% draft 接受 + 重复坍塌），把根因从模型权重推向引擎侧 MTP 状态管理——罕见的强归因证据结构。

## 跳过条目（去重/不收录）

| 条目 | 原因 |
|---|---|
| SGLang #37633（QSA extend forward IMA @ 并发） | 疑似与 #37448/#37606（6.183 prefill breakable graph 弱引用族）同根因的未定实例，待根因明确后编目 |
| vLLM #55322 伴随、#55259/#55339/#55196/#55129/#55017 等 RFC/Feature | 非正确性缺陷 |
| tt-metal #55394/#55211/#55200/#55125/#55015 | perf/CI/feature 类，非数值正确性 |
| SGLang #37813/#37792/#37150/#37349/#37519/#38069/#37372/#37405 | tracking/feature/guide 类 |
| vLLM #55350 已编目（见 5.31）；#55048/#55132/#55131 等已编目确认 | 主文档已有 |
| arxiv：TrainSDC(2608.30769)/SCOUT/LLM-PRISM/JITterFlip(2608.29745)/Groundhog(2608.25276)/ROBBIN(2608.23774)/指令级 FI(2601.19912)/共享 KV Bit-Flip 均已编目；GPUBreach(2605.03812)/LM-Fix(2511.02866) 为攻击提权/检测防御方法论文，不收录 | 已编目或不符编目范围 |

## API 调用记录

| # | 查询 | 命中 |
|---|---|---|
| 1 | vllm "wrong output" OR "incorrect result" is:issue created:>2026-08-28 | 6 |
| 2 | vllm "KV cache" corruption is:issue created:>2026-08-28 | 9 |
| 3 | sglang correctness OR "wrong result" is:issue created:>2026-08-28 | 11 |
| 4 | transformers label:bug "wrong output" OR "wrong result" created:>2026-08-28 | 0 |
| 5 | vllm quantization NaN OR "quant wrong" is:issue created:>2026-08-28 | 43 |
| 6 | TensorRT-LLM correctness OR "wrong output" is:issue created:>2026-08-28 | 0 |
| 7 | triton correctness OR "wrong result" OR miscompile is:issue created:>2026-08-28 | 2 |
| 8 | tt-metal PCC OR "incorrect" numerical is:issue created:>2026-08-28 | 23 |
| 9 | sglang NaN OR garbage OR corruption is:issue created:>2026-08-28 | 20 |
| 10 | vllm "silent" OR nondeterministic OR "bit flip" is:issue created:>2026-08-28 | 61 |
| A1–A6 | arxiv：SDC+LLM / bit flip+LLM / KV cache+corruption / LLM inference+correctness / GPU soft error / rowhammer+GPU | 全部已编目或不收录 |

- 条目核实直接使用 search API 附带的 issue 正文（无需额外 core REST 详情调用）
- 所有查询间隔 ≥8 秒，无 HTTP 403 限流

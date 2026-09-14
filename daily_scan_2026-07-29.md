# LLM 推理故障每日扫描 — 2026-07-29

> **扫描日期**：2026-07-29（UTC）
> **扫描窗口**：GitHub issues/PRs `created:>=2026-07-22`（7 天滚动，覆盖昨日未捕获条目）；arxiv 按提交日期降序取最近 5–8 篇
> **搜索范围**：6 个 GitHub 目标仓库（vllm / sgl-project·sglang / transformers / TensorRT-LLM / triton）× 8 组关键词全局查询 + 跨仓库关键词搜索 + arxiv 7 组关键词
> **对照基准**：[`FAULT_MASTER_REFERENCE.zh.md`](FAULT_MASTER_REFERENCE.zh.md)（215 条已编目故障，三层来源）
> **API 状态**：GitHub Search API 10/10 成功（remaining 50/60，无 rate-limit）；arxiv 7/7 成功（HTTPS，`-L` 跟随 301）

---

## 滚动汇总

| 维度 | 数值 |
|---|---|
| GitHub 命中条目（去重后相关） | **15** |
| arxiv 命中条目（强相关） | **6** |
| 标注为 `[已知类别]` 的新实例 | 17 |
| 标注为 `[新类别/新签名]` 的候选 | **3** |
| 今日重点关注（🔴 高优先级） | 4 |

> **趋势观察**：本日窗口内**正确性 bug 集中在三个领域**——(1) 硬件内核层的 **Inf/NaN 位模式比较错误**（tt-metal、triton），(2) **量化精度回归**（SGLang FP8 reciprocal-mul、TRT-LLM NVFP4 权重映射），(3) **推测解码 + 混合架构**的静默输出损坏（EAGLE/SA spec decoding）。其中 **tt-metal isclose 的 Inf/NaN 位模式比较** 是一个罕见的、值得收录的**硬件 SFPU 数值比较原语缺陷**，编目中此前以"数值/精度"类覆盖软件路径，硬件内核比较原语本身返回静默错误值是**新签名**。

---

## 🔴 重点关注：新故障签名 / 高价值案例

### 1. tt-metal #51220 / #51540 — SFPU `isclose` 把 **Inf/NaN 位模式当 float 量级比较**，静默返回错误值 `[新签名]`

- **链接**：https://github.com/tenstorrent/tt-metal/issues/51220 · https://github.com/tenstorrent/tt-metal/issues/51540 · 修复 PR #51537
- **来源**：tenstorrent/tt-metal（open，2026-07-27 / 2026-07-29）
- **现象**：Wormhole + Blackhole SFPU 内核 `ckernel_sfpu_isclose.h` 的 Inf/NaN 修正分支声明 `constexpr int32_t inf_bits = 0x7F800000`（文档标注为"IEEE-754 abs(+inf)"），但比较语句 `v_if(a_abs >= inf_bits ...)` 走的是 **float `>=` 重载**而非位模式比较。`sfpi` 没有 `(vFloat, int32_t)` 重载，也没有 `vFloat→vInt` 转换，因此实际把**整型位模式当作浮点量级**来比。结果：在 +Inf/NaN 输入上 `isclose` **静默返回错误值**——无 assert、无 NaN、无 crash，就是错的数字。
- **为什么是新签名**：编目第 4 类"数值/精度/不确定性"覆盖的是**软件层**数值路径（激活溢出、归一化、累积误差）。本条是**硬件 SFPU 内核比较原语本身因 C++ 重载解析陷阱返回静默错误值**——一个属于"**内核数值原语类型系统缺陷**"的新子签名。同一静态审计还发现了 **Quasar exp 缺失范围保护**（item 3），同属"硬件数学库静默错误值"新签名。
- **影响**：任何依赖 `isclose` 做 Inf/NaN 处理的上层算子（梯度裁剪、loss 监控、稳定性检查）在含异常值的张量上得到错误判定。报告者明确标注"high-severity numerical defects, silently wrong value on a supported input range"。
- **对照编目**：`数值/精度` 类的硬件侧新子签名；当前 215 条无"SFPU/内核原语位模式比较错误"条目。

### 2. triton #11047 — `warpgroup_mma_wait` 不跨 warpgroup 同步，破坏 Triton 编程模型的线程发散 `[新签名候选]`

- **链接**：https://github.com/triton-lang/triton/issues/11047
- **来源**：triton-lang/triton（closed，2026-07-25）
- **现象**：`warpgroup_mma_wait` 可被跨多 warpgroup 的代码调用，但它**不按多 warpgroup 同步**，导致"pernicious thread divergence that breaks the Triton programming model"。复现为一段 2-warpgroup Hopper NVMMADistributedLayout 内核：MMA wait 在错误的线程范围上释放，后续读取读到**未完成的 MMA 结果**。
- **为什么是新签名候选**：编目第 1 类"硬件/SDC"和扩展目录 GPU 类覆盖的是**物理故障**（ECC、rowhammer、磨损）。本条是**编译器/运行时在合法 API 使用下的同步语义错误**——内核算出一个**静默错误值**而非物理位翻转。属于"**内核同步语义静默错误**"的新子签名，是 LLM 推理 GEMM 内核可靠性的潜在根因。
- **影响**：Hopper/Blackwell 上手写 warpgroup Triton 内核（含 vLLM/SGLang 自定义 attention/MoE kernel）可能静默产出错误值。报告者强调是"pernicious"（隐蔽的）。

### 3. triton #11065 — `inline_asm_elementwise` 对复制元素（replicated elements）执行多余次，破坏副作用语义 `[新签名候选]`

- **链接**：https://github.com/triton-lang/triton/issues/11065
- **来源**：triton-lang/triton（open，2026-07-27）
- **现象**：当元素数 < 线程数（如 N=32 但 num_warps=4 → 4× 复制），`tl.inline_asm_elementwise` **每个复制元素都执行一次**而非每个逻辑元素一次。Triton 故意不暴露复制，因此带副作用（如 `red.global.add.u32`）的内联汇编会**对一个内存单元累加多次**，产生静默错误结果。
- **为什么是新签名候选**：与 #11047 同属"**内核原语在合法但非显然配置下的静默错误值**"。对 LLM 推理的影响：任何用 `inline_asm_elementwise` 写的 reduction/原子更新自定义内核在 tail（不满 warp）场景下静默错误。

---

## 🔄 已知类别的新案例（GitHub）

### 服务栈 / KV cache / 调度正确性（第 6 类、K§3/K§8）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 1 | [vllm #50297](https://github.com/vllm-project/vllm/pull/50297) | P/D 预emption race：最终步 prefill 被预emption，KV blocks 已 purge，async connector 的 `request_finished()` 看到**空 block 列表** | `K§8 control-flow` + `K§5 cross-node` | 分离式 P/D 下，被 preempt 的请求的 KV 被清空但传输未取消，下游 connector 持有空 KV。无 crash。 |
| 2 | [vllm #50302](https://github.com/vllm-project/vllm/pull/50302) | 统一块表宽度到 128 token，修复 MRV1/MRV2/DSA indexer 在对齐 + DCP 下的**indexer buffer mismatch** | `K§3 block-slot-alloc` | 块表宽度计算在多个路径间分叉，导致索引器缓冲不匹配。 |
| 3 | [vllm #50235](https://github.com/vllm-project/vllm/issues/50235) | Kimi-K3 prefix cache 在 prompt 长度恰为 1536-token（混合注意力块大小）边界时**cache miss** | `K§3 block-slot-alloc` | 边界对齐导致缓存键失效，性能问题（非正确性），但属块分配边界 bug。 |
| 4 | [vllm #50287](https://github.com/vllm-project/vllm/pull/50287) | Mamba 块表容量在 DCP（decoder checkpoint）下未被保留 | `K§7 arch-specific-cache` | 混合架构（Mamba）块表状态在检查点恢复后不一致。 |
| 5 | [vllm #50272](https://github.com/vllm-project/vllm/pull/50272) | 修复 `short_conv` (LFM2) 模型的推测解码 | `K§8 control-flow（推测解码）` | 推测解码控制流正确性修复。 |

### 量化正确性（第 7 类）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 6 | [sglang #32616](https://github.com/sgl-project/sglang/pull/32616) | **恢复 per-token group 量化的除法行为**：#30924 把 FP32-scale 乘子从直接除法改成显式倒数乘法，在 Hopper 上引入**小 FP8 舍入差异**，导致 **GLM-5.2 decode 回归**；DeepSeek-V3.2 MTP L3 accept length 从 3.459 掉到 2.462 | `第7类 量化（FP8）` | 典型的"算法等价但数值不等价"量化精度 bug，倒数乘法 vs 直接除法的 FP8 舍入差。恢复后 bitwise 与旧 v2 kernel 一致。 |
| 7 | [trt-llm #16899](https://github.com/NVIDIA/TensorRT-LLM/issues/16899) | **反向精度 bug**：Llama-3-70B **FP16/BF16 推理失败而 FP8 正常**（TRT-LLM 0.21.c-rc0，8×RTX 6000D，CUDA 13.0） | `第7类 量化` + 自定义 kernel | 反直觉：高精度路径（FP16/BF16）失败而量化路径（FP8）正常。可能是 FP16 自定义 kernel 缺陷。 |
| 8 | [trt-llm #16867](https://github.com/NVIDIA/TensorRT-LLM/issues/16867) | Gemma4 `attention_k_eq_v`：NVFP4 v_proj scale 张量未从 k_proj 复制 → fused-QKV loader `KeyError: 'v'` | `第7类 量化` + `第12类 权重加载` | NVFP4 权重映射在共享 K/V 投影时漏复制 scale 张量。 |
| 9 | [sglang #32229](https://github.com/sgl-project/sglang/pull/32229) | NVFP4：SM100 上 auto MoE 改用 `flashinfer_cutlass` | `第7类 量化（NVFP4）` | Blackwell 量化 kernel 路由修复。 |

### 推测解码正确性（K§8）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 10 | [sglang #32396](https://github.com/sgl-project/sglang/pull/32396) | **EAGLE fused top-k=1 在 draft logits 含 NaN 时选中 masked tail lane**，返回 `token_id == vocab_size`（OOB） | `K§8 control-flow` + `数值/NaN` | NaN 未映射到有限 floor 就 argmax → 选中超范围 token。修复把 NaN 映射到有限 floor。 |
| 11 | [trt-llm #16826](https://github.com/NVIDIA/TensorRT-LLM/issues/16826) | SA（后缀自动机）推测解码在 prompt 接近 `max_seq_len` 时**crash executor event loop**——`default_max_tokens` 按 draft 膨胀后的 `max_seq_len` 推导 | `K§8 control-flow` | 推测解码对序列长度预算的推导错误。 |
| 12 | [trt-llm #16967](https://github.com/NVIDIA/TensorRT-LLM/issues/16967) | DSpark speculative drafter 的 Structured Output 问题 | `K§8 control-flow` | 推测解码 + 结构化输出交互 bug。 |

### 分词器 / 位置编码 / 模型图（第 9、10、13 类）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 13 | [transformers #47496](https://github.com/huggingface/transformers/issues/47496) | `DataCollatorWithFlattening` 用 **0-based `position_ids`**，**静默降级** RoBERTa 家族模型 | `第9类 位置编码` | RoBERTa 的位置嵌入偏移 padding_idx，0-based position_ids 导致位置编码错位，**静默**降级质量。 |
| 14 | [transformers #47475](https://github.com/huggingface/transformers/issues/47475) | Zamba2 `torch_forward` **因果性违反**：inter-chunk state recurrence 在错误轴上 reduce，位置 P 的 token 改变 P 之前位置的 logits | `第13类 模型图/架构` | 纯 PyTorch 路径（无 mamba_ssm）下的因果性违反——典型"参考实现 bug"。 |
| 15 | [transformers #47530](https://github.com/huggingface/transformers/issues/47530) | Phi3/Phimoe `prepare_inputs_for_generation` 把 `logits_to_keep=None` 传入 forward → 4-dim logits → `torch.multinomial` crash | `第13类 模型图` | 生成路径参数传递 bug。 |

### 性能 / 架构集成（第 14 类）

| # | 链接 | 标题（简） | 归类 | 说明 |
|---|---|---|---|---|
| 16 | [vllm #50264](https://github.com/vllm-project/vllm/issues/50264) | RDNA 上混合 Mamba 模型回退到 Triton paged attention，**长上下文 decode 性能崩塌**（518→32K context: 82.5→235.3 ms/token） | `第14类 长上下文` + 架构集成 | `use_rocm_custom_paged_attention` 要求 head_size==128 且 block_size==16，但 head_dim=256 的混合模型不满足 → 回退到慢路径。性能问题，非正确性，但长上下文 decode 崩塌。 |
| 17 | [sglang #32182](https://github.com/sgl-project/sglang/issues/32182) | PD warmup 在 decode ep32 + dp attention 下损坏（high priority） | `第6类 服务栈` + 分布式 | 解码并行 + DP attention 的预热路径 bug。 |

---

## 📄 arxiv 论文发现

### 强相关（LLM 推理故障/可靠性）

| # | 链接 | 标题 | 日期 | 分类 | 说明 |
|---|---|---|---|---|---|
| A1 | [arxiv 2607.19623](http://arxiv.org/abs/2607.19623v1) | **From Bit-Position Sensitivity to Unequal Error Protection for DNN Inference Memory** | 2026-07-21 | `[已知类别·bitflip]` 深化 | 跨 16 个 workload（含 transformer）+ 3 种浮点格式刻画**逐位故障敏感度**。核心发现：最低有效浮点位翻转影响极小，而特定高位翻转造成极端外 outlier 传播。**量化了 bitflip 的位敏感度过渡**——为编目 bitflip 类提供精确的保护粒度依据。 |
| A2 | [arxiv 2605.11202](http://arxiv.org/abs/2605.11202v1) | **Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing** | 2026-05-11 | `[已知类别·服务栈]` 方法论 | 用模糊测试持续发现 LLM 服务栈漏洞——明确指出"最危险的故障来自服务层（KV cache、batching、prefix sharing、speculative decoding、adapters）而非模型行为本身"。**为服务栈正确性 bug 提供系统化发现方法**，直接对齐编目第 6 类 + K§ 系列。 |
| A3 | [arxiv 2510.17098](http://arxiv.org/abs/2510.17098v2) | **Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities (Malicious Token Injection)** | 2025-10-20 | `K§4 cross-request` 新攻击面 | 即使 prompt 和参数安全，KV cache 仍是被忽视的攻击面。提出 MTI 框架系统化扰动 cached key——**KV cache 缓存键污染**的新攻击实例。 |
| A4 | [arxiv 2510.22876](http://arxiv.org/abs/2510.22876v3) | **Batch Speculative Decoding Done Right** | 2025-10-26 | `K§8 control-flow` 重要结论 | **证明所有现有 batch speculative decoding 实现都违反输出等价性**（output equivalence）——即批量推测解码的输出分布与标准自回归生成**不恒等**。这是推测解码正确性的根本性结论，对编目 K§8 有重要补充价值。 |
| A5 | [arxiv 2603.16382](http://arxiv.org/abs/2603.16382v1) | **Rotated Robustness: Training-Free Defense against Bit-Flip Attacks on LLMs** | 2026-03-17 | `[已知类别·bitflip]` 缓解 | 免训练防御：bitflip 在量化权重上的脆弱性源于敏感权重位与特定轴的空间对齐，通过旋转解除该对齐。 |
| A6 | [arxiv 2603.10042](http://arxiv.org/abs/2603.10042v1) | **Targeted Bit-Flip Attacks on LLM-Based Agents** | 2026-03-07 | `[已知类别·bitflip]` 新攻击面 | BFA 扩展到 **LLM agent 多阶段管线 + 外部工具**的新攻击面（此前只针对单步推理如图像分类）。 |
| A7 | [arxiv 2510.00490](http://arxiv.org/abs/2510.00490v1) | **Single-Bit Flip → "Artificial Bad Intelligence" in LLMs** | 2025-10-01 | `[已知类别·bitflip]` | 单 bit 翻转即可触发 LLM "bad intelligence" 的实证，呼应编目第 3 类 BFA 家族。 |

### 弱相关（训练 SDC，推理参考）

| # | 链接 | 标题 | 日期 | 说明 |
|---|---|---|---|---|
| A8 | [arxiv 2604.10390](http://arxiv.org/abs/2604.10390v1) | LLM-PRISM: Characterizing SDC from Permanent GPU Faults in LLM Training | 2026-04-12 | 训练侧 SDC，永久性 GPU 故障刻画。 |
| A9 | [arxiv 2604.00726](http://arxiv.org/abs/2604.00726v1) | Exploring SDC as a Reliability Challenge in LLM Training | 2026-04-01 | 训练侧 SDC 综述。 |
| A10 | [arxiv 2502.12340](http://arxiv.org/abs/2502.12340v1) | Understanding Silent Data Corruption in LLM Training | 2025-02-17 | 真实 SDC 对 LLM 训练影响的首篇研究。 |
| A11 | [arxiv 2605.04563](http://arxiv.org/abs/2605.04563v1) | RangeGuard: Bounded Approximate Error Correction for Reliable DNNs | 2026-05-06 | DRAM 3D 集成后多 bit 错误不再罕见，随机 bit flip 可制造极端 outlier。 |

---

## 🆕 新类别 / 新签名汇总（建议编目收录）

1. **tt-metal #51220/#51540**：硬件 SFPU 数值比较原语因 C++ 重载解析陷阱把 Inf/NaN 位模式当 float 量级比较 → 静默错误值。建议新增"**硬件数学库原语静默错误**"子签名（硬件侧数值正确性）。
2. **triton #11047 / #11065**：内核同步语义（warpgroup MMA wait 跨 warpgroup）和复制元素副作用语义在合法但非显然配置下静默错误 → 静默错误值。建议新增"**编译器/运行时原语静默语义错误**"子签名。
3. **arxiv A4（Batch Speculative Decoding Done Right）**：所有现有 batch speculative decoding 实现违反输出等价性——这是**推测解码正确性的根本性缺陷**而非个案 bug，建议在 K§8 增设"批量推测解码输出等价性违反"条目。

---

## API 调用记录

| # | 查询 | 结果 |
|---|---|---|
| Q1 | vllm `label:bug` created:>=2026-07-22 | 280 命中（取 15） |
| Q2 | vllm `correctness` in title,body | 102 命中（取 15） |
| Q3 | sglang `label:bug` | 7 命中（全取） |
| Q4 | transformers `label:bug` | 12 命中（全取） |
| Q5 | TensorRT-LLM `label:bug` | 5 命中（全取） |
| Q6 | triton `label:bug` | 2 命中（全取） |
| Q7 | 全局 "silent data corruption"/"wrong output"/"bit flip"/"KV cache corruption" in title | 125 命中（取 20，多为非 LLM 仓库噪声） |
| Q8 | 全局 "quantization error"/"wrong result"/"NaN"/"numerical instability"/"non-deterministic" in title | 2150 命中（取 20，多数噪声） |
| Detail | 12 次单 issue GET（vllm/sglang/trt-llm/transformers/triton/tt-metal） | 全成功 |
| arxiv | 7 组关键词（SDC/bitflip/KV/correctness/GPU/specdec/quantfault） | 7/7 成功（HTTPS `-L`） |

> **arxiv 注意**：`export.arxiv.org` 强制 HTTP→HTTPS 重定向（301），必须用 `curl -L`；纯 `http://` 返回空体。

---

*报告生成于 2026-07-29 UTC，由 LLM 推理可靠性每日扫描 cron 作业自动生成。*

# 每日扫描报告 — 2026-08-23

- 扫描窗口：GitHub `created:>2026-08-16`（7 天滚动）；arxiv 最近提交排序
- 目标仓库：vllm-project/vllm · sgl-project/sglang · NVIDIA/TensorRT-LLM · huggingface/transformers · triton-lang/triton · tenstorrent/tt-metal
- 主文档基线：696 条（2026-08-22 合并后）

## 一、GitHub Issues / PR 发现

### 查询执行记录（10 次 Search API + 15 次 issue/PR 详情 API，1 次 504 重试成功，无 403 rate limit）

| # | 查询 | 仓库 | 命中 |
|---|---|---|---|
| q1 | `wrong OR incorrect OR corruption OR NaN in:title` | vllm-project/vllm | 3 |
| q2 | `quantization OR quant wrong in:title` | vllm-project/vllm | 0 |
| q3 | `wrong OR incorrect OR corruption OR NaN in:title` | sgl-project/sglang | 4 |
| q4 | `wrong OR incorrect OR correctness OR NaN in:title` | NVIDIA/TensorRT-LLM | 0 |
| q5 | `wrong output OR wrong result OR RoPE in:title` | huggingface/transformers | 0 |
| q6 | `incorrect OR numerical OR precision OR mismatch in:title` | triton-lang/triton | 1 |
| q7 | `numerical OR pcc OR mismatch OR incorrect in:title` | tenstorrent/tt-metal | 9 |
| q8 | `KV cache OR prefix cache OR speculative in:title` | vllm-project/vllm | 10 |
| q9 | `label:bug` | vllm-project/vllm | 12（窗口内） |
| q10 | `label:bug` | sgl-project/sglang | 2 |

### 已编目 / 已知（跳过）

- SGLang #35434（CPU causal mask）、#35305（Kimi-K3 DP vision grids）、#35274（DSA autotune NaN）、#35455（compressed-tensors KV scale）— 昨日已编目（8.51 / 10.34 / 6.142 / 7.52）
- tt-metal #54078（logsigmoid 修复）、#53720、#53476 — 昨日已编目/引用（1.102 / 1.28 / 1.87 补充来源）
- vLLM #52881（MM keep_on_cpu CI 回归）、#52600/#52588（docs 修正）、Triton #11381（#10928 cherry-pick，07-17 窗外性能类）— 昨日已评估不编目
- SGLang #35285（stale env-var 文档）、vLLM #53423/#53441/#53324/#53330/#53329/#53351/#53360（feature/CI/已编目修复 PR）、#53482 对应关系见下 — 非正确性或已覆盖

### 新发现 — 合并进主文档（8 条）

| # | 条目 | 来源 | 分类 | 判定 |
|---|---|---|---|---|
| N1 | **tt-metal #54048**（Bounty $2500，修复 PR #54098）— `ttnn.remainder` 标量/一元 fp32 路径：商经 host 预计算倒数乘法估计、仅单次 `-1` 过冲校正；\|被除数/除数\| ≳ 2^23 时舍入误差超过单次校正能力，**结果错整整一个除数的整数倍**（`remainder(1.42606e7, 3.0) → 1.0`，正确 2.0）；负误差更被 `v >= s` 符号门控的迭代循环**旁路**（`remainder(5.70425e7, 3.0) → -1.0`）。Wormhole 与 Blackhole 逐字节同算法双平台复现 | [#54048](https://github.com/tenstorrent/tt-metal/issues/54048) · [PR #54098](https://github.com/tenstorrent/tt-metal/pull/54098) | 第 1 类（硬件/SDC）→ **1.103** | `[新类别/新签名]` — **有限校正步数 × 近似商估计的区间外失效**（与 1.87 int32 remainder 同函数不同机制） |
| N2 | **vLLM PR #53456** — `strip_covered_mm_data()` 只把 M-RoPE 列为元数据豁免；XD-RoPE 模型（HunyuanOCR）以相同方式读 `image_grid_thw` 计算位置但 scheduler 只传 `uses_mrope`——prefix-cache 命中后 grid dims 被剥离、`gather_kwargs()` 跳过 feature，`get_xdrope_input_positions` 循环索引越界 `IndexError` → EngineCore fatal → **API server 关闭，一个坏请求终结整个进程** | [PR #53456](https://github.com/vllm-project/vllm/pull/53456) | 第 6 类（服务栈）→ **6.143** | `[新类别/新签名]` — **元数据豁免清单不完备**（枚举式豁免而非按数据依赖判定，新位置编码家族加入即失效） |
| N3 | **vLLM PR #53479**（fixes #52897）— `_mamba_block_aligned_split` 两耦合缺陷：(1) align-mode Mamba 状态只在 chunk 末物化，安静 prefill 只产生一个深位 chunk 末，sibling 共享 4278 token 中 2000 → **0** 复用；精确块对齐 prompt 唯一状态位于 `num_tokens`，超出 `num_tokens−1` 查找上限，**完全相同重复也零命中**；(2) `use_eagle` 回退 `last_cache_position -= block_size` 按 16-token block 设计，align block 为 544–2128 token——整个投测家族放弃一整块可复用前缀（实测损失一半 + ~9× warm prefill） | [PR #53479](https://github.com/vllm-project/vllm/pull/53479) | 第 6 类（服务栈）→ **6.144** | `[新类别/新签名]` — **状态物化粒度 × 查找上限错配**（与 6.126 priority 调度 0 命中同族但缺陷在物化策略本身） |
| N4 | **vLLM PR #53425** — FlashInfer main 上 SM120（GB10）DSV4 decode 以 64-token 页编译（`_DECODE_DSV4_PAGE_BLOCK_SIZE=64`），但 `DeepseekV4SparseMLABackend`/`IndexerBackend` 硬编码声明 `[256]` → `select_common_block_size` 无法切分 `--block-size 256`，**专用 sparse decode 内核被静默跳过**（page-64 内核本身健康：cosine vs torch 0.99966）。修复按 capability family 120 返回 `[64]` | [PR #53425](https://github.com/vllm-project/vllm/pull/53425) | 第 8 类（KV/注意力）→ **8.52** | `[新类别/新签名]` — **能力-页大小矩阵按 arch 失配 → 内核静默降级**（无 fallback 警告） |
| N5 | **vLLM PR #53444**（fixes #53431）— `gemma-4-26b-a4b-it` 实际发出两种 tool-call 开启形式：文档化的 `<|tool_call>call:name{...}` 与 bare 的 `<|tool_call>:name{...}`；parser FSM 只定义 `CALL_PREFIX`（"call:"）转移——bare 形式滞留 `TOOL_PREAMBLE`（无 content_events 出口）直至 `<tool_call|>`，**整个 tool call 静默丢弃**：无提取、无 content、无错误、无恢复路径 | [PR #53444](https://github.com/vllm-project/vllm/pull/53444) | 第 10 类（分词器/模板）→ **10.35** | `[新类别/新签名]` — **输出格式枚举不完备 × FSM 死态**（模型输出比文档多一种形式；状态机无可达终止转移） |
| N6 | **vLLM PR #53433** — `bad_words` 条目经某些分词器（SmolVLM）可编码为**空 token 列表**（非空白控制字符输入）；`update_from_tokenizer` 处理空格前缀形式时索引空列表 → `IndexError` → HTTP 500（AMD CI Schemathesis 捕获）。修复对空必填编码抛 `VLLMValidationError`（结构化 4xx） | [PR #53433](https://github.com/vllm-project/vllm/pull/53433) | 第 10 类（分词器/模板）→ **10.36** | `[新类别/新签名]` — **空 tokenization 边界值未校验**（分词器把不可分词输入映射为空序列，下游假设非空） |
| N7 | **vLLM PR #53449** — `DFlash2Qwen3Model` 声明 `decoder_layer_cls` 覆盖，但共享基类 `DFlashQwen3Model.__init__` **硬编码**基类 DecoderLayer——子类覆盖被遮蔽，DFlash2 draft 被静默建成无卷积的普通 DFlash 层，首个 `attention_conv` 张量加载即 `ValueError`。indirection 由 #52816 引入、被 #52560 再次移除（回归）。修复后 acceptance 7.98/8 | [PR #53449](https://github.com/vllm-project/vllm/pull/53449) | 第 13 类（模型图/架构）→ **13.32** | `[新类别/新签名]` — **硬编码构造 × 子类覆盖遮蔽**（OOP indirection 静默丢失，错误消息与根因相距整个继承链） |
| N8 | **vLLM #53430**（修复 PR #53440 / #53442 同日竞争）— MoE `TopKWeightAndReduceNaiveBatched` 用 `output.index_add_(0, t_idx, gathered)` 散射回源 token；top_k>1 时同行多专家写入，CUDA `index_add_` **原子浮点加法顺序跨 run 非确定**，BF16 低位差异在近 tie logits 处**足以翻转 greedy argmax**（破坏 temperature=0 确定性）。修复：scatter 到唯一 `[token, topk_slot]` 位置 + 固定 top-k 顺序 FP32 累加的 `moe_sum` | [#53430](https://github.com/vllm-project/vllm/issues/53430) · [PR #53440](https://github.com/vllm-project/vllm/pull/53440) · [PR #53442](https://github.com/vllm-project/vllm/pull/53442) | 第 4 类（数值/精度）→ **4.11** | `[已知类别]` — `var:fp_reduction_order_nondeterminism`（4.1 噪声底家族的 MoE 归约新实例） |

### 记录但不编目（非正确性域 / 纯性能 / fail-loud 无新签名）

- vLLM PR #53458（EAGLE3 full-vocab draft 走慢路径 vocab remap）— 纯性能（5× 慢），输出无影响
- vLLM PR #53474（KV offload 多节点 TP 行按 global world_size 计宽，本地 /dev/shm 一半闲置）— 容量/性能，非正确性
- vLLM PR #53460（Dots3 NOTE Omni KV 布局强制 BLHNC + 编码器优化）— 新模型接入修复，无独立故障报告，机制（混合 DSA/MLA/SWA 页大小表示）不足以立条
- vLLM PR #53461（LoRA resolver KeyError → 500 + resolver 循环中止）— fail-loud 错误处理瑕疵，无静默损坏
- vLLM PR #53478（yaml 下划线 key 的 `false` 布尔被静默丢弃，prefix caching A/B 测反）— **记录为候选**：配置语义静默丢弃值得关注，但属配置解析 UX 而非推理正确性；与 6.x config 处理族机制重叠度低，暂不立条
- vLLM PR #53482（Gemma4 MoE LoRA `get_expert_mapping` 缺失，引擎启动 AttributeError）— fail-loud 功能缺失，非静默
- tt-metal PR #54126（rotary_embedding_llama decode seqlen-mismatch 日志降级为 debug）— 日志噪声治理，非正确性（该 warning 是 decode 常态而非异常）
- tt-metal PR #54091/#54025/#54015（测试基建：off-by-one DRAM shard grid / pytest-xdist collection）— 测试自身缺陷
- tt-metal PR #53491（KV-PCC leg budget 移交 shield team）— 流程性

## 二、arXiv 论文发现（5 组查询）

| 查询 | 结果 |
|---|---|
| `silent data corruption LLM` | 4 篇全部已编目（SCOUT 2608.11034、LLM-PRISM 2604.10390、2604.00726、2502.12340） |
| `bit flip LLM inference` | 5 篇均已编目或引用（Bit-Position Sensitivity 2607.19623、RangeGuard、Rotated Robustness、Targeted BFA、BitFlipScope） |
| `KV cache corruption` | 全部已编目/引用（2608.07408=14.7、2608.05863、2606.28958、Moment-KV 方法论文、2606.07571=8.41） |
| `LLM inference correctness` | 无相关（金融校准/rEDMRec/自反思/协调规划等 agentic 主题） |
| `GPU soft error inference` | 2601.19912 已编目（1.98） |

**结论：arxiv 今日无新编目论文。**

## 三、API 调用记录

| 时间 (UTC) | 调用 | 状态 |
|---|---|---|
| 23:0x | GitHub Search ×6（q1–q6，间隔 ≥8s） | 全部成功，无 403 |
| 23:1x | GitHub Search ×4（q7–q10，间隔 8s） | 全部成功 |
| 23:1x | GitHub PR 详情 ×6（vLLM #53458/#53456/#53449/#53442/#53440/#53433） | 成功 |
| 23:1x | GitHub PR 详情 ×8（vLLM #53425/#53479/#53478/#53461/#53482/#53474/#53460 + tt-metal #54098） | 成功（一批 jq 引号错误重跑） |
| 23:1x | GitHub issue 详情 ×2（vLLM #53430 一次 504 后重试成功、tt-metal #54048）+ PR #53444 | 成功 |
| 23:1x–23:2x | arxiv query ×5（`curl -sL -G --data-urlencode`） | 全部成功 |

## 四、合并摘要

- **新增 8 条**（全部新子签名，其中 N8 标 `var:fp_reduction_order_nondeterminism`）：1.103、4.11、6.143、6.144、8.52、10.35、10.36、13.32
- 主文档总数 696 → **704**；第 1 层 565 → 573（每日扫描合并累计 473 → 481）
- 类别分布：第 1 类 102→103、第 4 类 10→11、第 6 类 142→144、第 8 类 51→52、第 10 类 34→36、第 13 类 31→32
- 轴分布更新：infra_off 547→554、ambiguous 73→74；config 148→151、partial 149→151、hw_specific 126→128、yes 94→95；documented 515→523；NEW 591→598、var 102→103

# 每日扫描报告 — 2026-09-07

- 扫描日期：2026-09-07（UTC）
- 扫描窗口：GitHub `created:>2026-09-05`（覆盖 09-06 扫描时窗之后的新条目）；arxiv 最近提交排序
- 目标仓库：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal
- 主文档基线：976 条（2026-09-06 合并后）

## 结果总览

- 共执行 10 组 GitHub Search API 查询（1 组因 >5 个 AND/OR 算子被 Validation Failed 拒绝，简化后重发成功；无限流 403）+ 5 组 arxiv API 查询（HTTPS，`curl -sL`）；条目核实另用 core REST 11 次（`--http1.1`，全部成功）
- 去重核实后：**13 条新条目合并进主文档**（2 个新子签名 `[新类别/新签名]` + 11 个已知类别变体 `[已知类别]`），976 → **989**
- 修复 PR #38355 作为新条目 11.69 的直接来源（其修复的 #38319 同步引用）
- 当日特征：vLLM 侧 Qwen3.5/3.8 hybrid GDN 前缀缓存命中 NaN（#55766）、NVFP4 三连（MoE TP2 静默腐蚀 / NVFP4 KV 95% 无效 / DFlash sq≥80）为当日重点新发现

## 新合并条目（13 条）

### 第 1 类 · 硬件/SDC（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 1.181 | CPU MoE kernel produces NaN logits under torch.compile for Qwen3_5MoeForConditionalGeneration（quant 无关，--enforce-eager 修复） | vLLM #55560 | `[已知类别]` var:1.12 | CPU 后端 + `Qwen3_5MoeForConditionalGeneration`：torch.compile/inductor 路径全词表 NaN logits（重排 garbage token），bf16 未量化与 GPTQ-Int4 均复现（报告者当日勘误：非 GPTQ 特定）；eager 干净——量化方法无关的 CPU MoE×inductor 编译缺陷 |

### 第 4 类 · 数值/精度/图编译（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 4.19 | `fuse_rope_kvcache`/`fuse_qk_norm_rope_kvcache` 在 MLA-only 模型静默注册零 pattern 但副作用照付 | vLLM #55720 | `[已知类别]` var:6.207 | 两 fusion pass 用 `get_layers_from_vllm_config(config, Attention)` 收集层，`Attention` 与 `MLAAttention` 是兄弟类——MLA-only 模型（GLM-5.3 `glm_moe_dsa`）dict 恒空、零 pattern 注册；但仍无条件多付两个编译端点、强推 `+rotary_embedding` custom op 覆盖 Inductor 自身 rope lowering——吞吐损失 + rope 数值路径被换，日志零提示 |

### 第 5 类 · 分布式/多GPU（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 5.35 | GLM-5.3-Flash 2 节点 16×H100 权重加载 `KeyError: layers.11.shared_transformer.self_attn.qkv_proj.weight`（单节点正常） | vLLM #55725 | `[新类别/新签名]` | shared_transformer 层的 `qkv_proj.weight` 键在 checkpoint 映射表中缺失：单节点 TP8 可跑、跨节点 TP16 在权重加载期 KeyError crash——切分/映射逻辑未覆盖 shared（非 per-TP-rank 复制）模块在更高切分系数下的键空间 |

### 第 6 类 · 服务栈/缓存正确性（+3）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 6.212 | Qwen3.5/3.8 hybrid GDN：前一个 prefill 结束在块边界后 4-10 token 时，后续请求前缀缓存命中即 NaN logits（mamba cache mode align） | vLLM #55766 | `[新类别/新签名]` | align 模式 attention block=816 对齐 mamba 页；prefill 恰在 N*816+r（r∈{4,6,8}必现，{4..10}偶数，r≤3/r≥12/奇数 r 干净）结束 → 块边界 checkpoint 损坏 → 后续请求恢复该状态后首步起 NaN（采样 id 0"!"×32768）；坏块被再缓存 → 同请求重试确定性失败；per-request `cache_salt` 或前部 padding 即愈——hybrid mamba 块边界状态 checkpoint 正确性新签名 |
| 6.213 | Encoder-decoder encoder-cache budget（num_free_slots）每个 decode 步膨胀 +E，admission control 失真 | vLLM #55733 | `[已知类别]` var:6.136 | `Scheduler._free_encoder_inputs` 每步释放全部 cached inputs，`EncoderDecoderCacheManager.get_cached_input_ids` 恒返回全部 id、`free_encoder_input` 无引用计数守卫地 `num_free_slots += E`——每 decode 步重复计回 E 个槽（纯记账单测：100 槽预算终值 300）；`can_allocate` 按虚增空槽放行 → 过量并发 encoder-decoder prefill → 运行期 encoder cache 真 OOM/crash |
| 6.214 | GLM-5.3-Flash PD-disaggregated 部署频繁输出离题/重复（"好的好的好的…"），非分离部署同样出现 | vLLM #55689 | `[已知类别]` var:6.163 | 8×H100 PD 分离 + kv-connector 部署 GLM-5.3-Flash：同 prompt 有时输出完全无关内容、有时同短语无限重复；非分离单节点也复现（较轻）——跨请求上下文污染/状态损坏家族在 GLM-5.3-Flash 上的低信息报告（无根因定位，documented 低粒度） |

### 第 7 类 · 量化（+3）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 7.78 | NVFP4 gated-MoE end-padding 在 TP2/B200 静默腐蚀 Gemma-4-26B（GSM8K flexible acc 2.0%，100% invalid） | vLLM #55649 | `[已知类别]` var:8.73 | per-expert 中间宽 704 / TP2 = 352 行，FlashInfer shuffle 要求 fused gate+up 行数被 128 整除 → end-padding；gated-MoE 中 pad 行与 gate/up 两投影对齐错位（PR #48624 的 split-pad 修复未覆盖此路径）→ 全 1,319 例 GSM8K 无引擎错误完成但 100% strict-invalid、91.8% 打满 1024 token 上限输出数字腐蚀 gibberish；TP1 不需 padding 正确 |
| 7.79 | FlashInfer TensorRT-LLM NVFP4 KV cache 在 Blackwell 上产出无效 Qwen3.5-397B 输出（93-95% invalid） | vLLM #55673 | `[已知类别]` var:8.53 | B200 + FlashInfer TRT-LLM attention backend + `--kv-cache-dtype nvfp4`：GSM8K flexible exact match 96.7%（FP8 KV）→ 4.5-6.5%（NVFP4 KV），93-95% 响应无可解析答案（空/立即终止），并发 64 与 512 均坏；引擎无错无警告——NVFP4 KV cache 静默精度塌方 |
| 7.80 | incoai glm-5.3-nvfp4+ 5.3 dflash sq>=80 时 unknown exit code | vLLM #55625 | `[已知类别]` var:7.69 | GLM-5.3-NVFP4+（社区量化变体）+ DFlash：序列长 ≥80 即 worker 以 unknown code 退出（正文仅启动命令+日志附件）；NVFP4 量化变体与 GDN/DFlash 路径的组合失败 |

### 第 8 类 · KV缓存/注意力（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 8.77 | GLM-5.3-Flash：FlashInfer SM90 sparse MLA backend 在 TP+EP + MTP decoding 下 CUDA IMA（H20 ×8） | vLLM #55626 | `[已知类别]` var:8.64 | `flashinfer_mla_sparse_sm90.py` 构建 attention metadata 时 `positions[:num_rows].cpu()` 同步点浮出 IMA（实际 faulting kernel 在更早异步流）；负载 65536 in/out × batch 32 中途崩；报告者自查排除 #55221（dtype 拒绝）/ #55280（ROCm）/ #53142（align precopy）三近邻——SM90 sparse MLA × TP+EP × MTP 组合的 IMA 新实例 |

### 第 10 类 · 分词器/模板（+2）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 10.57 | legacy qwen3_xml streaming parser 在首个 tool call 前发出 whitespace-only content | vLLM #55633 | `[已知类别]` var:10.42 | thinking 块后跟 whitespace-only 分隔符再接首个 XML tool call：parser 把分隔符作为普通 content delta 发布（`"\n\n"` Markdown/SSE 增量）——模型语义是 thinking→tool 过渡，客户端却收到 content 事件；parser 边界/流式状态机假设（通道终止符族） |
| 10.58 | Routed-expert metadata 请求 `routed_experts_prompt_start == prompt_len` 边界触发 EngineCore crash 并殃及后续请求 | vLLM #55750 | `[已知类别]` var:6.192 | `--enable-return-routed-experts` 下 offset 13（len 14）成功、offset 14（恰等于 rendered prompt 长度）→ scheduler `assert prompt_start < request.num_prompt_tokens` AssertionError → EngineCore fatal → 触发请求与**后续普通请求**均 HTTP 500；请求校验放行了调度器断言不可达的边界值 |

### 第 11 类 · 解码/采样/KV 控制流（+1）

| 编号 | 标题 | 来源 | 分类 | 简述 |
|---|---|---|---|---|
| 11.69 | chunked-prefill radix-insert 竞态腐蚀 QSA KV pages（#38319；修复 PR #38355） | SGLang #38319 + PR #38355 | `[已知类别]` var:6.163 | `MambaRadixCache.cache_unfinished_req()` 在 prefill 中途把每 chunk 的 KV 页插入 radix 树；请求被 retract/abort 时 `release_kv_cache()` 把这些页释放回分配器，radix 节点带悬垂引用存活 → 前缀命中者首个 decode 步读到垃圾（impossible token 248319 = lm_head 未训练 padding 行循环，`num_correct_drafts=0` 证明 target 自身产出）；QSA 压缩寻址 `full_slot // compress_ratio` 把单页释放放大为跨请求压缩槽污染；flush_cache 暂愈 |

## 补充来源引用（不占新编号）

| 条目 | 补充 |
|---|---|
| 7.78 | flashinfer #3206 / vLLM #39595（同一对齐问题的 fail-loud 形态）、修复 PR #48624（split-pad，未覆盖 gated-MoE TP2 end-padding 路径） |
| 8.77 | #55280（GLM-5.3-Flash hybrid block_size=1152 → 32-entry paged-MQA GPU fault，ROCm/gfx942）——报告者列为可能同族的 hybrid KV-geometry 候选 |

## 跳过条目（核实后不编目）

- vLLM #55613（AWQ/GPTQ WNA16 AVX2）——已于 2026-09-06 编目为 7.74
- vLLM #55633 中引用的 #49412（qwen3_xml 相关旧 issue）——上游引用，非当日新条目
- SGLang PR #38219/#38212/#38161——已于 2026-09-06 编目（6.207/6.193 补充来源）
- tt-metal #55587/#55585/#55582 + PR #55591——已于 2026-09-06 编目（1.177/1.178/1.179）
- triton #11604——已于 2026-09-06 编目（1.175）

## arxiv 结果（无新编目论文）

- `silent data corruption + inference`：Intel NCS2 电磁故障注入（2605.22437，2026-05，边缘 NPU 硬件域非 LLM 推理栈）、PVF（2405.01741，2024，AI SDC 漏洞评估框架，方法论文）、Low-cost Strategic Monitoring（2310.20349，2023，DNN 错误检测方法论文）——均不收录
- `bit flip LLM`：JITterFlip（已编目 3.18）、Groundhog（已编目 3.16）、Decision-Level Hijacking（已编目）、Unequal Error Protection（2607.19623，DNN 推理内存防护方法论文）
- `KV cache corruption`：Addressable Memory（已编目 14.7）、Runtime Observability（已编目 6.188 相关引用）、Nexus/Moment-KV（方法论文）、When Latent Agents Lie（已编目）
- `LLM inference correctness`：Compression-Aware Abstention（已编目 15.9）、Hollow-LLM Attack（已作为 4.x 远程推理完整性补充引用）、其余为测试/方法论文
- `GPU soft error`：Instruction-Level FI（已编目 1.98）

## API 调用记录

| # | 查询 | 结果数 | 状态 |
|---|---|---|---|
| Q1 | vllm issue `corruption/wrong output/NaN/silent created:>2026-09-05` in:title | 5 | ✅ |
| Q2 | sglang issue `corruption/wrong/incorrect/NaN/silent created:>2026-09-05` in:title | 1 | ✅ |
| Q3 | vllm `label:bug created:>2026-09-06` | 8 | ✅ |
| Q4 | transformers issue `wrong/incorrect/corruption/NaN created:>2026-09-05` in:title | 0 | ✅ |
| Q5 | TensorRT-LLM issue `correctness/wrong/incorrect/NaN created:>2026-09-05` in:title | 0 | ✅ |
| Q6 | triton issue `wrong/incorrect/miscompile/silent created:>2026-09-05` in:title | 1 | ✅ |
| Q7 | tt-metal issue `silent/incorrect/NaN/PCC/mismatch created:>2026-09-05` in:title | 3 | ✅ |
| Q8 | sglang PR `fix (silent/incorrect/corrupt) created:>2026-09-05` in:title | 首次 Validation Failed（>5 AND/OR） | ⚠ 简化后 ✅（4） |
| Q9 | tt-metal PR `silent/incorrect/PCC created:>2026-09-05` in:title | 1 | ✅ |
| Q10（并入 Q8 重试） | — | — | — |
| A1–A5 | arxiv 5 组关键词（HTTPS `-sL`） | — | ✅ 无新编目论文 |
| R1–R11 | core REST 条目详情（#55766/#55720/#55649/#55689/#55750/#55733/#55725/#55673/#55633/#55626/#55625/#55560/#38355，`--http1.1`） | — | ✅ 全部 200 |

> 注：Q8 首次因 6 个 OR 算子超过 GitHub Search 上限被 422 拒绝，简化为 3 关键词后成功；当日无限流 403。transformers 与 TensorRT-LLM 窗口命中 0。

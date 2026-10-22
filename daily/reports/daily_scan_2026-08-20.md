# 每日扫描报告 — 2026-08-20

**扫描窗口**：GitHub issue `created:>2026-08-18` · arxiv 最近提交
**目标仓库**：vllm-project/vllm · sgl-project/sglang · huggingface/transformers · NVIDIA/TensorRT-LLM · triton-lang/triton · tenstorrent/tt-metal

## 扫描结果总览

共执行 8 次 GitHub Search API 查询 + 5 次 arxiv API 查询 + 17 次 issue 详情抓取。发现 24 条相关内容，其中 **16 条确认为新案例**（去重后，issue 编号均未出现于主文档），已合并进主参考文档（656 → 672）；8 条为性能/非正确性/CI 噪声/低相关条目未编目。

---

## 一、新发现案例（已合并进主文档，全部为 [新类别/新签名] 子签名）

### A. vLLM 服务栈 / 缓存正确性（第 6 类，+6）

1. **[新签名] Mooncake 为 NULL recurrent-state 边界发布外部缓存键 → 静默输出分歧** — vLLM [#53084](https://github.com/vllm-project/vllm/issues/53084)（2026-08-19，bug）
   - `mamba_cache_mode=align` 下 vLLM 有意让部分状态边界不物化（`NULL_BLOCK_ID=0`），Mooncake Store 保存路径把 null block 解析为地址并照常发布键 → 消费端"成功"加载全零状态、跳过重算 → 与冷基线静默分歧。**部分情形 token 不变而 logprob 显著漂移**——输出级监控漏检。传输层字节完整性校验全部通过（错误在键/元数据层，非数据层）。
   - → 编目为 **6.127** `vllm_mooncake_null_block_kv_key_publication_silent_divergence`

2. **[新签名] 3-token prompt 被误派发进 spec-decode FULL cudagraph → GDN 状态全丢、确定性垃圾** — vLLM [#53051](https://github.com/vllm-project/vllm/issues/53051)（2026-08-19，rocm）
   - `_is_uniform_decode` 是纯形状检查；MTP k=2 时任意 3-token prompt 的 prefill 恰满足 `1+k` 形状判据 → 误入 spec-decode FULL graph，hybrid GDN 的 recurrent-state 写被全部跳过 → 后续 decode 从零状态计算。逐块校验和定位根因。无 crash。
   - → 编目为 **6.128** `vllm_prefill_shape_misdispatch_spec_cudagraph_gdn_state_loss`

3. **[新签名] empty-splitting-ops 防护被早退旁路 → piecewise cudagraph attention 进图，spec decode 静默全拒 draft** — vLLM [#53030](https://github.com/vllm-project/vllm/issues/53030)（2026-08-19，rocm）
   - `set_splitting_ops_for_v1` 在非 VLLM_COMPILE 模式早退，跳过"splitting_ops 空 ⇒ cudagraph_mode=NONE"防护；attention 被捕获进 piecewise graph 且以 per-request query length 为键 → spec decode 形状不匹配时静默错值/全拒 draft。**防护逻辑本身正确，只是控制流不可达**。stock vLLM 经 `VLLM_USE_BREAKABLE_CUDAGRAPH` 可达。
   - → 编目为 **6.129** `vllm_empty_splitting_ops_guard_skip_piecewise_cudagraph_spec_reject`

4. **[新签名] CLIP pooling 持续多模态负载下文本请求全部退化为空串 embedding（200 正常返回）** — vLLM [#53091](https://github.com/vllm-project/vllm/issues/53091)（2026-08-19，multi-modality）
   - 混合负载下 completion 式文本请求输出全部等于空串 embedding（cos=1.0000，健康基线 0.4651）；图像 embedding 与 /tokenize 始终正确；fresh pod 一接生产流量即坏（排除状态累积）。文本检索系统静默全灭。**输出恒定（非随机）**，一致性监控也漏检。
   - → 编目为 **6.130** `vllm_clip_pooling_mm_load_empty_prompt_embedding_collapse`

5. **[新签名] hybrid mamba align precopy 用错 block size 播种 state_idx → 第二请求 illegal memory access** — vLLM [#53142](https://github.com/vllm-project/vllm/issues/53142)（2026-08-19，quantization）
   - `add_request` 以 attention 的 `cache_config.block_size`（CLI 128）而非 mamba 组 block size（2176）计算 state_idx → prefix-cache resume 读 `block_table[req, 781]`（仅 61 列）→ CUDA illegal memory access crash。fail-fast 型。
   - → 编目为 **6.131** `vllm_mamba_align_precopy_state_idx_wrong_block_size_ima`

6. **[新签名] Kimi-K3 chunked prefill vs one-shot hidden states 在首个 PP stage 前即分歧** — vLLM [#52909](https://github.com/vllm-project/vllm/issues/52909)（2026-08-19 closed，kimi/k3）
   - 同一 32K 输入 one-shot（32768）vs 4×8192 chunked：首个 8K chunk 逐位一致、分歧自第二 chunk 引入，超出 BF16/kernel 容差。同一请求不同 `--max-num-batched-tokens` 得到不同隐状态。issue closed（机制定论未在 issue 内留档）。
   - → 编目为 **6.132** `vllm_kimi_k3_chunked_prefill_hidden_state_divergence`

### B. 量化（第 7 类，+2）

7. **[新签名] int4 per-token-head KV cache 写入内核往返错误（确定性 54.7% 错配 + flaky 翻转）** — vLLM [#53086](https://github.com/vllm-project/vllm/issues/53086)（2026-08-19，quantization）
   - `triton_reshape_and_cache_flash_per_token_head_quant` int4 分支 dequant 往返 vs bf16 参考：35/64 元素错配、abs diff 3.05（容差 0.5）、rel diff 3651；`[int4-0-16-128-1-42]` 与 `[int4-0-16-64-4-7]` **同代码逐运行 pass/fail 翻转**（提示位操作未定义/竞态成分）。int8/fp8 全通过。pytest 确定性复现。
   - → 编目为 **7.48** `vllm_int4_per_token_head_kv_quant_kernel_roundtrip_wrong`

8. **[新签名] draft 模型量化配置 packed_modules_mapping 恒空 → fused 投影不量化构建 → 加载 crash** — vLLM [#53116](https://github.com/vllm-project/vllm/issues/53116)（2026-08-19，bug+spec+quant）
   - `get_draft_quant_config()` 不流经 `configure_quant_config()` → mapping 空 → `gate_up_proj` 无法映射 → 不量化构建 → 加载期 `AttributeError: 'MergedColumnParallelLinear' object has no attribute 'data'`。非 fused 层量化正常 → **崩溃表面与量化无关，归因陷阱**。无 GPU 即可复现。
   - → 编目为 **7.49** `vllm_draft_model_quant_config_packed_mapping_missing`

### C. 解码 / 采样（第 11 类，+1）

9. **[新签名] spec-decode verify 行全 NaN logits 被 rejection sampler"洗"成恰好 vocab_size 的越界 token id** — vLLM [#53029](https://github.com/vllm-project/vllm/issues/53029)（2026-08-19）
   - NaN 行经 one-hot residual + gumbel（仍 NaN）→ NaN 污染 argmax 链产出 `token_id == vocab_size`，传播进 input_ids，1–3 步后以 `srcIndex < srcSelectDimSize` device assert 异步引爆——**崩溃点远离故障根源**。引擎唯一 NaN 检测器对 verify 路径全盲（NaN-metric 盲区）。#42261（gemma4+MTP 间歇同 assert）疑同源。附带一族 uninit-scratch 审计发现。post-forward 结构性盲点的极端形态。
   - → 编目为 **11.50** `vllm_rejection_sampler_nan_row_laundered_oov_token_id`

### D. 硬件 / SDC（第 1 类，+4）

10. **[新签名] Triton AMD gfx1151 运行时 trip-count `scf.for` 循环 miscompile** — triton [#11378](https://github.com/triton-lang/triton/issues/11378)（2026-08-19）
    - TTGIR 语义正确但运行时错误：num_warps=1 确定性错值（~2/3 packed 输出错）；2/4 先错后 HIP illegal-address；8 首 launch 即 fault。`tl.static_range` 变体逐位一致（排除驱动/固件）。跨后端共享 IR 层缺陷的迁移风险参照。
    - → 编目为 **1.95** `triton_amd_gfx1151_runtime_trip_count_loop_miscompile`

11. **[新签名] tt-metal multicore argmax 工作划分三缺陷（越界读/挂死/错值）** — tt-metal [#53769](https://github.com/tenstorrent/tt-metal/issues/53769)（2026-08-19，bug）
    - cores>blocks 时末核 read size 下溢 → 越界 NOC read；≥3 个 core-range 时全落入 cores0 而 multicast 只取前两 range → 第 3+ range 挂死；单核路径亦挂死。argmax 是采样/top-k 底层原语，错值时下游静默选错 token。
    - → 编目为 **1.96** `ttmetal_argmax_multicore_range_partition_gaps`

12. **[新签名] LayerAck 在 enqueue 时触发而非 device 写完成时——ack 驱动 KV 流迁移的时间炸弹** — tt-metal [#53815](https://github.com/tenstorrent/tt-metal/issues/53815)（2026-08-19）
    - GPT-OSS per-layer ack 无 device 同步（"层已提交"≠"KV 已可读"）；当前序列化屏障掩盖，激活 overlapped driver 后 decode 读部分写入的 cache、**无任何报错**。selftest 下迁移 PCC 校验失败。DeepSeek 版已用同 CQ D2H ack 修复。
    - → 编目为 **1.97** `ttmetal_layerack_enqueue_time_semantics_kv_stream_race`

13. **[研究编目] LLM 推理 GPU 软错误指令级故障注入研究（首个 LLM 指令级 FI）** — arXiv [2601.19912](https://arxiv.org/abs/2601.19912)（2025-12-25）
    - 架构/参数规模/任务复杂度三轴刻画 LLM 软错误韧性；为 SDC 注入位图与容错设计提供依据。与 LLM-PRISM（RTL 级）构成"RTL → 指令 → token"三级软错误刻画谱系。
    - → 编目为 **1.98** `llm_gpu_soft_error_instruction_level_fi_study`

### E. 比特翻转攻击（第 3 类，+1）

14. **[新签名] .gguf 单比特翻转定向语义攻击（BitSifter 脆弱位定位）** — arXiv [2510.00490](https://arxiv.org/abs/2510.00490)（2025-10-01）
    - DeepSeek/Qwen 等 .gguf 量化权重中存在单比特脆弱位：翻转 1 bit 即可诱发 Artificial Flawed/Weak/Bad Intelligence（事实错误/推理退化/有害内容）；信息论敏感熵 + BitSifter 概率扫描高效定位。**无 crash、无 NaN——语义层攻击，比数值破坏型 BFA 更隐蔽**。
    - → 编目为 **3.15** `gguf_single_bit_semantic_failure_bitsifter`

### F. 分布式（第 5 类，+1）

15. **[新签名] shm_broadcast 启动竞态杀死 rank 0，`/v1/models` 仍 200——readiness 盲区实证** — vLLM [#53157](https://github.com/vllm-project/vllm/issues/53157)（2026-08-19）
    - 8×GB10 TP8 启动期 shm_broadcast 块饥饿 ~60s → rank 0 死、EngineDeadError（5/7 复现）；**3 次受控试验中 2 次 HTTP API 照常应答 models 端点 200 而生成已死**。附录 C.6 服务栈活性盲区的直接实证：只有真实 completion 请求能暴露。
    - → 编目为 **5.21** `vllm_shm_broadcast_startup_wedge_models_endpoint_healthy`

### G. 模型图 / 架构（第 13 类，+1）

16. **[新签名] NemotronParse lm_head↔embed_tokens 权重绑定缺失 → 加载"成功"但全垃圾输出** — vLLM [#53019](https://github.com/vllm-project/vllm/issues/53019)（2026-08-19）
    - vLLM 两个版本复现 garbage 输出（HF 正常）；缺陷在架构实现的 tie_weights 语义。无 crash——模型加载与 serving 全部"正常"。跨栈行为分歧典型实例。
    - → 编目为 **13.31** `vllm_nemotron_parse_lm_head_weight_tying_missing`

---

## 二、已编目/未编目的其余条目

- vLLM #53157（见上，已编目）
- vLLM #53041（tiered SWA/Mamba checkpointing 讨论）— 设计讨论，非缺陷，不编目
- vLLM #53042（KV offload 1.92× host 内存超配）— 资源泄漏/性能域，无正确性影响，不编目
- vLLM #53031（DFlash capture 日志误导）— 可观测性缺陷，轻量，不编目
- vLLM #53105（bare assert 无诊断）— DX 改进，不编目
- vLLM #53137（recipe_json_to_vllm_config -cc 解析错）— 离线工具配置解析 bug，产出错误 config.yml（可间接致错误部署），但非推理运行时正确性，不单独编目
- TRT-LLM #18035（RoPE SMEM 去重 perf PR）、#17977（类型注解）— 非正确性
- sglang #35420（SM80 fallback 特性请求）— 非缺陷
- transformers Q5：0 结果
- tt-metal #53782/#53770/#53765/#53760（性能/重构类 community 条目）、#53748/#53731/#53730/#53709 等（CI triage 噪声）— 不编目

## 三、重点新发现（监控视角）

1. **#53029 NaN 洗钱进 token id**：NaN 前向故障不再是"输出 NaN/空"——spec-decode 采样器可把 NaN 行转成**合法形态的越界 int**，监控窗口仅剩一步且崩溃点远离根源。对本项目 post-forward 结构性盲点清单是重要补强。
2. **#53157 僵尸服务形态**：API 活、生成死。活性故障绕过全部输出正确性监控——附录 C.6 的又一实证，端到端生成探针是唯一可见手段。
3. **#53084 元数据层缓存污染**：分布式 KV 卸载的传输字节校验全通过，错误在键发布层——"数据完整性校验"类监控对这类故障原理性无效。

## 四、API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH search | vllm correctness/wrong output (>08-18) | 11 hits |
| 2 | GH search | vllm NaN/garbage/corrupted | 6 hits |
| 3 | GH search | vllm kv cache/prefix cache/quant wrong | 16 hits |
| 4 | GH search | sglang correctness/wrong/NaN | 1 hit |
| 5 | GH search | transformers wrong output/numerically | 0 hits |
| 6 | GH search | TensorRT-LLM correctness/mismatch | 2 hits |
| 7 | GH search | triton wrong result/miscompile/NaN | 1 hit |
| 8 | GH search | tt-metal numerics/correctness/pcc | 24 hits |
| 9-13 | arxiv | SDC+LLM / bit flip weights / KV cache corruption / inference correctness / GPU soft error | 各 1-5 条 |
| 14-30 | GH detail | issue 详情抓取（#52909/#53019/#53029/#53030/#53051/#53084/#53086/#53091/#53116/#53137/#53142/#53157/#11378/#53769/#53815 等） | OK |

无 rate limit 错误（全部查询间隔 ≥8s / 2s）。

## 五、合并结果

- **主文档 656 → 672 条**（+16：1.95–1.98、3.15、5.21、6.127–6.132、7.48–7.49、11.50、13.31）
- 滚动汇总 / 目录 / 三层来源表 / 如何使用 全部同步更新；导出脚本 Parsed 672 entries ✓

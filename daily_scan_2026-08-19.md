# 每日扫描报告 — 2026-08-19

**扫描窗口**：GitHub issue `created:>2026-08-12`（部分 `>2026-08-15`）· arxiv 最近提交
**目标仓库**：vllm-project/vllm · sgl-project/sglang · NVIDIA/TensorRT-LLM · triton-lang/triton · tenstorrent/tt-metal · huggingface/transformers

## 扫描结果总览

共执行 10 次 GitHub Search API 查询 + 6 次 arxiv API 查询 + 12 次 issue 详情抓取。发现 17 条相关内容，其中 **12 条确认为新案例**（去重后），已合并进主参考文档；5 条为已编目 issue 的重复/补充。

---

## 一、新发现案例（已合并进主文档）

### A. [新类别/新签名] 类

1. **[新签名] XPU/多后端 qnorm+RoPE 融合内核单程序内双 store 竞态** — vLLM [#52415](https://github.com/vllm-project/vllm/issues/52415)（2026-08-15）
   - 同一 Triton 程序内先整体 store 未旋转 KV、再以 stride-2 地址覆写旋转区，无顺序保证；调度器差异决定哪个 store 落地 → **非确定性 KV cache RoPE 区损坏**（MTT S5000/MUSA 上 4096 token 中 30 个损坏，两次运行 30→29）。CUDA/NVIDIA 调度通常掩盖该竞态（found because MUSA schedules differently）。一行 mask 修复（`mask=offs < NOPE_DIM`）。
   - 归类：**KV缓存/注意力（第 8 类）** → `vllm_xpu_qnorm_rope_unordered_twin_store_race`

2. **[新签名] 动态 NTK 缩放在 served_len < trained_len 时负底数复指数 → NaN embeddings** — vLLM [#52665](https://github.com/vllm-project/vllm/issues/52665)（2026-08-17）
   - `inner = factor*served/trained - (factor-1)` 在 served=512、trained=2048 时为 **-0.5**，`base*(-0.5)^(dim/(dim-2))` 产生复数 effective_base ≈ 1.86e38 复 RoPE cache，cast 到实 dtype 后 NaN embeddings，静默返回。NomicBert embedding 模型（nomic-embed-text-v1）。
   - 归类：**位置编码/RoPE（第 9 类）** → `vllm_dynamic_ntk_negative_base_rope_nan`

3. **[新签名] 长序列 punica LoRA 内核 NaN（>400 token 全部输出 "!" = token 0）** — vLLM [#52568](https://github.com/vllm-project/vllm/issues/52568)（2026-08-17）
   - Qwen3.5-9B hybrid-GDN + dynamic LoRA（H20）：短序列正常、>~400 token 时 logits 全 NaN → argmax(NaN)=token 0（"!"）。base 模型/merged 模型/HF PEFT 均正常 → 缺陷锁定在 vLLM punica Triton LoRA kernel 长序列路径。静默错误回答（HTTP 200）。
   - 归类：**权重/适配器（第 12 类）** → `vllm_punica_lora_long_seq_nan_token0`

4. **[新签名] AITER sparse PA prototype × 推测解码 → 输出损坏、acceptance 7%** — vLLM [#52860](https://github.com/vllm-project/vllm/issues/52860)（2026-08-15）
   - MiniMax-M3 FP8 on MI325X：`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` 自动选 AITER sparse paged-attention prototype 路径，叠加 DSpark 推测解码后输出从正确退化为重复/mojibake（greedy "1,1,1,1..."）。关 shuffle（走 Triton sparse 路径）即恢复正确且 acceptance 7.3%→44%。**短补全常以正确答案开头，smoke test 会漏检**。无 crash、无日志。
   - 归类：**KV缓存/注意力（第 8 类）** → `vllm_aiter_sparse_pa_spec_decoding_corruption`

5. **[新签名] ELWMUL dest-reuse 高 dst_index face-3 数值损坏（Blackhole）** — tt-metal [#53693](https://github.com/tenstorrent/tt-metal/issues/53693)（2026-08-19）
   - Blackhole 上 ELWMUL（FPU 乘法）配 dest reuse 时，累加目标为最高 `dst_index`（32x32 tile 下 index 7）的最后 tile 返回错误值，损坏限于 face 3；faces 0-2 与 golden 一致。28/224 输出 tile 受影响，恒为每 8-tile 组的第 8 个。`DEST_TO_SRCA`/`DEST_TO_SRCB` 均复现，LoFi/HiFi2/HiFi4 均复现。（2026-08-19）
   - 归类：**硬件/SDC（第 1 类）** → `ttmetal_elwmul_dest_reuse_last_tile_face3_corruption`

6. **[新签名] LLK workaround 移除回归：RMSNorm PCC drop（DeepSeek e2e）** — tt-metal [#53649](https://github.com/tenstorrent/tt-metal/issues/53649)（2026-08-19）
   - PR #53419（用 mul_reuse_dest_tiles 替换 LLK workaround）导致 DeepSeek v3_d_p `test_rmsnorm` PCC 断言失败（0.9843 < 阈值，3.2K/4K 长度）。bisect 定位到该 PR；与 #53693（ELWMUL dest-reuse 错误）同根：**dest-reuse 数值路径本身有缺陷，移除 workaround 暴露真实硬件/SFPU 缺陷**。
   - 归类：**硬件/SDC（第 1 类）** → `ttmetal_llk_workaround_removal_rmsnorm_pcc_regression`

7. **[新签名] TRT-LLM batch>3 并发 prefill 输出损坏（Qwen3.5，逐 token 退化累积）** — TensorRT-LLM [#17964](https://github.com/NVIDIA/TensorRT-LLM/issues/17964)（2026-08-19）
   - Qwen3.5-2B pytorch backend：单请求正确；20 并发下 19/20 输出幻觉重复；**batch≤3 正确、>3 损坏**；损坏随输出长度累积（短输出看似正常）；关 CUDA graph 无效；vLLM 同模型任意并发正确。批处理 prefill 路径数值/索引缺陷。
   - 归类：**服务栈/缓存（第 6 类）** → `trtllm_batched_prefill_gt3_progressive_corruption`

8. **[新签名] DSV4-Flash SM120 逐位置恒定 argmax+恒定 logprob 退化输出（FLASHINFER_MLA_SPARSE_DSV4）** — vLLM [#52938](https://github.com/vllm-project/vllm/issues/52938)（2026-08-19）
   - RTX PRO 6000 Blackwell SM120：greedy 下**每个 decode 位置返回完全相同的特殊 token + 完全相同 logprob（15 位小数一致）**，与位置/上下文无关——典型 forward-pass 断裂签名（decode 步注意力上下文恒定/清零）。TP 与 DP+EP 均复现，非环境问题。HTTP 200 无报错。
   - 归类：**KV缓存/注意力（第 8 类）** → `vllm_dsv4_sm120_constant_argmax_logprob_degenerate`

9. **[新签名] Mooncake 多 rank 共享 ssd_offload_path 碰撞 → L3 KV 静默损坏** — SGLang [#35484](https://github.com/sgl-project/sglang/issues/35484)（2026-08-19）
   - `MooncakeStore` 把同一 `ssd_offload_path` 传给每个 rank 的 embedded client，但 Mooncake LOCAL_DISK bucket tier 假设**一目录一客户端**。共享目录**静默损坏 KV 数据**且跨重启恢复歧义。DeepSeek-V4-Flash MLA TP8 HiCache L3。修复 PR #31926 已存在但无 tracking issue。
   - 归类：**服务栈/缓存（第 6 类）** → `sglang_mooncake_shared_ssd_path_multi_rank_kv_corruption`

10. **[新签名] 8 个 tool-call parser 逐字符流式与 one-shot 不一致（丢调用）** — SGLang [#35564](https://github.com/sgl-project/sglang/issues/35564)（2026-08-19）
    - `cohere_command4/gemma4/glm/glm45/glm47/minimax-m2/mistral/step3` 8 个 parser：空 `{}` 参数调用在逐字符 streaming 模式下与 `detect_and_parse` 结果不一致——调用丢失（minimax-m2）、第 2 个调用丢失（mistral）、`parse_streaming_increment` 拆词错误。agentic 工具调用静默丢失。
    - 归类：**分词器/模板（第 10 类）** → `sglang_toolcall_parsers_streaming_vs_detect_mismatch`

11. **[新签名] Session continuation M-RoPE 位置表部分切片未防护 → 全引擎 crash** — SGLang [#35482](https://github.com/sgl-project/sglang/issues/35482)（2026-08-19）
    - Qwen3.5-family session 续聊第二请求时 scheduler 在 `cuda_graph_buffer_registry._foreach_copy` 崩：mm-input M-RoPE 位置表 partial slice 尺寸 1842≠1816。4/4 模式矩阵复现（native/streaming、graph on/off、text-only 也崩）→ 排除 mm payload 与 CUDA-graph runner，缺陷在共享 buffer loader 的未防护切片。
    - 归类：**位置编码/RoPE（第 9 类）** → `sglang_session_mrope_table_partial_slice_crash`

12. **[新签名] `find_mm_placeholders` 重复 token 掩护下返回错误插入位置** — vLLM [#52924](https://github.com/vllm-project/vllm/issues/52924)（2026-08-19）
    - `PromptInsertion` 插入的 placeholder 与原 prompt 相邻 token 重叠时（如 [32000,32000] 插到 [32000] 旁），`find_mm_placeholders` 扫描找到**更早的匹配**（含原 token）而非实际插入位置 → 多模态嵌入绑定到错误 token 位置。静默错误输出。
    - 归类：**分词器/模板（第 10 类）** → `vllm_find_mm_placeholders_ambiguous_match_position`

### B. [已知类别] 新实例（部分也作为新条目编目，见上）+ 补充来源

13. **[已知类别] Triton MoE / block-FP8 GEMM K-tile 处理缺陷（OOB 读 + 跨组错 scale）** — vLLM [#52576](https://github.com/vllm-project/vllm/issues/52576)（2026-08-17）
    - `fused_moe_kernel_gptq_awq` 权重 load 不带 k_mask（OOB 读，数值被 masked scale 抵消——内存安全 bug）；`w8a8_triton_block_scaled_mm` 单 scale-per-K-tile 假设 `BLOCK_SIZE_K | group_k` 不成立且注释要求写反，跨两组的 tile 用错 scale。默认配置在安全范围内故未爆。归入量化（第 7 类）→ `vllm_triton_blockfp8_ktile_scale_oob_pair`。

14. **[已知类别] Block-FP8 TP 分片小于量化块加载失败** — vLLM [#52997](https://github.com/vllm-project/vllm/issues/52997)（2026-08-19）
    - Qwen3.6-35B-A3B-FP8 TP8：rank-local K=64 < block_k=128、N 分片 < block_n=128，两次不同 ValueError/RuntimeError。属"TP 分片 × 量化块整除性"已知族（12.18 同族），loud 加载失败非静默。归入量化（第 7 类）→ `vllm_blockfp8_tp_subblock_shard_load_fail`。

15. **[已知类别] NIXL HMA 接收失败仍返回 HTTP 200 损坏 token（fail-open）** — vLLM [#52234](https://github.com/vllm-project/vllm/issues/52234)（2026-08-14）+ DeepSeek-V4 变体 [#52276](https://github.com/vllm-project/vllm/issues/52276)（2026-08-14）
    - NIXL KV 接收失败后请求继续从 invalid/uninitialized KV 解码并返回 HTTP 200 携带损坏 token；DSV4 变体：损坏进 `reasoning`、`content` 为空。KV load failure policy: fail 未生效（fail-open）。与既有 6.x NIXL 计数不匹配族（#50754）同域不同签名：本条是**策略声明 fail 但实际 fail-open**。归入服务栈/缓存（第 6 类）→ `vllm_nixl_hma_recv_fail_open_corrupted_200`。

16. **[已知类别] Voxtral 实时音频空嵌入 → 全引擎永久 hang** — vLLM [#52926](https://github.com/vllm-project/vllm/issues/52926)（2026-08-19）
    - 特定 WAV 触发 `Realtime model received empty multimodal embeddings ... Returning zero embeddings` 后无限解码（零嵌入永不产生 EOS），GPU 100%，/health 停止响应。确定性复现（同文件两次）。归入服务栈/缓存（第 4 类数值域外的活性故障，按机制归第 6 类）→ `vllm_voxtral_zero_embed_fallback_infinite_decode_hang`。

17. **[已知类别] align-mode prefix cache 在 priority 调度下命中率结构性 0（静默）** — vLLM [#52897](https://github.com/vllm-project/vllm/issues/52897)（2026-08-19）
    - hybrid GDN + Mamba align cache mode（block 1600）：priority scheduler 切 token budget 不保块对齐 → 快照永不出现在精确边界 → 99.6 万查询 0 命中、无警告。性能静默退化（非正确性）但仍编目（缓存策略 × 调度器交互）。归入服务栈/缓存（第 6 类）→ `vllm_align_prefix_cache_priority_sched_never_hit`。

---

## 二、去重跳过的条目

| issue | 原因 |
|---|---|
| vLLM #52617（LMCache P/D full-hit 但 decode 发散） | 已编目（grep 命中 1 处） |
| vLLM #52520（Mamba align ceiling requeue 死循环） | 已编目（命中 2 处） |
| vLLM #52475（MTP × turboquant 重复坍缩） | 已编目（命中 1 处） |
| vLLM #52531（Kimi-K3 cudagraph 3 种失败模式） | 已编目（命中 1 处） |
| vLLM #52413（causal_conv1d 对齐特化竞态 NaN） | 已编目（命中 1 处） |
| vLLM #52150（GLM-5.2-FP8 冷启动 garbage） | 已编目（命中 2 处） |
| vLLM #52308（NVFP4 MoE ReLU² 损坏） | 已编目（命中 3 处） |
| transformers #48057（Qwen2-VL get_rope_index 回归） | 已编目（9.13） |
| transformers #48029（MPS bf16 加载 segfault） | 已编目（12.20） |
| transformers #48080（Zamba torch.compile hang） | 编译 hang 非数值损坏，报告记录但不编目 |
| vLLM #52404（B200 DP4+EP 乱码） | 根因未定位（无 maintainer 分析），报告记录待后续 |
| tt-metal #53655/#53631（MoE dest-reuse LLK 修复/调查） | 与 #53693 同族线索，作为 #53693 补充来源 |
| vLLM #53013/#52950/#52965（内存 profiling/packed transfer/TRITON_CACHE_DIR） | 资源管理/工程问题非正确性 |
| SGLang #35533（NUL byte regex segfault） | 输入校验 DoS 非推理正确性 |
| triton #11352/#11327/#11322/#11321（编译器 crash） | 编译期 crash 非数值正确性 |

**arxiv**：`silent data corruption LLM`（4 篇均已编目）、`bit flip LLM inference`（0 结果）、`KV cache corruption`（2606.28958 已编目；其余主题不符）、`LLM inference correctness`（2607.28884 已编目）、`GPU soft error / bit-position sensitivity`（2607.19623 已编目 3 处）。**无新编目论文**。

---

## 三、重点新发现总结

1. **单程序双 store 竞态（#52415）**：NVIDIA 上被调度掩盖的 Triton 内核竞态在 MUSA 上显形——「后端特定的调度顺序掩盖数据竞态」是新签名，提示 x86/CUDA 上"正确"的内核在异构后端上可能非确定损坏。
2. **TRT-LLM batch>3 进行性损坏（#17964）**：批处理 prefill 数值损坏随输出长度累积，短输出掩护缺陷——典型"smoke test 通过但生产损坏"。
3. **NIXL fail-open（#52234/#52276）**：配置 `fail` 策略的 KV 传输失败实际 fail-open，HTTP 200 + 损坏 token 静默返回——分离式 PD 栈的静默正确性缺口。
4. **TT dest-reuse 家族（#53693/#53649）**：硬件数值原语缺陷 + 移除软件 workaround 暴露回归——第 1 类硬件/SDC 域的"软件掩盖硬件缺陷"双向案例。
5. **动态 NTK 负底数（#52665）**：纯数学边界 bug（served<trained）导致复指数 NaN——RoPE 配置域新签名。

## 四、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GH search: vllm label:bug created:>08-15 | 4 hits |
| 2 | GH search: sglang label:bug created:>08-15 | 0 hits |
| 3 | GH search: TensorRT-LLM label:bug created:>08-15 | 2 hits |
| 4 | GH search: triton created:>08-15 numerical/correctness | 0 hits |
| 5 | GH search: vllm incorrect/wrong in:title created:>08-12 | 2 hits |
| 6 | GH search: vllm KV cache in:title created:>08-12 | 9 hits |
| 7 | GH search: sglang correctness/wrong/incorrect created:>08-12 | 4 hits |
| 8 | GH search: transformers label:Bug created:>08-12 | 4 hits |
| 9 | GH search: tt-metal numerics/correctness/pcc created:>08-12 | ~20 hits |
| 10 | GH search: vllm NaN/garbled/corrupt in:title created:>08-12 | 11 hits |
| 11 | GH search: vllm nondeterministic/silent created:>08-12 | 3 hits |
| 12 | GH search: triton label:bug created:>08-10 | 4 hits |
| 13-24 | issue 详情抓取（repos/../issues/N） | 12 次 |
| 25-30 | arxiv API × 6 组关键词 | 0 新论文 |

GitHub rate limit 未触发（search remaining 充足，均间隔 ≥7s）。arxiv 全部走 `curl -sL` HTTPS。

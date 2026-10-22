# 每日扫描报告 — 2026-08-22

**扫描日期**：2026-08-22（UTC）
**扫描窗口**：GitHub issues `created:>2026-08-15`（7 天滚动）+ arxiv 最近提交
**主文档基线**：680 条（合并前）

---

## 一、GitHub Issues 发现

### 查询执行记录（10 次 Search API + 19 次 issue/PR 详情 API，无 rate limit 错误）

| # | 查询 | 仓库 | 命中 |
|---|---|---|---|
| q1 | `wrong OR incorrect OR corruption OR NaN in:title` | vllm-project/vllm | 11 |
| q2 | `quantization OR quant wrong in:title` | vllm-project/vllm | 3 |
| q3 | `wrong OR incorrect OR corruption OR NaN in:title` | sgl-project/sglang | 7 |
| q4 | `wrong OR incorrect OR correctness OR NaN in:title` | NVIDIA/TensorRT-LLM | 1 |
| q5 | `wrong output OR wrong result OR RoPE in:title` | huggingface/transformers | 1 |
| q6 | `incorrect OR numerical OR precision OR mismatch in:title` | triton-lang/triton | 1 |
| q7 | `numerical OR pcc OR mismatch OR incorrect in:title` | tenstorrent/tt-metal | 16 |
| q8 | `KV cache OR prefix cache OR speculative in:title` | vllm-project/vllm | 85 |
| q9 | `label:bug` | vllm-project/vllm | 303（取前 15） |
| q10 | `label:bug` | sgl-project/sglang | 2 |

### 已编目（issue 编号已在主文档，跳过）

- vLLM **#53142**（mamba align precopy IMA）、**#53086**（int4 per-token-head KV quant 错值）、**#53084**（MooncakeStore invalid recurrent states）、**#53029**（All-NaN logits → OOV token id = 11.50）、**#52924**（find_mm_placeholders = 10.32）、**#52627**（Kimi-K3 1P1D 静默损坏）、**#52576**（Triton MoE/block-FP8 K-tile OOB）、**#52568**（hybrid-GDN+LoRA NaN = 12.21）、**#52475**（TurboQuant MTP 重复坍塌 = 11.34）
- SGLang **#35564**（tool-call parser 流式错误 = 10.31）、**#35257**（DSA top-k tie-overflow）
- TensorRT-LLM **#17964**（Qwen3.5-2B batch>1 错误输出）
- tt-metal **#54006**（TTTv2 四缺陷 = 1.101）、**#53883**（Kimi-K2.6 PCC≈0 flap = 1.100）、**#53823**（NOP 注入器竞态 = 1.99）、**#53649**（RMSNorm PCC）、**#53389**（remainder int32 = 1.87）
- HF transformers **#48051**（convert_to_rgb alpha = 10.33）

### 新发现 — 合并进主文档（16 条）

| # | 条目 | 来源 | 分类 | 判定 |
|---|---|---|---|---|
| N1 | **tt-metal #53787**（Bounty $5k，修复 PR #54078）— `ttnn.log_sigmoid` SFPU 内核三重精度缺陷：x≤−4 分支缺失（原样返回输入，0.452% 误差）、正尾部单项级数（0.91% 误差）、中段多项式在 x=0 偏 ln2 达 7.12e-4；峰值 fp32 误差 **131,755 ULP**，改用 `min(x,0)−log1p(exp(−|x|))` 恒等式后 ~2 ULP。bf16 几乎不受影响（2.1 ULP）——bf16-only sweep 曾长期掩盖缺陷 | [#53787](https://github.com/tenstorrent/tt-metal/issues/53787) · [PR #54078](https://github.com/tenstorrent/tt-metal/pull/54078) | 第 1 类（硬件/SDC） | `[新类别/新签名]` — **SFPU 数学函数分支缺失 + 区间分裂缺陷**（静默大误差，非溢出非竞态） |
| N2 | **vLLM #53083**（修复 PR #53330）— PD 多层 KV offload：producer 供给集与 consumer 需求集由两台引擎上的**两条独立代码路径**各自计算，无任何机制校验一致；#51840 把 `lookup` 语义改为返回 `HIT_PENDING` 后，一个文件的语义变化**静默改变了 consumer 需求哪些 key**，测试套件毫无察觉；#52808/#53062 是同一弱点的两个独立实例 | [#53083](https://github.com/vllm-project/vllm/issues/53083) · [PR #53330](https://github.com/vllm-project/vllm/pull/53330) | 第 6 类（服务栈） | `[新类别/新签名]` — **跨引擎 supply/demand 契约漂移**（分布式不变量无验证，语义变化单向静默传播） |
| N3 | **vLLM #53062**（修复 PR #53329）— `_cascade_existing_blocks_to_request_level_tiers` 只保留 primary-tier `lookup==HIT` 的 key：**`HIT_PENDING`（写入仍在途）的 key 被永久丢弃**——`prepare_store` 视其已存在而排除、cascade 视其不可读而丢弃、scheduler 推进 `next_stored_chunk_idx` 越过该 chunk，再无任何路径重新供给；P2P producer 侧无 parked entry，peer 烧完 `LOAD_TIMEOUT_S` 后全量重算 | [#53062](https://github.com/vllm-project/vllm/issues/53062) · [PR #53329](https://github.com/vllm-project/vllm/pull/53329) | 第 6 类（服务栈） | `[新类别/新签名]` — **HIT_PENDING 被当作 "never" 而非 "not yet"**（三处代码共同把在途状态折叠成永久缺失） |
| N4 | **vLLM #49250**（修复 PR #53298）— 外部 KV load 被拒（`kv_load_failure_policy="recompute"`）时 scheduler 状态回卷，但 Model Runner V2 **保留旧的 device 侧 computed-token offset**；重叠批次下已在途的 output 还可**改写已重置的 scheduler 状态**或在请求重排后才送达——结果**不等于全量重算**（静默错误输出） | [#49250](https://github.com/vllm-project/vllm/issues/49250) · [PR #53298](https://github.com/vllm-project/vllm/pull/53298) | 第 6 类（服务栈） | `[新类别/新签名]` — **回卷路径的部分回滚状态分歧**（scheduler 回卷 vs device 状态未回卷 vs 在途 output 竞态，三方不一致） |
| N5 | **vLLM PR #53271** — `compute_sub_block_ptrs()` 用 `data_ptr()+stride(0)` 推导裸设备地址，假设**平坦无 padding 行布局**；DeepSeek-V4 MLA 页 padding 到 576B/584B（`fp8_ds_mla`）时地址走出 buffer——swap kernel 的 mask 只对 declared size 边界、不对 allocation 边界，指针本身无校验；启用 `OffloadingConnector`+`fs_python` tier 直接在启动期杀死 worker（CUDA error） | [PR #53271](https://github.com/vllm-project/vllm/pull/53271) | 第 6 类（服务栈） | `[新类别/新签名]` — **padded 页 stride 算术 OOB**（布局假设 × 派生裸指针，mask 边界与分配边界错位） |
| N6 | **vLLM #53366**（修复 PR #53367）— `SpeculativeConfig.compute_hash()` **漏掉 `num_speculative_tokens`**：同一模型扫 k=7/9/11 时，torch.compile 复用为其他 k 编译的 Inductor 图（图内嵌具体张量尺寸）→ `AssertionError: expected size 7==9`；**顺序依赖**——首个 k 填 cache，后续 k 命中 stale 图则崩、干净 cache 则正常，极难归因 | [#53366](https://github.com/vllm-project/vllm/issues/53366) · [PR #53367](https://github.com/vllm-project/vllm/pull/53367) | 第 6 类（服务栈） | `[新类别/新签名]` — **config-hash 键遗漏 → 编译图跨配置碰撞**（缓存键完备性缺陷，症状随历史漂移） |
| N7 | **SGLang PR #35274** — ROCm gfx950 + Triton 3.7.0 下 DSA sparse-MLA prefill 内核 autotune 可选中 `(BLOCK_N=128, num_warps=1, num_stages=2)`，该 config 编译出的内核**产生 NaN/inf**，损坏 prefill 输出并传导至 decode（garbage tokens）；autotune 会把坏选择**按 shape 缓存**，首次 prefill 即定型 | [PR #35274](https://github.com/sgl-project/sglang/pull/35274) | 第 6 类（服务栈） | `[已知类别]` — autotune 选中坏 config 族（6.21 KDA autotune 竞态）的新实例，但机制为**编译器发射 NaN 内核**而非数据竞争；防御性排除修复 |
| N8 | **vLLM #53339**（修复 PR #53346）— `use_ue8m0=True` 下 `per_token_group_quant_fp8` 的 C++ op 把 scale floor 在硬编码 `1e-10` 后 round-up 到 2 的幂，**in-file Triton fallback 只 floor absmax 不 floor scale**：全零组 C++ 得 `2^-33`、fallback 得 `2^-42`；absmax < 2.61e-8 的任何组两后端发出**不同 scale 与不同 fp8 codes**——同一算子在 contiguous/非 contiguous 输入下 bitwise 分歧 | [#53339](https://github.com/vllm-project/vllm/issues/53339) · [PR #53346](https://github.com/vllm-project/vllm/pull/53346) | 第 7 类（量化） | `[新类别/新签名]` — **量化 helper 双后端 scale 分歧**（floor 阈值单侧缺失；与 6.46 融合契约丢失不同源） |
| N9 | **vLLM #53334** — TurboQuant KV cache 在 GB10（sm121）评测暴露两点：(1) hybrid 模型（GDN+full attention）init 即 `ValueError: Unknown TurboQuant cache dtype: 'auto'`（fail-loud 但报错误导）；(2) `triton_turboquant_store` 的 per-vector value scale/zero-point **恒用 fp16**——bf16 输入含 attention-sink 级 outlier（|min|>65504）时 zero-point 溢出，重建值返回 **∞**：一条被投毒的 cache line 而非优雅降级（probe 级 store→decode 往返已验证） | [#53334](https://github.com/vllm-project/vllm/issues/53334) | 第 7 类（量化） | `[新类别/新签名]` — **KV 量化元数据 fp16 溢出投毒 cache line**（outlier 超过 fp16 表示范围 → ∞ 注入 cache） |
| N10 | **SGLang PR #35455**（closed）— compressed-tensors checkpoint 声明 `kv_cache_scheme` 时附带的校准 per-tensor `k_scale`/`v_scale` **加载期静默丢弃**（`Parameter ... not found in params_dict`）→ `--kv-cache-dtype fp8_e4m3` 跑**无 scale 的 cache**；三个缺口叠加：`from_config` 不转发 scheme、`get_quant_method` 无 `RadixAttention` 分支、weight mapper 只认 ModelOpt 序列化路径 | [PR #35455](https://github.com/sgl-project/sglang/pull/35455) | 第 7 类（量化） | `[新类别/新签名]` — **KV 量化 scale 加载静默丢弃**（量化元数据管线的"找不到参数→丢弃"模式） |
| N11 | **vLLM PR #53376** — `CutlassFP8ScaledMMLinearKernel.is_supported()` 只检查平台是 CUDA、从不探测算力——`cutlass_scaled_mm_supports_fp8()` 拒绝一切低于 SM89 的平台，但该 kernel 在 `_POSSIBLE_FP8_KERNELS` 中排在 Marlin 之前，**pre-Ada GPU 上赢得自动选择且下游无法恢复**；在某 commit 移除 force-fallback 环境变量后变为可达 | [PR #53376](https://github.com/vllm-project/vllm/pull/53376) | 第 7 类（量化） | `[新类别/新签名]` — **kernel 能力门控缺失 → 不支持平台上自动选中**（capability probe 缺失于分发矩阵） |
| N12 | **SGLang PR #35434**（open）— Intel AMX CPU 后端 `extend_attention_cpu` 对 extend 阶段**恒用 causal mask**：对 decoder 自注意力正确，对 encoder-only 双向注意力（bge-reranker 族）**静默给出因果掩码的注意力分数**——rerank 场景下所有文档得到完全相同分数（2.611328125），修复后分数正确分化（6.1875 / −8.15625） | [PR #35434](https://github.com/sgl-project/sglang/pull/35434) | 第 8 类（KV/注意力） | `[新类别/新签名]` — **extend 阶段强制 causal mask**（硬编码 mask 假设 × 双向架构；"全部同分"是可编程检测签名） |
| N13 | **SGLang #35949** — `RadixArk/Qwen3.8-27B-NVFP4` 在 SGLang 上 2D grounding 坐标严重错误（中心误差 178.3px、IoU 0），同 checkpoint/图/提示在 vLLM 误差仅 2.9px（IoU 0.71）；A/B 隔离到 `qwen3_5.py` 的 **CUDA fused QK-RMSNorm+RoPE+output-gate 路径**，强制 native 路径即恢复 | [#35949](https://github.com/sgl-project/sglang/issues/35949) | 第 9 类（RoPE） | `[已知类别]` — fused QK-norm+RoPE 族（9.8/9.9）新实例，但含 **output-gate 融合**这一新变体，且显形为空间坐标任务退化 |
| N14 | **SGLang PR #35305**（open）— Kimi-K3 DP-sharded vision + cpu transport：batch 内 ≥2 张**不同分辨率**图片即 `ValueError: deferred GPU preprocessing produced wrong grids` 崩服务；根因：deferred 路径用 **shard-local 位置**而非全局图片位置读 `grid_thws_host`；相同 grid 的图片掩盖该 bug（回归自 #34404，单测因用 identical grids 而漏检） | [PR #35305](https://github.com/sgl-project/sglang/pull/35305) | 第 10 类（分词器/模板→多模态预处理） | `[新类别/新签名]` — **DP 分片局部索引 vs 全局图片索引错位**（shard 边界上的坐标语义换算缺失；fail-loud） |
| N15 | **vLLM PR #53379**（open）— `AutoWeightsLoader` 按相邻前缀分组：`language_model.* / vision_tower.* / language_model.*` 交错流使语言模型 loader 被调用**两次**；每次调用都 finalize MegaMoE 权重并删除原始 weight/scale 参数 → 后到组缺 `w13_weight_scale` 加载失败 | [PR #53379](https://github.com/vllm-project/vllm/pull/53379) | 第 12 类（权重加载） | `[新类别/新签名]` — **交错权重流的重复 finalize**（"分组相邻"假设 × 流式 loader；有消耗性的 finalize 副作用不可重入）。注：编号与 tt-metal #53379（已编目 13.x）撞号但为不同仓库不同问题 |
| N16 | **arXiv:2608.07408** "Addressable Memory for Video World Models" — 视频 world model 的 KV cache 作为增长视觉记忆，**rollout 超出训练 horizon 后 temporal RoPE 偏移超出训练所见范围**，模型无法经注意力取回相关视觉信息；且在 RoPE 旋转后空间做朴素 cache 压缩会**平均掉不同相位的 key 而损坏记忆** | [arXiv:2608.07408](http://arxiv.org/abs/2608.07408) | 第 14 类（长上下文能力） | `[新类别/新签名]` — **temporal RoPE 越界寻址失效 + 旋转空间压缩损坏**（model_on：健康权重的能力边界；KV 交叉引用） |

### 补充来源（不占新编号，追加到已有条目）

- **tt-metal PR #53720** → 1.28（logaddexp naive 溢出）：第二版数值稳定实现（fused SFPU `max(a,b)+log1p(exp(-|a-b|))`），同修 #52037
- **tt-metal PR #53476** → 1.87（remainder int32 garbage）：修复确认 scalar overload 走 float SFPU 快路径、INT32 落入 float kernel 被 reinterpret（正股息→denormal≈0、负股息→NaN 位模式）
- **vLLM PR #53406** → 11.34（TurboQuant MTP 重复坍塌）：根因定位——`TurboQuantMetadataBuilder` 声明自相矛盾（`_cudagraph_support=UNIFORM_BATCH` × `supports_spec_as_decode=False`）：UNIFORM_BATCH 允许 FULL 捕获 query_len=1+k 的 verify 批，而 `reorder_batch_threshold=1` 使批走单 token 路由；k=1 直接 IMA crash
- **SGLang PR #35228** → 7.44（compressed-tensors lm_head 量化分发缺失）：同机制的修复 PR（fp8 weight 被 value-cast 进 bf16 参数、`lm_head.weight_scale` 静默丢弃）

### 记录但不编目（非正确性域 / 信息不足 / 纯性能）

- vLLM **#53407**（ROCm MRV2 uniform decode 派发 padded FULL cudagraph）——性能域：一半 decode batch 静默回退 eager
- vLLM **#52881**（revert MM keep_on_cpu CI 回归）——CI-only，无正确性数据
- vLLM **#53351**（ROCm CI 恢复 KV-cache layout refactor 后的注意力覆盖）——测试覆盖修复
- vLLM #53324/#53333/#53360/#53365/#53367 之外 q9 命中的 feature/telemetry/parser 项（#53368 等）——非正确性
- SGLang **#35285**（docs 修复：过期环境变量 SGLANG_SANITIZE_NAN_LOGITS 等）——文档域（旁证 NaN logits 处理机制存在）
- Triton **#11381**（cherry-pick of #10928，AMD GFX9 direct-to-LDS blocked/shared order mismatch）——合并于 07-17（窗外），且主要是 warp coalesced writes 性能问题 + lowering 误拒合法组合
- tt-metal **#54025/#54015**（pytest-xdist collection mismatch）——测试基础设施
- tt-metal **#53491**（KV-PCC leg budget 移交 shield team）——CI 编排
- TensorRT-LLM q4 仅 #17964（已编目）；HF q5 仅 #48051（已编目）

---

## 二、arXiv 论文发现（5 组查询，23 篇返回）

| 查询 | 结果 |
|---|---|
| `silent data corruption LLM` | 4 篇全部已编目（SCOUT 2608.11034、LLM-PRISM 2604.10390、2604.00726、2502.12340） |
| `bit flip LLM inference` | 4 篇均已编目或引用（Bit-Position Sensitivity / RangeGuard / Rotated Robustness / Targeted BFA / BitFlipScope） |
| `KV cache corruption` | **2608.07408（Addressable Memory for Video World Models）为新发现 → 编目为 14.7**；2608.05863 已引用；2606.28958 已编目；Moment-KV（2605.29873）为压缩方法论文非故障报告；2606.07571 已编目（8.41） |
| `LLM inference correctness` | 全部 agentic/benchmark 主题（金融校准退化、rEDMRec 等），与推理正确性故障无关 |
| `GPU soft error inference` | 2601.19912 已编目（1.98）；2104.09455（2021）经典容错 |

---

## 三、API 调用记录

| 时间 (UTC) | 调用 | 状态 |
|---|---|---|
| 23:0x | GitHub Search ×8（q1–q8，`--data-urlencode`，间隔 ≥8s） | 全部成功，无 403 |
| 23:1x | arxiv query ×5（`curl -sL`） | 全部成功 |
| 23:1x | GitHub 详情 ×4（vLLM #53334，SGLang #35949/#35434/#35274） | 成功 |
| 23:1x | GitHub 详情 ×4（vLLM #53330/#53329/#53298/#53271） | 成功 |
| 23:1x | GitHub 详情 ×4（tt-metal #54078/#53787，SGLang #35305，tt-metal #53476） | 成功 |
| 23:1x | GitHub Search ×2（q9/q10 label:bug，间隔 8s） | 成功 |
| 23:1x | GitHub 详情 ×4（vLLM #53346/#53379/#53367，SGLang #35455） | 成功 |
| 23:1x | GitHub 详情 ×3（vLLM #53406，SGLang #35228，vLLM #52881）+ Triton PR #10928 | 成功 |
| 23:2x | GitHub 详情 ×3（vLLM #53366/#53339，tt-metal #53720） | 成功 |
| 23:2x | GitHub 详情 ×2（vLLM #53407/#53376） | 成功 |

---

## 四、合并摘要

- **16 条新条目**合并进 `FAULT_MASTER_REFERENCE.zh.md`：
  - 第 1 类 +1（1.102 `ttmetal_logsigmoid_sfpu_missing_branch_range_splits`）
  - 第 6 类 +6（6.137 `vllm_kv_offload_supply_demand_contract_drift`、6.138 `vllm_tiering_cascade_hit_pending_key_permanent_drop`、6.139 `vllm_kv_load_reject_partial_rollback_state_divergence`、6.140 `vllm_kv_offload_sub_block_ptr_padded_page_oob`、6.141 `vllm_compile_cache_key_omits_spec_tokens_stale_graph`、6.142 `sglang_dsa_triton_prefill_autotune_nan_config_rocm`）
  - 第 7 类 +4（7.50 `vllm_ue8m0_group_quant_backend_scale_divergence`、7.51 `vllm_turboquant_fp16_zero_point_overflow_poisoned_cache`、7.52 `sglang_compressed_tensors_kv_cache_scheme_scales_dropped`、7.53 `vllm_cutlass_fp8_kernel_pre_sm89_ungated_autoselect`）
  - 第 8 类 +1（8.51 `sglang_cpu_amx_extend_forced_causal_mask_bidirectional`）
  - 第 9 类 +1（9.16 `sglang_qwen38_nvfp4_fused_qknorm_rope_outgate_grounding`）
  - 第 10 类 +1（10.34 `sglang_kimi_k3_dp_vision_shard_local_grid_index`）
  - 第 12 类 +1（12.22 `vllm_kimi_k3_interleaved_weight_stream_double_finalize`）
  - 第 14 类 +1（14.7 `video_world_model_temporal_rope_horizon_addressing_failure`，arXiv:2608.07408）
- **4 条补充来源**：PR #53720→1.28、PR #53476→1.87、PR #53406→11.34、PR #35228→7.44
- **总计 680 → 696 条**（第 1 层 549 → 565）
- 14 条为 `[新类别/新签名]`、2 条为 `[已知类别]` 新实例（N7 autotune 族、N13 fused QK-norm+RoPE 族）

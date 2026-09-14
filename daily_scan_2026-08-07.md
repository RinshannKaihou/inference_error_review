# LLM 推理故障每日扫描报告 — 2026-08-07

**扫描日期**：2026-08-07（UTC）
**搜索窗口**：GitHub `created:>2026-07-31`（7 天滚动窗口）+ arxiv 近期
**主参考文档基线**：349 条（合并前）

---

## 扫描范围与 API 调用记录

| # | 目标 | 查询 | 结果数 |
|---|------|------|--------|
| 1 | vLLM | `label:bug created:>2026-07-31` | 279 |
| 2 | vLLM | `silent OR wrong OR incorrect OR corruption OR NaN OR determinism in:title created:>2026-07-31` | 21 |
| 3 | SGLang | `label:bug created:>2026-07-31` | 2 |
| 4 | SGLang | `incorrect OR silent OR corruption OR correctness in:title created:>2026-07-31` | 16 |
| 5 | TRT-LLM | correctness title search | 0（null——无匹配） |
| 6 | triton | `incorrect OR correctness in:title created:>2026-07-31` | 1 |
| 7 | triton | `determinism OR deterministic in:title,body created:>2026-07-31` | 2 |
| 8 | tt-metal | `correctness OR silent OR incorrect in:title created:>2026-07-31` | 14 |
| 9 | transformers | `incorrect OR silent OR NaN in:title created:>2026-07-31` | 4 |
| 10 | vLLM | `offload silent OR wrong created:>2026-07-31` | 38 |
| A1 | arxiv | `silent data corruption LLM` | 3 |
| A2 | arxiv | `bit flip LLM inference` | 5 |
| A3 | arxiv | `KV cache corruption inference` | 5 |
| A4 | arxiv | `LLM inference correctness` | 5 |
| A5 | arxiv | `GPU soft error learning` | 3 |
| A6 | arxiv | `quantization LLM error` | 5 |

GitHub Search API 调用约 12 次（限额 10/min，每次间隔 ≥7s，未触发 403）；arxiv 6 次。

---

## 去重结果

对照 FAULT_MASTER_REFERENCE.zh.md 现有 ~384 个 `#编号` 引用，本日发现的下列条目经逐一比对**未出现在主文档中**（全部为新条目）。以下条目为确认新增。

---

## 发现条目（按故障领域分类）

### 硬件 / SDC（SFPU / 内核原语 / block-float 缺陷）

1. **[已知类别 var:1.x] tt-metal #52532 — `ttnn.sort` ROW_MAJOR DRAM 多核路径返回非排列垃圾值**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52532
   - 机制：`ttnn.sort` 对 ROW_MAJOR 输入且宽度足够触发 `SortProgramFactorySingleRowMultiCore` 时返回非排序、非输入排列的结果——含输入中不存在的值（含 ±inf），无错误抛出。TILE 布局同形状正确，ROW_MAJOR 其他两条 factory 也正确。Wormhole B0 复现。

2. **[已知类别 var:1.x] tt-metal #52307 — Welford Float32 在 `fp32_dest_acc_en=false` 时走 FPU 路径（静默精度损失）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52307
   - 机制：`layernorm_pre_all_gather` 的 Welford 程序对 Float32 输入在 `fp32_dest_acc_en=false` 时走 FPU 路径——unpacker 截断到 tf32/bf16 风格数学，op 仍返回结果但表现为正常数值差异而非配置错误。`welford_unpack_fp32_active` 应选 SFPU/`UnpackToDestFp32` 路径但未强制为唯一有效 Float32 模式。

3. **[已知类别 var:1.x] tt-metal #52040 — sharded softmax 接受 `subblock_w > Dest` 静默返回错误结果（行和达 47.42）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/52040
   - 机制：sharded softmax 对 `subblock_w` 的唯一约束是 `block_w % subblock_w == 0`；当 `subblock_w`（12）超过 Dest 容量时返回行和高达 **47.42** 的垃圾张量（softmax 应行和为 1），无错误/警告/日志。Blackhole P300 复现，sentence_bert 真实形状触发。

4. **[已知类别 var:1.x] tt-metal #51921 — `topk` 单核路径静默忽略 `indices_tensor`（payload 丢弃）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/51921
   - 机制：`topk_single_core_program_factory` 硬编码 `GENERATE_INDICES = "1"`，忽略传入的 `indices_tensor` payload（如全 777），返回 generated iota `[0,1,2,...]`。仅多核 factory 读取该 tensor。单核形状（W<8192、非 2 的幂、W≥65535、k>64 等）的调用者静默获得错误索引。同时 index dtype 未验证。

5. **[已知类别 var:1.x] tt-metal #51860 — `moe_routing_remap` 静默接受错误的 `expert_parallel_size`（C++ 运算符优先级 bug）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/51860
   - 机制：`validate_on_program_cache_miss` 中 `TT_FATAL` 的条件因 C++ 运算符优先级（`==` 比 `?:` 紧）始终为非零 mesh 维度——检查永不触发；三元臂还与树的其余部分互换。错误配置因此运行而非报错，program factory 用截断的 `non_zero_per_device` 产生静默错误的 per-device 路由权重。

6. **[新类别/新签名] tt-metal #51831 — `bfloat8_b` reciprocal/rsqrt/log/div 对子 tile 4-D 张量返回全零（静默 SDC，含 RMSNorm rsqrt）**
   - 来源：https://github.com/tenstorrent/tt-metal/issues/51831
   - 机制：`bfloat8_b`（block-float）、TILE_LAYOUT、4-D 且末两维各小于 32×32 tile 时，`reciprocal`/`rsqrt`/`log`/`log2`/`log10` 返回**全零输出**。严格正输入的 reciprocal 不可能为 0 → 这是 SDC 而非精度不足。`div`（reciprocal 基）更广——tile 对齐 2-D 也错。**rsqrt 是 RMSNorm 核心原语**：bfloat8_b 下小 trailing 空间维的 RMSNorm 会静默将归一化输出清零——无 NaN、无 assert。缺陷在 block-float unary/reciprocal compute path。Blackhole 复现。

7. **[已知类别 var:1.x] triton #11186 — `OptimizeThreadLocality` 错误 reordering 导致 miscompile（reduction 外维度也被重排）**
   - 来源：https://github.com/triton-lang/triton/pull/11186
   - 机制：`OptimizeThreadLocality` 引入 `can_reorder` reshape，可能重排 reduction 维度**之外**的元素 → miscompile。修复改用显式 reshape+permute 序列构造张量以保持 intra-thread reduction 正确。属编译器 codegen 静默错误（同 1.39/1.40 族）。

### 服务栈 / 缓存正确性

8. **[已知类别 var:6.x] vLLM #51396 — `cpu_offload_gb` 在 Model Runner V2 上静默 no-op（offloader 从不安装）**
   - 来源：https://github.com/vllm-project/vllm/issues/51396
   - 机制：`--cpu-offload-gb`（UVA 权重卸载）在 V2 GPU model runner 激活时被**静默忽略**——无卸载、无警告。多个架构（`DeepseekV2ForCausalLM`、`Qwen2MoeForCausalLM` 等）默认用 V2 runner。对需要卸载的负载（模型大于 VRAM），可见失败是 OOM，错误信息建议无关 remedy，用户无从知晓卸载设置未生效。

9. **[已知类别 var:6.x] vLLM #51401 — YAML 配置中 falsey 值被静默丢弃（`false` 不写入 argv）**
   - 来源：https://github.com/vllm-project/vllm/issues/51401
   - 机制：`--config <file>.yaml` 加载时，`FlexibleArgumentParser` 的 config→argv 转换中，bool `False` 被静默丢弃（`if value: append; # else: nothing`），而非转为显式 off 信号。对默认非 False 的三态 flag，用户的 `false` 静默回退到真实默认——无警告、无错误。

10. **[已知类别 var:6.x] vLLM #51441 — DSv4-MTP 混合稀疏注意力 + Eagle：非 256 倍数 prompt 长度全 cache miss**
    - 来源：https://github.com/vllm-project/vllm/issues/51441
    - 机制：DeepSeek-V4-Pro MTP + Eagle 推测解码，prefix caching 对 prompt_length 非 256 倍数时 cache_hit_length=0（如 16384 → 0 hit，20000 → 16128 hit）。Hybrid sparse attention 与 Eagle 的块对齐在非对齐长度上失效，prefix cache 实为静默 no-op。

11. **[已知类别 var:6.x] vLLM #51094 — `OffloadingConnector` 在精确 chunk 边界 + `mamba_cache_mode=all` 时静默返回错误输出**
    - 来源：https://github.com/vllm-project/vllm/issues/51094
    - 机制：`OffloadingConnector` 在 prompt 长度恰为 offloaded Mamba chunk 整数倍且 `mamba_cache_mode="all"` 时，CPU cache refetch 后静默产生不同输出。`resolve_mamba_align_size()` 仅对 `align` 模式启用 hit-window 对齐，但 `"all"` 模式也存 recurrent 状态；prefix lookup 上限 `num_tokens-1`（末 token 须重算 logits），精确边界处 refetch 返回 `N-1` token 但恢复的 state 含 token N → token N 被应用两次，输出改变无异常。混合 attention+Mamba 模型（Nemotron-Nano-9B-v2）复现：cold vs refetch 首 token 序列在第 5 个 token 开始分叉。

12. **[已知类别 var:6.x] SGLang #34043 — DFlash/DSpark 推测池 corrupt sconv（short-conv）state（radix cache hit 时全零 sconv 张量）**
    - 来源：https://github.com/sgl-project/sglang/pull/34043
    - 机制：DFlash/DSpark 推测池在 radix cache 对 hybrid-SWA 模型（Inkling/Inkling-Small）启用时 corrupt sconv state——mamba tracking gate 在 prefill-graph capture 上，radix cache hit 的请求用全零 sconv cache 张量 resolve。长上下文尤甚：SWEBench Verified/Pro/Multilingual 的 70 题子集，DFlash 将 pass rate 从 55% 降至 35%（长上下文 Inkling rollout 卡住、遗忘）。

13. **[已知类别 var:6.x] SGLang #33397 — DeepSeek-V4-Flash-0731 并发下渐进式输出损坏（DP attention）**
    - 来源：https://github.com/sgl-project/sglang/issues/33397
    - 机制：2× H200 服务 DeepSeek-V4-Flash-0731 + `--attention-backend dsv4` + DP attention，输出损坏随 batch size 缩放：从细微语言退化 → 畸形/错信道 tool-call markup → 完整 token soup。两种几乎不共享任何东西的配置均复现（含自带 #33098 patch 的镜像）；vLLM 0.26.0 同节点同 checkpoint 同 fp8 KV 不复现。疑似 KV/page-index 在 DSA（"HiSparse"）注意力路径的问题。

### 量化

14. **[已知类别 var:7.x] vLLM #51419 — MXFP4 `weight_scale` 存为 FP32 时 casting 错误 + per-expert checkpoint 映射缺失**
    - 来源：https://github.com/vllm-project/vllm/pull/51419
    - 机制：部分 MXFP4 checkpoint 将 `weight_scale` 存为 FP32，权重加载时 casting 到 uint8 出错；同时 per-expert 量化 checkpoint 的权重映射缺失。

15. **[已知类别 var:7.x] vLLM #51411 — INT8 W8A8 MoE 在 TritonExperts 中 crash（未识别 int8 激活 dtype）**
    - 来源：https://github.com/vllm-project/vllm/pull/51411
    - 机制：`TritonExperts` 未识别 `torch.int8` 为有效输入 dtype，INT8 W8A8 MoE 模型处理量化激活时 crash（`assert hidden_states.dtype in [...]` 失败）。修复添加 INT8 到支持类型并用 BF16 做 compute type。（注：本条为 crash 而非静默损坏，但属量化路径正确性 bug。）

16. **[已知类别 var:7.x] SGLang #33541 — MXFP4 per-expert checkpoint 加载：expert block scale 全部静默清零（服务正常，输出 garbage）**
    - 来源：https://github.com/sgl-project/sglang/pull/33541
    - 机制：SGLang MXFP4 仅被 GPT-OSS checkpoint（所有 expert 打包进单 tensor，走 fused fast path）验证过。DeepSeek 每 expert 一个 tensor，走 per-expert 加载路径，命中 6 个串行问题——前 3 个大声失败，**后 3 个静默**：每个 expert block scale 加载时清零，模型启动、服务、生成流利文本但**停止跟随 prompt**，日志仅一行 "weights not initialized" 列表项。涉及 DeepSeek name remap 将 `.scale` 重写为 fp8 block-scale 参数名。

### KV 缓存 / 注意力机制故障

17. **[已知类别 var:8.x] SGLang #33656 — DeepSeek-V4 hierarchical cache 确定性 SWA KV position 损坏（kv-canary TAIL_K_SWA write_position）→ 下游 NaN 采样 crash**
    - 来源：https://github.com/sgl-project/sglang/issues/33656
    - 机制：DeepSeek-V4-Flash + `--enable-hierarchical-cache` v0.5.16，生产 crash：采样概率张量含 NaN（`Assertion 'probability tensor contains either inf, nan or element < 0' failed`）杀所有 TP rank。轻载（1 请求、~24K 上下文）下复现。`--kv-canary=log` 捕获**确定性** SWA 组件 KV 损坏：`kv_canary violation: launch_tag=TAIL_K_SWA fail_reason=write_position slot_idx=53504 position=512 stored_token=1299 expected=...`。

18. **[已知类别 var:8.x] SGLang #33579 — DSPARK/DFLASH spec v2 `page_size=1`：`req_to_token` 行过窄，context 边界静默 KV slot 泄漏**
    - 来源：https://github.com/sgl-project/sglang/issues/33579
    - 机制：DFLASH-family spec v2 每 decode step reserve `committed + 2*num_draft_tokens` KV slot，但 `req_to_token` 行仅在 `page_size>1` 时按此宽度分配。`page_size=1` 时行宽 `context_len + 4 + num_draft_tokens` 小于 reserve 水位。DSPARK gamma=16（`num_draft_tokens=17`）：行 headroom=21，per-step reserve 超 committed=34。请求接近 context 边界时（`committed > context_len-13`）：行写入溢出到**邻居请求的行**，release slice 静默 clamp 到行尾 → 溢出 tail slot 永不释放。

19. **[已知类别 var:8.x] SGLang #33268 — HiCache 存储键省略 `kv_cache_dtype` → 跨运行静默缓存碰撞**
    - 来源：https://github.com/sgl-project/sglang/issues/33268
    - 机制：HiCache 存储键含 model_name、TP/PP/CP geometry，但**不含 KV cache dtype**。`HiCacheStorageConfig` 本身无 dtype 字段。后果：同一模型两次运行（如 bf16 然后 fp8_e4m3）产生相同 key，第二次运行静默复用 dtype 错误的缓存条目字节。

20. **[已知类别 var:8.x] vLLM #51313 — Kimi-K3 `--kv-cache-dtype fp8` 在 H200 不可用（assertion 循环：要求被静默忽略的 flag）**
    - 来源：https://github.com/vllm-project/vllm/issues/51313
    - 机制：Kimi-K3 + `--kv-cache-dtype fp8` + `--attention-backend FLASHMLA` 在 H200（SM 9.0）运行时 assertion 失败，要求启用 `--attention-config {"use_prefill_query_quantization": true}`。**设置该 flag 不解决错误**——同 assertion 再触发。非 Blackwell（非 cap-100）设备上该 flag 被静默忽略，错误循环且不可满足。

### 解码 / 采样故障

21. **[已知类别 var:11.x] SGLang #33697 — seeded 采样对空分布静默选择 token 0**
    - 来源：https://github.com/sgl-project/sglang/issues/33697
    - 机制：`multinomial_with_seed()` 对 log 概率加确定性 Gumbel 噪声返回 argmax。若每行 log 概率均为 `-inf`，所有得分仍为 `-inf`，argmax 返回 index 0。与 unseeded `torch.multinomial`（拒绝全零分布）不同。空分布静默产出 token ID 0 而非请求级失败。

### 模型图 / 架构故障

22. **[已知类别 var:13.x] vLLM #51395 — SM120 FlashInfer sparse MLA 声称支持 dense prefill 但仅实现 sparse MQA（AttributeError）**
    - 来源：https://github.com/vllm-project/vllm/pull/51395
    - 机制：SM120 sparse backend 从 `MLAAttentionImpl` 继承 `supports_dense_mha_prefill = True`，但只实现 sparse `forward_mqa` 路径。长 prompt 选 dense/masked MHA prefill 时，共享 MLA router 访问该实现未定义的 dense-path state（如 `masked_mha_available`）→ 失败。修复 override 为 `False`，路由将 prefill 全走 sparse-MQA。

23. **[已知类别 var:9.x] vLLM #33967 — B200+FP8 QK Norm+RoPE 融合 pattern 静默失败（Inductor CSE 未合并 duplicate splits）**
    - 来源：https://github.com/vllm-project/vllm/pull/33967（修复 #33295）
    - 机制：B200 + FP8 量化模型（如 Qwen3-30B-A3B-FP8），PyTorch Inductor CSE 未合并相同 `split_with_sizes` 调用。QK Norm+RoPE pattern matcher 期望一个 split 节点带 3 个 getitem user（q/k/v），但图中含 3 个独立 split 节点各 1 user → 融合**静默失败**。修复加 `coalesce_equivalent_splits()` 先合并 duplicate split。

### 分词器 / 模板故障

24. **[已知类别 var:10.x] transformers #47838 — `additional_special_tokens` 在含 `extra_special_tokens` 时被静默丢弃**
    - 来源：https://github.com/huggingface/transformers/issues/47838
    - 机制：两个 v5 back-compat shim 用 `init_kwargs.setdefault("extra_special_tokens", init_kwargs.pop("additional_special_tokens"))` 转换 deprecated kwarg。Python 先求值参数，`.pop(...)` 总执行——若 `extra_special_tokens` 已存在，`setdefault` 不覆盖，pop 出的 `additional_special_tokens` 被静默丢弃。

### 分布式 / 多 GPU 故障

25. **[已知类别 var:5.x] SGLang #33967（Ascend）— `int32` vs `int64` dtype 不匹配导致错误内存地址访问**
    - 来源：https://github.com/sgl-project/sglang/issues/33967（注：此为 SGLang Ascend issue，与 vLLM #33967 同号不同仓）
    - 机制：SGLang 中定义为 int32，但 `ascend_backend` 内核读为 int64 → 不正确的内存地址访问。Ascend910B2C（16× NPU）测试时发现多个 SGLang Ascend 后端 bug。

---

## arxiv 论文（强相关但多为训练/防御/方法论视角，非推理服务故障新模式）

- **arXiv:2608.05863 — "Runtime Observability for Heterogeneous Attention Memory"**（2026-08-06）：异构注意力内存的运行时可观测性。方法论/工具，非新故障机制，但与 KV cache 监控相关。
- **arXiv:2608.04074 — "Spend Bits Where Queries Look"**（2026-08-04）：KV cache 向量量化 with attention-preserving transforms。量化方法，非故障。
- **arXiv:2608.03854 — "Quantization Effects on Biomedical LLM Reliability"**（2026-08-04）：量化对生物医学 LLM 可靠性影响。质量评估，非新机制（前次扫描已记）。
- **arXiv:2607.27275 — "Flat Score, Amplified Failures"**（2026-07-29）：量化 LLM agent 中 error budget 掩盖损坏。量化可靠性分析，可作为量化类别补充视角。
- **arXiv:2607.19623 — "From Bit-Position Sensitivity to Unequal Error Protection"**（2026-07-21）：DNN 推理内存位位置敏感性与不等错误保护。BFA/位翻转防御，第 3 类视角补充。
- **arXiv:2605.11202 — "Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing"**（2026-05-11）：LLM 服务系统漏洞 fuzz 发现。方法论，与本项目监控/注入目标高度相关，可作为方法论引用。

---

## 重点新发现（突出）

本日最值得关注的新案例：

1. **tt-metal #51831（bfloat8_b reciprocal/rsqrt 全零）** — **新子签名**：block-float dtype 下子 tile 4-D 张量的 unary/reciprocal 静默返回全零，rsqrt 直接影响 RMSNorm（Llama/Qwen/Mistral/Gemma 等），无 NaN/assert。这是"block-float compute path 子 tile 静默 SDC"的全新签名，此前编目仅有 SFPU 近似域外错误和格式寄存器残留，未覆盖 block-float dtype 的 compute path 缺陷。
2. **vLLM #51094（OffloadingConnector chunk 边界静默错误输出）** — **完全静默**：精确 chunk 边界 + mamba_cache_mode=all 时 refetch 后输出分叉，无 crash/NaN。hybrid attention+Mamba 模型的 CPU 卸载与 recurrent state 对齐新签名。
3. **SGLang #34043（DFlash/DSpark sconv state corruption）** — radix cache hit 的请求用全零 sconv 张量，长上下文 SWEBench pass rate 降 20 个百分点。推测解码 + radix cache + SSM state tracking 交互的新签名。
4. **SGLang #33656（hierarchical cache SWA KV position 损坏 → NaN crash）** — kv-canary 确定性捕获的 SWA 组件 write_position 损坏，是 KV cache 层级化 + SWA 的新签名。
5. **SGLang #33268（HiCache 键省略 dtype → 跨运行碰撞）** — 缓存键设计缺陷导致跨 dtype 静默碰撞，是 HiCache 持久化层的新签名。

无全新故障领域类别发现；全部为已有类别的新实例/新子签名。

---

## 将合并进主文档的条目（25 条，全部为新）

见主文档合并结果。

# LLM 推理故障每日扫描报告 — 2026-09-01

- **扫描日期**：2026-09-01（滚动窗口 created:>=2026-08-31，与昨日报告重叠部分已按主文档 issue 编号查重剔除）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 815 条（第 0 层 17 + 第 1 层 684 + 第 2 层 114）
- **结果**：新发现并合并 **29 条**（8 个新子签名 + 21 个已知类别变体）；另有 1 篇 arxiv 论文（TrainSDC 2608.30769）作为 §3.1 补充来源引用不占新编号。arxiv 另命中 1 篇强相关新论文（JITterFlip 2608.29745，宿主侧 JIT 控制平面 BFA）已编目为 3.18；Compression-Aware Abstention（2608.29934）编目为 15.9（KV 压缩逐出证据→幻觉的行为面缓解，属幻觉/忠实性域）；其余命中均为已编目论文（SCOUT、LLM-PRISM、Groundhog GBFA、ROBBIN、Decision-Level Hijacking、When Latent Agents Lie、TurboQuant 统计论文、Moment-KV、Runtime Observability、Batch Speculative Decoding ragged-tensor 系）或不收录的方法论文（RAP: RoPE-Aligned Pruning、Continuous Fuzzing）。
- **API 事件记录**：GitHub Search API 当日 8 组查询全部一次成功（无 403）；core REST API 共 34 次调用全部成功。transformers 仓库 7 天窗口 correctness 关键词命中 0 条。

---

## 一、新编目条目（29 条，已合并进主文档）

### 第 1 类 · 硬件 / SDC（数值内核原语）（+7）

1. **tt-metal #55106** — [LLK] Dest→Src bank waits still miss the math-pipeline drain at 33 sites (WH + BH) — https://github.com/tenstorrent/tt-metal/issues/55106
   `[已知类别]`（→ 1.130，var:1.98）PR #52256（已编目为 1.98 来源）给 Blackhole destination-reuse 握手加 math-pipeline drain 的修复只落在**一个文件的两个函数**；同样的裸 `TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD)` 仍存在于 **16 个文件的 33 处站点**（含 canonical `transpose_dest` LLK 与 TTNN-private LLK 拷贝，被生产内核编译）——修复未传播模式的规模化形态。
2. **tt-metal #55017** — ttnn.grid_sample accepts FLOAT32 input and returns unrelated values — https://github.com/tenstorrent/tt-metal/issues/55017
   `[新类别/新签名]`（→ 1.131）grid_sample 对 FLOAT32 输入校验通过、不报错，返回值与参考**毫无关系**且有限、bit-stable——同为 n150 与 p150b 两代硬件上返回**不同的垃圾**；BFLOAT16 输入正确。(dtype 接受面 × 内核未实现静默垃圾。)
3. **tt-metal #55032** — concat: sharded s2s uses absolute dim 2/3 instead of rank-relative H/W — https://github.com/tenstorrent/tt-metal/issues/55032
   `[新类别/新签名]`（→ 1.132）`ConcatS2SMultiProgramFactory` 用**绝对 dim==2** 判 height，rank-3 张量的 width concat（dim=rank−1=2）被误判为 height：分片**堆叠在下方而非并排**，输出列 16–31 为零；大 shard 下还可能 L1 OOB。静默 in-bounds 错数据。
4. **tt-metal #55075** — layer_norm/rms_norm_pre_all_gather 2D core grid path: same wide-row L1 over-allocation, plus a hang at H=1024 W=1024 — https://github.com/tenstorrent/tt-metal/issues/55075
   `[已知类别]`（→ 1.133，var:1.105 同验证域）2D core grid 路径复现 #54697 的 per-row x² buffer 过分配 L1 超限，且 2D 路径在普通 shape 上**直接挂起**（hang）。
5. **vLLM #54785** — [Bug][XPU] XPU graph capture + MTP num_speculative_tokens=4 produce non-deterministic, wrong logits at temperature=0 — https://github.com/vllm-project/vllm/issues/54785
   `[已知类别]`（→ 1.134，var:1.110）XPU graph capture × MTP k=4 在 temperature=0 输出 2–3 个变体循环、top-1 logprob 摆幅 ~1.5 nats、40 请求后退化为重复；k≤3 干净、eager 干净、bs≥2 eager 回退干净、compile 无关——**边界恰为 k=4 的悬崖**。Battlemage XPU 持久状态损坏（1.110 token-0 损坏）在推测解码方向的新显形。
6. **tt-metal #55064** — test_model.py: cache_pcc is unreachable, so the KV cache PCC gate has never run — https://github.com/tenstorrent/tt-metal/issues/55064
   `[已知类别]`（→ 1.135，var:1.105）`cache_pcc = layers==1 and not dummy_weights` 在仅有的两个 parametrize 组合下恒 False——**KV cache PCC 门自引入（#21306）以来从未运行过**；块内还潜伏一个 per-layer 下标 bug（#53816 修）。
7. **vLLM #54698** — [Bug]: EngineCore hangs indefinitely in torch.xpu.graphs.replay() under concurrent load on Intel Arc B70 (Battlemage) — https://github.com/vllm-project/vllm/issues/54698
   `[已知类别]`（→ 1.136，var:1.110）Battlemage 单卡并发负载下 EngineCore 在 `torch.xpu.graphs.replay()` 无限挂起（无 crash、无日志推进）——XPU 持久状态损坏族的 liveness 显形。

### 第 3 类 · 比特翻转攻击（+1，arxiv）

8. **arXiv:2608.29745** — JITterFlip: Uncovering Fault Attack Surfaces in JIT-Compiled LLM Serving — https://arxiv.org/abs/2608.29745
   `[新类别/新签名]`（→ 3.18）首个针对 LLM 服务**宿主侧 JIT 控制平面**的 BFA：不翻模型权重、不翻执行计算的代码，而是翻转 CPU 驻留的 JIT 服务决策（artifact 选择/调度）——同时实现 gibberish 输出（PCL 比率 15.45×–2.48e6×）与 correct-output sponge（延迟放大 2.03×–181.90×），跨模型可迁移，绕过既有 BFA 防御；端到端 Rowhammer 单 bit 翻转实证。

### 第 4 类 · 数值 / 精度（+1）

9. **vLLM #54569** — [Bug]: FunASR get error result with fp16 dtype — https://github.com/vllm-project/vllm/issues/54569
   `[已知类别]`（→ 4.15，var:4.4）FunASR-Nano 语音模型 `--dtype float16` 下 ASR 转写输出无限重复 "locklocklock…"（token 退化循环）；H100 上复现，dtype 相关的 fp16 溢出/精度族在 audio 模型的新实例。

### 第 5 类 · 分布式 / 多 GPU（+1）

10. **vLLM #54723** — [Bug]: FlashInfer TRTLLM MoE Backend stuck under DP/EP (Kimi K2 base, DeepSeek v3 base) — https://github.com/vllm-project/vllm/issues/54723
    `[已知类别]`（→ 5.28，var:5.25）GB200 TP8/DP+EP 下 FlashInfer TRTLLM MoE 后端在 `_apply_quant_method` 路径 wedged（worker 停在 dummy_run/inductor 调用链，无进展、无 NaN）；建议临时切 triton/cutlass 后端规避——与 5.25（symm-mem NCCL 注册静默死锁）同为分布式 silent-wedge 族。

### 第 6 类 · 服务栈 / 缓存正确性（+6）

11. **vLLM #54796** — [Bug][XPU] inductor-compiled decode pieces × speculative decode livelock both GPUs at >=32k ctx — https://github.com/vllm-project/vllm/issues/54796
    `[新类别/新签名]`（→ 6.177）inductor 编译 decode 分段 × MTP 推测解码在 ≥32k 上下文双 GPU livelock（Compute+Copy Engine 100%、HBM ~80GB/s 空转、ttft 200s）；同电池下 whole-step graph capture + eager 内核完全干净——缺陷在 **inductor 编译区 × spec decode 交互**，非 graph capture 本身。
12. **SGLang #37326** — [Bug] NEXTN/MTP draft acceptance decays to ~0 over server uptime on qwen4_exp — https://github.com/sgl-project/sglang/issues/37326
    `[新类别/新签名]`（→ 6.178）draft acceptance 随服务器 uptime 约 24 小时衰减到 ~0（accept rate 0.47–0.76 → 0.00，吞吐 -2.9×），无 crash 无错误无警告，重启完全恢复且可复现——与 #21138（恒 0.33）/#19796（NaN crash）均不同：**慢性累积型状态退化**。
13. **vLLM #54831** — [Bug]: KV offloading impossible for GLM-5.3 (DSA indexer): tail_cache block_size=4 fails hash-alignment assertion — https://github.com/vllm-project/vllm/issues/54831
    `[已知类别]`（→ 6.179，var:6.168）GLM-5.3-Flash 稀疏 MLA indexer 的 `tail_cache` KV 组 block_size=4 无法对齐 prefix-hash 粒度（tokens_per_hash=8704），`--mamba-cache-mode align` 不达该组——offload 连接器配置 assert 拒绝；报错指向的 `--enable-prefix-caching` 是死胡同。与 6.168（hybrid 页对齐巨块坍塌）同"hybrid 模型多 KV 组对齐"域。
14. **SGLang #37263** — [Bug] /v1/responses silently drops prompt_cache_key instead of applying it to radix-cache identity — https://github.com/sgl-project/sglang/issues/37263
    `[已知类别]`（→ 6.180，var:6.169）OpenAI Responses API 的 `prompt_cache_key` 被 Pydantic 忽略未知字段静默丢弃，未桥接到原生 `cache_salt`——不同 caller cache key 的请求全部落入同一全局 radix-cache 命名空间（Codex 每 task 一个稳定 key 的语义失效）。6.169（LMCache cache_salt 丢弃）的 OpenAI 协议层同构。
15. **vLLM #54808** — [Bug]: qwen3_coder / qwen3_xml parser silently ignores tool_choice "required" and named function — https://github.com/vllm-project/vllm/issues/54808
    `[已知类别]`（→ 6.181，var:10.40）`tool_choice: "required"` 与具名函数在 qwen3_coder/qwen3_xml parser 下不被强制，请求成功、行为如同 `auto`，`strict:true` 无效、`VLLM_ENFORCE_STRICT_TOOL_CALLING` 默认 True 也无效——约束被静默降级。与 #50477（gemma4）/#49712（poolside_v1）/#51804（step3p5）同根因（不区分 parser 的横切缺陷）。
16. **vLLM #54726** — [Bug]: Qwen3.8-Flash-Next-FP8 + MooncakeStoreConnector fails at startup — https://github.com/vllm-project/vllm/issues/54726
    `[已知类别]`（→ 6.182，var:6.179）官方 recipe 声称支持 Mooncake 分布式 KV，实际启用即分级失败：`block_size must be divisible by hash_block_size` → 加 `--prefix-match-unit 8` 后又 `mamba_cache_mode != 'align'` 不支持——hybrid GDN 模型 offload 路径配置断言链。

### 第 7 类 · 量化（+2）

17. **vLLM #54765** — [Bug]: Qwen3.8-Flash-Next: ModelOpt NVFP4 checkpoint with FP8-quantized PLE n-gram table fails to load — https://github.com/vllm-project/vllm/issues/54765
    `[已知类别]`（→ 7.66，var:12.25）NVFP4 主检查点 + FP8 量化 PLE n-gram 表（`model-plefp8-*` 独立分片、全局 `ngram_embedding.weight_scale`）恰为 `Qwen4ExpPLEFp8EmbeddingMethod` 设计布局，但该方法因忽略列表/量化方法选择逻辑从未被选中——混合量化 checkpoint 加载路由缺失（7.65 同周第二例、12.25 NVFP4 反量化丢 global scale 的同族）。
18. **SGLang #37342** — [Bug] DeepSeek-V4-Flash-0731 fails to load on v0.5.17/v0.5.18: Hidden size mismatch in mxfp4 MoE path — https://github.com/sgl-project/sglang/issues/37342
    `[已知类别]`（→ 7.67，var:7.57）混合量化（fp8 attention/dense + I8+UE8M0-scale mxfp4 packed routed experts）在 tag 版本 v0.5.17/0.5.18 的 `flashinfer_mxfp4` runner 上 hidden-size assert 崩溃，main（2026-08-30）已修但未发布——**修复存在但发布链未携带**（6.156 uv resolver 回退复活族的模式：用户拿到的版本仍坏）。报告者自行撤回了两个错误主张，仅此一项成立。

### 第 8 类 · KV 缓存 / 注意力（+2）

19. **tt-metal #55102** — [Gemma4/Wormhole] Deterministic garbled long-prose output at specific encoded lengths >=100k — https://github.com/tenstorrent/tt-metal/issues/55102
    `[新类别/新签名]`（→ 8.60）Wormhole T3K 上 Gemma4（12B/31B）在特定编码长度（100,793 / 109,857 / 127,574）**确定性**产出 garbled 输出（'s la's 循环 / token salad），同长度 Blackhole 全干净；**非单调**（100,462 clean → 100,793 corrupt → 104,954 clean）；path 无关（metal demo 与 vLLM serving byte-identical）、chunk-invariant、build-stable——WH 特定长度窗口的前向损坏，非服务层状态。
20. **vLLM #54740** — [Bug]: [XPU] gdn_attention asserts on ragged ngram draft lengths (single stream) — https://github.com/vllm-project/vllm/issues/54740
    `[已知类别]`（→ 8.61，var:6.172）XPU `gdn_attention` 内核要求 batch 内 draft 长度均匀，ngram 部分匹配产出 ragged 长度 → `Expected spec_token == num_spec_decodes * (k+1)` assert、EngineDeadError——单流即可触发；是 6.172（PD Mamba spec padding 窗口穿过 align 裁剪）的 ragged-draft 显形面在 XPU 路径。

### 第 10 类 · 分词器 / 对话模板（+2）

21. **SGLang #37283** — [Bug] A non-object entry in a tool-call array discards the valid calls beside it — https://github.com/sgl-project/sglang/issues/37283
    `[已知类别]`（→ 10.44，var:10.42）tool-call 数组含一个非 object 条目（`"junk"`/`[1,2]`/`null`）时，`parse_base_json` 的 `act.get("name")` 抛 AttributeError，把旁边**格式良好的有效调用一并静默丢弃**——一条畸形邻居让整个合法 tool call 消失。10.42（parser 通道终止符假设静默丢 tool call）的相邻故障模式。
22. **vLLM #54739** — [Bug]: Qwen3.8-Flash-Next generates corrupted Thai orthography under vLLM; llama.cpp at same QSA budget is clean — https://github.com/vllm-project/vllm/issues/54739
    `[新类别/新签名]`（→ 10.45）vLLM 下 Qwen3.8-Flash-Next 系统性产出畸形泰文（声调标记错位/双写 combining mark/字符替换，12–18 处/1000 字符），同预算 llama.cpp 零缺陷；已排除 #54521 的 persistent_topk 非确定性与 QSA block 选择——**栈特定（vLLM-only）的 tokenizer/预处理正字法损坏**，泰文作为高敏感检测器（~3× token/字符、近同 combining mark 词元）放大可见。

### 第 11 类 · 解码 / 采样（+3）

23. **vLLM #54843** — [Bug][CPU][Spec Decode]: --use-fp64-gumbel silently ignored in CPU recovered-token sampling — https://github.com/vllm-project/vllm/issues/54843
    `[已知类别]`（→ 11.58，var:4.13 同"保证 flag 静默失效"域）CPU shim 接受 `USE_FP64_GUMBEL` 后在 argmax 前 `inv_q.to(torch.float32)` 降精度，注释声称相反——flag on/off 行为逐位相同；lower-tail draw q≈1e-40 在降转时 `inv_q` 溢出 inf、微概率 token 窃取 argmax（fp64 应答 1、实答 0）。GPU Triton 内核遵守 flag——**CPU/GPU 同 draw 分歧**，PR #43150 建立的 parity 在 CPU 不成立。
24. **SGLang #37475** — [Bug] DFlash target_layer_ids fallback guesses wrong tap layers; count collision fails silently — https://github.com/sgl-project/sglang/issues/37475
    `[已知类别]`（→ 11.59，var:11.41）draft config 缺 `target_layer_ids` 时 fallback 公式猜出的 tap 层数恰好相同但层号不同（[1,16,31,46,61] vs 训练的 [5,19,33,47,61]）——权重干净加载、无任何报错，draft **静默读错目标层**、acceptance 塌缩；20 个已发布 checkpoint 全设此键故今日为死代码，但手写 config/格式变更即触发。DFlash 系（11.41）第 4 例。
25. **vLLM #54690** — [Bug]: Speculative kv_cache_dtype=fp8 crashes startup on hybrid GDN models — https://github.com/vllm-project/vllm/issues/54690
    `[已知类别]`（→ 11.60，var:7.52）`--speculative-config {"kv_cache_dtype":"fp8"}` 的 draft-only KV dtype 以 raw `auto` 传入 FlashInfer metadata builder → `Unrecognized dtype: auto` ValueError；显式设 target bf16 绕过后又在 SM89 attention init **原生崩溃**（无 Python traceback）——spec 配置的 dtype 语义未做 target/draft 分别的归一化。

### 第 12 类 · 权重 / 适配器 / 检查点加载（+3）

26. **tt-metal #55014** — tt_dit weight cache silently drops non-Parameter module state — https://github.com/tenstorrent/tt-metal/issues/55014
    `[已知类别]`（→ 12.27，var:12.15）`Module.save/load` 只序列化 `named_parameters()`，`_prepare_torch_state` 内从 checkpoint 派生的宿主侧值（非 Parameter）不进 weight cache、load 时不重算——**静默回退构造器默认值**；已在 LTX-2.5 text encoder 上毁掉 prompt adherence，树中已有 4 处独立 workaround 而框架仍允许该失败模式。
27. **vLLM #54839** — [Bug]: /v1/unload_lora_adapter never removes the adapter from the engine — https://github.com/vllm-project/vllm/issues/54839
    `[已知类别]`（→ 12.28，var:6.149）`unload_lora_adapter` 只在前端 `del lora_requests[lora_name]` 并打日志"Removed"，从不调 `AsyncLLM.remove_lora`——adapter 仍驻留 worker GPU slot/CPU cache 直到 LRU 逐出；名称槽位泄漏使 unload 后同名 reload 行为错乱。6.149（sleep level=1 wake 损坏 LoRA 状态）的"控制面与引擎状态不同步"同族。
28. **vLLM #54775** — [Bug]: KDA/gated-delta-rule chunked-scan buffers are outside memory profiling — https://github.com/vllm-project/vllm/issues/54775
    `[已知类别]`（→ 12.29，var:6.136）KDA chunked-scan 中间 buffer 随 `--max-num-batched-tokens` 线性增长、在 forward 内瞬态分配、**不在启动期 memory profiling 覆盖内**——KV cache 按该内存可用来定容，8192–65536 稳定、81920 起 16 并发 38k 请求 11/16 运行中 OOM 引擎关闭。6.136（hidden-states KV budget 高估启动拒绝）的"profiler 盲区瞬态分配"运行期形态。

### 第 15 类 · 幻觉 / 忠实性（+1，arxiv）

29. **arXiv:2608.29934** — Compression-Aware Abstention: Teaching LLMs to Refuse When KV-Compression Masks Remove Answer Evidence — https://arxiv.org/abs/2608.29934
    `[新类别/新签名]`（→ 15.9）首个把「KV 压缩逐出答案证据 → 模型幻觉而非识别上下文不足」公式化为**学习问题**：用 compressor survival mask + 答案承载 span 构造 Confident/Abstain 监督，10.1M LoRA 在 ~2.6K 样本上训练——截断式幻觉 -97%、多压缩器联合训练在真实压缩解码下对证据保留样本相对提升 6–22×；受控删除实验证明习得行为由证据内容而非长度驱动。为 K§6（驱逐/压缩）故障族补上行为面（弃权）缓解路径，并证实「逐出证据性 token → 幻觉」因果链。

### 补充来源（不占新编号）

- **arXiv:2608.30769 TrainSDC** — LLM 训练 SDC 跨前向/反向计算接口的系统刻画 + Q/K 路径重算、residual-gain 监控、指数感知梯度缩放防护——追加为 §3.1（1.1/1.5 研究刻画类）补充来源：Q/K 路径位置依赖的持续训练偏差结论与 A* 注入位图选择直接相关。

---

## 二、查重后跳过/仅记录不编目的条目

- **vLLM #54829 / #54749 / #54704 / #54766 / #54777 / #54779 / #54712 / #54677** — feature/RFC/perf/OOM 类（metrics RFC、dynamic-SD RFC、fp8 KV for Inkling RFC、GDS offload RFC、dummy load OOM、吞吐回归），无正确性故障机制，不编目。
- **vLLM #54740 已编目**（→ 8.61）；**#54785 已编目**（→ 1.134）；**#54723/#54726** 已编目（→ 5.28/6.182）。
- **SGLang #37405 / #37372 / #37349** — 测试清理 RFC / HiCache IPC feature / V100 指南，非故障。
- **tt-metal #55035 / #54965 / #54962 / #54968 / #54974 / #54975 / #55015 / #54952** — 测试组织/bring-up/微基准/月度活动/性能计数器命名（非正确性数值语义）——不在编目范围。
- **triton #11526** — tf32x3 GEMM SMEM 用量回归（3.6→3.8），性能回归非正确性。
- **vLLM #54671 / #54670 / #54667 / #54665 / #54657 / #54664 / #54607** — 窗口外（8-30 前后）或属断言/除零 fail-loud 配置校验类（ZeroDivisionError 直炸），非静默正确性损坏；其中 #54664（dsv4 flash PD 混部流式 usage 默认输出）为行为语义争议非损坏，暂不编目。
- **arxiv RAP: KV-Cache Compression via RoPE-Aligned Pruning (2602.02599)** — 压缩方法论文（K§6 相关但非故障刻画），不收录（与 Moment-KV/TurboQuant 统计论文同判）。
- **arxiv Continuous Discovery of Vulnerabilities in LLM Serving Systems with Fuzzing (2605.11202)** — 模糊测试方法论文，无新故障签名，不收录。

---

## 三、重点新发现摘要

1. **JITterFlip（3.18）** — 把 BFA 攻击面从权重/计算代码扩展到**宿主侧 JIT 服务控制平面**，Rowhammer 单 bit 实证 + 跨模型迁移 + 绕过既有防御，是 BFA 家族的全新子类（正确输出海绵 = 活性面攻击、gibberish = 正确性面攻击，二者由同一控制平面故障分别达成）。
2. **SGLang #37326（6.178）** — 首例"**draft acceptance 随 uptime 慢性衰减到 0**"：无 crash/NaN/警告、重启即愈、可复现——服务栈累积状态退化的新时间维度签名，监控面需关注 accept rate 时间序列而非阈值。
3. **tt-metal #55102（8.60）** — WH 特定编码长度（≥100k 非单调窗口）的**确定性**前向损坏、跨 metal/serving 路径 byte-identical、BH 同长度干净——"长度门控硬件特定损坏"新签名。
4. **vLLM #54796（6.177）** — inductor 编译区 × spec decode 交互 livelock（graph capture 单独干净），编译优化与推测解码组合的新活性故障面。
5. **vLLM #54739（10.45）** — vLLM-only 的泰文正字法系统性损坏（llama.cpp 同预算零缺陷），多语言正字法作为栈特定预处理损坏的敏感检测器。
6. **tt-metal #55106（1.130）** — 已修复 LLK 同步缺陷在 16 文件 33 站点的规模化残留——"修复传播不完整"从个案升级为系统性模式（1.122/1.129 之后的第三例）。

---

## 四、API 调用记录

| # | 查询 | 命中 | 状态 |
|---|---|---|---|
| 1 | vllm correctness/wrong/corruption/silent/NaN (created:>=2026-08-31) | 32 | OK |
| 2 | vllm quant/fp8/KV cache | 60 | OK |
| 3 | sglang correctness/wrong/garbage/NaN/corruption | 9 | OK |
| 4 | transformers wrong output/NaN/rope/tokenizer inference | 0 | OK |
| 5 | TensorRT-LLM correctness/wrong/NaN/corruption/output | 2 | OK |
| 6 | triton wrong/incorrect/NaN/miscompile | 1 | OK |
| 7 | tt-metal numerical/pcc/wrong/NaN/corruption | 24 | OK |
| 8 | vllm bit flip/determinism/SDC/ECC | 28 | OK |
| A1–A5 | arxiv: SDC+LLM / bit flip+LLM inference / KV cache+corruption / LLM inference+correctness / GPU soft error+ML | 5+5+5+5+3 | OK |

core REST 详情调用 34 次全部成功；无 rate-limit 事件。

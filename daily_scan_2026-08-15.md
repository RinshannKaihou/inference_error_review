# LLM 推理故障每日扫描报告 — 2026-08-15

- 扫描日期：2026-08-15（UTC 23:00 执行）
- 滚动窗口：GitHub issue/PR `created:>2026-08-08`（7 天）+ arxiv 最近提交
- 目标仓库：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal
- 基线：FAULT_MASTER_REFERENCE.zh.md 合并前共 **527 条**（第 0 层 17 + 第 1 层 396 + 第 2 层 114）

## 汇总

共评估 30+ 条候选，**21 条确认为新条目合并进主文档**（全部为新子签名，Cov `NEW`），另有 2 条作为已有条目的补充来源引用（不占新编号）。合并后主文档共 **548 条**。

分布：第 1 类 +2、第 5 类 +2、第 6 类 +3、第 7 类 +2、第 8 类 +3、第 10 类 +1、第 11 类 +5、第 12 类 +2、第 13 类 +1。

## 重点新发现（新类别/新签名）

### 🔴 静默输出损坏类（无 crash、无 NaN，最难检测）

1. **[新签名] vLLM #52475 — MTP 推测解码 + turboquant 量化 KV cache 在 sm120 上长 prompt 重复坍塌** → 合并为 11.34
   - Qwen3.8-27B-NVFP4（hybrid GDN）+ MTP：KV dtype=fp8 正常，`turboquant_4bit/3bit_nc` 时 ~8k token 以上 prompt 退化为单 token 重复 ~230 次；短 prompt（"2+2"）答对 → 极易漏检。服务器健康启动、KV pool 正常。
   - https://github.com/vllm-project/vllm/issues/52475

2. **[新签名] vLLM #52413/#52476 — causal_conv1d 对齐 specialization + 共享 Triton cache 写竞态 → 静默 all-NaN** → 合并为 6.96
   - metadata 指针的 16-byte alignment specialization 使 unaligned 变体必在推理期首次编译；多节点共享 `TRITON_CACHE_DIR` 时 cache 写竞态损坏二进制 → GRPO RL（6×GB200, Nemotron-3-Super-120B）静默 all-NaN 输出。
   - https://github.com/vllm-project/vllm/issues/52413

3. **[新签名] vLLM PR #52383 — Inductor pad alias view 在 CUDA graph replay 时冻结为零 → MoE 语义静默损坏** → 合并为 6.98
   - `F.pad` 在 trace 区被 Inductor 优化为 stride-remap alias view，喂给 opaque `vllm::moe_forward` custom op 后 graph replay 读到捕获时刻冻结的零（`hidden_states` 全零、`router_logits` 真实）。与 7.35（#52308）为同一 issue 的第二根因。
   - https://github.com/vllm-project/vllm/pull/52383

4. **[新签名] vLLM #52442 — ROCm Kimi-K3 SiTU MoE 双 flag 错配 → 静默 garbage** → 合并为 7.36
   - `AITER_SITUV2_A8W4`（库侧选 interleave kernel）与 `VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4`（vLLM 侧 weight shuffle）独立命名；只设前者 → interleave kernel 读 separated 布局权重 → `2+2=` → `' 0. 4. 0. 0. 4'`，无任何告警。
   - https://github.com/vllm-project/vllm/issues/52442

5. **[新签名] transformers #47970/#47331 — MPS MetalConfig 量化 kernel 2D 假设 × 3D batched 输入 → batch>1 时后续序列损坏** → 合并为 7.37
   - `MetalLinear.forward` 直接把 [B,S,in] 传给假设 2D 的 `kernel.affine_qmm_t`，只算第一个 batch 元素。
   - https://github.com/huggingface/transformers/pull/47970

6. **[新签名] vLLM #52469 — MTP 推测解码流式输出在 quote/byte-merge 边界间歇丢词（raw /v1/completions 也复现，非 parser bug）** → 合并为 11.35
   - https://github.com/vllm-project/vllm/issues/52469

7. **[新签名] vLLM PR #52452/#43338 — 推测解码 + 结构化输出：scheduler 在 grammar 校验前 commit accepted blocks → 不变量破坏** → 合并为 11.37
   - https://github.com/vllm-project/vllm/pull/52452

8. **[新签名] vLLM #52461 — async scheduling 下 stateful logits processor 读到 -1 placeholder（修复机制只统计 CLI 注册的 processor）** → 合并为 11.36
   - https://github.com/vllm-project/vllm/issues/52461

### 🟠 硬件内核原语缺陷（tt-metal）

9. **[新签名] tt-metal #53267 — `global_avg_pool2d` 对 non-tile-divisible 形状 [1,7,7,2048] 输出结构性错误值（PCC 0.55，确定性）** → 合并为 1.76
   - https://github.com/tenstorrent/tt-metal/issues/53267

10. **[新签名] tt-metal #53196 — `interleaved_to_sharded` dispatch cache hash 键域（padded_shape）宽于参数域（logical shape）且命中后不重放 RTA → 静默错误 row stride** → 合并为 1.77
    - https://github.com/tenstorrent/tt-metal/issues/53196

11. **[新签名] tt-metal #53207 — sdpa_fw `AttentionMaskType::None` 死锁 + WH fw/bw softmax scale 不一致 + fully-masked rows NaN** → 合并为 8.40
    - https://github.com/tenstorrent/tt-metal/issues/53207

### 🟡 驱动/分布式层

12. **[新签名] SGLang #34641 — KFD PeerDirect 多 NIC 注册同一 buffer → 显存记账损坏（free 超物理总量）→ KV pool 过度分配 OOM** → 合并为 5.15
    - amdgpu/KFD 驱动层缺陷，最小 HIP+ibverbs 复现（MI355X）。
    - https://github.com/sgl-project/sglang/pull/34641

### 📚 arxiv 论文

13. **[新签名] arXiv:2606.07571 — DLM（扩散语言模型）双向注意力违反「KV 一旦计算即不变」假设，LLM 共享前缀缓存技术直接迁移会 corrupt 前缀 KV** → 合并为 8.41
14. **[新签名] arXiv:2606.28958 — 多智能体系统把 KV-cache 隐藏状态作为跨 agent 信任通道传递，可被篡改且文本审计不可见** → 合并为 8.42
15. **[新签名] arXiv:2607.19490 — P2P 分布式推理中恶意节点篡改其层输出静默腐蚀最终结果（含 Hollow-LLM arXiv:2607.28884 作为相关引用）** → 合并为 5.16
16. **[已知类别补充来源] arXiv:2604.10390（LLM-PRISM）** — 永久 GPU 故障致 LLM 训练 SDC 的 RTL 级刻画 → 引用进 §3.1（1.1）补充来源，不占新编号

## 已知类别的新案例

17. **vLLM #52465** — Muse Glimmer reasoning parser 截断时丢弃已完成 reasoning block、流式/非流式不一致 → 合并为 10.20（分词器/解析器领域新子签名）
18. **vLLM #52398/#52399** — `/inference/v1/generate` 在 `n>1` 时静默丢弃 choices → 合并为 6.97（服务栈响应组装）
19. **vLLM #52454/#52474** — Quark 结构化量化配置 list 被 module-name mapper 处理 → 加载 crash（`AttributeError: 'dict' object has no attribute 'endswith'`）→ 合并为 12.17（量化配置加载）
20. **vLLM #52434/#52451** — `CompressedTensorsW8A8Fp8.create_weights` 未初始化 `ParallelLMHead` partition metadata → 加载 crash → 合并为 12.18（权重加载）
21. **vLLM #52445** — Kimi-K3 MegaMoE 对 DeepGEMM pinned commit 的 kwarg 名错配（`activation_beta` vs `situ_beta`）→ 首次 forward `TypeError` crash → 合并为 13.28（模型集成/图）
22. **vLLM PR #52436** — DSpark adaptive verification `draft_budget==0`（常规选择而非边角）× structured outputs grammar mask 映射 → assert crash → 合并为 11.38（解码/采样）
23. **vLLM PR #52470** — MooncakeStoreConnector preemption 时 abort 在飞 async store 任务 → 作为 6.93 的补充修复来源引用（不占新编号）

## 已在编目（去重跳过）

- SGLang #34899（Unified Radix Cache bit-exact 覆盖——测试基建，对监控有参考价值，不合并）、#34629（7.34 已编目）、#34392（已编目）、#34297（已编目）、#34189（8.25 补充来源已引用）
- transformers #47885/#47894（已编目）、#47942（已编目）
- TRT-LLM #17699（已编目）
- arxiv：SCOUT 2608.11034、Exploring SDC 2604.00726、Understanding SDC 2502.12340、Decision-Level Hijacking 2607.25227、UEP 2607.19623 均已引用

## 记录但不合并（相关性不足）

- TRT-LLM #17561 — B200 `kda_verify` NaN test waiver（信号弱，无独立机制细节）
- triton #11279 — AOT `--target` flag 类型错误（非数值正确性）
- tt-metal #53186 — sparse matmul 测试期望值修正；#53038 — CI hugepage NOC 地址不匹配 flake（基础设施）
- vLLM #52460 — Mamba cache 'all' 模式回退 'align'（鲁棒性修复，无正确性后果细节）
- arxiv 2608.05863（Runtime Observability for Heterogeneous Attention Memory）— 监控/可观测性契约论文，与本项目监控使命相关，暂作参考不编目
- 全局 "silent data corruption" 搜索 91 条均为非 LLM 仓库（数据库/内核/文件系统），无推理相关新条目

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GitHub search | vllm wrong/incorrect/corruption/NaN/silent (7d) | 241 hits，取 20 |
| 2 | GitHub search | sglang 同关键词 (7d) | 6 hits |
| 3 | GitHub search | transformers 同关键词 (7d) | 6 hits |
| 4 | GitHub search | TensorRT-LLM 同关键词 (7d) | 2 hits |
| 5 | GitHub search | triton wrong/incorrect/numerical (7d) | 1 hit |
| 6 | GitHub search | tt-metal numerical/correctness/pcc/mismatch (7d) | 5 hits |
| 7 | GitHub search | vllm label:bug (6d) | 274 hits，取 15 |
| 8 | GitHub search | "silent data corruption" in:title 全局 | 91 hits，无 LLM 相关 |
| 9–25 | GitHub issues API | 逐条拉取 #52475/#52469/#52461/#52442/#52413/#52398/#52465/#52436/#52451/#52445/#52383/#52452/#52434/#52474、transformers #47970、sglang #34641、tt-metal #53267/#53207/#53196/#53038 正文 | 全部成功 |
| 26–30 | arxiv API | silent data corruption LLM / bit flip LLM / KV cache corruption / LLM inference correctness / GPU soft error | 各 5–6 条，`curl -sL` 跟随重定向 |

无 rate limit 错误（HTTP 403 未出现）。

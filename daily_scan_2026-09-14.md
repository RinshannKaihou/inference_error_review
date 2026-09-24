# LLM 推理故障每日扫描 — 2026-09-14

- **扫描日期**：2026-09-14（UTC 23:00 执行）
- **扫描窗口**：GitHub `created:>=2026-09-12`（上次扫描 09-13 覆盖 `>=2026-09-11`，本窗口重叠 09-12/09-13 两天，已逐条对主文档去重；arxiv 按提交日期取最新）
- **基线**：FAULT_MASTER_REFERENCE.zh.md 共 **1073 条**（1.1–15.10 共 959 + K1.1–K10.31 共 114；2026-09-14 重构后无 A* 前缀条目）

## 结果概览

- GitHub Search API：11 组查询全部成功（q02 首次返回 Unicorn HTML 错误页后重发成功；无限流 403）
- core REST 详情：16 次调用，15 成功、1 次 403（#56653，bench 指标类不收录，无影响）
- **arxiv：5 次尝试全部 IP 级 "Rate exceeded"（含 60s 退避 + UA 重试），记录跳过**——连续第二天同症状（09-13 亦全 429），建议下次补查
- 合并：**9 条新增**（第 1 类 +4、第 4 类 +1、第 6 类 +1、第 7 类 +1、第 8 类 +1、第 11 类 +1 = 5 个新子签名 + 4 个已知类别变体），1073 → **1082**；另有 1 条补充来源（vLLM PR #55738/#56896 → 8.84）

## 🆕 新故障模式 / 新子签名（本日合并）

| # | 条目 | 归类 | 为什么是新签名 |
|---|---|---|---|
| 1 | [vLLM #56792](https://github.com/vllm-project/vllm/issues/56792) DSpark fill-in 布局 checkpoint 缺可选布局旗标 → 默认 anchor-first 采样器静默服务，接受率 0.9% 无任何报错/警告 | 第 11 类 11.77 | **draft checkpoint 布局元数据缺失 → 服务端默认值静默选错采样语义**：布局旗标（`sample_from_anchor`）是可选字段，旧导出路径不写；加载器无探测手段时按默认方向"解析"歧义布局。与 12.31/12.35（量化 DFlash draft 0% 接受）同症状但根因在**配置模式缺省方向**，非数值损坏 |
| 2 | [vLLM #56868](https://github.com/vllm-project/vllm/issues/56868) GLM-5.3-Flash 长 decode 退化：`.0.0.0.` 重复低熵崩塌；关键证据——**76 个完整请求的累计 decode/reasoning token 是触发条件**（prefill-only 预热 276 万 prompt token 后第 77 个请求仍健康；同一请求冷重放健康） | 第 6 类 6.235 | **累计 decode 深度依赖的迟发退化**：非上下文长度、非 KV 复用、非采样器状态——是跨 76 请求 ~37 万 decode token 累计后进入的状态；SGLang 同病（栈无关）；MTP/prefix cache 双开关无效 |
| 3 | [tt-metal #56550](https://github.com/tenstorrent/tt-metal/issues/56550) `chunk_gated_delta_rule` 衰减因子拆分 `exp(g_sum−decay)·exp(decay)` 在 FTZ/DAZ 下 `+∞×0=NaN` 毒化整个 recurrent state S（或有限×0=0 全抹除记忆）——数学值 `exp(g_sum)` 本身完全正常 | 第 1 类 1.214 | **代数恒等式拆分 × FTZ 边界**：中间量单独溢出/下溢而真值正常；gated-delta-rule（KDA/FLA 家族）SSM 硬件内核新 locus |
| 4 | [tt-metal #56518](https://github.com/tenstorrent/tt-metal/issues/56518) `ttnn.pow` fp32：`z = m*recip − recip` 用部分融合 SFPMAD（4 extra bits）构造约化变量，近 1.0 底数 catastrophic cancellation——3 ULP 承诺差 5 个数量级（3e6 指数 4.37% 相对误差）；注释声称"等价"的减法形式按 Sterbenz 引理本可精确 | 第 1 类 1.216 | **部分融合 FMA 的融合度幻觉**：ISA 文档明示 SFPMAD 只带 4 extra bits，开发者按真 FMA 无限精度假设改写；与 1.118（KDA 倍增求逆）同族相消但根因是**硬件融合指令精度假设** |
| 5 | [tt-metal #56522](https://github.com/tenstorrent/tt-metal/issues/56522) `ttnn.where` fp32：谓词按 INT32 位模式判零，`-0.0`(0x80000000) ≠ 0 选 true 分支——与自家 golden `torch.ne(pred,0)` 分歧 | 第 1 类 1.217 | **−0.0 符号位 vs 数值判零**：ISA 明文要求软件先冲刷 −0.0，内核未做；1.109（比较器 NaN/±0 dtype 分裂）族新维度（where 三元分支方向翻转） |

## 🔄 已知类别的新案例（本日合并）

| # | 条目 | 归类 | var 关系 |
|---|---|---|---|
| 6 | [tt-metal #56517](https://github.com/tenstorrent/tt-metal/issues/56517) `ttnn.gelu(variant=Tanh)` 负尾 `1+tanh(u)` 相消：x<−6 恒 0、[−6,−4] 达 200% 相对误差（bf16/fp32 双精度）；#56490（erf 变体）之外**独立**的第二个 GELU 内核 | 第 1 类 1.215 | var:1.162（logsigmoid SFPU 分支缺失）同族"近似激活负尾相消/分支缺失"；与 09-13 已编目 #56490 平行 |
| 7 | [vLLM #56770](https://github.com/vllm-project/vllm/issues/56770) compressed-tensors **MXFP4 W4A16**（`input_activations:null`）在 SM100+ 被无条件分派 `CompressedTensorsW4A4Mxfp4`——权重格式单变量判别吞掉激活量化元数据 | 第 7 类 7.89 | var:7.58（mxfp4 MoE runner else 吞 9 值 fail-late）族：mxfp4 分派判别不完整 |
| 8 | [SGLang #39300](https://github.com/sgl-project/sglang/issues/39300) TorchNativeAttnBackend 非因果（encoder 双向）滑窗注意力被硬编码因果掩码静默忽略 → 全局注意力（ModernBERT 族）；修复 PR #39298 | 第 8 类 8.86 | var:8.51（CPU AMX 强制 causal mask 双向语义静默反转）族 |

## 补充来源（不占新编号）

- **vLLM PR #55738**（merged 2026-09-14）：GLM-5.3-Flash NoPE MLA prefill allow-list + `k_pe` 零宽 concat 短路；#56896 为旧树移植文档 → 并入 **8.84**（SGLang #39302 同族：NoPE MLA 后端可用性矩阵）。

## 本窗口内其余命中——已编目（跳过）

- vLLM #56389 → 6.233；SGLang #39147 → 6.234；#39193 → 7.85；#39311 → 7.86；#39271 → 6.232；#39235 → 4.20；flashinfer #5190 → 7.87；triton #11735/#11733 → 1.211；tt-metal #56414 → 1.207；#56424 → 09-13 不收录维持（TT_FATAL fail-loud）

## 本窗口内其余命中——本日判定不收录（附理由）

| 条目 | 理由 |
|---|---|
| vLLM #56900 Qwen1.5-MoE torch.compile 退化输出 | 症状仅限 2024 年代老 MoE 架构 × VLLM_COMPILE；与 1.181（CPU MoE compile NaN）族重叠、量级更小；留观 |
| vLLM #56889 HiSparseConnector GLM-5.2 DEP8 输出不连贯 | 无根因定位（MultiConnector+EP+fp8KV+Rust 前端组合过多变量）；GSM8K 0% 但未隔离；留观 |
| vLLM #56887 moe_wna16 split-K 16-bit atomicAdd 非确定 | 与 4.11（index_add 原子非确定）同机制（原子浮点顺序非确定），wna16 路径新实例不占新编号；flashinfer #5221（b12x_fused_moe 非确定）同族——两者作为 4.11 补充语境记录于此 |
| vLLM #56815 Engram async prefetch 回归 decode flake | 修复回归追踪（#56512 reintroduces），吞吐抖动非性能；留观 |
| vLLM #56830 内存剖析 assert（co-tenant 释放内存） | fail-loud 启动 abort，无静默损坏；治理类 |
| vLLM #56871 hybrid KV offload OffloadKey 每组一行容量÷组数 | 容量核算效率缺陷（永不命中），非性能正确性；与 6.136 族相邻但无损坏；留观 |
| vLLM #56894 offload namespace cache_dtype 随 executor 分歧 | 缓存键漂移导致跨引擎零共享（性能损失），无损坏；留观 |
| vLLM #56797 DSpark 接受长 2.82 vs golden 3.51 | 自证"非正确性失败"（greedy 93.4% 正常）；性能回归；与 11.77 症状相邻但非静默损坏 |
| vLLM #56790 RDNA3 fused MoE 硬编码 2x gated 因子 | fail-loud shape assert（336 vs 672）启动即崩；#44489 同概念；无静默形态 |
| vLLM #56774 extract_hidden_states 混合模型 KV 预算 24× | 容量核算（6.136 族性能面）；本地分支（V2 ported #49811）未对上游复现 |
| vLLM #56759 Humming MoE permute scratch 随层数缩放 | 内存治理；无正确性 |
| vLLM #56696 OTLP tracer 不发 span / #56668 CPU 单测 / #56653 bench 并发计数 | 观测/测试/bench 工具类（09-13 已判维持） |
| SGLang #39400 Bootstrap PUT /route 未鉴权 | 安全面（路由投毒/DoS），非数值正确性；超出编目范围 |
| SGLang #39299 MoE deferred finalize 不可达 | 功能可达性（无损坏路径）；性能 |
| flashinfer #5175 mm_mxfp8 K%128 回退 3.4× 慢路径 | 性能回退（无错值） |
| flashinfer #5192 DSV4 双缓存 prefill 模板缺失 | fail-loud "Unsupported configuration" 启动拒绝；与 8.84 族相邻但已由 09-13 覆盖族谱 |
| flashinfer #5213/#5209/#5184/#5162 | CI/feature/tracking |
| tt-metal #56532 FFT bf16 截断舍入 | bf16 精度偏差（−0.27%~−2.4% 系统性）——信号处理域、非 LLM 推理路径；数值族（1.207 族外延）留观 |
| tt-metal #56521 ISCLOSE 测试基建 mask 未编程 | 测试基建缺口（生产 isclose 不受影响） |
| tt-metal #56384 fmod host-rounded reciprocal | **撞号注意**：主文档 09-13 "不收录"清单中的 #56384 为 vLLM #56384（测试竞态）；本条 tt-metal #56384（fmod(x,0.1)≡0、|divisor|<2.9e-39 → +inf）为不同 issue。评估：host 倒数近似商缺陷与已编目 1.133（fp32 标量 remainder 静默错一个除数整数倍——近似商精度预算耗尽）机制高度重合（host-rounded reciprocal 正是近似商来源），作为 1.133 族**语境补充**记录于本报告，不另立条目 |
| tt-metal #56520 logical_not(−0.0)=0 | 与 #56522（已并合为 1.217）同根因同 ISA 契约（#43831 六算子修复未覆盖），并入 1.217 记述，不重复立条 |
| triton #11747 tcgen05 consan abort | 09-13 已判维持（fail-loud LLVM abort） |
| q09 全域 "silent data corruption" 搜索 | 全部为无关仓库（digest/PR 噪声），无 LLM 推理命中 |
| transformers / TensorRT-LLM 窗口 | 0 命中（q04/q05 total_count=0） |

## API 调用记录

| # | 查询 | 结果 |
|---|---|---|
| q01 | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-12` | 18 命中 |
| q02 | `repo:vllm-project/vllm is:issue created:>=2026-09-12 wrong in:title` | 首次 Unicorn HTML 错误页（非 JSON）；重发成功 1 命中（#56792） |
| q03 | `repo:sgl-project/sglang ... correctness OR wrong OR corruption` | 10 命中 |
| q04 | `repo:huggingface/transformers ... wrong output` | 0 |
| q05 | `repo:NVIDIA/TensorRT-LLM ... correctness OR wrong` | 0 |
| q06 | `repo:triton-lang/triton ... NaN OR miscompile OR wrong` | 3 |
| q07 | `repo:tenstorrent/tt-metal ... incorrect OR NaN OR numerics` | 18 |
| q08 | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-12` | 8 |
| q09 | `"silent data corruption" LLM created:>=2026-09-12` | 22（全部无关噪声） |
| q10 | `repo:vllm-project/vllm ... quantization OR KV cache OR NaN` | 51 |
| core REST ×16 | vLLM #56900/#56889/#56868/#56815/#56871/#56790/#56770/#56797 + tt-metal #56550/#56532/#56522/#56520/#56518/#56517 + SGLang #39300 + vLLM #56653(403) | 15 成功 / 1 限流 |
| arxiv ×5 | `"silent data corruption" AND LLM`（含 20s×3 退避 + 60s+UA 重试） | 全部 "Rate exceeded"（IP 级限流），跳过 |

# 每日扫描报告 — 2026-09-13

- **扫描窗口**：GitHub `created:>=2026-09-11`（覆盖 09-11 至今，含 09-12 缺跑日的缺口），arxiv 最近提交（当日被限流未完成，见下）
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **结果**：新发现并合并 **18 条**（8 个新子签名 `NEW` + 10 个已知类别变体 `var:`），主文档 1058 → **1076**
- **昨日已编目条目**（今日再现均查重跳过）：vLLM #56417（11.76）、#56396（09-11 已判 fail-loud）；SGLang #39087（12.35）、#39092/#38981（09-11 已判）；tt-metal #56295（1.200）、#56283（1.201）、#56263（5.37）、#56248（1.202）、#56298/#56242/#56254（09-11 已判）；flashinfer #5128（7.83）、#5157（09-11 已判）；TRT-LLM #19063（09-11 已判）
- **误报甄别**：vLLM #56569（prompt 含 `and <数字>/` 时零字节断连）经评论区核验为**报告者自证的误报**——网络边缘 WAF 把该模式当 SQL 注入签名重置 TCP 连接，与 vLLM/tokenizer 无关（报告者已请求关闭），不收录

---

## 一、新合并条目（18 条）

### 第 1 类 · 硬件/SDC（+6）

1. **[已知类别]** **vLLM Triton attention softcap exp 溢出 NaN**（vLLM #56578）— `apply_softcap` 把 `x·tanh(S/x)` 展开为 exp 商，|S/x|>~88 双 exp 溢出 → `inf/inf=NaN` 毒化整行 attention；Gemma-2（softcap=50）pre-softcap score>4400 即触发；unified/DiffKV/INT4 三内核共用该 helper。kernel 复现 `[50,50,nan,nan,nan,inf]`。归类 var:1.162（超越函数手工展开大参数域溢出族）。→ **1.206**
2. **[已知类别]** **tt-metal BF16 trunc/floor/ceil/frac 把 NaN 变 ±inf**（tt-metal #56414）— bf16 路径 NaN 按符号位返 ±inf、frac(±inf)=+inf；fp32 路径全对；binary_fmod/rdiv/remainder 共享 `_trunc_body_` 疑似同病。NaN 哨兵被静默改写为有限值。归类 var:1.197（NaN/±inf 特殊值语义静默改写族）。→ **1.207**
3. **[已知类别]** **tt-metal bitwise_right_shift 对 UINT32 做算术移位**（tt-metal #56385）— LLK 只给 UInt16 走无符号分支，UInt32 落入符号扩展 else；UInt32 域 **50.00%** 输入静默错值（`0x80000000>>31` 返 4294967295 应为 1，4.29e9× 高估）；**golden 函数本身编码了该缺陷**（纯 Python 复现，无需硬件），CI 无法捕获。归类 var:1.176（bitwise_right_shift 语义缺陷族）。→ **1.208**
4. **[已知类别]** **tt-metal moreh_softmin 含 ±inf 全零行 + 死 define**（tt-metal #56371）— 程序工厂定义 `SOFTMIN=1` 但 kernels/ 下**无任何 `#ifdef SOFTMIN` 分支**——define 从未被消费，全部内核按 softmax 变体算，+inf 行 `e=−(x−inf)` 全 NaN→全零；W/H/NC/mask 全并行路径一致错，softmax/logsoftmax sibling 正常。归类 var:1.202（moreh 族特殊值未守卫）。→ **1.209**
5. **[新类别/新签名]** **tt-metal ttnn.pad 单核 row-major 忽略 front padding**（tt-metal #56323）— `PadRmReaderWriterProgramFactory` 的 reader 从不读 `input_tensor_start`，前向填充静默落到数据尾部（`[x0..xn,0..0]` 而非 `[0..0,x0..xn]`）；tt-mlir 恒用多核、单测全零前向填充——零覆盖。→ **1.210** `NEW`（布局算子前缀量被忽略静默错位子签名）
6. **[已知类别·跨仓库]** **triton 3.8.0 release wheel 缺 13 个 main 正确性修复**（triton #11735，#11733 为同触发面实例并入）— main 测试对 wheel 逐条运行：mxfp4 MN-packed `dot_scaled` sm_120 返回 garbage（524214/524288 元素错）、E8M0 scale=0/subnormal 全 16384 元素错、`OptimizeThreadLocality` 错误重排 reduction（256/256 错）、`tl.range(flatten=True)` 空内环丢迭代、`tl.softmax dim≠-1` 轴错、constexpr 列表推导 if 过滤忽略、`tl.full(-0.0)` 丢符号位 + TMA 不对齐 stage 设备故障等；cherry-pick 构建全部修复。归类 var:7.83（修复未跨副本传播族——flashinfer vendored 的 release 分支维度镜像）。→ **1.211**

### 第 4 类 · 数值/精度/不确定性（+1）

7. **[新类别/新签名]** **SGLang SM120 decode q 64 头 pad 移除可复现改变 greedy 输出**（SGLang #39235）— 去除"应当惰性"的 padding 后端到端 greedy 分歧（未打补丁双 boot 4/4 一致 vs 打补丁 1/4 一致），但 `sparse_mla_sm120_decode_dsv4` 内核隔离 A/B **bit-identical**（0/49152 元素差，含随机填充头对照）、kernel inventory diff 0/299——报告者明示无法解释；代码注释自认"padded heads 可影响 real-head 输出"。根因开放的配置敏感性观察，编入噪声底维度。→ **4.20** `NEW`（speculative）

### 第 6 类 · 服务栈/缓存（+3）

8. **[已知类别]** **SGLang DP scheduler batch_is_full 粘性 wedge**（SGLang #39271）— input 长度近 KV 容量（190720+0+64≥190784）时 `max_new_tokens` 剪 0 仍过验证但 `PrefillAdder` 永不接纳；`batch_is_full=True` 置位后请求取消也**无失效路径**——该 rank 永久拒新请求，仅重启可恢复；无 GPU 状态级复现。归类 var:6.135（调度器粘性标志永不复位族）。→ **6.232**
9. **[新类别/新签名]** **vLLM DSpark CUDA-graph padded 行流入 dsv4_topk 路由 → IMA**（vLLM #56389）— FULL_DECODE_ONLY graph 的 padded 行被 `dsv4_topk` Triton 快速路径当真实 token 路由（非法 expert id）→ 异步 IMA → EngineCore 死亡；8×H20 生产复现 + 第二用户 1024 并发复现；本地补丁（padded 行发零权重/−1 expert 或 draft 改 generic router）1024/1024 两轮通过；`max_num_seqs=256` 稳定。修复方向 PR #53577。→ **6.233** `NEW`（graph padding 行跨入 MoE 路由子签名）
10. **[新类别/新签名]** **SGLang HiCacheFile 混合池存在性查询 min 端点误报**（SGLang #39147）— `batch_exists_v2` 取各池最长前缀的 min，对 TRAILING_PAGES 池的合法端点洞（{3}/{2} 报 2、{1,3}/{1,2} 报 2 漏掉合法公共端点 1）误报 hit——报告不可恢复的混合前缀为可恢复；MooncakeStore 同接口已用合法端点交集。报告者明示未复现端到端输出错误（契约级复现），作为静默损坏放大器收录。→ **6.234** `NEW`（混合池存在性"端点集合代数"子签名）

### 第 7 类 · 量化（+3）

11. **[新类别/新签名]** **SGLang DSV4.1 fp8 wo_a absorb 非 2 幂激活 scale 静默截断**（SGLang #39193）— `DEEPGEMM_SCALE_UE8M0=false` 分支给 `deep_gemm.fp8_einsum` 非 2 幂 fp32 激活 scale；内核要求 2 幂（smxx_layout.cuh:131 断言 sign+mantissa 清零）但**只对权重 scale 强制、激活 scale 静默截断 mantissa** → ~25% 错、无错误。真实 checkpoint 权重隔离复现（SM121；flag 分支理论其它 arch 可达）。→ **7.85** `NEW`（内核前置条件单侧强制子签名）
12. **[新类别/新签名]** **SGLang fp8 专家 + F32 scale checkpoint 跨引擎精度分歧**（SGLang #39311）— 同机同权重同 harness：SGLang 0.5.17 AIME25 pass@1 **84.58%** vs vLLM **94.79%**（480 样本，−10.2pp 远超 SEM）；排除 parser（排除 no_answer 仍 −7.1pp）、模板、截断、marlin；官方 fp4 专家基模无差距（95.83%）。走 Fp8MoEMethod+DeepGEMM 预期路径、零警告——多数是**错答**非丢答；根因开放。→ **7.86** `NEW`（documented）
13. **[新类别/新签名]** **flashinfer MoE 路由枚举缺 sqrtsoftplus → 静默错选专家风险**（flashinfer #5190）— `trtllm_fp4_block_scale_moe` 的 10 种 RoutingMethodType 全部 softmax/sigmoid/top-k 组合，DSV4.1 的 sqrtsoftplus 评分不可表达；SGLang 现传 DeepSeekV3 触发误导性 `topkGroup<=4` assert（指向错误根因），若用户改传其它 method 则**静默按错误评分函数选专家**（正确性 bug 而非配置不匹配）。→ **7.87** `NEW`（documented）

### 第 8 类 · KV 缓存/注意力（+1）

14. **[已知类别]** **SGLang GLM-5.x NoPE MLA 在 SM120 无任何可用 DSA backend**（SGLang #39302）— `qk_rope_head_dim=0`（absorbed q dim 512 而非 576）：tilelang 需 202KB 动态 smem（设备上限实测 101376B）；GLM+fp8 KV+SM120 代码强制的 FlashInfer 后端内核拒 NoPE——fp8 KV 下 unrunnable by construction，bf16 KV 其余 backend 各自受阻。几何与 flashinfer `nope_mla_dimensions` preset 完全一致（上游一等公民）。归类 var:8.62（sparse-MLA 后端可用性矩阵缺口族）。→ **8.84**

### 第 10 类 · 分词器/模板（+4）

15. **[新类别/新签名]** **vLLM GLM-5.3-Flash reasoning 退出边界 tool-call token bleed → word salad**（vLLM #56605，修复 PR #56635）— 多轮 agentic 长会话间歇退化成重复 token word salad（(tmp)(tmp)…）；维护者确认"known issue because of tool call / vllm frontend"；修复 = 跳出 reasoning 时显式检查 tool call token 防 bleed。build A/B 干净（08-26）vs 退化（09-08）；合成请求 30+ 对照全干净——仅真实 agentic 长会话触发。→ **10.62** `NEW`
16. **[已知类别·跨类]** **SGLang b12x-vision 分支 DSML arguments 双重归一化回退**（SGLang #39277）— dev-v4f-2dgx 预览镜像（#28035 合并的后代）回退修复：serving_chat 把 arguments 归一化为 dict，`encode_arguments_to_dsml` 又无条件 `json.loads` → TypeError → 包成 `{"arguments": dict}` 错误 DSML 参数层；多轮工具调用历史理解静默损坏。归类 var:6.156（已修复 bug 经发布分支回退复活族）。→ **10.63**
17. **[已知类别]** **vLLM xgrammar 多分支 allOf 静默不约束且 gate 不回退**（vLLM #56556，修复 PR #56557）— xgrammar 0.2.3 把多分支 allOf 编译成不强制任何分支的文法（4 个违规实例全 accept）；feature gate `has_xgrammar_unsupported_json_features` 不列 allOf → auto 不回退 guidance（后者正确强制）。Pydantic/OpenAPI 继承天然产出该形态；200 + schema 违规输出。归类 var:10.47（xgrammar schema 特性静默丢约束族）。→ **10.64**
18. **[已知类别]** **vLLM /responses 忽略 include_reasoning**（vLLM #56428，closed）— /chat/completions 与 /completions 都实现该过滤，/responses 不实现（`reasoning.effort=none` 同样无效）；GLM-5.3+ 无 thinking 开关模型把 reasoning 合并进 text content。归类 var:10.52（输出过滤参数被 API 路径忽略族，10.61 同族第三成员）。→ **10.65**

---

## 二、检视后不收录（含理由）

| 条目 | 理由 |
|---|---|
| vLLM #56569（`and <数字>/` 零字节断连） | **报告者自证误报**：网络边缘 WAF 把该模式当 SQL 注入签名重置 TCP（对不存在路径同样 reset、绕开 WAF 的主机拿到 400、WAF 修正后正常）——非 vLLM 缺陷 |
| vLLM #56646（2 节点 TP mamba_cache_mode 派生不达远端 rank） | init 期 KV-spec merge assert，fail-loud crash；workaround 显式传参已验证（同 #56396 标准） |
| vLLM #56540（ROCm GLUON 回退 TRITON 刷日志） | 后端可用性/性能回退 + 日志噪声，非正确性 |
| vLLM #56559（s390x xgrammar 无条件 import 启动崩） | fail-loud 启动 crash（平台移植） |
| vLLM #56410（cpu-offload-gb 用 ~1.9× host 内存） | 内存超额治理，非正确性 |
| vLLM #56696/#56668/#56653/#56384/#56727/#56506/#56361 | 观测缺失/单测环境差异/bench 工具/镜像分发/RFC×2/metrics 计数 |
| vLLM #56396 | 09-11 已判（裸 assert fail-loud），今日窗口再现，维持 |
| SGLang #39072（disagg decode + dp-attention + spec crash） | fail-loud crash 且栈截断无根因——留观 |
| SGLang #39173（Engram SPS 表 CUDA-graph capture 崩） | fail-loud capture 错误（"expects one equal block per request"） |
| SGLang #39070（FLUX.2 sage_attn 白名单拒绝） | fail-loud 启动拒绝 |
| SGLang #38980（sgl_kernel sm_89 无 cubin + ver 死参数） | fail-loud CUDA error at launch；契约不一致（is_fa3_supported vs wheel 内容）已记录于本表 |
| SGLang #39107 / #39279 | feature |
| SGLang #39092 / #38981 | 09-11 已判（profiling 工具缺陷 / fail-loud invariant crash），维持 |
| tt-metal #56424（quantize/dequantize bf16 scale 深处 TT_FATAL） | fused 路径 fail-loud abort（同 #56254 标准）；composite 分支静默 cast 后正确——接口不一致但无静默损坏 |
| tt-metal #56345（Copilot cloud runtime abort） | CI/基础设施 |
| tt-metal #56298/#56242/#56254 | 09-11 已判（hang/留观/fail-loud），维持 |
| triton #11747（tcgen05 sm_120a LLVM Cannot select） | 编译器 abort，fail-loud |
| flashinfer #5157 | 09-11 已判（CI 基础设施），维持 |
| TensorRT-LLM #19063 | 09-11 已判（C++20 编译失败），维持 |
| transformers 仓库 | 窗口内 0 命中 |

## 三、arxiv

**当日 5 组查询均未完成**：首批 5 连发触发 IP 级 429 限流（`Rate exceeded.`），此后 45s/90s/240s 指数退避重试仍 429，扫尾补试遇连接超时（HTTP 000）与 503 Service Unavailable——累计 7 次尝试跨约 25 分钟均未成功。按任务规则记录并跳过——建议下次扫描优先补查（`silent data corruption LLM` / `bit flip LLM inference` / `KV cache corruption` / `LLM inference correctness` / `GPU soft error ML` 五组）。昨日（09-11）结论仍有效：TrainSDC/SCOUT/LLM-PRISM/JITterFlip/GBFA/ROBBIN 等均已编目，方法论文不收录。

## 四、API 调用记录

| 类别 | 调用 | 结果 |
|---|---|---|
| GitHub Search API | 10 组（vllm bug / vllm wrong / sglang wrong / hf wrong / trtllm wrong / triton wrong / ttm bug / flashinfer wrong / vllm quant / sglang kv），窗口 `created:>=2026-09-11`，间隔 ≥7.5s | 全部一次成功（HTTP 200），无限流 403 |
| GitHub core REST | issues 评论 4 次 + 单评论 1 次（`--http1.1`） | 5/5 成功 |
| arxiv API | 5 组（`curl -sL` HTTP→HTTPS） | **全部 429 限流**，三次退避重试均失败，跳过 |
| 去重核验 | 主文档全文正则匹配 46 个候选 issue/PR 编号 | 8 个已编目/已判（跳过），其余确认全新 |

## 五、合并操作

- 主文档新增：1.206–1.211、4.20、6.232–6.234、7.85–7.87、8.84、10.62–10.65（triton #11733 作为 1.211 同触发面实例并入，不占新编号）
- 滚动汇总/目录/分布数字已按实际条目重新计算：1058 → **1076**（第 1 层 927 → 945；每日扫描合并累计 835 → 853）
- 分布变化：第 1 类 205→211、第 4 类 19→20、第 6 类 231→234、第 7 类 84→87、第 8 类 83→84、第 10 类 61→65
- 导出：`export_faults.py` → fault_catalog.xlsx + fault_catalog.csv（Parsed = **1076** 行；脚本将 A1–A17 作为一等条目纳入导出，导出行数 = 主文档总条目数）

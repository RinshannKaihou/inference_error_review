# LLM 推理故障每日扫描报告 — 2026-10-09

- **扫描日期**：2026-10-09（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-29`（`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算为 2026-10-02；因上次扫描为 09-29、中间存在 **10 天空窗**（09-30 至 10-08 九天无扫描运行），本次窗口前移至 2026-09-29 以完整覆盖空窗——不小于规范 7 日窗口。与 09-29 扫描窗口 created:>=2026-09-22 的重叠区间条目全部按既有判定跳过，本日新判定集中于 09-29 扫描执行（23:40 UTC commit `8345d1f`）后创建的 **407 个 issue/PR**）；arxiv 6 组查询全部 200 有效
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1252 条（2026-09-29 扫描后）；编号编目 1138 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `8345d1fba93b4adee9168a57dda860914f7aaa6c` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-29 wrong OR incorrect OR corruption OR NaN` | 75 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-29` | 78 命中（单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-29 wrong OR incorrect OR corruption OR NaN` | 30 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-29 wrong output OR incorrect OR NaN` | 6 命中（单页取全） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-29 wrong OR incorrect OR corruption` | 8 命中（单页取全） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-29 wrong OR miscompile OR NaN` | 9 命中（单页取全） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-29 numerical OR wrong OR pcc OR silent` | 213 命中（**三页取全 100+100+13**） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-29 wrong OR incorrect OR NaN` | 18 命中（单页取全） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-29 quantization OR "KV cache" OR deterministic` | 151 命中（**两页取全 100+51**） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-29 silent OR "bit flip" OR "weight corruption"` | 56 命中（单页取全） |
| 11 | GH Search | `repo:sgl-project/sglang is:issue label:bug created:>=2026-09-29` | 0 命中 |
| 12 | arxiv | `"silent data corruption" AND LLM` | 200，6 命中全已编目（含 SProbe 2609.19743、TrainSDC、SCOUT、LLM-PRISM 等） |
| 13 | arxiv | `"bit flip" AND LLM AND inference` | 200，8 命中全已编目或已判（BARE-AI=2605.11202 已编目、JITterFlip、UEP、RangeGuard 等） |
| 14 | arxiv | `"KV cache" AND corruption` | 200，命中已编目或方法论文不收录（Contiguity/Runtime Observability/Moment-KV/Nexus 等） |
| 15 | arxiv | `"soft error" AND GPU AND inference` | 200，3 命中已判（2609.16742 CNN 边缘域、2601.19912 已编目、2104.09455 非 LLM 域） |
| 16 | arxiv | `LLM AND inference AND "numerical correctness"` | 200，3 命中均为非故障域论文 |
| 17 | arxiv | `LLM AND inference AND "wrong output"` | 200，0 命中 |

GitHub Search API 11 组查询全部有效、**分页完整（fetched == total 逐组核对：75/78/30/6/8/9/18/151(100+51)/56/0/213(100+100+13)，`incomplete_results=false` 逐组核验）**、无限速 403、无 Validation Failed；查询间 ≥8s 间隔。去重后 **513 个唯一 issue/PR**（跨 11 组查询重叠），其中 **407 个为 09-29 扫描执行后新建需判定**、106 个重叠区按既有判定跳过。候选正文取自 Search API 缓存（响应自带 body）；core REST **24 次调用全部成功**（`--http1.1`：18 次来源回访快照 + 6 次评论/PR 深查；未认证运行，环境无 GITHUB_TOKEN，额度 60 制内）。

## 🆕 新发现（本次合并，共 65 条：37 个新子签名 + 28 个已知类别变体）

10 天空窗带来迄今最大单轮合并。按类别：

### 第 1 类 +19（1.272–1.290）

1. **1.272** [新签名] vLLM #59404 — vendored FLA `chunk_o.py` 丢上游 gate mask 且 mask 计算移到 `exp` 之后：部分边界 chunk 未定义 lane 被 exponentiate，NaN/inf 静默进 GDN 输出累加器（双侧源码对照）。verified。
2. **1.273** [新签名] TRT-LLM #19955 — GroupRMSNorm 双缺陷：共享内存初始化竞态（397,748/397,774 调用错）+ one-warp large-batch 路径从不归一化 input 1（401,268/401,268 全错）。fp64 参照 harness。verified。
3. **1.274** [新签名] TRT-LLM #19943 + #20035 — `util_convert_type`/`flash::convert_type` 返回**局部数组视图**（悬垂 UB）：nvcc 读陈旧值、clang -O3 下整条 SwiGLU+MMA 链被 DCE（SASS 三方对照 96/64/12/72 → 64/40/0/48）。documented。
4. **1.275** [变体 var:1.151] triton #12099 — argmax/argmin 对 reshape/ravel/flip 输入：`allow_reorder` 重排 value 不重排 index，返回索引静默错。documented。
5. **1.276** [新签名] triton #12102 — bf16 标量常量经 `std::to_string`（6 位小数）截断：`1e-8`→0、`1.5e-6`→2e-6。verified。
6. **1.277** [新签名] triton #12104 — loop-carry 检测按 dry-run IR 值判等：`prev=cur` 同值初值不进 iter_args，carry 静默丢失。documented。
7. **1.278** [变体 var:1.127] triton #12192 — FuseNestedLoops flatten=True 在内层迭代推进外层 carried 索引。documented。
8. **1.279** [新签名] tt-metal #58669 — logaddexp bf16 近零符号翻转、最高 222k ULP。documented。
9. **1.280** [变体 var:1.126] tt-metal #58682 + #59462 — digamma LUT 无域守卫：负轴全域外推（−2.5 符号反）+ 正数 <1.2125e-4 段外推（1e-7 达 3.0e7 ULP、digamma(0) 返 +6.68e6）。documented。
10. **1.281** [变体 var:1.266] tt-metal #59126 — fp32→bf16 helper 穷举 16,777,214 NaN 位型：131,072 个变非 NaN（±0/±Inf）。verified。
11. **1.282** [变体 var:1.267] tt-metal #59016 — Quasar 通用 typecast 把 Int32 当 sign-magnitude（原生 2SC）：负值转换垃圾，**golden 打包同错掩盖**。documented。
12. **1.283** [新签名] tt-metal #58695 — ttnn.sum/mean fp32 ROW_MAJOR W>256 错值。documented。
13. **1.284** [新签名] tt-metal #59133 — ttnn.embedding TILE indices batch>1 仅 batch 0 正确（校验只对 ROW_MAJOR 跑）。documented。
14. **1.285** [变体 var:1.197] tt-metal #59184/#59185/#59186/#59247 — erf/erfc/i1/gelu 把 NaN 静默映射为有限值（SFPSWAP 按位排序 NaN）。documented。
15. **1.286** [变体 var:1.269] tt-metal #59818 — sharded softmax 全族 padding 列入行和（24/24 配置泄漏，min row sum 0.73–0.79）。documented。
16. **1.287** [新签名] tt-metal #59645 — trace 回放覆写捕获后分配的 buffer（MiniMax-H3 实测 12.6 dB PSNR 偏差、逐块 instrumentation 定位）。documented。
17. **1.288** [新签名] tt-metal #59125 — WH matmul unpack MOP 只覆盖前 8 streamed tiles：9–16 tile 隔次调用错。documented。
18. **1.289** [变体 var:1.119] tt-metal #59689（+四重复/姊妹 #60162/#60165/#60170/#60181）— dst_full_sync_en 下校验放行内核算错的 subblock。documented。
19. **1.290** [新签名] tt-metal #58785 — BH custom matmul Bfp2_b plain path 输出与乘积无关（Bfp4/Bfp8/压缩路径正确）。documented。

### 第 4 类 +3（4.28–4.30）

20. **4.28** [变体 var:4.12] vLLM #59764 — Qwen3.6 hybrid GDN 多序列路径 **run-to-run 非确定**（同批重复提交 logprobs 差 0.168；272 生产 prompts 10–17% 决策翻转；`max_num_seqs=1` 精确 0.0）。documented。
21. **4.29** [变体 var:4.17] vLLM #60007 — VLLM_BATCH_INVARIANT=1 下 TRITON_MLA chunked prefill 位一致违约（enforce_eager 仍失败；疑 merge_attn_states LSE 非结合）。documented。
22. **4.30** [变体 var:4.17] vLLM #59940 — BI 冷 model-info cache 下 import 顺序使 attention 未包裹即入 CUDA graph：短 prefill 静默错 token（默认多进程 LLM 引擎中招）。documented。

### 第 6 类 +9（6.264–6.272）

23. **6.264** [新签名] vLLM #59409 — V2 runner 抢先做 request-state 更新（清零/COW）再调 connector `handle_preemptions`：异步 save 读中页可被覆写（mock 确定性复现的回调顺序违约）。documented。
24. **6.265** [新签名] vLLM #59569 — P2P kv_offload store 超时解钉**在途 RDMA 传输**的槽、传输完成仍报 success=True：消费者解码被覆写块（静默 KV 损坏）。documented。
25. **6.266** [变体 var:6.258] vLLM #60004 — LHBNC + native CPU offload：stride 启发式误判布局、每块只搬 layer0/head0（12.9 GB→33.5 MB），回读命中 gibberish。verified。
26. **6.267** [新签名] vLLM #60494 — GLM-5.2-FP8 + prefix cache 命中后 `content: null` + 乱语 reasoning、200 无错（强制冷 prefill 0/50 失败）。documented。
27. **6.268** [新签名] vLLM #59926 — DSV4-Flash 前缀缓存复用 × tool-call 格式失败、replica 局部（63% vs 23% 按命中形状；机制未知、混杂自述如实记录）。documented。
28. **6.269** [新签名] vLLM #60905 — DSV4-Pro `--long-prefill-token-threshold=256` 下 >380k prompt **自信答错**（1,033k 全错）；去 flag 全对——单变量 A/B 因果闭合。documented。
29. **6.270** [新签名] SGLang #43075 — GPU JPEG 解码覆写仍在被排队 GPU 工作读取的内存。documented。
30. **6.271** [变体 var:6.233] vLLM #59686 — CUTLASS fp8 MoE 对 CUDA-graph padding 的 `topk_ids=-1` 算错 `expert_first_token_offset`：真实 token 进错专家（单请求首 token 坍塌；triton/marlin/eager 正确）。verified。
31. **6.272** [变体 var:6.259] vLLM #59933 — RecoverSSM align 恰块边界把 state 提交到未写块（floor vs ceil−1）：spec 下 10–11/16 输出损坏、两行修复 0/16。verified。

### 第 7 类 +7（7.101–7.107）

32. **7.101** [变体 var:7.52] SGLang #42369 — nvfp4 KV 把 checkpoint 的 fp8 校准 `k/v_scale` 当 NVFP4 global scale（~75× 失配）：**删 64 字节单变量证明** needle 1/4→4/4，全程无 scale 日志。verified。
33. **7.102** [变体 var:7.62] vLLM #59403 — Marlin int8-activation 负 group scale 按 uint16 读（50.8% 组错、反量化误差 170×；仍加载仍"benchmark 正常"）。verified。
34. **7.103** [变体 var:12.45] vLLM #60608 — Exllama/RDNA3 W4A16 `group_size%32≠0` 静默错值（CUDA+ROCm 双平台；诚实范围：无已知公开 checkpoint 触发）。documented。
35. **7.104** [新签名] TRT-LLM #19954 — `calculate_nvfp4_global_scale` hidden>8192 非倍数时 `__syncthreads()` 进发散循环：309/512 token scale 错。verified。
36. **7.105** [新签名] flashinfer #5996 — CuTe-DSL MXFP8 E8M0 字节 0（=2^−127）被当零 scale：非零 BF16 块整块归零。verified。
37. **7.106** [新签名] SGLang #42917 — compressed-tensors W4A16 int4 服务质量劣化（wikitext PPL 9.98 vs vLLM 6.05 同 checkpoint）。documented。
38. **7.107** [新签名] vLLM #60752 — grouped_topk 多组内核每 warp 评一组：>8 组从不评分、返回错专家（9/16/32 组 20/20 失败）。verified。

### 第 8 类 +5（8.95–8.99）

39. **8.95** [变体 var:8.53] flashinfer #5965 — FA2 NVFP4 KV head_dim=64 SM120/121 prefill 错值（#4748 SM80 closed-not-planned 的当前版本再现）。documented。
40. **8.96** [新签名] tt-metal #59561 — update_cache fp32 写 1 token 毒化同 tile 其余 31 行（bf16/TF32 截断；fill_cache 精确）。documented。
41. **8.97** [新签名] tt-metal #58683 — fill_cache 多 head 时 core 块跨头边界把 head h+1 写进 head h 页（源码 TODO 承诺的 assert 不存在）。documented。
42. **8.98** [变体 var:8.83] vLLM #59741 — GLM-5.3-Flash ROCm kpool `always_select_tail` 只保不完整尾池：完整 recent 池可被 2048 截断挤掉（尾部检索非确定失败；强制 128 尾池修复）。documented。
43. **8.99** [变体 var:8.54] vLLM #60160 — Qwen3.8-27B-FP8 MI355X 高并发准确率坍塌（AITER on 触发、off 恢复）。documented。

### 第 9 类 +2（9.31–9.32）

44. **9.31** [变体 var:9.18] vLLM #60165 — Nomic pooling 忽略 `rotary_scaling_factor=2`（effective `rope_type="default"`）：4096 token embedding min cosine 0.9007 vs NTK 参照 0.99999。documented。
45. **9.32** [变体 var:9.29] SGLang #43158 — `get_rope_config` 对 v5 per-layer-type 配置**凭空造 rope_theta=10000**——与 9.29 构成跨栈姊妹（vLLM overrides 通道 / SGLang 解析通道各自丢/造 base）。documented。

### 第 10 类 +11（10.93–10.103）

46. **10.93** [变体 var:10.64] vLLM #60237 — auto 结构化输出分派谓词（allOf≥2）不覆盖 xgrammar 真实缺陷域（组合子×同节点约束关键字）：schema 编译为 `{}` 等价规则**静默全放开**（xgrammar 接受全部 9 个非法实例、guidance 拒绝全部）。documented。
47. **10.94** [新签名] vLLM #59876 — request 级 `chat_template_kwargs`（即使 no-op 值）静默丢请求全部图像（prompt_tokens 276→22、模型答"没有图像"、200 无告警）。documented。
48. **10.95** [新签名] vLLM #60340 — Olmo3 reasoning 流式尾 delta 为 marker 子串（"think"）时缓冲永不 flush。documented。
49. **10.96** [变体 var:10.42] vLLM #60341/#60348 — InternLM2 parser `<|action_start|>` 后无 `<|plugin|>` 丢全部正文 + 流式参数丢字符。documented。
50. **10.97** [变体 var:10.42] vLLM #60342/#60343/#60421 — DeepSeek-V3 多行 JSON 空 / xLAM 嵌套>2 层截断 / Olmo3 换行坏字符串（三 parser 批量审计）。documented。
51. **10.98** [变体 var:10.42] vLLM #60703 — glm47 流式正文零散 `<tool_call>` 后吞掉剩余整条流（含真 tool call）。documented。
52. **10.99** [变体 var:10.42] vLLM #60865 — DotsToolParser 忽略 init 期 tools（request.tools=None 时全部 call 被当文本）+ FunctionTool 500。documented。
53. **10.100** [变体 var:10.69] SGLang #43149/#43017/#43153 — detector 参数值强制转换无视声明 schema 类型批（可空字符串 `"123"`→123、"null"→null、布尔空白翻 false）。documented。
54. **10.101** [新签名] vLLM #59738 — deepseek_r1 reasoning 流式 delta 发**原始 BPE 字形**（CJK 乱码；content 通道正确、非流式正确）。documented。
55. **10.102** [新签名] SGLang #43155 — LLAMA2 conversation 风格按消息**序号奇偶**推 role 标签。documented。
56. **10.103** [变体 var:10.47] vLLM #60580 — xgrammar 允许控制 token 入 JSON 字符串 + muse_glimmer parser 截断答案。documented。

### 第 11 类 +3（11.91–11.93）

57. **11.91** [新签名] vLLM #59785 — CPU 路径 top-p 内核丢 top-p 必须保留的 token（8-token 行丢 7.5% 质量 token；514,800 行扫描 0–4% 行差；引擎级 4000 请求复现被丢 token 采样 0 次 vs 期望 ~300）。verified。
58. **11.92** [新签名] vLLM #60897 — eagle_prepare_inputs 无下界索引：PP>1 spec 下 query 无 bonus 行时下溢——首请求 OOB gather、后续请求**串线到前一请求的采样 token**。documented。
59. **11.93** [新签名] TRT-LLM #19902 — beam search logits processor 收 per-slot 列表而非 beam hypothesis（steering 作用于错误谱系）。documented。

### 第 12 类 +5（12.46–12.50）

60. **12.46** [新签名] vLLM #59647 — Fast Start WeightCacheKey 只哈希 safetensors **header**：同布局 checkpoint 同 key，ipc_cache daemon 静默服务错 checkpoint（e2e：q_mod 引擎输出 q 的结果，CLI 显示 B）。verified。
61. **12.47** [变体 var:12.42] vLLM #59799 + #60223 — PEFT rank_pattern/alpha_pattern 被丢（OLMoE adapter vLLM 丢失 ~40% 增益 PPL 11.28 vs 10.06）+ 未实现 PEFT 特性静默加载静默错。verified。
62. **12.48** [变体 var:7.52] vLLM #60333 — MiMoV2MTP loader 三命名族全丢 fp8 KV scale、draft 层 unit scale 运行（唯一痕迹是可误读的启动 warning）。documented。
63. **12.49** [新签名] vLLM #59920 — legacy fused MoE LoRA 内核无条件 cast 操作数到 bf16（fp16 模型 10→7 bit 尾数；one-shot 路径正确使分歧难诊断）。documented。
64. **12.50** [新签名] vLLM #60811 — GPT-OSS mixed_moe_lora_format gate/up 双切片连续写 vs 基座交错布局：delta 加错通道进 SwiGLU-OAI。documented。

### 第 13 类 +1（13.42）

65. **13.42** [新签名] transformers #49337 — MetalLinear 3D 输入仅第 0 行正确（MPS 批式前向静默错）。documented。

## 评估后不收录（记录备查，摘要）

- **vLLM**：#59381（Qwen3.5-35B merged SFT H20 间歇 NaN logits——HF 对照有限、单环境无根因留观）、#59415/#59551（Intel B70 TP>1 gibberish——第三方 GPU 栈无诊断信息留观）、#59642（Qwen3.8 disagg MTP 0% 接受，closed——11.63 族新实例、根因未定，并入族语境跟踪不占号）、#59449（routed experts 层级重载后陈旧路由，closed——观测元数据域）、#59539（GLM-5.3-Flash 视觉缓存拒收——容量校验域）、#59520（GDN wrapper fallback 性能已修）、#59607（EP16 Xid 31 MMU——fail-loud 崩溃域）、#59671（报告者自行关闭、未证实的 int4 KV 精度疑虑）、#59907（优先级调度 pytest 五败——调度逻辑域留观）、#59926→已编目 6.268、#60124（MultiConnector 静默禁用 hybrid manager 致 Nixl 拒绝——拒绝面 fail-loud 留观）、#59457（ARM CPU W4A8 garbled——无根因留观）、#59756（Qwen4Exp layer_type ValueError fail-loud）、#59798（AutoRound PLE 拒绝 INCConfig fail-loud）、#59867（NVFP4+MTP draft 打包 crash fail-loud）、#59946（gemma4 KV-shared 层要求不存在的 k_proj fail-loud 拒载）、#60119（NVFP4 B300 CUDA illegal access——崩溃域）、#60350（FP8 KV 预算漏 CUDA graph 内存——OOM 域）、#60455（EPLB async rearrangement AssertionError fail-loud）、#60471（CUTLASS MoE NVFP4 serving OOM——内存域）、#60489（QSA workspace 逃逸 profiling——OOM 域）、#60551（speculators nvfp4_ds_mla 假设 split k/v crash fail-loud）、#60560（DFlash2 HMA 接受率 −5.5pp——**纯性能**、target 输出正确）、#60587（DSV4-Flash-Vision deep_gemm shape mismatch fail-loud）、#60685（EPLB step_interval<window_size 计数错位——负载均衡/性能域）、#60765（Qwen3.5 PP×dflash 缺 opt-in 报错 fail-loud）、#60771（DSV4-Flash-0731 重复乱码——环境特定无诊断留观）、#60830（MTP × json_schema 请求期 400 fail-loud）、#60859（kv fp16×bf16 dtype mismatch capture crash fail-loud）、#60860（DSV4.1-Flash 长 decode unspecified launch failure——崩溃域）、#60914（MXFP4 MoE GB200 IMA——崩溃域）、#60420（HiSparse host staging OOM——内存域）、#60137（paused streaming 满 KV 死锁——挂起域）、#60165 已编目 9.31、#59828（CPU 采样 buffer 别名 closed——11.51 族语境补充不占号）、#60059（/inference/v1/generate 缺 reasoning 参数——API 行为域）、#60212（DeepStream moov ZeroDivision——视频工具域）、#60264/#60629/#60794（bench/绘图工具域）、#60465/#60476（API/性能域）、#60744（pause_generation 挂起流会话 raise——fail-loud）。
- **SGLang**：#41939（GLM-5.3-Flash NVFP4 TP4 推理循环无终答——无根因留观）、#42749（CI GSM8K 0.87<0.93 门回归 closed——7.99 族语境跟踪）、#42465（HiCache write_through TP 死锁——挂起域）、#42508（unified-memory 忽略 max-total-tokens——配置治理）、#43324（EP dispatch −1 哨兵别名+OOB crash——crash 面主导留观）、#42981（AMD fp8e4m3fnuz 溢出转换分歧——CI 单测域、内核数值面真实但无 e2e 证据留观）、#42907（coredump 管道名不匹配——诊断工具域）、#43157（thinker_config dtype 被抹——dtype 回退证据弱留观）、#43175（ComfyUI 插件 import 错误归因——报错质量域）、#42397（speculative-token-map TP>1 device assert——crash 面）、#41866（NPU INT8 KV RFC）、#41828→已编目（PEFT bias 41828 与 59799/60223 同族，vLLM 侧已按 12.47 编目；SGLang 侧本条并入族语境）。
- **triton**：#12038（gfx1151 重 spill 错值/设备不可用，closed——待修复版本与根因核实留观）、#12061（TRITON_INTERPRET 调试器 bf16 位型错——仅解释器域非生产路径留观）、#12155（gfx12 OCP fp8 走软件路径——性能）、#12170（mbarrier bra.uni——PTX 生成正确性讨论留观）、#12173（文档契约）。
- **flashinfer**：#5878（top_k/min_p 按输出行索引 per-request 参数 OOB——**与 11.83（#5339）同机制**（验证按输入行、索引按输出行），作为该条重复/补充证据并入语境不占号）、#5966（attention-sink wrapper × NVFP4 不可用——能力缺失拒绝面）、#5732/#5768/#5825/#5786（feature/tracking）。
- **tt-metal**（LLK 审计系列与窄域 op）：#58670（xielu 负分支零交叉 MAD 相消——窄域 op、证据充分可后续补录）、#59177（fmod/remainder |a/b|≳2^20 域错——1.251 族语境留观）、#59182（rms_norm 0-D 忽略 epsilon——与已编目 1.150 同判、重复）、#59138（tilizeA_B fp32×fp32 face 叠读——LLK 审计待运行时验证）、#59140（odd tiles Fp8 越界 32 字节——同上留观）、#59141/#59142（face compressed matmul 竞态/行计数——同上）、#59213（SDPA REDUCE_ROW SUM 混合格式 swap，closed——待修复版本核实）、#59300（SDPA decode 桶分裂 logits 分歧，closed——调度拆分域留观）、#59368（pack_untilize 少 tile 仍按 init 数打包越写——留观）、#59453（eltwise sharded view 布局失配——留观）、#59488（comp_pcc 常量张量 PCC 0.0——测试工具域）、#59499（WH 硅片 vs ttsim Qwen3-VL 零输出——模型级待定位留观）、#59510（approx exp SFPNOP 缺失——flag 组合域留观）、#59542（MiniMax-M3 prefill 非确定 hang——Fabric 2D 挂起域）、#59561 已编目 8.96（#59660 为重复已 closed）、#59605（topk 65536 off-by-one，closed——8.55/8.56 族语境）、#59610/#59611（CCL×matmul async 忽略 transpose/activation、核不交叠校验——防御缺失、当前无活触发路径留观）、#59616（load_checkpoints 路径穿越——安全面）、#59663（GELU 无参数静默走近似核——1.137 族语境留观）、#59689 已编目 1.289、#59700/#60119（custom_mm 单列转置错 tile——窄触发留观）、#59726（deepseek_v3_b1 sampling softmax 读串脸——采样 op 域待复现证据）、#59752（pack untilize narrow_row——LLK 留观）、#59765（transpose 16 字节 pitch——留观）、#59768（chunk_gated_delta_rule CB 越写——留观）、#59824（update_cache num_users>32 OOB 页——8.96 关联、待修复版本）、#59826（upsample 分数 scale sharded 错页 closed）、#59829（norm 残差 dtype 错 closed）、#59845/#59847（scale_mask_softmax ROW_MAJOR mask 崩/错——1.269 族姊妹待定位）、#59848（group_norm fp32 ≥5 tile 统计错——**证据充分强候选**，本日合并量已大、记入未覆盖清单下轮补录）、#59849（untilize_with_unpadding 非最外维裁剪错）、#59914（六 op program-cache 陈旧编译值——审计帖整批沿 #57360 判例留观待单项运行时证据）、#59928（零填充缓存迁移——工程项）、#59931（jit build 头文件先于 defines——防御域）、#59969（QK rope cos/sin 分片核心集不一致——留观）、#59983（ttrocc addrgen rs1 编码错——Quasar 预生产）、#59998（SCRATCH_SEC0 破坏 ring MLA 精度——留观）、#60008（WH reduce F16→F32 MAX/标量和错——证据充分可后续补录）、#60010（logaddexp2 −ln2 变体 69 ULP——1.279 族姊妹）、#60023（pow +inf 解码 2^128——窄域）、#60027（Quasar DFB drain 挂起——预生产）、#60029（K3 派发容量溢出读未初始化槽——MoE 域留观）、#60134（Blaze CI 权重缓存零覆写——CI 定位帖留观）、#60155（gemma4_d_p 小 op 开销——性能）、#58533/#58579/#58581/#58588（GLM4 ND PCC/Quasar 测试基建/2x formats——留观）、#58593（tile_cnt 归一化——测试基建）、#58566（embedding_bw 重复索引不累加——训练侧、#36341 根因候选）、#58643（vLLM CI garbage 一次性——不可复现留观）、#58653（mcast fused-bias hang 2/4——挂起域）、#58696（layer_norm ROW_MAJOR hang——挂起）、#58717/#58770/#58771/#58794/#58855（LLK perf 系列——性能）、#58871（dram_prefetcher NoC 越写 closed——硬件域）、#59017（topk_xl K=512 索引面未写——留观）、#59048（GELU_TANH 无快路径——性能）、#59247 已编目 1.285、#59462 已编目 1.280、#59645 已编目 1.287、#59752 系列（Deslop 工程项）、#59836（Quasar RoCC 间距——编译器域）。
- **transformers**：#49324（Qwen3_5 causal-LM 映射类指向 flat 类，closed——影响加载路径、未见静默错值证据留观）、#49272（code_revision TypeError fail-loud）、#49434（无效报告）。
- **TensorRT-LLM**：**#19861（fp8 cross-attention KV 池 overrun——pool<2× 需求时 IMA 或落入其他映射内存时静默错 token，L4 e2e 复现齐备。实为强候选，正文已读、证据充分；因本日合并量已达 65 条，记入未覆盖清单，下轮优先补录为第 6 类）**、#19913（XGrammar qwen_xml 升级 0.2.7 修复——上游已修发布工程域）、#19726（beam 活跃行压缩 RFC）、#20035 已编目 1.274。

## 来源回访（近 7 日新增条目引用来源 + 额度轮巡旧来源）

本日回访 18 个主来源 + 6 个修复 PR 状态（core REST 快照全读，含评论深查）：

- **6.262（vLLM #59104）**：closed/completed（10-05 17:35）；修复 PR #59105 同刻 merged（merged_at=10-05T17:35:52）。合并不等于已发布——条目补回访记，判定不变。
- **6.259**：修复 PR #55601 已 merged（10-07T17:09，core REST merged=true）。"发布带病 + 修复 open"状态终结；含修复的 release 版本待查。
- **7.100（SGLang #41569）**：closed/completed（10-01 22:33）；修复 PR #41668（janbernloehr 09-30 开出）同刻 merged。机制记录不变。
- **6.260（SGLang #41494）**：closed/completed（10-05 22:02）。closed 不自动证明修复已发布；修复 PR 未逐一核实。
- **1.269（tt-metal #58495）**：closed/completed（10-06 15:39）——维护者 jberkowitzTT 确认 bounty 完成并启动支付（评论区核验）。
- **6.261（SGLang #41617）**：仍 open；09-29 sohom-cs 评论确认 #41281 合并后 main 仍复现（finish/second-retraction 及 4 个 page_size=4 用例双释放），一行 `owned_kv_len` 修复验证有效。
- **10.87（vLLM #58824）**：仍 open；最后评论 09-26（neevmodh 推测试到 #58829），无新增。
- **4.27（vLLM #58973）**：仍 open；最后评论 09-28（twu3202 DFlash-b16 复现），无新增。
- **6.263（vLLM #59110）**：open、2 评论、09-29 后无更新。**11.90（#59161）**：open 0 评论无变化。**12.44（#59184）**：open 1 评论无变化。
- **1.266/1.267/1.268/1.271/1.270（tt-metal #58371/#58372/#58467/#58370/#58272）**：全部 open、0 评论、无 triage 响应。
- 未检查完说明：近 10 日新增条目引用来源约 200 个（本日 65 条 + 09-29..10-08 未运行扫描期间并入项），本日按 core 额度回访上述 24 个高价值来源（昨日合并主来源 + 6 个修复 PR 状态 + 旧 open 来源轮巡）；其余约 180 个经本日 q01–q11 search 快照确认无 state 翻转/评论数突变（快照含全部窗口内活跃 issue 的 state/comments 字段），评论级深查缺口滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **65 条**（37 个新子签名 + 28 个已知类别变体）：1.272–1.290、4.28–4.30、6.264–6.272、7.101–7.107、8.95–8.99、9.31–9.32、10.93–10.103、11.91–11.93、12.46–12.50、13.42
- 总条数 1252 → **1317**；编号编目 1138 → **1203**（K* 扩展 114 不变）
- 分类分布：第 1 类 270→289、第 4 类 27→30、第 6 类 263→272、第 7 类 100→107、第 8 类 94→99、第 9 类 30→32、第 10 类 92→103、第 11 类 89→92、第 12 类 45→50、第 13 类 41→42
- 轴分布更新（按解析器实测）：infra_off 1083→1148、config 268→299（+6⚠ 不变）、other_stack 211→215（+25⚠ 不变）、partial 245→249、hw_specific 327→353、documented 711→762、verified 534→548、NEW 902→936、var:x 347→378
- 条目更正（回访补记，判定不变）：6.262（closed+修复 merged）、6.259（修复 PR merged）、7.100（closed+修复 merged）、1.269（closed+bounty 完成）、6.260（closed）、6.261（仍 open+复现确认）
- 观察：10 天空窗带来迄今最大单轮合并，三条主线——①**跨栈同机制姐妹对再现**（9.29/9.32 RoPE base 静默替换两栈各一、7.52 族在 7.101/12.48 两端（scale 错用 vs 丢弃）、10.42 parser 族单轮 +5 站点）；②**"校验存在但校验错对象"形态扩张**（7.107 组数≤32 被接受但内核只评 8 组、10.93 unsupported 检查盯 allOf≥2 而 xgrammar 真缺陷在组合子×约束共存、1.282 golden 与内核共享同错假设）；③**缓存/复用路径是静默损坏主产区**（6.266–6.269 四条全在复用路径，12.46 把"键窄于身份"推进到权重内容寻址）
- 未覆盖范围：q07（tt-metal 213 命中）与 q01/q02/q09 中既有判定条目未逐一重读正文（按既有判定跳过）；tt-metal LLK 审计系列（#58717–#58855、#59138–#59142、#59914 六 op program-cache 批等）维持整批留观未逐项展开；**TRT-LLM #19861（fp8 cross-KV 池 overrun 静默错 token）为已读正文的强候选，因单轮合并量控制未编目，下轮优先**；tt-metal #59848（group_norm fp32 ≥5 tile）/#58670（xielu）/#60008（WH reduce F16→F32）证据充分待下轮补录；约 180 个来源仅经 search 快照确认无状态翻转（无评论级深查）；arxiv 本日查询词面之外的论文不在覆盖范围
- 导出与校验：`python3 export_faults.py`（1317 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1317 条：正文汇总、CSV、XLSX、字段边界一致**

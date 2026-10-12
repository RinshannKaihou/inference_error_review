# LLM 推理故障每日扫描报告 — 2026-10-12

- **扫描日期**：2026-10-12（UTC）
- **扫描窗口**：GitHub `created:>=2026-10-05`（`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算）；与 10-09 扫描窗口（created:>=2026-09-29）的重叠区间条目全部按既有判定跳过，本日新判定集中于 10-09 扫描执行（commit `8b928c2`）后创建的 issue；arxiv 6 组查询全部 200 有效
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1317 条（2026-10-09 扫描后）；编号编目 1203 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin` 成功；`HEAD` = `origin/main` = `8b928c21efe3d37418f872877279c7e281a7a8ba` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-05 correctness OR "silent wrong output" OR "wrong output"` | 30 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-05 KV OR quantization OR "kernel numerics" OR SDC` | 98 命中（单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-10-05 correctness OR "wrong output" OR "silent" OR garbage` | 31 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-10-05 correctness OR "wrong output" OR numerics` | 3 命中（单页取全） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-10-05 correctness OR "wrong result" OR "silent" OR numerics` | 7 命中（单页取全） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-10-05 miscompile OR "wrong result" OR numerics OR correctness` | 3 命中（单页取全） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-10-05 "Bad Outputs" OR correctness OR "wrong value"` | 37 命中（单页取全） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-10-05 correctness OR "wrong output" OR numerics` | 14 命中（单页取全） |
| 12 | arxiv | `all:"silent data corruption" AND all:inference` | 200，5 命中全已编目或已判（SProbe=2609.19743、PVF、NCS2-EMFI、Strategic Monitoring、Saggitarius 非 LLM 推理域） |
| 13 | arxiv | `all:"bit flip" AND all:LLM` | 200，8 命中——**1 新论文：2609.38477 Security-Enhanced Seed-Based Weight Quantization for LLMs（量化随机种子安全加固，防御方向方法论文，非推理故障编目域）**；其余全已编目或已判（BARE-AI=2610.08739 即 2605.11202 v2、JITterFlip、GBFA、Decision-Level、UEP） |
| 14 | arxiv | `all:"KV cache" AND all:corruption` | 200，命中全已编目或方法论文不收录（When Latent Agents Lie、Contiguity、Addressable Memory、Runtime Observability、Nexus、Moment-KV、2606.07571） |
| 15 | arxiv | `all:"soft error" AND all:GPU` | 200，3 命中已判（Carry-Through Checksum=2609.16742 CNN 边缘域、2601.19912 已编目、其余非 LLM 域） |
| 16 | arxiv | `all:"numerical correctness"` | 200，命中全非故障域论文（Citation Swaps/Praxa 审计/MakoXC/BALMS/BF1/FinRank/BMOA——2610.06251 HQQ 量化早停议题留观、BMOA 编译器数值偏差归因与 1.137 族主题相邻但为方法论文不收录） |
| 17 | arxiv | `all:"wrong output" AND all:LLM` | 200，0 命中 |

GitHub Search API 8 组查询全部有效、**分页完整（fetched == total 逐组核对：30/98/31/3/7/3/37/14，`incomplete_results=false` 逐组核验）**、无限速 403、无 Validation Failed；查询间 ≥7s 间隔。去重后 **139 个唯一 issue**（q1∩q2 重叠 13 个、跨组其余无重叠），其中 10-09 扫描已判定条目按既有判定跳过，本日逐一重读新判定约 60 个。候选正文取自 Search API 缓存（响应自带 body）；core REST **37 次调用成功**（`--http1.1`，正文深读 32 + 状态快照 5），中途触碰未认证 60/h 额度上限 403 一次（~02:04 UTC），等待 ~10 min 重置后恢复，剩余 3 个未读正文（tt-metal #59530、SGLang #43331/#42544）改经 Search API 单号查询补齐（额度独立）。arxiv 用 urllib 脚本 6 组查询全部 200 有效。

## 🆕 新发现（本次合并，共 25 条：全部为新子签名）

> 撰写中计数勘误（2026-10-12，本文件内自洽修正，不改写任何历史事实）：初稿曾拟 13.43（作废，从未编目）与 12.51 var:12.42（核对 12.42 实际机制为 rsLoRA √r 公式族，与本条 API 参数契约缺口不同机制，按 var: 使用规范须写清共享的具体失效机制，故 12.51 定为 NEW）；最终口径见"更正计数"一节，各条目列表为实际合并清单的忠实记录。

包含上一轮（10-09）明确记入未覆盖清单的 4 个强候选补录（TRT-LLM #19861 → 6.273、tt-metal #59848 → 1.291、#60008 → 1.292、#58670 → 1.294）。按类别：

### 第 1 类 +4（1.291–1.294）

1. **1.291** [新签名] tt-metal #59848（10-09 留观补录）— group_norm fp32 DEST（fp32 输入或 bf16+fp32_dest_acc_en）half-sync 只有 4 tile 容量，interleaved 工厂把通道 subblock 取到 5–8 tile：一组跨 ≥5 channel tile 起确定性错值（C=1024 误差 1.00 vs 正确 0.011）；sharded 工厂已有 fp32 half-sync 封顶 4 的正确守卫、interleaved 漏掉。documented。
2. **1.292** [新签名] tt-metal #60008（10-09 留观补录）— WH reduce Float16 输入 + Float32 输出：MAX（Row/Column/Scalar）与 Scalar Sum/Average 全错（1.10→6e4、630→0.01），Row/Column Sum 正确；CI 从未把 Float16 列入输入格式参数化。documented。
3. **1.293** [新签名] tt-metal #60223 — Blackhole `ttnn.where`：bf16 predicate + fp32 数值混合组 65,503–65,536/65,536 lane 错（predicate 位型被当数值语义重解释，输出 2.2e-22 量级垃圾）；全 bf16 或全 fp32 组正确。documented。
4. **1.294** [新签名] tt-metal #58670（10-09 留观补录）— xielu FLOAT32 输出负分支在自身零交叉 x* 处 catastrophic cancellation：MAD 相消 1,000–21,000×，exp 近似残差主导答案，fp32 输入最坏 53.6% 相对误差（bf16 输出 ≤0.55 ULP 掩盖）。documented。

### 第 4 类 +1（4.31）

5. **4.31** [新签名] flashinfer #6173 — RoPE FP8 quantize trace 模板对 quantize_dtype=e5m2 仍按 e4m3 ±448 clamp：e5m2 表示范围被静默压缩 128×；CUDA 内核 `cvt.rn.satfinite.e5m2x2` 本身正确、参考实现侧错且三个 correctness 测试从未传 quantize_dtype。documented。

### 第 6 类 +8（6.273–6.280）

6. **6.273** [新签名] TRT-LLM #19861（10-09 强候选补录）— PyTorch backend encoder-decoder + KvCacheConfig(dtype="fp8")：cross-attention KV 池按 fp8 尺寸计但访问按 bf16 步长越池——池 <2× 在飞需求时 IMA 崩溃**或落入其他映射内存时静默错 token**（L4 flan-t5-base e2e 复现，1.3.0rc29/1.4.0rc0 均复现）。documented。
7. **6.274** [新签名] vLLM #60174 — Qwen3.8-27B NVFP4 + DFlash2 + prefix caching：第三次请求起（前缀命中后）单 token（`duct`）无限重复、drafter 接受恒 0；v0.29.0 同请求全对，0.30.0 引入；隔离表排除 PR #57128/GDN prefill 后端/linear 后端/KV 池尺寸。documented。
8. **6.275** [新签名] SGLang #43731 — 并发启动两个实例时 scheduler ZMQ bind 失败静默重试 port+42：客户端仍连配置端口，**请求被送达并执行于另一个实例**（输出写入错误实例的 output file、另一实例永久等待）；~3/40 并发启动触发。documented。
9. **6.276** [新签名] SGLang #43514 — hybrid GDN + DFLASH spec + grammar 约束流量：整批请求一步坍塌为 token id 0 无限重复（含无 grammar 的纯文本请求）、accept 率三步内 0.48→0.00、无错误无 crash；两次生产事故（v0.5.19/v0.5.20 各一实例，~12% 流量乱语 5 天）。documented。
10. **6.277** [新签名] SGLang #43239 — NVFP4 KV + hierarchical cache + kernel io-backend：host pool 行几何按 device **元素**数而非字节数计算（FP4 打包 2 元素/字节），大前缀逐出回载后 **spec decode 静默永久失效**（accept 钉 1.00、target 输出字节精确正确、吞吐反降 25%、/flush_cache 不恢复）；生产 3h 触发 + 确定性 3/3 复现。verified。
11. **6.278** [新签名] flashinfer #6151 — trtllm_fmha_v2_prefill 在共享 workspace 前部留下非零 tile 计数器，XQA decode 把同一 workspace 前 8 MB 当零初始化 semaphore：同引擎共享 workspace 时**每次 prefill 后首个 decode 静默错注意力输出**（SM90/SM120；SGLang 已 workaround 切片偏移）。documented。
12. **6.279** [新签名] vLLM #61100 — Apertus-70B FP8 SM120 v0.25.1 全量输出退化（temp=0 无关的重复乱语直到 length），v0.22.1 同 checkpoint 正常；引擎健康 HTTP 200，CutlassFP8ScaledMM + FLASHINFER + FlashInfer 采样栈。documented（版本回归强证据、具体内核未定位）。
13. **6.280** [新签名] SGLang #43331（closed 10-09）— GDN cutedsl decode 内核按 `[N,HV,K,V]` 读写 #20283 起改为 `[N,HV,V,K]` 的 state pool：K==V（128）无形状错误，每步读转置 state → 全量垃圾（`The. be Ha人 II`）；对照 Triton 后端正确、精度测试 #20436 起被 skip；解释 #22472（B200 v0.5.10 同症状）。verified。

### 第 7 类 +4（7.108–7.111）

14. **7.108** [新签名] flashinfer #6316 — b12x W4A16 BF16 激活 × MXFP4：E8M0 scale 解码链每权重恒乘 2^-119、epilogue 2^119 全局补偿——**scale 字节 <121 时权重落 BF16 subnormal**（<114 时 FP4 0.5 归零），真实 checkpoint 字节分布 100–130 正中陷阱；cos 0.99999 门不可见。verified。
15. **7.109** [新签名] flashinfer #6210 — mm_mxfp8 对非连续 `out`（cutlass/cute-dsl/auto 后端）按紧凑 row-major 写入、无视真实 stride：结果垃圾且越界覆写相邻内存；仅 cudnn 后端正确、auto 默认解析到 cutlass 受影响。verified。
16. **7.110** [新签名] flashinfer #6310 — trtllm-gen context（prefill）内核 NVFP4 KV head_dim=256：V scale stage offset 8 按 linear 应用、漏 ×4 token interleave——上半 head（dims 128–255）用错 scale，每 prefill 损坏注意力输出（decode 内核与 head_dim 64/128 正确、K 不受影响；错误寻址参考模型与内核吻合到 FP8 底噪）。verified。
17. **7.111** [新签名] vLLM #61003 — DFlash2 draft 的 fp8 `modules_to_not_convert` 全局层号别名只补 `exclude_modules` 不补 `ignored_layers`：分组卷积投影被建成 fp8 线性、bf16 权重载入 fp8 参数（log 可见）；target 校验保输出正确但 acceptance 4.3→1.03、吞吐 210→46 tok/s。verified（含 workaround 复测）。

### 第 9 类 +1（9.33）

18. **9.33** [新签名] tt-metal #59969 — fused QK rope 的 cos/sin/trans_mat 分片按 2×batch 行填充 11 宽 core 集，Q/K 实际落 8 宽行：布局分歧处用户被别人的 cos/sin 旋转或读到无 cos 的 L1；validate 只查数量不查 core 集合。documented。

### 第 10 类 +5（10.104–10.108）

19. **10.104** [新签名] SGLang #43147 — GLM4 MoE tool 参数 `parse_arguments` 走 `ast.literal_eval`：含逗号标量（`1,000`）解析成 Python tuple 序列化为 JSON 数组 `[1,0]`。documented。
20. **10.105** [新签名] SGLang #43156 — HarmonyParser 工具调用后 filter-commentary 状态把 stripped/lower 后为 "commentary" **前缀**的 normal 事件静默丢弃：流式首 chunk 恰为 `"c"`/`"Co"`/`"comment"` 时答案开头被删。documented。
21. **10.106** [新签名] SGLang #43523 — deepseekv3 detector 贪心 `(.*)<｜tool▁sep｜>` 从**最后一个**分隔符匹配：一个流增量含两个完整 tool call 时只发第二个（第一个永不达客户端）；stream-interval 大或积压时必现。documented。
22. **10.107** [新签名] SGLang #43409 — `/v1/images/edits` 声明并接收 `mask` 上传但 handler 从不读取：inpainting 请求 200 静默跑全图无掩码编辑（自家 ComfyUI 集成正在发送被忽略的 mask）。documented。
23. **10.108** [新签名] SGLang #43107（closed 10-08）— ComfyUI executor conditioning 键 = shape + 首尾标量：共享前 ~75 CLIP token 的两个 prompt 键碰撞，后者 T5 上下文被静默替换为前者（Flux/Qwen-Image-Edit 实测）。documented。

### 第 12 类 +2（12.51–12.52）

24. **12.51** [新签名] vLLM #61026 — Qwen3-ASR `/v1/realtime` 静默丢弃 `session.update` 的 prompt 与 language（REST 路径同参数生效）：per-call 上下文（专有名词）实时流式会话全部丢失，无错误。documented。（Cov 归 NEW——API 参数声明/消费契约缺口为独立子签名，非 12.42 加载器公式族）
25. **12.52** [新签名] SGLang #42544 — `/update_weights_from_disk` 四个 API 参数（is_async/keep_pause/token_step/num_paused_requests）声明+文档化但树内无任何读取方；响应字段硬编码 0 且 DP>1 返回类型与文档不符——RL 训练侧静默错误契约。documented。

### 更正计数（最终口径）

- 初稿曾拟 13.43（并入 6.274/6.280 语境不占号，从未编目）与 12.51 var:12.42（经核对 12.42 为 rsLoRA √r 公式族、与本条不同机制，按 var: 规范定为 NEW）。
- 最终：第 1 类 +4、第 4 类 +1、第 6 类 +8、第 7 类 +4、第 9 类 +1、第 10 类 +5、第 12 类 +2 = **25 条，全部新子签名（NEW），无第 13 类新增**。

## 拒绝 / 留观（本日主要候选）

- **vLLM**：#61117（CPU rejection sampler 接受 -1 placeholder draft → 结构化输出约束行被绕过——机制清晰含内核级 repro，但 CPU 无 Triton 构建（macOS arm64）非生产主路径，**强留观**，e2e 任务影响证据补齐后优先）、#61027（CPU MXFP4 MoE AMX 门拒绝 EPYC——fail-loud 启动失败）、#60860（DSV4.1-Flash 长 decode unspecified launch failure——崩溃域）、#60119（GLM-5.3-Flash NVFP4 B300 IMA——崩溃域，10-09 已判维持）、#60160（Qwen3.8-27B-FP8 MI355X 高并发 accuracy collapse——**强候选**：ROCm AITER 域、与 8.99（vllm_rocm_qwen38_27b_fp8_aiter_accuracy_collapse_high_concurrency）同模型同栈同症状，为该条 SGLang 侧姊妹实例，并入 8.99 语境跟踪不占号）、#61069（GB10 NVFP4+Marlin+MTP 日级批量崩溃——崩溃域）、#61003 已编目 7.111、#60845（CUDA graph 死锁 TP——挂起域）、#60928（非输出 rank 异常后 ranks desync——错误传播治理域）、#60523/#60366/#60931/#60932/#61024（RFC/Roadmap）、#61072（27B draft 配对回归分析——分析帖）、#60771（DSV4-Flash-0731 重复乱码——10-09 已判环境特定无诊断留观维持）、#60765（Qwen3.5 PP×dflash 缺 opt-in 报错——10-09 已判 fail-loud 维持）、#60700（W4A8 %256 assert——10-09 已判 fail-loud 维持）。
- **SGLang**：#42981 AMD fp8e4m3fnuz、#43324、#41939（10-09 已判留观维持）、#43146（apply_reasoning_enabled 读侧规则不镜像——读侧过滤规则域，行为语义错误真实但属 chat 呈现层过滤而非生成损坏，留观）、#43133（required plugins fail-fast——feature）、#43098/#43314/#43519/#43699（Roadmap）、#43609（安全中间件 feature）、#43411（ComfyUI worker 批 RFC）、#43116（Qwen-Image-Edit 丢首图后参考图——diffusion 执行器域强候选，与 10.107 同插件族、本日 43107 已编目，此条留观下轮）、#43184/#43487/#43488（FluxAdapter/Kontext/ControlNet 丢条件，closed——ComfyUI 扩散域整批留观）。
- **flashinfer**：#6298（B300 2-CTA FP4 MoE UTCBAR 线程级延迟注入下 sticky error——**非天然触发**（NVBit SASS 延迟注入），崩溃面、无错值行为（明示 every clean trial bitwise correct），竞态存在性证据留观不占号）、#6112（GB300 autotune 跨进程 tactic 不一致——性能/复现域，与 4.12 族相邻但无错值证据留观）、#6121/#6232（测试断言栈错误）、#6165/#6189/#6304/#6313（feature）、#60660 已编目 6.270 同期条目（MNNVL allreduce IMA——崩溃域留观）。
- **tt-metal**：#59663（无参 GELU 融合激活静默走近似核——1.137 族语境，10-09 已判留观维持）、#59824（update_cache num_users>32 OOB 页——8.96 关联姊妹，10-09 已判待修复版本留观维持；**证据与 8.96 同级，机制为写侧越界而非行内毒化，子签名成立，下轮优先补录**）、#59761（pow/rpow fp32 精度 362 ULP 随指数增长——LLK 精度议题、ULP 量级低于已编目 digamma/logit 案例且 pow 非推理主路径热区，留观）、#59930（logit fp32 近 0.5 处 8,767 ULP——同上精度议题域，现有测试已捕获（eps=0.25 案例失败），修复方向明确，留观）、#60023（pow +inf 解码为 2^128——10-09 已判窄域留观维持）、#59752/#59765/#59849/#59914（LLK 审计/留观系列维持）、#59556/#59674/#59713–#59717（性能/tracker 工作项）、#59757（gelu_bw 饱和——训练侧）、#59825（group norm backward bf16 累加饱和——训练侧）、#59247 已编目 1.285、#59818 已编目 1.286、#59561 已编目 8.96、#59689 已编目 1.289、#59530（F32 宽度 concat HEIGHT_SHARDED 截断 TF32——**强候选**：数据搬运 op 且 mantissa 丢失 13 bit、root cause 定位到 UnpackToSrc vs UnpackToDest 且姊妹工厂已有正确实现、8190/8192 元素错。本日合并量控制未编目，**下轮优先**，预计 var:1.266（数据搬运 dtype= 转换语义缺失族））。
- **triton**：#12169（Gluon setmaxnreg warpgroup 内不一致——生成器讨论域）、#12170（mbarrier bra.uni 分支——PTX 正确性讨论留观，10-09 已判维持）、#12173（文档契约）。
- **transformers**：#49337 已编目 13.42（MetalLinear）、#49488（架构讨论）、#49497（ZeRO-3 每 rank 全量读 checkpoint——训练侧性能/正确性边界，非推理域）。
- **TensorRT-LLM**：#20083（Ray NcclEP 从 MPI_COMM_WORLD 建 1-rank communicator——分布式编排域、失败模式为挂起/错误而非静默错值，留观）、#19954 已编目 7.104、#19955 已编目 1.273、#20035 已编目 1.274、#19994（Whisper serving parity——feature）。
- **arxiv**：2609.38477（seed-based weight quantization 安全加固——防御方法论文不收录）；2610.06251（HQQ 量化早停改变答案——量化早停议题与第 7 类相邻但为方法评估论文，非栈实现故障，不收录）。

## 来源回访（近 7 日新增条目引用来源 + 额度轮巡旧来源）

本日回访 5 个旧来源（core REST 快照）+ 借 search 快照确认窗口内既有条目无状态翻转：

- **8.96（tt-metal #59561）**：仍 open、2 评论；姊妹 #59660 重复报告已 closed、#59824（num_users>32 OOB 写）为写侧独立子签名维持留观（见上）。
- **6.272 族（#59716 等迁移 tracker）**：M3 prefill tracker 步骤推进中，非来源状态翻转。
- **10-09 合并的 65 条引用来源**：约 200 个中本日 core 额度（60/h 上限、中途耗尽一次）不足以全量评论级深查；窗口内活跃 issue 的 state/comments 已由 q1–q8 search 快照覆盖确认无异常翻转（快照含 state/comments 字段），评论级深查缺口滚动至下次扫描。
- 旧 open 来源轮巡不足说明：本日 core REST 额度大部分用于新候选正文深读（37 次成功），按"高价值优先"原则回访上述来源；3 个旧 open 长期来源（如 6.261 #41617、10.87 #58824、4.27 #58973）的评论级复查**未在本日执行**，列入未覆盖范围，下轮回访。

## 合并动作摘要

- 主文档新增 **25 条**（全部新子签名）：1.291–1.294、4.31、6.273–6.280、7.108–7.111、9.33、10.104–10.108、12.51–12.52
- 总条数 1317 → **1342**；编号编目 1203 → **1228**（K* 扩展 114 不变）
- 分类分布：第 1 类 289→293、第 4 类 30→31、第 6 类 272→280、第 7 类 107→111、第 9 类 32→33、第 10 类 103→108、第 12 类 50→52
- 含 10-09 未覆盖清单 4 个强候选的全部补录（6.273、1.291、1.292、1.294）
- 观察：①**FP4/MXFP4 解码链的 scale 语义错误集中爆发**（7.108 b12x E8M0 2^-119 恒偏、7.110 trtllm-gen V scale 漏 ×4 interleave、6.277 host pool 元素/字节混淆）——同一量化格式的 scale 寻址在三个独立栈各有空洞；②**共享 workspace / 共享 state pool 的隐式契约**（6.278 prefill 计数器污染 XQA semaphore、6.280 K==V 掩盖 state 转置、6.275 port+42 静默重绑）；③**"声明但不读"契约缺口新形态**（12.52 API 参数无消费者、10.107 mask 上传无读取方）与"校验只查数量不查集合"（9.33）延续"校验存在但校验错对象"主线。
- 未覆盖范围：tt-metal #59530（F32 concat TF32 截断，强候选下轮优先）、#59824（num_users>32 OOB 写，下轮优先）、vLLM #61117（CPU rejection sampler -1 接受，CPU 域留观）、SGLang #43116（ComfyUI 丢参考图，下轮）；10-09 条目约 200 来源与 3 个旧 open 来源的评论级深查未执行（search 快照确认无 state 翻转）；arxiv 本日查询词面之外的论文不在覆盖范围。
- 导出与校验：`python3 export_faults.py`（1342 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1342 条：正文汇总、CSV、XLSX、字段边界一致**（结果见下）

## Git 提交

- 暂存：`FAULT_MASTER_REFERENCE.zh.md`、`daily_scan_2026-10-12.md`、`fault_catalog.csv`、`fault_catalog.xlsx`（仅本轮实际变更文件）
- 提交与推送结果见最终摘要（Git 操作在摘要前完成）。

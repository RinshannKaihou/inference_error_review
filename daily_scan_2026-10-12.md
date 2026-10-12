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

---

# 晚间轮增补（2026-10-12 23:01 UTC 起，当日第二次扫描）

- **执行时刻**：2026-10-12 23:01 UTC（cron）；Git 前置检查通过（main 干净、fetch 成功、HEAD == origin/main == `251ce97`，即上方早轮 02:28 UTC 的提交）。
- **扫描窗口**：GitHub `created:>=2026-10-12`（cutoff 按 UTC 计算；语义为"早轮 commit 后新建的 issue"，与早轮 created:>=2026-10-05 窗口的重叠区间条目按早轮判定跳过）。arXiv 无 created 限定、按提交日降序核对（近月命中全已编目）。

## 晚间轮 GitHub Search API 查询清单（全部带 created:>=2026-10-12）

| # | 查询 | total/fetched | incomplete | 结果 |
|---|---|---|---|---|
| 1 | vllm is:issue label:bug | 6/6 | false | #61252 #61229 #61206 #61197 #61172 #61170 |
| 2 | vllm silent wrong output | 4/4 | false | #61150 #61229(重复) #61225 #61157 |
| 3 | vllm quantization correctness | 0/0 | false | 无命中 |
| 4 | vllm KV cache | 11/11 | false | #61225 #61208 #61160 #61131 #61229 #61281 #61144 #61150(重复) #61157(重复) #61239 #61252(重复) |
| 5 | sglang label:bug | 0/0 | false | 无命中（sglang 不用该 label 流） |
| 6 | sglang wrong OR incorrect OR corrupt OR gibberish | 7/7 | false | #43857 #43765 #43929 #43874 #43738 #43888 #43741 |
| 7 | triton wrong OR incorrect OR numerical | 0/0 | false | 窗口内无命中 |
| 8 | tt-metal label:bug | 11/11 | false | #60342 #60338 #60294 #60286 #60285 #60277 #60262 #60246 #60245 #60244 #60243 |
| 9 | tt-metal wrong OR incorrect OR silently OR corrupt OR PCC | 21/21 | false | 上表 + #60325 #60321 #60318 #60319 #60241 #60347 #60249 #60247 #60252–#60256 |
| 10 | TensorRT-LLM 全部 | 6/6 | false | #20170 #20169 #20168 #20141 #20138 #20126 |
| 11 | transformers label:bug | 5/5 | false | #49540 #49519 #49516 #49513 #49510 |
| 12 | flashinfer 全部 | 8/8 | false | #6391 #6389 #6354 #6352 #6350 #6348 #6347 #6344 |

- 逐组 fetched==total、incomplete=false；查询间 ≥7s 间隔，无限速 403、无 Validation Failed。
- **执行事故记录**：第一批执行时 shell 变量未展开，vllm KV 与 sglang 两组曾退化为全时间窗查询（命中 5601/旧库）——该两次结果作废，已带 cutoff 重发（表中数字为重发值）。
- **core REST 限额**：issue 详情调用触 403（未认证 60/h 被占）。正文深读改经两条合规路径补齐：① search API 快照自带 body（3000 字符）；② GitHub issue 页面 HTML 内嵌 JSON 直接抓取全文——14 个候选 issue 全部取得完整正文，无正文缺口。状态回访经 search 单号查询完成。

## 晚间轮新增合并条目（20 条：15 NEW + 5 var；1342 → 1362）

### vLLM
- **6.281** [vllm-project/vllm#61150](https://github.com/vllm-project/vllm/issues/61150) PIECEWISE cudagraph + splitting_ops 含 moe_forward：重放陈旧地址静默垃圾（四臂对照 0/10 vs 10/10）→ NEW（config）。
- **6.282** [#61229](https://github.com/vllm-project/vllm/issues/61229) NixlConnector lease 过期无条件释放 KV 块 × 在途远端 READ：decode 静默消费复用后他请求内容，5/5+5/5 确定性复现 → NEW（other_stack）。
- **6.283** [#61160](https://github.com/vllm-project/vllm/issues/61160) NIXL PD 分离 + 默认 prefix caching 共享前缀跨请求内容污染：46/48 顺序、122/128 并发串答，关 prefix caching 即消失 → NEW（other_stack；机制未定位留待上游）。
- **6.284** [#61208](https://github.com/vllm-project/vllm/issues/61208) MooncakeStoreConnector 把 list 值层缓存 reduce 成 cache[0]，V 视图被丢：save/lookup/load 全成功后 fresh consumer 损坏输出 → NEW（other_stack）。
- **8.100** [#61225](https://github.com/vllm-project/vllm/issues/61225) nvfp4 KV 写内核 head-major 内部公式 × reader 实际 stride，QSA 后端默认 BLNHC token-major 布局下双向错位：全上下文长度乱、NIAH 0/N、自建 buffer 测试结构性失明 → var:8.53（nvfp4 KV 布局契约族第三站）。
- **10.111** [#61172](https://github.com/vllm-project/vllm/issues/61172) Anthropic /v1/messages system 非文本块（image）静默丢弃，200 成功 → NEW（config）。
- **10.112** [#61206](https://github.com/vllm-project/vllm/issues/61206) qwen3 reasoning parser 剥离 tool 参数内字面 `</think>`（双 parser 串联边界）+ 同 issue 另两缺陷（cutoff 报告、tool result 配对错序）→ var:10.42。
- **12.53** [#61197](https://github.com/vllm-project/vllm/issues/61197) MTP draft 不继承 target --revision、从默认分支加载（0.29→0.31 回归，ZeroDivisionError 形态；静默错误权重风险为推断、留观）→ NEW（config）。

### flashinfer
- **8.101** [flashinfer-ai/flashinfer#6389](https://github.com/flashinfer-ai/flashinfer/issues/6389) trtllm-gen sparse MLA FP8 64-head persistent KeepsAb 内核随机 16-head tile 损坏（NaN LSE）；32×2/128/BF16 对照全对 → NEW（hw_specific，B300）。

### SGLang
- **9.34** [sgl-project/sglang#43857](https://github.com/sgl-project/sglang/issues/43857) sglang-mm token layout u32 回绕（≥2^32 token prompt）：M-RoPE 守卫用真实 64 位长度校验回绕 pair 恒过，位置表静默错位；CPU-only 源码级 repro → NEW（other_stack；verified）。
- **9.35** [#43738](https://github.com/sgl-project/sglang/issues/43738) DSV4.1-Flash 多轮会话命中钉死 11,520（恰系统提示 45 页），60–80K 前缀逐轮全量重 prefill；GLM-5.3-Flash 同栈正常、三成因排除、机制未定位 → NEW（other_stack；主影响性能，正确性面留观）。

### TensorRT-LLM（Qwen3-VL 视频管线双缺陷）
- **10.109** [NVIDIA/TensorRT-LLM#20141](https://github.com/NVIDIA/TensorRT-LLM/issues/20141) total_num_frames 重写不重算 fps：默认路径视频恒 4 帧摘要 + 时间戳按 decoded/source 压缩 → NEW（other_stack）。
- **10.110** [#20138](https://github.com/NVIDIA/TensorRT-LLM/issues/20138) 多模态 E/PD 交接丢 per-frame 时间戳 token 与帧标记，PD 侧对 mRoPE ids 无校验按短 prompt 切片 → NEW（other_stack）。

### transformers
- **9.29 补充来源**（非新条目）[#49519](https://github.com/huggingface/transformers/issues/49519)：v4 风格 rope_scaling setter 整体替换 rope_parameters 丢 rope_theta——9.29/9.32（vLLM/SGLang 静默换 base 10000）的上游本体根因坐实（PR #49515 修复在途）；判定不变、根因面上收。

### tt-metal（含早轮"下轮优先"2 项补录）
- **1.295**（补录）[#59530](https://github.com/tenstorrent/tt-metal/issues/59530) F32 HEIGHT_SHARDED 宽 concat 经 SrcA unpack 截断 TF32（8190/8192 错、与 TF32 参考 bit-exact）；早轮预测 var:1.266 族，实读后判独立机制（unpack mode 差错）→ NEW（hw_specific）。
- **1.296**（补录）[#59824](https://github.com/tenstorrent/tt-metal/issues/59824) update_cache num_users>32 且非 32 倍数：tile padding 行被当真实 user 越界写 cache 外 DRAM（源码行级 + worked example）→ var:8.96（写侧姊妹：地址面 vs 数值面）。
- **1.297** [#60245](https://github.com/tenstorrent/tt-metal/issues/60245) convert_u32_to_bfp exp-A 下溢分支后无条件加 hidden bit：微小值膨胀 ±2^-15（1e-6→3.05e-5）→ NEW（verified）。
- **1.298** [#60244](https://github.com/tenstorrent/tt-metal/issues/60244) tilize 转置×非方 face：heap overflow READ/WRITE + 往返不闭合、边界检查打印后仍 memcpy → NEW（verified）。
- **1.299** [#60262](https://github.com/tenstorrent/tt-metal/issues/60262) 单行 sort 哨兵值与真实值并列：pad index（≥n）泄入返回（values 全对、indices 错）→ var:1.23。
- **1.300** [#60246](https://github.com/tenstorrent/tt-metal/issues/60246) equal_within_n_sig_figs 字符串比较不对称假匹配（0.5 vs 5.5 判真）→ var:1.234（验证面恒绿族）。
- **10.113** [#60342](https://github.com/tenstorrent/tt-metal/issues/60342) Phi-3 强制 trust_remote_code：零 mismatch 报告下 130 参数随机初始化、双设备垃圾输出而 accuracy demo 恒绿 → var:12.46（报告成功≠权重到达族；verified）。

## 晚间轮不收录（判定与理由）

- vLLM **#61157** GLM hybrid-SSM+MTP 并发请求到达瞬间 deep-ctx agent session 退化（伪造 tool args、自失忆；#58454/#58368 修复后仍现）——取证极详尽但机制未定位、无合成复现，强留观下轮优先。
- vLLM **#61281** Mamba align 边界状态块中途释放/驱逐次序倒置——纯 KVCacheManager 级无 GPU 复现、主影响性能，正确性面留观。
- vLLM **#61239** FlashInfer metadata builder 异步调度 pinned workspace 覆写竞态（#57214 起）——IMA/hang 崩溃域主导，留观。
- vLLM #61144（MI300X SR-IOV VF XGMI 崩溃 fail-loud）、#61252（Ministral 3 启动失败 fail-loud closed）、#61170（audio 边界拒绝）、#61131（CPU 无 Triton fail-loud）。
- SGLang #43765（aiter 移除 pa_decode_sparse import 失败）、#43929（bench 计数工具域）、#43874（ComfyUI 插件 14 bug 打包帖，待单项证据整批留观）、#43888（计费 u32 回绕，计费元数据域）、#43741（fp8 KV QSA crash fail-loud）。
- flashinfer #6391（fast_decode_plan 采纳 host 元数据 IMA——拒绝面/防御缺失，无静默错值证据）、#6344/#6347/#6348/#6350/#6352/#6354（CI/测试基建/feature）。
- transformers #49540（tp_plan DTensor 混用 fail-loud）、#49516（T5 SDPA 回退纯性能）、#49513（closed）、#49510（auto_map 加载）。
- TRT-LLM #20170（NVRTC uv 环境 fail-loud）、#20168/#20169（EAGLE-3 延迟性能；20168 为 20169 重复帖已 closed）、#20126（IPv6 部署域）。
- tt-metal #60338（mesh trace 挂起）、#60294（concat NOC burst 挂起——hang 域）、#60277/#60285/#60286（Quasar 实验性 API 窄域）、#60241（LLK 硬件条件静态审计）、#60243（FreeListOpt OOM 泄漏——资源治理）、#60247（matmul bias 逻辑 h 校验缺失——候选：h∈(1,M) padding 行入输出真实、PR #59819 在途，合并量控制留观下轮优先）、#60249（1D mcast 校验用 device grid——弱触发）、#60252–#60256（squad-plan 重构批）、#60318/#60319/#60321/#60325（MiniMax-M3 batching 工作项）、#60347（CI 超时）。
- triton 窗口 0 命中——无候选可判。

## 晚间轮来源回访

1. **9.29 族根因坐实**：transformers #49519（open）——9.29（vLLM）与 9.32（SGLang #41227，GSM8K 161→106/200）的 RoPE base 静默替换统一上收到 transformers v4 setter 缺陷；PR #49515 修复在途。判定与条目数不变，9.29 补充来源与说明。
2. **6.261**（SGLang #41617）仍 open、修复 PR #41767 仍 open（search 快照核实）。
3. **早轮"下轮优先"两项**：tt-metal #59530 仍 open（修复 PR #59639 open；#59634 为 closed 前身）、#59824 仍 open——正文本日深读并补录为 1.295/1.296。
- 未覆盖：core REST 限额致本轮无法做评论级复查（10.87/4.27 两个旧 open 长期来源仅确认无 state 翻转，评论正文未读——与早轮相同的未了项，顺延下轮）。

## 晚间轮 arXiv

9 组查询全部 HTTP 200（引号短语+AND）：`"silent data corruption" AND "LLM inference"`、`"bit flip" AND "LLM inference"`、`"KV cache corruption"`、`"numerical correctness" AND inference`、`"soft error" AND GPU` 及 4 组窄化变体（`"silent data corruption" AND inference`、`"inference SDC"`、`"KV corruption"`、`"wrong output" AND vLLM`，后 3 组 0 命中=窄查询无匹配非失效）。命中全部已编目或已判：SProbe（2609.19743）、JITterFlip（2608.29745）、2601.19912、2609.16742 等；无新编目。

## 晚间轮合并与观察

- 合并后总数 **1362**（1342 + 20；编号编目 1228 → 1248）。分布：第 1 类 +6、第 6 类 +4、第 8 类 +2、第 9 类 +2、第 10 类 +5、第 12 类 +1。滚动汇总、轴表、目录计数均已按实际解析结果更新。
- 观察：①**nvfp4 KV 布局契约族第三站**（8.100）——早轮 7.110（scale interleave）、6.277（pool 几何）之后 writer/reader stride 公式分裂再现"测试视图窄化 × 真实 allocator 布局"主线；②**跨请求内容泄漏三连发**（6.282 lease 竞态、6.283 PD prefix cache 串扰、6.275 早轮 ZMQ 重绑）——三种独立机制、全部 HTTP 200 零告警；③**Qwen3-VL 视频管线双缺陷同日暴露**（10.109 单位制错配、10.110 分离交接无校验切片）；④**"报告成功≠权重到达"**（10.113 零 mismatch 报告下 130 参数随机、demo 恒绿）与 12.46 成对。

## 晚间轮 Git 交付

- 变更文件：`FAULT_MASTER_REFERENCE.zh.md`、`daily_scan_2026-10-12.md`、`fault_catalog.csv`、`fault_catalog.xlsx`（导出生成）。
- 提交：`daily scan 2026-10-12: inference`（第二次扫描增量）；推送 `origin HEAD:main` 后以 `git ls-remote` 回读核对远程 SHA == 本地 HEAD。结果见最终摘要。

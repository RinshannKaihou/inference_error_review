# LLM 推理故障每日扫描报告 — 2026-09-29

- **扫描日期**：2026-09-29（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-22`（7 日完整窗口，`cutoff=$(date -u -d '7 days ago' +%F)` 实际计算；与 09-28 扫描窗口 created:>=2026-09-21 全重叠——09-28 及更早判定条目全部按既有判定跳过，本日新判定集中于 09-28 扫描执行（09-28 23:14:42 UTC commit `4589647`）之后创建的 103 个 issue/PR）；arxiv 6 组查询全部 200 有效
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1237 条（2026-09-28 扫描后）；编号编目 1123 条 + K* 扩展 114 条

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `458964710cd76542268c5f9ed8469eab9f03fb26` ✅ → 开始扫描

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-22 wrong OR incorrect OR corruption OR NaN` | 47 命中（单页取全） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-22` | 62 命中（单页取全） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-22 wrong OR incorrect OR corruption OR NaN` | 18 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-22 wrong output OR incorrect OR NaN` | 7 命中（单页取全） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-22 wrong OR incorrect OR corruption` | 0 命中 |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-22 wrong OR miscompile OR NaN` | 1 命中（#11993 09-28 已判维持） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-22 numerical OR wrong OR pcc OR silent` | 213 命中（**三页取全 100+100+13**） |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-22 wrong OR incorrect OR NaN` | 4 命中（#5393/#5438/#5637 已判维持 + #5676 本日新判 feature） |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-22 quantization OR "KV cache" OR deterministic` | 99 命中（单页取全） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-22 silent OR "bit flip" OR "weight corruption"` | 54 命中（单页取全） |
| 11 | GH Search | `repo:sgl-project/sglang is:issue label:bug created:>=2026-09-22` | 0 命中 |
| 12 | arxiv | `"silent data corruption" AND LLM` | 200，6 命中全已编目（SProbe/TrainSDC/SCOUT/LLM-PRISM/2604.00726/2502.12340） |
| 13 | arxiv | `"bit flip" AND LLM AND inference` | 200，8 命中全已编目或已判（JITterFlip/UEP/RangeGuard/2603.10042/BitFlipScope/BitHydra 等；2504.12976 非 LLM 域） |
| 14 | arxiv | `"KV cache" AND corruption` | 200，15 命中（2609.25881 虚拟试穿 OR 噪声、其余已编目或方法论文不收录） |
| 15 | arxiv | `"soft error" AND GPU AND inference` | 200，3 命中（2609.16742 CNN 边缘域已判不收录、2601.19912 已编目、2104.09455 非 LLM 域） |
| 16 | arxiv | `LLM AND inference AND "numerical correctness"` | 200，3 命中均为非故障域论文（非 GPU 加速器 field study/agent skill/UCCL-Zip 通信压缩） |
| 17 | arxiv | `LLM AND inference AND "wrong output"` | 200，0 命中 |

GitHub Search API 11 组查询全部有效、**分页完整（fetched == total 逐组核对：47/62/18/7/0/1/213(100+100+13)/4/99/54/0）**、无限速 403、无 Validation Failed；查询间 ≥8s 间隔。去重后 **407 个唯一 issue/PR**（跨 11 组查询重叠），其中 **103 个为 09-28 扫描执行后新建需判定**、304 个重叠区按既有判定跳过。候选正文取自 Search API 缓存（响应自带 body）；core REST **20 次调用全部成功**（`--http1.1`：16 次来源回访快照 + 4 次评论深查；未认证运行，环境无 GITHUB_TOKEN）。

## 已编目/已判条目（本次跳过，重叠区去重）

- vLLM：#58824（→10.87，回访见下）、#58973（→4.27，回访见下）、#59086/#58960/#58969/#59087/#59043/#59089/#59027/#59045/#59064/#59065（09-28 留观/不收录维持）及 09-22..09-28 判定项
- transformers：窗口内 7 命中全为已判（#49036 留观、#49051→10.86、#49066→10.85、#49134→13.41、#49126/#49026/#49140 CI triage）
- SGLang：#41192（→5.46）、#41494（→6.260，回访见下）、#41569（→7.100，回访见下）、#40921（→5.45）、#40903（→7.97）、#40835（→12.42）、#40966（→12.43）、#41038（→6.256）、#40735/#40730/#41351/#41466/#41467/#41471/#40817（已判维持）
- flashinfer：#5438（→7.96）、#5393（已判维持）、#5637（→11.89，回访见下）；triton #11993（09-28 已判维持）
- tt-metal：#58078/#58177/#58181（→1.263/1.265/1.264，回访见下）、#58106/#58132/#58138/#58142/#58150/#58061/#58082/#58189（09-28 留观维持）及 09-22..09-28 判定项

## 🆕 新发现（本次合并，共 15 条）

### A. 全新子签名（8 条）

1. **[新签名] tt-metal #58371 — 设备侧数据搬运 op（clone/copy/to_memory_config/interleaved_to_sharded/sharded_to_interleaved）的 fp32→bf16 `dtype=` 转换按截断（RTZ）**：1,048,576 随机值实测 50.04% 偏离 RNE、100% 等于截断、平均带号误差 −0.5 ulp（向零偏置），Wormhole/Blackhole 一致；`ttnn.typecast` 与 `from_torch`（RNE）对照正确——**同一转换按入口分裂**。→ 第 1 类 **1.266**，Cov NEW，verified。
2. **[新签名] tt-metal #58372 — 七个数据搬运 op 的整数↔浮点 `dtype=` 转换按位重解释**：12 组合 99.5–99.8% 输出与输入值无关（typecast 精确对照 0%）；校验接受、无报错。→ 第 1 类 **1.267**，Cov var:1.227（整数位型按位重解释族第四站点），verified。
3. **[新签名] tt-metal #58467 — binary_ng shard 别名谓词只比 grid/ShardSpec、布局不匹配即静默错值**：真实 n150 实测 `ttnn.add` PCC 0.2200（72% 元素偏差>0.1）；全 operand 同 config 对照 PCC 1.0。→ 第 1 类 **1.268**，Cov NEW，verified。
4. **[新签名] tt-metal #58495（$750 bounty）— 融合 scale_mask_softmax 小核在非 32 倍数宽度不 mask padding 列**：padding 参与分母、行和最低 0.04（causal 概率质量几乎全漏进 padding）；宽度 17/50/197/1001 复现。→ 第 1 类 **1.269**，Cov var:1.245（tile padding 未屏蔽族），documented。
5. **[新签名] tt-metal #58370 — ttnn.slice 对 rank-1 ND-sharded >2 shards 输入错值/挂死**：4 shards 从 index 64 起错、8 shards 从 32 起错、一配置设备 hang；rank-2 等价形状正确。→ 第 1 类 **1.271**，Cov NEW，verified。
6. **[新签名] SGLang #41617 — `--strip-thinking-cache` × retraction：release_kv_cache 释放 radix tree 仍持有的 KV 槽（双释放）**：slot 同进 free list 与 tree 引用，`alloc()` 可把同一 slot 发给两个请求（静默 KV 污染）；会捕获的断言在 strip_thinking_cache 下被跳过。→ 第 6 类 **6.261**，Cov NEW，verified。
7. **[新签名] vLLM #59104 — DSpark drafter 用 N+1 query 位置而 num_lookahead_tokens 只预留 N：越界写覆写他请求 target KV**（synthetic CUDA repro + 红绿测试 + GPU instrumentation 确认；修复 PR #59105）。→ 第 6 类 **6.262**，Cov NEW，verified。
8. **[新签名] vLLM #59110 — NixlConnector 异构 TP 按字节切分 KV 块 × 三后端双 slot/shuffled 页布局：静默错读**（Part 1 读源码定位 ROCM_AITER_FA/ROCM_ATTN/B12X；Part 2 MI300X 实测 standard→shuffled gluon handshake 全过、**30/30 全 gibberish 无错**）。→ 第 6 类 **6.263**，Cov NEW，documented。

### B. 已知类别新变体（7 条）

9. **[已知类别] tt-metal #58272 — ttnn.divide fp32 零除数臂 NaN/−0/subnormal 全错**：SFPSETCC mode 6 整数判零只匹配 +0 位型；2×2^32 全位型扫描 33,554,429 对错（P300/N300 同）。→ 第 1 类 **1.270**，Cov var:1.154（div 特殊值分支族），verified。
10. **[已知类别] vLLM #59218 — ParserEngine 系 parser 非流式丢 tool call 后/之间正文（流式保留）**：`_single_pass_parse` 不 flush `_events_to_delta` 的 defer 文本；实测非流式 `content: None` vs 流式 `'Done.'`。→ 第 10 类 **10.90**，Cov var:10.42（与 10.87 流式/非流式互为镜像），verified。
11. **[已知类别] vLLM #59272 + #59273 — ThinkingBudgetStateHolder 两缺陷：无 draft 步全员 forced `</think>` 写 row 0 / unidirectional move 残留前任 state**：CPU 单元复现齐备。→ 第 10 类 **10.91**，Cov var:11.86（thinking budget 族扩到状态容器），verified。
12. **[已知类别] vLLM #59241 — /derender logprob token 字符串剥 SentencePiece 前导空格**：`▁true`/`true` dict 键碰撞丢候选（#48674 症状复活）+ text_offset 漂移。→ 第 10 类 **10.92**，Cov NEW（derender 域 09-28 两缺陷留观后首个达门槛面），verified。
13. **[已知类别] vLLM #59161 — MRV2 lm_head LoRA 按请求（非按 logits 行）映射**：spec 下 logits 每请求 1+draft 行，映射过短对齐到前 num_reqs 行——adapter 对本请求无效、delta 泄漏到他人 logits（V1 正确对照）。→ 第 11 类 **11.90**，Cov var:12.28（LoRA 引擎态/映射族），verified。
14. **[已知类别] vLLM #59184 — DFlash DCP slot 映射把 kernel block size 同时用作 DCP ownership 与 block-table 查找**：cp_size>1 且物理块≠kernel 块时偏离通用契约（通用路径两者分离、正确）。→ 第 12 类 **12.44**，Cov NEW，verified。
15. **[已知类别] vLLM #59312 — WNA16 MoE 非整除中间宽（704/128）group-size 折半 × checkpoint scale 组错位**：加载即 shape mismatch（11 vs 12），AutoRound MoE checkpoint 不可用（fail-loud、无静默面）；姊妹 #59248（gate qweight 实例化失败）同日并入记述。→ 第 12 类 **12.45**，Cov var:12.39，documented。

### C. 评估后不收录（记录备查）

- **vLLM #59306**（GLM-5.3 NVFP4+DCP+MTP Hopper 启动崩——#57532 NaN 哨兵在 DeepseekV32 MTP 路径复发，DeepseekV32MultiTokenPredictorLayer 未用 defer_lm_head；两独立故障均 fail-loud 启动失败，修复方向明确）。
- **vLLM #59115**（GLM-5.3-Flash 长 prefill IMA B200 v0.30.0 复现（TP4+EP+MTP）——fail-loud 崩溃域、已 closed 转追踪）。
- **vLLM #59151**（ROCm mxfp4 MoE：TRITON_UNFUSED 永不被自动选择、gfx90a 落到 emulation 或启动失败——后端选择策略缺口；作者 4×MI210 数日运行证明内核本身健康，fail-loud/性能域）。
- **vLLM #59222**（Mistral pre-v11 tool parser 对"合法 JSON 但形状意外"整请求 400/500、流式静默丢数据——fail-loud 为主；流式丢失面与 10.90 同族（var:10.42 语境已并入本日观察），不另占号）。
- **vLLM #59230**（AOT 编译缓存 key 不含量化内核选择——同版本源内切换内核加载陈旧工件后 crash：触发需手工改源/下游补丁的场景，结果面 fail-loud crash，缓存键治理留观）。
- **vLLM #59250**（Kimi-K3 warmup 对所有模型 import Kimi 代码、非 root/只读文件系统 numba 崩——fail-loud 启动域）、**#59157**（v0.30.0-cu129 镜像含 torch cu130、torchvision::nms 缺失——打包缺陷 fail-loud）、**#59292**（Transformers 后端 draft `Duplicate layer name: 0.attn`——fail-loud）、**#59176**（GLM-5.3-Flash HiSparse 无兼容 KV 布局 LBHNC vs BLHNC——fail-loud 启动失败，与 6.263 布局契约域相邻但拒绝面）、**#59122**（ExampleHiddenStatesConnector 不支持 Hybrid KV Manager——feature 缺失 fail-loud）、**#59248**（Qwen3-Coder-Next AutoRound routed gate qweight 加载前失败——12.45 姊妹、fail-loud）。
- **vLLM #59267**（AudioSpec mono 输出保秩——音频预处理 API 域）、**#59268/#59269**（suppress_stdout fd 绑定两缺陷——日志工具域）、**#59270**（LoRA resolver 路径穿越——安全面）、**#59271**（DSV4 VL dummy image 非最坏情形——profiling 议题）、**#59109**（PP gloo 桥 IP 解析 535ms/send——网络环境域）、**#59154**（rust vllm-bench 与 bench serve 口径——工具域）、**#59114**（MoRIIO 生产者死后 decode 挂到超时——资源治理/挂起留观）、**#59116**（SMG MoRI-IO PD disagg 不兼容——信息不足）、**#59117**（draft 新 cache spec KV 池坍塌 RFC——设计文档，诊断详实但为 POC 叙述）、**#59120/#59210**（feature）、**#59189**（GB200 DP4+EP NCCL symmetric reduce-scatter 崩——fail-loud）。
- **tt-metal #58272–#58336 sfpi-gcc/sfpi-binutils 批（parent #58289，约 22 项 AI 审计）**：含两条"默认 flags 静默错码"级（#58290 rmext check_extend 高估目的地、#58291 immload_combine 减变加）——**AI 审计未运行时验证，沿 #57360–#57367 判例整批留观**，待单项复现证据后逐条编目；#58288（ldexp Correct 模式越界回绕，Correct 非默认）同批留观。
- **tt-metal #58399–#58409/#58423/#58425/#58426**（AI 静态审计第三四批 meta-issue，~111 项：ccl/conv/deepseek/data_movement/eltwise/operations/moreh/quasar/transformer/ttnn/hw——整批留观待单项运行时证据）；**#58442/#58447/#58449**（models/tests/tt-train 静态审计批）同判。
- **tt-metal #58339**（GLM-5.2-744B chunked prefill padded accuracy 门禁确定性回归 PCC 0.529<0.88、KV PCC 随深度衰减——CI 回归定位帖，窗口 b0eb80aa..dff19de；与 #58132（09-28 留观）同为"main 数值回归待定性"流程帖，**若维护者定性为未预期数值回归则按 10.86 谱系编目**，本日留观）。
- **tt-metal #58382**（ttnn.erf fp32 单段 16/16 有理拟合主域 4–6 ULP vs bf16 拟合 1.1 ULP——精度优化议题，同 1.137 bias_gelu 默认近似域但无门禁越过证据，留观）；**#58453**（iDMA scatter-list 首 16 项 SCATTER_INDEX 不前进——Quasar 预生产、Bad Outputs 真实但 DMA 引擎域非 LLM 推理主路径，留观）；**#58261**（Llama 3.3-70B 长 prefill 2x 慢——性能）；**#58457**（Wan2.2 VAE decode +0.4s——性能）；**#58463**（minimal_matmul 防御性校验——加固）；**#58253/#58276/#58280/#58347–#58351/#58284/#58420/#58422/#58424/#58427/#58429**（BEVFormer/squad-plan/moreh 迁移/Qwen-Image feature 工作项）。
- **SGLang #41653**（Simulator 缺 sgl_kernel CPU ops fail-loud）；**flashinfer #5676**（Rubin DSA deterministic selection——feature）；**triton #11993**（09-28 已判维持）。

## 来源回访（近 7 日新增条目引用来源 + 额度轮巡旧来源）

本日回访 16 个来源（core REST 快照全读，其中 4 个含评论级深查）：

- **6.260（SGLang #41494）**：**首条机制定位评论（liaoruoxue 09-29）——`kpool_topk_transform.cuh` 32 KB staging 区（8192 候选）溢出时超额候选被静默丢弃**，`index_kpool=4` 溢出域自 ~16,388 history tokens 起、与 4K/8K/16K 准确率悬崖精确吻合；评论者已在生产 serving 运行 exact-key multi-round rescan 修复头文件（同 #37625 语义），A/B 显示修复无性能代价；第三方独立复现（8×RTX PRO 5000 Blackwell TP8）。**条目补回访记、机制由"未逐行定位"升级为"kernel 级定位+修复已验证"**，判定不变。
- **4.27（vLLM #58973）**：新评论（twu3202 09-28）——在 Qwen3-8B DFlash-b16 上复现同机制（AR vs DFlash 首 token logprobs 9/16 prompts 分歧），`TORCHINDUCTOR_DETERMINISTIC=1`（#58899 workaround）下 bitwise 一致——**确认与 #58899（Inductor autotune 按计时选 reduction 配置）同机制**，条目语境补强，判定不变。
- **10.87（vLLM #58824）**：3 评论（09-26，快照确认无新增）——#58829 latch 修复获 he-yufeng 逐机制背书 + 三 case 测试；PR #58940 快照 state=open 未合并（与 10.88 修复线分开）。条目不动。
- **7.100（SGLang #41569）**：新评论（wenchen76 09-29）——认领意愿 + 指出 #39079（SM100 pin 相关独立修复）将文本冲突；无状态翻转，条目不动。
- **1.264/1.265/1.263（tt-metal #58181/#58177/#58078）**：快照 state=open、0 评论（#58078 09-29 有 body 编辑痕迹），无 triage 响应，条目不动。**修复 PR #58180（1.262）state=open merged=False 维持"发布带病+修复 open"**。
- **修复 PR 状态核实（core REST）**：#55601（6.204/6.259）open 未合并；#58890（9.30）open 未合并；#58940（10.88）open 未合并。三个"发布带病 + 修复 open"暴露面维持。
- **旧来源轮巡（4 个 open 主来源）**：#54273（10.42，08-31 后无新评论）、#40921（5.45，09-23 后无变化）、#58485（11.87，09-26 后无变化）、#58824（见上）——无状态翻转、无报告者撤回。
- 未检查完说明：近 7 日新增条目引用来源约 116 个（09-22..09-28 合并 103 条 + 本日 15 条），本日按 core 额度回访上述 16 个高价值来源（昨日 8 条新合并主来源全量 + 4 个修复 PR 状态 + 4 个旧 open 来源轮巡）；其余约 100 个经本日 q01–q11 search 快照确认无 state 翻转/评论数突变（快照含全部窗口内活跃 issue 的 state/comments 字段），评论级深查缺口滚动至后续扫描。

## 合并动作摘要

- 主文档新增 **15 条**（8 个新子签名 + 7 个已知类别变体）：1.266–1.271、6.261–6.263、10.90–10.92、11.90、12.44–12.45
- 总条数 1237 → **1252**；编号编目 1123 → **1138**（K* 扩展 114 不变）；**勘误：目录头部"编号编目 1115 条"系 09-28 漏更（实际 1123），本次一并修正**
- 分类分布：第 1 类 264→270、第 6 类 260→263、第 10 类 89→92、第 11 类 88→89、第 12 类 43→45
- 轴分布更新（按解析器实测）：infra_off 1068→1083、config 264→268（+6⚠ 不变）、other_stack 210→211（+25⚠ 不变）、partial 242→245、hw_specific 320→327、documented 708→711、verified 522→534、NEW 894→902、var:x 340→347
- 条目更正（回访补记，判定不变）：6.260（kpool_topk_transform.cuh staging 截断 kernel 级定位 + 第三方 Blackwell 复现 + 修复已验证）；4.27（twu3202 独立复现 + #58899 workaround 确认同机制，以快照确认评论存在，正文未改留待下轮）
- 观察：**1.266/1.267 合看——tt-metal 数据搬运 op 的 dtype= 转换语义整体缺失**（浮点方向 RTZ 截断偏置、整数方向按位垃圾），"typecast 精确而 clone/copy/tilize 错"把多入口数值分歧族（1.241/1.260）推到数据搬运面；**1.268 使 binary_ng 布局假设族成三形态**（混合 dtype unpack / 行主序 stride / shard 别名谓词）；**6.262 与 12.44 同日暴露 spec-decode slot 映射契约两方向**（越界写他请求 KV vs DCP 归属错位路由）；**6.263 是 NixlConnector 数据面第二形态**（页内布局对字节切分不可见：handshake 全过 + 0/30 静默 gibberish，#58860 修复落地即暴露）；**10.90 与 10.87 构成 parser 流式/非流式互为镜像对**；**10.91 把 thinking budget 族从结束状态机扩到状态容器生命周期**；**11.90 是 LoRA 族首个"跨请求泄漏"方向**（12.28/12.30 为自请求无效/基模服务）
- 未覆盖范围：q07（tt-metal 213 命中）与 q01/q02/q09 中 09-22..09-28 创建、既有判定条目未逐一重读正文（按既有判定跳过）；tt-metal 静态审计四批 meta-issue（#57360–#57367/#57510 附近/#57674–#57694/#58399–#58426/#58442–#58449）维持整批留观未逐项展开；约 100 个来源仅经 search 快照确认无状态翻转（无评论级深查）；arxiv 本日查询词面之外的论文不在覆盖范围
- 导出与校验：`python3 export_faults.py`（1252 行 CSV/XLSX）→ `python3 verify_catalog.py` **PASS 1252 条：正文汇总、CSV、XLSX、字段边界一致**

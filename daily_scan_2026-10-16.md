# LLM 推理故障每日扫描报告 — 2026-10-16

- **扫描日期**：2026-10-16（UTC）
- **扫描窗口**：GitHub `created:>=2026-10-09`（cutoff 实际计算）；重点判定 10-14 报告（23:56Z commit）之后创建的 issue。NPU 共享批次两个：`batch-20261015T230051994971-bhc7sd0o`（覆盖 2026-10-14T22:00:52Z → 10-15T23:00:51Z updated 增量）与 `batch-20261016T230043154367-2mujqpkn`（2026-10-15T22:00:51Z → 10-16T23:00:43Z）。**10-15 无日报**（当日扫描未执行，cron 前置批次存在），本轮补判两批次并集，判定范围实际覆盖 10-14T22:00Z → 10-16T23:00Z 全部 NPU 增量。
- **目标仓库**：原 GitHub 7 仓 + arXiv + NPU 19 推理源（GitCode/AtomGit 16 + GitHub 3）
- **编目基线**：1389（active）→ **1407**（active 1389 + pending_audit 18）

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `841c608d9df2192b859ef2a6f5f437ef4a0dd03b` ✅ → 开始扫描

## 一、原 GitHub 7 仓 + arXiv

### API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-09 correctness OR "silent wrong output" OR "wrong output"` | 37 命中（单页取全 incomplete=false） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-09 KV OR quantization OR "kernel numerics" OR SDC` | 114 命中（单页 100+incomplete=false，>100 部分本次未取第二页，见未覆盖） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-10-09 correctness OR "wrong output" OR silent OR garbage` | 38 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-10-09 correctness OR "wrong output" OR numerics` | 3 命中（无新候选） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-10-09 correctness OR "wrong result" OR silent OR numerics` | 10 命中 |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-10-09 miscompile OR "wrong result" OR numerics OR correctness` | 0 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-10-09 "Bad Outputs" OR correctness OR "wrong value"` | 27 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-10-09 correctness OR "wrong output" OR numerics` | 16 命中 |
| 9-14 | arxiv | 6 组查询（SDC+inference / bit flip+LLM / KV cache+corruption / soft error+GPU / numerical correctness / wrong output+LLM） | 全部有效返回；唯一新论文 2610.20980（kernel 优化脚本训练，非故障域拒收） |

`GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥9s 间隔、core REST 60/h）。8 组 Search 全部 incomplete=false（第 2 组 114>100 例外，已列未覆盖）。core REST 单号详情 7 次（61730/44735/6533/61803/20375 全文 + 2 组 search-body 核对），未触限流等待。去重 206 唯一 issue，其中 10-14 报告后创建的 56 个逐一标题判定，10 个静默正确性强候选读全文。

### 🆕 新发现（原 7 仓，6 条）

1. **INF-1391** [NEW] [vllm-project/vllm#61730](https://github.com/vllm-project/vllm/issues/61730) — lightning_attention prefill 历史状态少乘一次衰减（差 exp(s) 因子）：MiniMax/Bailing 线性注意力层 token 256 起全错、chunked prefill 续存量态首块即错；kernel 自身不一致（对角块对、stored-state 路径错到 RMS 2.6 倍）；单测因 0.01 缩放+0.15 容差漏检。documented。
2. **INF-1392** [NEW] [sgl-project/sglang#44735](https://github.com/sgl-project/sglang/issues/44735) — 混合 GDN 模型 HiCache host 层命中恢复后输出静默错误：命中计数满、延迟 20s→0.4s，needle 0/2、0/3；两种 io backend 同败；冷/device 命中正确。issue 已 closed 未见修复链接。documented。
3. **INF-1404** [NEW] [NVIDIA/TensorRT-LLM#20375](https://github.com/NVIDIA/TensorRT-LLM/issues/20375) — AttentionOp 从不填 cumSeqLensQ/Kv 指针：grouped MLA 生成核（mSelectsGroupedMla）读之越界，batch 4/16 静默错（sum -209.5/-218.6 vs -243.7）或图捕获 illegal address；默认核不读故 main 潜伏无症状。documented。
4. **INF-1402** [NEW] [tenstorrent/tt-metal#60821](https://github.com/tenstorrent/tt-metal/issues/60821) — ttnn.acosh_bw 域保护查 x²<1 而非 x<1：x<-1 负射线返回有限错误梯度（-2→+0.577，PyTorch NaN）、x=-1 且 grad=0 返回 +inf。documented。
5. **INF-1405** [NEW] [tenstorrent/tt-metal#60966](https://github.com/tenstorrent/tt-metal/issues/60966) — sigmoid 近似 LUT 对 ±inf 返回 NaN（0*inf+B 段；tt-llk#1701 tanh 同类硬件已确认）。documented。
6. **INF-1406** [NEW] [tenstorrent/tt-metal#60960](https://github.com/tenstorrent/tt-metal/issues/60960) — softplus bf16 对 x<-5（默认参数）尾部钳位为精确 0.0，torch bf16 为 0.0025~0.0067；-5.03125 处不连续跳变。documented。
7. **INF-1407** [NEW] [flashinfer-ai/flashinfer#6533](https://github.com/flashinfer-ai/flashinfer/issues/6533) — plan() 复用 pinned int workspace 异步 H2D 后无同步：下一次 plan 覆写致前次 run 读撕裂 plan info，FA2 paged prefill 5/5 MISMATCH（0.1~2.05）、偶发 kernel 卡数分钟；--guard 对照 5/5 OK；main 3637083a3 仍在。documented。

### 拒绝 / 留观（原 7 仓主要候选）

- **vLLM**：#61803（unified_attention block_table 越界读 used columns 之外——越界读面真实但正文仅环境块+标题，症状/静默影响证据未取到，留观下轮补正文）、#61912/#61905/#61768（KV Connector load-failure 上报三连：preempt 后迟到上报块留 cache、block-ID 复用后迟到失败影响无关请求、abort 前未处理失效块复用——与 10-12~14 HiCache/"恢复未完成即消费"族同主线，机制描述清晰但本轮未逐条深读正文，留观下轮优先）、#61775（offload 背压忽略失败二级存储——性能/冗余域为主）、#61730 已收录、#61766/#61738/#62094/#62003/#62028（fail-loud crash/startup）、#61806（DSpark draft token 数虚增——speculative 采样语义候选，本轮未深读，留观）、#61998/#61866/#61854（性能）、#61850（assert fail-loud）、RFC/Tracking 类（#61886/#61880/#61787/#62100/#62083/#62033）不占号。
- **SGLang**：#44504（update_weights_from_disk 未知视觉权重名静默跳过仍报 success——RL 权重更新静默旧权重域，**候选强**（源码行号+语义完整），本日合并量控制留观下轮优先）、#44789（processor 硬编码 mean/std——输入预处理错值域候选，留观）。
- **TensorRT-LLM**：#20369（词表外 token id 设备侧 assert——fail-loud）、#20365/#20371（性能/特性未启用）、#20376/#20377/#20382（grouped MLA 配套缺陷，作为 INF-1404 关联记录）。
- **tt-metal**：#60778（fold L1 sharded 路径 fp32 经 TF32 unpack 非 bit-exact——测试断言范围问题+已知 sharded 行为，文档域；但 Tf32 unpack 现象与既有序列相关，不另占号）、#60752（pool2d 配置 hash 漏 ceil_pad_hw 共享 program——结果错风险候选，本轮未深读，留观）、#60710（concat gcd 混单位——量纲 bug 候选，留观）。
- **flashinfer**：#6533 已收录。
- **transformers/triton**：窗口内无新判定候选（triton 0 命中）。

## 二、NPU 来源扫描（batch-20261015T230051994971 + batch-20261016T230043154367）

### 覆盖账本

- 10-15 批次：22/22 仓 complete、failures=[]、inference.jsonl 9144 条；窗口 10-14T22:00:52Z→10-15T23:00:51Z。
- 10-16 批次：22/22 仓 complete、failures=[]、registry_hash 2f6422d0…；inference.jsonl 10064 条；窗口 10-15T22:00:51Z→10-16T23:00:43Z。分仓 fetched（两批次合并去重后）：Ascend/pytorch 332、op-plugin 13、torchair 54、ops-transformer 264、ops-nn 372、ops-math 178、ge 86、runtime 112、hccl 77、hcomm 111、mindspore 11、mindformers 18、msprobe 13、mssanitizer 8、MindSpeed-LLM 13、MindIE-LLM 0、msmodelslim 16、vllm-ascend 335、sgl-kernel-npu 7、triton-ascend 48。
- 两批次按 updated_at 增量合并 10064 唯一项；10-14 报告后创建的 1436 项为判定池。

### 筛选与判定过程

1. 中英关键词分层（zh+en：精度/静默/错值/乱码/不一致/溢出/NaN/mismatch/wrong/incorrect/corrupt/precision/accuracy/silent 等）：强标题候选 74 条。
2. 74 条逐条读题分类；其中 24 个重点候选读本地正文（bodies/*.md，零 API 成本）；文档域（~8 条：README/参数名/docs 不一致）、错误码整改类（hcomm #6021/#6022/#6024/#6032、ops-nn #12014/#1206710、runtime #5575/#5595 等）、CI/风格类不占号。
3. 近 14 天日报无本轮拟收录 URL 判定记录（grep 核对）；catalog `find <规范URL>` 全部拟收录 URL 两库查重 0 命中（GitCode 完整 URL、GitHub 完整 URL）。

### 🆕 新发现（NPU 来源，11 条）

**vllm-ascend**
1. **INF-1390** [NEW] [#18769](https://github.com/vllm-project/vllm-ascend/issues/18769) — FA3 按紧密页（131072 元素）寻址真实 K/V 交错布局（stride(0)=262144，[K0][V0][K1][V1]）：K 页 1 读到 V0；Qwen3-0.6B eager+FA3 HTTP 200 输出 'ConvertedConverted...' 乱码；单文件对照（仅差 #18729 修复）归因闭合；修复 PR #18729 open。documented。
2. **INF-1393** [NEW] [#18794](https://github.com/vllm-project/vllm-ascend/issues/18794) — AscendStoreConnector 多 KV 组请求部分块加载失败（-704）只记日志不上报：scheduler 不知前缀不完整，decode 用无效 KV 生成损坏输出；重复同 prompt 复现、换前缀绕过；CPU mock 可复现。documented。
3. **INF-1394** [NEW] [#18898](https://github.com/vllm-project/vllm-ascend/issues/18898) — MRV2 Mamba align 模式 KVPool 恢复时序颠倒（preprocess_state 先拷贝、start_load_kv 后恢复源块）：同 20K token 请求第二次 L2 命中 20,736 后正文空、reasoning 重复标点，finish_reason=length；冷请求正常。documented。
4. **INF-1398** [NEW] [PR #19016](https://github.com/vllm-project/vllm-ascend/pull/19016)（open）— C8 MXFP8 PD KV scale 损坏双机制（K-scale 物化 strided 视图写回覆盖并发传输页；异步加载跳过整页清零残留无效 V scale）→ NaN 与重复'!'输出、MTP 接受率 0；聚合先证 #17956（Qwen3.6-27B W8A8+C8 首请求后全 128 个'!'，GDN SSM 写覆盖 V-scale 字节）、#18618（MooncakeV2 用生产端缓存字节覆盖 decoder 静态 V scale）、#18801（页步进布局数值退化改连续区段恢复）。documented。

**torchair / torch_npu / op-plugin**
5. **INF-1399** [NEW] [torchair#868](https://gitcode.com/Ascend/torchair/issues/868)（closed，PR #3844/#3845）— autotune 选优后立即 dlclose 落选 kernel .so：HBM 代码段可被复用而 icache 无一致性校验，偶发精度异常或 aic error；修复=dlclose 延迟到进程退出。**首个"指令缓存与内存代码段不一致"类静默损坏条目**。documented。
6. **INF-1400** [NEW] [op-plugin#586](https://gitcode.com/Ascend/op-plugin/issues/586) — aten._index_put_impl_ float32 尾部 undefined indices 静默错写（[idx,None] maxdiff≈3、[idx,None,None] 通道交换，exact 形状一次 SIGSEGV）；fp64/CPU 正确；影响 ATen 层调用方（Python 前端剥 None 不直接触发）。documented。
7. **INF-1396** [NEW] [Ascend/pytorch#5451](https://gitcode.com/Ascend/pytorch/issues/5451)（修复 PR #48029 open）— _make_reduction_inner 缺 reduction_type 致 argmin/argmax 索引错。issue 正文极简（一句描述），证据薄但 issue+修复 PR 双向印证；触发面未知，标 trigger_review 留观补全。documented。

**cann 算子仓**
8. **INF-1395** [NEW] [ops-math#3616](https://gitcode.com/cann/ops-math/issues/3616) — Cumsum 多核分支（blockDim=56）系统性 GM 越界写：fp16 N≥4095 同时算错（16 档 N 隔离实测、3324 条 memcheck 告警、aiv1~55 全覆盖）；fp32 N≥4097 越界但结果侥幸对。documented。
9. **INF-1397** [NEW] [ops-nn#6698](https://gitcode.com/cann/ops-nn/issues/6698) — MatMulV3 mode4 前序 NaN fill 历史污染：全有限输入输出 16 NaN（50/50、10/10 触发）；参考内核 UB dump 定位 TO_MUL 路径 A 尾部未初始化（NaN×0 污染归约）；A 清零 5/5 恢复；mode0/零历史正常。documented。
10. **INF-1401** [NEW] [ops-transformer#6221](https://gitcode.com/cann/ops-transformer/issues/6221) — LI/LI v2/QLI v2 分页地址 block_id*stride0 按 uint32 计算回绕：47663×90112 期望 4295008256 实际 40960，TopK 校验失败；修复 PR #13938。documented。
11. **INF-1403** [NEW] [ops-nn#6783](https://gitcode.com/cann/ops-nn/issues/6783)（修复 PR #12124 open）— PadConv2dFusionPass 对 FP32 constant_values 用 int32 截断判零：0<v<1（如 0.5）被当 0 融合，PadV3 语义静默变隐式零填充（例：输出 4→0）。documented。

### NPU 留观与拒收（主要项）

- **留观（强候选，下轮优先）**：hccl#827（单 rank V 算子遗漏 displacement 读写位置错——源码级定位+算子影响表完整，**写错位置属静默错值**，但 950PR/950DT 环境且未附实测输出，与 #825 同仓双发；本日合并量控制留观下轮优先）；ops-math#3604（RadixTopK WS 变体三缺陷：V→MTE3 同步缺失/无符号下溢越界写/int32 前缀和溢出+UT 恒测不到 int64 主路径——机制极详，sortLen>2^31 触发面窄，closed+PR #6396）；Ascend/pytorch#4941（上轮留观"下轮优先"——本轮批次显示 closed 10-13，正文验收口径 36/0 完整，**下轮改判入编目**）。
- **训练域转介（不占推理号）**：ops-nn#6737（SiluGrad exp(-x) 中间溢出设溢出标志致 AMP 跳更新——影响为训练 AMP 行为，推理侧 SiluGrad 不在 serving 路径；已确认训练库未收，转训练扫描判定）；ops-nn#6730（HingeLossGrad hinge 边界梯度衰减——训练损失域）；ops-nn#6708（InplaceApplyAdaMax 标量 shape 静默截断——优化器域）；ops-nn#6706/PR#12006、PR#12075（卷积反向整数溢出/下溢——训练反向）；ge#659（A5 allow_fp32_to_fp16 不生效——性能两头落空但数值**更准**（cos 1.0），配置静默不生效域非错值，与 #5108 同族观察，不占号）。
- **拒收/不占号**：ge#663（GE DataType Proto 枚举序号错——GEIR 序列化互操作缺陷，模型类型错读面存在但属构建/序列化域+探针验证，closed 方向明确，留观）；ops-math#3596（SquareSumAll FP32 末级合并三方精度超差——RMSE 3.77 vs 1.5，**候选接近收录**，但修复承诺"不承诺任意输入逐位一致"、属算子精度容差治理域，留观下轮族化评估）；ops-math#3591（RadixTopK sorted=true 被错误拦截——拦截后走 RadixTopK 但 kernel 不支持 sorted，输出未排序仍正确值，语义面，留观）；hcomm#1055（EI0012 段错误——fail-loud）；hcomm#1058（2P 绕行逆映射错——通信路由域，中间节点错误+预期绕行边缺失，机制描述清晰但无端到端数值影响证据，留观）；mindspore#29700（NoFallbackGuard 屏蔽 MsDispatchMode——dispatch 回调跳过、算子正常执行，扩展/ instrumentation 域非计算错）；ops-transformer#6219/PR#13930（QFA-MXFP8 随机 NaN——已修（vscale L1 尾块清零避 0*nan），closed+merged，机制已定位不留观面）；ops-transformer#6223（MoE 通信 buffer 容量估算不一致——23MiB 差导致容量校验失败 fail-loud）；ops-nn#6760/#6787/#6790/#6793（Proto/注册元数据不一致系列——文档/契约域）；PR 类（#12064/#12075/#12083/#12084/#13889/#13888 等修复型，标题含机制摘要，无静默影响新证据面）；msprobe#1103（精度预检工具自身报错修复——观测工具域）；msprobe#644（文档域）。
- **留观队列更新**：上轮 vllm-ascend#17476（msmodelslim 量化路径 expert 集错位，closed）本轮批次无新评论数据，维持留观；#18755（910B2C vs H100 logprob 平台差异）无更新，维持。

### NPU 扫描未覆盖范围

- 两批次均按 updated_at 增量；10-14T22:00Z 前创建且其后无活动的项不在索引（首轮全量已覆盖大部分，见 10-14 报告未覆盖说明）。
- gitcode API 无评论端点：全部 open 候选（#18769/#18794/#18898/#586/#3616/#6698/#6221/#827 等）的后续评论级回访待下轮。
- vllm 主仓 Search 第 2 组 114>100 命中，per_page=100 单页取全声明 incomplete=false 但第 2 页 14 条未取——KV/量化关键词组可能有 ≤14 条未判，列入下轮补查。
- ops-nn 27 条强标题候选中约 12 条（Proto dtype 系列、错误码系列）仅按标题判定，未读正文。

## 三、来源回访（近 7 日条目引用来源 + 旧来源轮巡）

- **状态翻转（3）**：vllm-ascend#18746（INF-1374）→ closed 10-15T02:02Z，未见修复 PR，机制未定位维持 active → update 记录；ops-transformer#6114（INF-1387）closed 更新至 10-16T03:36Z，无修复链接 → update；ops-nn#6516（INF-1383）/#6482（INF-1382）closed，修复状态未核实 → update。
- **活跃无变化（2）**：vllm-ascend#18754（INF-1379，10-14 15:20Z 后无更新）；triton-ascend#2531（INF-1375，10-15 08:44Z 有更新但批次无评论内容）→ update 记录。
- **旧来源轮巡（≥3，本轮 4）**：INF-1003（vllm#58824）、INF-0343（vllm#58973）、INF-0654（sgl#41617）、INF-0860/1056（NPU 域旧条目）——本轮批次与 Search 窗口均无新活动记录（上轮 10-14 已回访，48h 内无变化），写明原因：Search 窗口 7 日覆盖+批次数窗口内无 update，跳过重复 API 消耗。
- watchlist 33 项中 29 项批次内状态已核对（见上留观节）；#4941 改判计划下轮执行。

## 四、编目操作与校验

- 新增 18 条（`catalog_tool.py new`，全部 pending_audit）：INF-1390–INF-1407
  - vllm 主仓：INF-1391（lightning_attention 衰减差一）、INF-1404（TRT-LLM grouped MLA cumSeqLens）
  - sglang：INF-1392（HiCache host 恢复错值）
  - tt-metal：INF-1402（acosh_bw 域保护）、INF-1405（sigmoid LUT ±inf NaN）、INF-1406（softplus bf16 尾部钳零）
  - flashinfer：INF-1407（plan pinned workspace 竞态）
  - vllm-ascend：INF-1390（FA3 交错页寻址）、INF-1393（AscendStore 加载失败不上报）、INF-1394（Mamba KVPool 时序）、INF-1398（C8 MXFP8 scale 损坏族）
  - torchair：INF-1399（autotune dlclose icache 不一致）
  - op-plugin：INF-1400（index_put 尾部 undefined）
  - torch_npu：INF-1396（argmin/argmax 索引）
  - cann：INF-1395（cumsum 多核越界+错值）、INF-1397（MatMulV3 NaN 历史）、INF-1401（LI 分页地址回绕）、INF-1403（PadV3 小数填充融合）
- 更新 7 条（update --note）：INF-1374/1382/1383/1387（状态翻转回访）、INF-1379/1375（活跃无变化）、INF-1390（证据边界补充）
- `build` → views/ 三文件再生成；`validate` → **ok:true，1407 条，0 errors**（87 warnings 为既有模式层待定标题类，非本轮引入）
- 编目总数 **1389 → 1407**（active 1389 + pending_audit 18）

## 五、观察

1. **"缓存恢复/传输完整性"族继续爆发且跨栈同构**：INF-1392（sglang host 层 GDN 恢复错值）、INF-1393（AscendStore 失败不上报）、INF-1394（Mamba 状态恢复时序）、INF-1390（FA3 布局寻址）+ vLLM 主仓 KV Connector 三连（#61912/#61905/#61768）——与 10-12~14 HiCache 主线完全同向：**命中/加载成功信号与数据实际正确性之间没有校验层**，GPU/NPU 两栈并行恶化。
2. **NPU 首现"指令缓存不一致"静默损坏**：INF-1399（dlclose 后 icache 与 HBM 代码段不一致）是编目内首个该类机制，与硬件 SDC 族（不可复现、概率性）行为相似但根因在软件生命周期管理——监控面值得单独评估。
3. 算子层"边界值语义"缺陷成排：INF-1400（尾部 undefined）、INF-1403（0<v<1 截断判零）、INF-1405（LUT ±inf）、INF-1406（尾部钳零）、INF-1402（域保护平方）——**边界输入（0、±inf、(0,1) 小数、负射线）在 NPU/GPU 两侧 SFPU 与融合 pass 中系统性失守**，与 10-14 triton-ascend floor-mod 链同模式。
4. 留观队列显式化（下轮优先）：hccl#827（V 算子位移）、Ascend/pytorch#4941（TE 布局原语物化，已 closed 拟改判收录）、sglang#44504（权重静默跳过）、ops-math#3596（三方精度超差）、vllm#61803（block_table 越界读）、KV Connector 三连、ops-math#3604（RadixTopK 三缺陷）。

## Git 提交

- 暂存：`catalog/entries/INF-1390..INF-1407.yaml`（新增 18）、被 update 的 7 个条目 YAML、`catalog/id_registry.tsv`、`views/CATALOG.zh.md`、`views/fault_catalog.csv`、`views/fault_catalog.xlsx`、`daily_scan_2026-10-16.md`
- 提交与推送结果见最终摘要。

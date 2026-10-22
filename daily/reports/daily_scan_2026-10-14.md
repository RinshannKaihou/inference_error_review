# LLM 推理故障每日扫描报告 — 2026-10-14

- **扫描日期**：2026-10-14（UTC）
- **扫描窗口**：GitHub `created:>=2026-10-07`（cutoff 实际计算）；本日新判定集中于上次扫描（10-13 ~23:40 UTC commit）后创建的 25 个新 issue。NPU 新来源共享采集批次 `batch-20261014T230052044826-vwpnnvwy`（22/22 仓 complete，采集窗口 02:14–23:00 UTC updated_at 增量，含历史创建近期更新项）。arXiv 6 组查询全部有效。
- **目标仓库**：原 GitHub 7 仓 + 新增 NPU 19 推理源（GitCode/AtomGit 16 + GitHub 3）
- **编目基线**：1370 条（active）；本轮后 1389（active 1370 + pending_audit 19）

## Git 前置检查（第 3 节）

- 分支 `main`、上游 `origin/main`、`git status --porcelain -uall` 为空 ✅
- `git fetch origin main` 成功；`HEAD` = `origin/main` = `fbb23ae06090bd58642e0662349990548e5a5ba1` ✅ → 开始扫描

## 一、原 GitHub 7 仓 + arXiv

### API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-07 correctness OR "silent wrong output" OR "wrong output"` | 33 命中（单页取全，incomplete=false） |
| 2 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-10-07 KV OR quantization OR "kernel numerics" OR SDC` | 117 命中（两页取全 100+17，incomplete=false） |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-10-07 correctness OR "wrong output" OR silent OR garbage` | 46 命中（单页取全） |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-10-07 correctness OR "wrong output" OR numerics` | 3 命中（均已判/非候选） |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-10-07 correctness OR "wrong result" OR silent OR numerics` | 10 命中（无新判定候选） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-10-07 miscompile OR "wrong result" OR numerics OR correctness` | 3 命中（无新候选） |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-10-07 "Bad Outputs" OR correctness OR "wrong value"` | 44 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-10-07 correctness OR "wrong output" OR numerics` | 19 命中（4 条 CAKE feature，非候选） |
| 12 | arxiv | `all:"silent data corruption" AND all:inference` | 6 命中：**2610.18557（CSNB）新方法条目**；其余已编目/已判 |
| 13 | arxiv | `all:"bit flip" AND all:LLM` | 15 命中全已编目或已判（BARE-AI v2=2610.08739、2609.38477 等 10-13 已判维持） |
| 14 | arxiv | `all:"KV cache" AND all:corruption` | 15 命中全已编目或已判（Contiguity 2609.17983、2604.17249 等维持） |
| 15 | arxiv | `all:"soft error" AND all:GPU` | 9 命中已判维持 |
| 16 | arxiv | `all:"numerical correctness"` | 15 命中：2610.06251（HQQ）留观维持、2610.08675 非故障域维持 |
| 17 | arxiv | `all:"wrong output" AND all:LLM` | 6 命中非本域 |

`GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥8s 间隔、core REST 60/h）。8 组主查询 + 1 组二页分页全部有效（fetched==total 逐组核对，`incomplete_results=false`）；去重 250 唯一 issue，其中 10-13 23:30Z 后创建的 25 个新 issue 逐一正文判定。core REST 一次 60/h 耗尽（11 次详情后），前台分段 sleep（每段 ≤60s、逐段重查 remaining）至重置后完成剩余深读，未启动后台进程。

### 🆕 新发现（原 7 仓，4 条）

1. **INF-1379** [NEW] [vllm-project/vllm-ascend#18754](https://github.com/vllm-project/vllm-ascend/issues/18754)（NPU 首轮）— 采样侧流 q 竞态（详见 NPU 节）。
2. **INF-1385** [NEW] [vllm-project/vllm#61673](https://github.com/vllm-project/vllm/issues/61673) — V1 cascade attention + SimpleCPUOffloadConnector：CPU store 在途给共享头块 +1 引用 → 15/16 请求持有的块被当全体持有，无共享请求 O 用第一请求的 KV 解码（贪心 token 2-3 偏离 solo 9.56-12.94 nats 处）；五臂隔离（关 cascade 0/12、去 connector 0/60、V2 恒 common_prefix_len=0 不触发），60 次时序试验 15/18~0/12。documented。
3. **INF-1388** [NEW] [vllm-project/vllm#61526](https://github.com/vllm-project/vllm/issues/61526) — NixlConnector P/D 异构 TP 下 Mamba2 conv state 静默错传：n_groups 不被较大 TP 整除（granite-4.0-h n_groups=1）时 decode 输出错，无错误无警告；模型矩阵实测（Nemotron-H n_groups=8 / GDN heads 整除均正确）。documented。
4. **INF-1386** [NEW] [tenstorrent/tt-metal#60613](https://github.com/tenstorrent/tt-metal/issues/60613) — ttnn.sort fp32 NaN 按位比较幅值大于 inf，NaN 排到 ±inf padding 之后被 slice 丢弃，padding 索引（≥n）泄漏进返回值：三工厂×升降序×stable 全中，out-of-range indices=4、输出 NaN=0（期望 4）。documented。

另：**INF-1380** [NEW] [vllm-project/vllm#61681](https://github.com/vllm-project/vllm/issues/61681) — `--mm-encoder-tp-mode data` 下 GLM-5.3-Flash（NVFP4 TP2 B200）图像/视频答案错乱（48 图首尾帧答案相同且都错），唯一变量 flag、weights 模式 5 次全过；机制未定位。documented。
**INF-1381** [NEW 方法类] [arXiv 2610.18557](https://arxiv.org/abs/2610.18557) — CSNB 替代 ABFT 做 SDC 检测：稠密线性层保留 99% 检出（开销 -87%），MoE router gate 仅 55%——router 小 GEMM 仍需 checksum。documented（受控注入实验值，非生产实测）。

### 拒绝 / 留观（原 7 仓主要候选）

- **vLLM**：#61616（NIXL 失败传输 HMA 下被当已加载、decode 用未初始化 KV——**与 INF-1363/1364 同"恢复未完成即消费"主题且源码级 TODO 定位清晰，closed(重复方向)，候选强，本日合并量控制留观下轮优先**）、#61532（P2P consumer 按 timer 释放槽位而 producer WRITE 可能仍在途——代码审读+NIXL 1.4.1 CPU 探针（5/5 次晚到写），未端到端复现 KV 损坏，留观）、#61658（tiering 异步 lookup 缓存 absent 结论跨实例陈旧——共享二级层下漏块重算，性能/冗余域为主，CPU-only 可复现，留观）、#61617（Mamba-only NIXL 握手 StopIteration——fail-loud 崩溃域）、#61608（tower/connector LoRA 8 模型 punica wrapper 尺寸低估——startup/首批 assert fail-loud）、#61621（attention metadata builder 同步 .cpu() 持锁——性能域）、#61604/#61665（性能）、#61539/#61567/#61626/#61698（RFC/ADR）。
- **SGLang**：#44171（多模态身份保真 RFC）、#44186（PD 传输失败后 decode KV 页释放 ADR）——设计文档不占号；#44304（ROCm 性能优化）。
- **tt-metal**：#60614（selective_reduce_combine 迁移 tracker）。
- **flashinfer**：#6485/#6486/#6487/#6491（CAKE 训练 kernel feature）。
- **transformers/triton/TensorRT-LLM**：窗口内无新判定候选。

## 二、NPU 新来源首轮扫描（batch-20261014T230052044826-vwpnnvwy）

### 覆盖账本

前置脚本输出：22/22 仓 complete、failures=[]、registry_hash 2f6422d0…。inference.jsonl 8200 条（issue 2501 + PR 5699）；采集按 updated_at 增量（since 2026-10-14T02:14:21Z，until 23:00:52Z），批次内含历史创建近期更新项（最早 2023-10）——首轮无水位、实际为全量索引+当日增量，本报告按批次实际窗口记录，不伪称"仅 7 日"。

分仓 fetched（issues+pulls 两端点分页全取）：Ascend/pytorch 248、op-plugin 19、torchair 60、cann/ops-transformer 472、ops-nn 234、ops-math 118、ge 46、runtime 55、hccl 80、hcomm 120、mindspore 16、mindformers 14、msprobe 10、mssanitizer 7、MindSpeed-LLM 6、MindIE-LLM 2、msmodelslim 28、vllm-ascend 336、sgl-kernel-npu 2、triton-ascend 39（MindSpeed/MindSpeed-RL 属训练 scope 未入 inference 视图）。

### 筛选与判定过程

1. 标题关键词分层（zh+en：精度/静默/错值/乱码/不一致/wrong/mismatch/precision/corrupt/logprob 等）：L1 636 条 → 排除明显 fail-loud/hang/纯性能/文档不一致类标题后 L2 347 条。
2. L2 逐条过题分类：**大量为文档参数名不一致、golden/CI 工具链、README-dtype 虚标类**（cann 各仓 ~60%），不占故障号；对 40 个静默正确性强候选读本地正文（bodies/*.md），11 个重点候选全读。
3. 近 14 天日报无任何 NPU 来源 URL 判定记录（grep 全量核对）；catalog `find <规范URL>` 对全部拟收录 URL 查重均 0 命中（GitCode 完整 URL、GitHub 完整 URL，无裸编号匹配）。

### 🆕 新发现（NPU 来源，14 条）

**vllm-ascend**
1. **INF-1379** [#18754](https://github.com/vllm-project/vllm-ascend/issues/18754) — `random_sample()` 侧流 q 竞态：q 在 global_stream() 上 exponential_() 填充、回主流仅 wait_stream 无 event，graph+async scheduling 下 `div_` 读未初始化/陈旧 q → `<pad>`/垃圾 token 间歇插入；四臂差分（唯一改 q 路径的 enable_async_exponential=true → 0/60）归因闭合。规避零成本。documented。
2. **INF-1374** [#18746](https://github.com/vllm-project/vllm-ascend/issues/18746) — MRv2 FULL graph 下 DeepSeek V4.1 Flash 输出损坏重复：五臂对照隔离到 FULL graph 路径（eager/MRv1/dspark 开关均正确）；根因未定位。documented。
3. **INF-1389** [PR #17343](https://github.com/vllm-project/vllm-ascend/pull/17343)（open 未合并）— fused MC2 dispatch 容量静默截断：post-dispatch 计数达 max_output_size 即丢工作不报错，实测负载达均值 4.9 倍，8192 复现 273/288 截断+NaN 行。documented。
4. 留观：#18755（910B2C vs H100 逐 token logprob 系统性偏差 ~2.2×自噪声、集中于低置信 token、eager/graph 与精度开关不变——**平台数值实现差异的真实量化证据**，但根因未定位且属"实现差异 vs 缺陷"边界，留观下轮评估入族）、#18710（0.27.1rc1 kimi-k3 首 token 乱码+跨请求同碎片+8 节点零 ERROR——静默 KV 串扰面真实但机制未定位、配置极复杂，留观）、#17593（0.20.2rc1 图模式乱码关图恢复——旧版本+机制未定位，留观）、#17476（msmodelslim 描述文件存在即触发 ascend 量化路径、FLOAT skip 丢 tid2eid 致 sqrtsoftplus 层 expert 集错位——机制清晰 closed，候选强，本日合并量控制留观下轮优先）、#13853（block-size 2048 乱码，9 月旧单，留观）、#17385（流式末 trunk 空 delta——输出协议兼容域，留观）、#17391（GLM P/D 静默截断 closed，机制未定位，留观）、#18173（multi-branch allOf 静默禁用 guided decoding——**非 NPU 特有**（xgrammar+V1 通用），vLLM 主仓域，留观下轮在主仓上下文重评）、#18197（Qwen3-VL bbox 坐标错 0% vs transformers 100%——量化/多模态混合归因未闭合，留观）、#18705/#18706/#18735（fabric memory 精度 case 关闭——测试环境调整非机制证据）。

**triton-ascend**
5. **INF-1375** [#2531](https://github.com/triton-lang/triton-ascend/issues/2531) — 静态 r-numel 编译期常量特化致 fused softmax 行 max 归约误编译（fp16 SDPA attn_mask 97% 错；同内核仅常量化差异数值翻转；排除矩阵 22 项 autotune 全过）。documented。
6. **INF-1376** [#2291](https://github.com/triton-lang/triton-ascend/issues/2291) — 3-D 首轴 tl.sum 归约 stride 交换静默误编译（3D maxdiff 2025.0/2D 0.0；inductor broadcast backward 33 opinfo 行暴露）。documented。
7. **INF-1372** [#2290](https://github.com/triton-lang/triton-ascend/issues/2290) — 向量 bool != 真值标量折叠全真（负除数 floor-mod 修正链整体失效；三组隔离对照证明罪在 != 组合）。documented。
8. **INF-1377** [#2518](https://github.com/triton-lang/triton-ascend/issues/2518) — 整型 %0 编译态恒返 -1 vs eager 返回被除数（三除数形态全中；floor-mod 链第四缺陷、唯一直接命中生产数值路径）。documented。
9. 留观：#2399/#2362/#2500（修复 PR 已有——dot_scaled f16 精度、atan2、OpClassifier——对应 issue 未见同号判定，机制已修不留观面，仅作修复证据记录）。

**Ascend/pytorch（torch_npu）**
10. **INF-1371** [#5114](https://gitcode.com/Ascend/pytorch/issues/5114) — npugraph_ex 回放覆写前次输入存储（同形状换入参即静默损坏、无告警；npugraphs/npu/aot_eager 对照受保护）。documented。
11. **INF-1378** [#5296](https://gitcode.com/Ascend/pytorch/issues/5296) — gather/scatter/cross_entropy/nll_loss 缺 host 越界校验（CPU 抛异常处 NPU 静默错值/丢写；embedding/select 对照仍 raise）。documented。
12. 留观：#5108（set_float32_matmul_precision/allow_tf32 在 NPU 静默无效——**配置静默不生效域**，与 10-12 DFlash2 SWA block-drop 同族候选；无错值证据本日不占号，下轮族化评估）、#5366（SDPA 完全 decompose+编译后数值错 97.9%——**与 INF-1375（#2531）同现象**：同环境同 97% 量级同 attn_mask 场景，#2531 已把根因定位到 r-numel 常量特化；#5366 从 torch_npu 侧报告同一缺陷的 torch.compile 表现，作为 INF-1375 姊妹证据不另占号——已验证两单环境链一致（910B2+triton_experimental+SDPA fp16））、#5362（eval BN 分解链精度差被 BiLSTM 放大——修复分支已有、分解层差 3.8e-6 属实现差异边界，留观）、#4941（triton_experimental blanket-fallback 物化 view 致 copy_ 变异被 DCE 丢失——**强候选**（机制+修复+36/0 验收口径完整），本日合并量控制留观下轮优先）、#5328（streaming create_block_mask 静默回退省显存路径失效——性能优化静默不生效域，与 #5108 同族观察）、#5392（mul 混合精度 GPU/NPU 不一致——数值边界未定位，留观）、#5287（Guard Filter 过滤 y=None 后静默复用错误图——众测报告+官方示例路径，候选，留观下轮）。

**cann 算子仓**
13. **INF-1373** [cann/ops-transformer#6133](https://gitcode.com/cann/ops-transformer/issues/6133) — GatherPaKvCache ND kernel 32B 向下取整丢尾（head_size=17 整 token 丢、33 丢尾 1B；违反 README 按位一致承诺）。documented。
14. **INF-1387** [cann/ops-transformer#6114](https://gitcode.com/cann/ops-transformer/issues/6114) — ScatterPaKvCache 空/不足 slotMapping 静默接受（numTokens 取 key.dim0 索引 slotMapping；ctypes 实测写 64/128 元素仍 SUCCESS）。documented。
15. **INF-1382** [cann/ops-nn#6482](https://gitcode.com/cann/ops-nn/issues/6482) — AddRmsNorm 950 tiling 不拦 Pre/Post 模式（绕过 aclnn 路径 numRow 少算输出大半未初始化+越界写；PR #11564 修复）。documented。
16. **INF-1383** [cann/ops-nn#6516](https://gitcode.com/cann/ops-nn/issues/6516) — HardSwishV2 BF16 分支 Duplicate 覆盖原输入 tmp0（输出退化为 clamp(x+3,0,6)，48/64 真机失败；FP32 对照 0 失败）。documented。
17. 留观：ops-math#3446（Trunc/Ceil/Floor/Round/Sinh 负零丢符号位——真机逐位证据+arch35 对照线索，**候选强**，数值面窄（-0.0）本日合并量控制留观下轮优先）、ops-nn#6472（DynamicMxQuant BF16 FP4 floor 负数下溢 -0——量化域候选，修复分支已有，留观）、ops-nn#6477（DynamicMxQuant 非尾轴尾块偶现 FP4 错值——偶现+用户环境未稳定复现，留观）、ops-transformer#5819（mega_moe MTE3/Scalar 共享 LocalTensor 缺同步——**代码审读级**（"可能会导致"），无实测证据，留观）、hccl#825（CCU AllReduce 归约槽序随 rank 漂移跨卡 bitwise 不一致——**需求单形式**、机制描述清晰（one_shot 本卡垫底+2die 分支不对称），修复 PR #3607 open；跨卡不一致域与 4.28 族相邻，留观下轮族化评估）、hcomm#1007（确定性开关多次读取竞态——closed，未展开）、ops-nn#6595（tiling key 空格静默丢 key——构建过滤域）、ops-nn#6513（sparse_apply_adagrad_v2 FMA 融合 inf/nan 场景精度——训练侧）、ops-nn#5906（GroupNormSiluV2 近常数输入 rstd 偏离——closed，量化敏感域候选，留观）。
18. 不占号类：msprobe#638（比对工具 118 行假阳性——golden 侧 dump 缺失/错位+NPU 未写缓冲区 NaN，**监测工具误报**不转述为计算故障，符合 NPU_SCAN 纪律 5）、msprobe#633/#641（采集覆盖/工具崩溃）、msmodelslim#615（feature）、cann 各仓文档参数名不一致系列（~80 条，文档域）、ops-transformer#6060/#6061/#6064、ops-nn#6588/#6589（精度汇总 CSV/判通过工具缺陷——观测工具域）、ops-transformer#5778（FfnWorkerBatching 缺 dim 检查 GM 越界读——探针 C 静默空输出**是静默面**但属非法输入被吞+closed 已修，候选弱于 #6114 同型，留观）、mindspore#29697（CPU 路径 roll/MatMulExt 修复聚合单——CPU 域非 NPU 推理）、mindformers#8794（MTP 超参静默建 mHC 参数——训练域）。

### NPU 扫描未覆盖范围

- 批次按 updated_at 增量：**历史久远且近期无更新的 issue/PR 不在本批索引内**（首轮全量采集受单日窗口限制），10-07 前创建且 10-14 02:14 后无活动的静默故障存量未评估——下轮起水位滚动补齐。
- L2 347 条中约 40 条 cann/hccl PR 类（修复型 merge_requests）仅读题未逐一读正文（标题已含修复对象+机制摘要，无静默影响新证据面）——如需深挖按 body_path 补查。
- gitcode API 无评论端点数据（批次只含正文），全部候选的后续评论级回访待下轮（对 open 候选 #18754/#18746/#2531/#2291 等尤为必要）。

## 三、来源回访（近 7 日条目引用来源 + 3 个旧来源轮巡）

48+3 来源经 Search/单号 REST 快照（同上 API 表），活跃深查 5 组：

- **状态翻转（1）**：tt-metal#60393（INF-1368）→ closed/completed 10-14T16:07Z，修复 PR #60395（GDN prep kernel wrong dl/k_dec_t at chunk_size 64）merged 同刻、commit 7086ed4c7a06 → update 记录（closed+merged 不自动证明已发布）。
- **活跃新证据（4）**：sgl#44070（INF-1364）修复 PR #44081 已由报告者 agent 提交 + 社区认领讨论 → update；vllm#61359（INF-1367）两个修复 PR 进展（#61524 Draft tokenizer 侧方案 vs #61363 双侧全修待 review，报告者指出 embedding 左侧截断路径）→ update；vllm#61490（10-13 留观）修复 PR #61547 已提交（scheduler 标记 stale + V1 runner 重注册需求）——**该留观项有新修复证据，但触发面（管理端点×in-flight prefill 时序）判定不变，维持留观并记录**；vllm#61486（INF-1365）仅 ROCm bot 模板回复 → update 无实质进展。
- **旧来源轮巡（3）**：vllm#58824（INF-1003）、vllm#58973（INF-0343）、sgl#41617（INF-0654）均无新评论/状态变化 → update 记录；另 NPU 域旧条目 sgl#38196（INF-0860）、PR#34794（INF-1056）首轮回访（closed 未 merged）→ update。
- INF-1363/1366/1369/1370 快照无变化 → update 记录。

## 四、编目操作与校验

- 新增 19 条（`catalog_tool.py new`，全部 pending_audit）：INF-1371–INF-1389
  - vllm-ascend：INF-1374（MRv2 FULL graph）、INF-1379（采样侧流竞态）、INF-1389（fused MC2 截断）
  - triton-ascend：INF-1372（!=折叠）、INF-1375（r-numel 常量）、INF-1376（3-D 归约换轴）、INF-1377（整型 %0）
  - torch_npu：INF-1371（npugraph_ex 覆写输入）、INF-1378（越界校验缺失）
  - cann 算子：INF-1373（GatherPaKv 尾丢）、INF-1382（AddRmsNorm 950）、INF-1383（HardSwishV2 BF16）、INF-1387（ScatterPaKv slotMapping）
  - op-plugin：INF-1384（变长 causal mask 错位）
  - vLLM 主仓：INF-1385（cascade×offload 前缀误享）、INF-1386（ttnn.sort NaN padding）、INF-1380（mm-encoder-tp-mode data）、INF-1388（Nixl 异构 TP conv state）
  - arXiv：INF-1381（CSNB 方法，kind=方法）
- 更新 11 条（update --note）：INF-1363/1364/1365/1366/1367/1368/1369/1370（回访）、INF-0343/INF-0654/INF-1003/INF-0860/INF-1056（评论级复查/首轮 NPU 域回访）
- `build` → views/ 三文件再生成；`validate` → **ok:true，1389 条，0 errors 0 warnings**
- 编目总数 **1370 → 1389**（active 1370 + pending_audit 19）

## 五、观察

1. **NPU 栈"图/编译加速路径 × 静默损坏"集中爆发**：INF-1371（ACL graph 回放覆写输入）、INF-1374（MRv2 FULL graph）、INF-1375/#5366（编译期常量特化误编译）、INF-1379（graph+async 采样竞态）——与既有 PIECEWISE 捕获契约族（6.281/INF-1365）同主题：**执行加速层假设与运行期真实值/所有权不匹配**。
2. **"非法输入被静默接受"在 CANN host 层成模式**：INF-1387（slotMapping）、#5778（schedule_context）、INF-1378（越界索引）——CPU/CUDA 会 fail-loud 的边界在 NPU 栈被吞掉，且全部返回 SUCCESS。
3. triton-ascend floor-mod/谓词折叠链（INF-1372/1377 与 #2290/#2289/#2518）四连发：**编译器对边界输入（0、负数、标量 bool）的 folding 全部改变数值语义**，与 GPU Triton 行为分裂——Inductor 不变量（compiled==eager）在 NPU 上系统性脆弱。
4. vLLM 侧"恢复未完成即消费"族再添两员：INF-1385（store 引用污染 cascade 前缀判定）、#61616（HMA 失败不报 invalid blocks）——与 10-12/10-13 的 HiCache 三连发同主线，跨 KV 层的完整性契约缺失在 GPU/NPU 两栈并行出现。
5. 留观队列显式化（下轮优先）：vllm#61616、vllm-ascend#17476、Ascend/pytorch#4941、ops-math#3446、vllm-ascend#18755（平台数值差异量化）；族化评估：Ascend/pytorch#5108+#5328（配置静默不生效）、hccl#825（跨卡 bitwise 不一致 vs 4.28 族）。

## Git 提交

- 暂存：`catalog/entries/INF-1371..INF-1389.yaml`（新增 19）、被 update 的 13 个条目 YAML、`catalog/id_registry.tsv`、`views/CATALOG.zh.md`、`views/fault_catalog.csv`、`views/fault_catalog.xlsx`、`daily_scan_2026-10-14.md`
- 提交与推送结果见最终摘要。

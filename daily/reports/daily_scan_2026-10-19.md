# 每日扫描日报 2026-10-19（推理库）

- **扫描窗口**：GitHub `created:>=2026-10-12`（cutoff=2026-10-12）；重点判定 10-18 报告之后的新增。NPU 共享批次 `batch-20261019T230043098795-dh2bh0lv`（22/22 仓 complete、0 failures、registry_hash 2f6422d0…，采集窗口 2026-10-18T22:00:43Z → 2026-10-19T23:00:43Z updated_at 增量，candidates=11687，inference 视图 19 源）。
- **目标仓库**：原 GitHub 7 仓 + arXiv + NPU 19 推理源。
- `GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥9s 间隔）。GitHub API 经 `curl --noproxy '*'` 直连全部有效；arXiv 直连首轮 ax2/ax3 超时，重试后 6 组全部 200 有效。core REST 5 次单号状态核验（vllm#61225、sglang#44997/#44758/#44999、tt-metal#61268），未触限流等待。累计限流等待 0 秒（仅 arXiv 超时重试，无配额等待）。
- Git 前置检查：main 干净、fetch 成功、HEAD == origin/main（570c5568）。

## 一、原 7 仓新增条目（8 条，pending_audit）

1. **INF-1457** [NEW] [vllm#62417](https://github.com/vllm-project/vllm/issues/62417)（c2026-10-19）— DSpark 投机解码损坏 DeepSeek-V4.1-Flash-1017 思考/chat 输出（k=3/4/5/7 全触发，0.32.0 release+nightly；关投机干净、/v1/completions 干净、原版 Flash checkpoint 干净；tokenizer/模板/分词/采样逐变量排除）。documented。
2. **INF-1458** [NEW] [vllm#62554](https://github.com/vllm-project/vllm/issues/62554)（c2026-10-19）— packed NCCL 权重同步 + layerwise reload：接收缓冲在重放拷贝在飞时被 allocator 归还复用，下一 pack 覆盖未消费字节，MoE w13 权重静默写坏（RL 部署实测 weight checker 报不匹配、副本互不一致；num_buffers=k 时层跨 ≥k+2 packs 暴露）。documented。
3. **INF-1459** [NEW] [vllm#62283](https://github.com/vllm-project/vllm/issues/62283)（c2026-10-18）— KV offload direct 布局误标 parallelism-agnostic 致 namespace 丢 TP：跨 TP 引擎共享 fs/obj/p2p tier 加载对方 KV 静默乱码（gsm8k 0.332→0.0；TP1→TP1 与 canonical_layout=True 正确；V2 有指纹门控安全）。修复 PR #55191 开放中。documented。
4. **INF-1481** [NEW] [vllm#62483](https://github.com/vllm-project/vllm/issues/62483)（c2026-10-19）— Parser._apply_structural_tag 组 OrFormat([tool_call_tag, response_format]) 并清空原字段：GLM-5.3-Flash 同带 tools+json_schema 的请求 schema 不被强制，多轮 agent 单请求可烧 ~130K tokens；0.32 nightly 新增 glm_4_7 builder 后生效（0.31.0 早退不受影响）。documented。
5. **INF-1460** [NEW] [vllm#62124](https://github.com/vllm-project/vllm/issues/62124)（c2026-10-17，本轮窗口内深读收录）— FP8 per-tensor MoE + CUDA graphs：padding 行计入 per-tensor amax（内容为上一步残留 hidden states），同一贪心 prompt 答案随前一请求内容变化、首 token 即分叉（114 对 88 不同、270 对 269 不同；enforce-eager/per-block/bf16 干净；SKIP_PADDING=0 仅缩小不消除）。documented。
6. **INF-1461** [NEW] [sglang#44998](https://github.com/sgl-project/sglang/issues/44998)（c2026-10-18）— NVFP4 Marlin 回退（SM80-90）对小 block scale checkpoint（weight_scale_2=1 导出）全输出乱码（192/192 请求、~49,133/49,152 token）；`(scales*2**7).view(int16)<<1` 假设顶位恒置位；vLLM #34577（2026-03-17 merged）同根因已修，SGLang 未移植。documented。
7. **INF-1462** [NEW] [triton#12348](https://github.com/triton-lang/triton/issues/12348)（c2026-10-18）— TritonGPUOptimizeThreadLocality 归约外提用循环更新算符而非归约算符累积：`acc -= tl.sum(x,1)` 静默得 +512（应 -512）、`acc += tl.max` 得 -inf；main 与 3.8.0 均复现。documented。
8. **INF-1463** [NEW] [triton#12369](https://github.com/triton-lang/triton/issues/12369)（c2026-10-19）— Hopper warp_specialize 第二 consumer：sliceOp 把 off_m+64 记入共享映射，指针偏移再加一次，64~127 行被存到 128~191 行（静默越 tile 错位）。documented。
9. **INF-1464** [NEW] [triton#12368](https://github.com/triton-lang/triton/issues/12368)（c2026-10-19）— triton_kernels.matmul persistent kernel 接受未 padding ragged-K 输入（检查自 #8765 起永不失败），slice 尾后至末 tile 行静默加进结果（[128,128,100] vs [100,100,100]）；3.7.0 回归。documented。
10. **INF-1465** [NEW] [tt-metal#61489](https://github.com/tenstorrent/tt-metal/issues/61489)（c2026-10-19）— SDPA causal+滑窗、S 非 q_chunk 倍数时部分块行错（PCC 0.68~0.984；Muse-Glimmer-30B prefill 命中）。documented。
11. **INF-1466** [NEW] [tt-metal#61398](https://github.com/tenstorrent/tt-metal/issues/61398)（c2026-10-19）— ttnn.move/reallocate DRAM ROW_MAJOR、buffer 重叠、行>核数且行字节非 64 倍数时静默搬坏（20/24 配置错；L1/TILE 精确）。documented。
12. **INF-1467** [NEW] [tt-metal#61402](https://github.com/tenstorrent/tt-metal/issues/61402)（c2026-10-19）— fp32_dest_acc_en=False 时 ttnn.sum/mean 沿 W 超 128 tile 结果钉死 8192 不再增长（全 1 行 262144→8192；随机行偏差数倍）。documented。
13. **INF-1468** [NEW] [tt-metal#61490](https://github.com/tenstorrent/tt-metal/issues/61490)（c2026-10-19）— moreh mean/norm backward keepdim=False + dim=末维：形状假设错误致广播读 tile padding（992/1024 元素错、错值可为 1.06e37 垃圾）。documented。

### 拒绝 / 留观（原 7 仓，主要项）

- **vLLM**：#62554/#62483/#62417/#62124 本轮已收录；#62558（TurboQuant GDN 管理块不可切+KV 池膨胀——资源/容量域为主，量化 KV drafter 数值面未证实，留观）；#62308（Qwen3.8-Flash-Next 量化 q/k/v+QSA 加载失败——fail-loud 加载域）；#62340（LoRA+Gemma4 MTP 启动崩溃——fail-loud）；#62351（FunASR adaptor 长度用错 olens+四编码器全 T×T score——前者输入编码错位候选强、后者性能域；正文已读大半，本轮合并量控制留观下轮优先）；#62356（Qwen3-Omni 音频编码跳过 quant_config——量化配置域候选，未深读留观）；#62466（persistent_topk 尾部 -inf 桶下标越界——closed，报告者自证无 shipped config 端到端触发，留观）；#62510（SSM 组 placeholder BlockStored 路由高估——观测/调度域，非数值错误）；#62252（tool-call 参数 0017→int 强转——schema 强转域，未深读留观）；#62225 已收录（10-17 INF-1408，窗口重叠不重判）；#62122/#62051/#61738 维持 10-16~10-18 判定；#61157 维持 10-16 留观；#61681 维持 10-18 留观（本轮窗口无新评论迹象）。
- **SGLang**：#44998 本轮已收录；#45235（qwen3_coder 一个 strict 工具使非 strict 零参工具被发明参数——xgrammar 结构 tag 把 strict 传染给所有工具+换行包裹，机制与 vLLM INF-1481 相邻但不同库不同实现，本轮未逐段深读，留观下轮优先）；#45159（GLM-5.3-Flash EAGLE grammar vocab mask illegal memory access——fail-loud 崩溃域）；#44735 已 closed（HiCache host-tier hybrid GDN 错输出——10-16 留观维持，closed 未详查）；#44789/#44504 维持 10-16/10-17 判定（#44504 已收录 INF-1413）。
- **TensorRT-LLM**：#20603（CSA2 sparse MLA kernel -1 索引 padding 越界读——fail-loud illegal address 域，约 40% 服务器首 20 请求命中，间歇崩溃非静默，留观）；#20485（NGramPoolManager 跳过剪枝违反 invariant——CPU 级 Python 复现，draft 质量影响未量化，留观）；#20634（性能）；#20504 已收录（10-18 INF-1450）；#20509 维持 10-17 留观。
- **tt-metal**：#61489/#61398/#61402/#61490 本轮已收录；#61329/#61328（设备 logprob 归一化含 padding 列/报告惩罚后分数——报告值错误域，GPT-6.1 审阅源码序但 TT 设备误差量级未验证，留观下轮族化评估）；#61471（dispatch-hang 检测解耦——工具域）；10-17 静态审计族（#61175 等）维持不重判。
- **flashinfer**：#6533 已收录（10-16 INF-1407，本轮 find 复核）；#6570 closed（cake_moe_grouped_gemm 权重梯度 TMA 竞态——训练反传域，closed）；#6479/#6389 维持（10-17/#6389 本轮窗口内无新活动迹象未重读）。
- **triton 主仓**：#12348/#12369/#12368 已收录；#12370（与 #12369 同 pass 同日提交、正文首段实为 split-K scale 批次错配的独立缺陷——标题与正文错位，未逐段深读留观）；#12351（warp_specialize 循环结果未用即整循环连同 store 被删——编译器域候选强，本轮合并量控制留观下轮优先）；#12349（同签名 helper 常量捕获冲突——语言语义域留观）；#12366（多结果 helper hint 传染——向量域留观）；#12347 维持 10-18 留观；#12254（tuple reduce axis=None 配错索引——标题判定留观）；#12240/#12243（指针算术/Gluon 类型——未深读留观）。
- **transformers**：#49734（serve 正温度不启用采样——10 次调用输出全同，修复 PR #49732 已提交；采样域候选，本轮未深读留观）；#49578（5.18 tokenizer 丢失 normalizer——embedding 模型每输入多一个换行 token 改变全部 embedding，输入编码域候选强，未深读留观下轮优先）；#49519（rope_scaling 赋值丢既有 RoPE base——位置编码域候选留观）；#49540 维持 10-12 判定。

## 二、NPU 来源扫描（batch-20261019T230043098795-dh2bh0lv）

- 22/22 仓 complete、0 failures、registry_hash 2f6422d0…（22 源全选）；本轮增量判定 = updated_at > 2026-10-18T23:00:43Z 的 **2,437 项**，其中标题强信号（精度/静默/错值/mismatch/NaN/越界/污染/残留等）231 项、关键词命中分桶 578 项逐一过题，其余按标题分类（文档/CI/InferShape 整改/错误码整改/性能族占多数）；重点候选正文深读 22 篇。上一报告（10-18，覆盖至 10-18T23:00:43Z）之前创建且无新更新的条目不重复判定。

### 新增条目（11 条，pending_audit）

1. **INF-1469** [NEW] [op-plugin#605](https://gitcode.com/Ascend/op-plugin/issues/605)（c2026-10-19）— SDPA bool mask 全 False 行被忽略：NPU 输出完整注意力值（0.4767）而非 CPU 语义 0，float -inf mask 双端一致定位 bool→bias 转换分支；生成/解码静默产生非法 token。A3 双芯、CANN 9.1.0。documented。
2. **INF-1470** [NEW] [op-plugin#602](https://gitcode.com/Ascend/op-plugin/issues/602) + 修复 PR [!6055](https://gitcode.com/Ascend/op-plugin/merge_requests/6055)（open）— masked_scatter src 不足时 NPU 静默接受并填越界读设备内存垃圾（CPU 抛 RuntimeError）；确定性、无告警。documented。
3. **INF-1471** [NEW] [op-plugin#598](https://gitcode.com/Ascend/op-plugin/issues/598) — cross_entropy 越界 target 静默吞：mean 悄悄排除该样本、none 槽位跨芯不一致垃圾值（npu:0=0.624 vs npu:1=1.797，真 OOB 读）；CPU 抛 IndexError。documented。
4. **INF-1472** [NEW] [op-plugin#601](https://gitcode.com/Ascend/op-plugin/issues/601) — norm 族（batch/instance/group_norm）fp16 输出 44~155 ULP 系统性偏差、running stats 逐位一致（定位归一化中间精度）；fp32 全族逐位一致。documented。
5. **INF-1473** [NEW] [op-plugin#596](https://gitcode.com/Ascend/op-plugin/issues/596)（c2026-10-18T23:17，本轮窗口内）— cumsum fp16 累加误差沿扫描维增长至 2.3e-2；bf16 路径与 CPU 逐位一致（存在正确实现路径，fp16 降级）。documented。
6. **INF-1474** [NEW] [ops-math#3652](https://gitcode.com/cann/ops-math/issues/3652) — aclnnAddN 混合 dtype 不校验静默错值（fp32+int32 位模式重解释 8388609.0、fp16+fp32 拆位+越界读；dtype 一致性检查的 return false 被当 ACLNN_SUCCESS 吞掉）；仅白名单外 dtype 报 161002。A3、CANN 9.2.0-beta.2。documented。
7. **INF-1475** [NEW] [ops-transformer#6325](https://gitcode.com/cann/ops-transformer/issues/6325) — 310P3 PromptFlashAttention FP16 未归一化 P×V 累加逼近 65504 的位置起输出 +Inf（V=35、position 1871 起 177 个位置；V=4 正常）；黑盒定位待维护者确认中间 dtype。documented。
8. **INF-1476** [NEW] [ops-nn PR !12304](https://gitcode.com/cann/ops-nn/merge_requests/12304)（c2026-10-19，修复单含完整根因+逐 bit 验证）— Ascend950 WeightQuantBatchMatmulV2 N_FIRST_TAIL_RESPLIT 模板限核 16/22 时 PreloadA 越界写 L1 A 区踩 WeightL1_P1 与反量化写竞态：rel_l2 0.2~0.6 且同进程重跑互不相同；修复钳制预取长度至 L1 实际容量。documented。
9. **INF-1477** [NEW] [ops-nn#6879](https://gitcode.com/cann/ops-nn/issues/6879) — 6 个 Loss/Norm 算子 arch35 两类缺陷：SyncAll 前未排空 MTE3 致固定种子两轮红例集不一致（偶现）+ SFL 残差槽未清零（1198 例 111 红）/Poisson 裸加（rmse 比值 8 倍）/SFLGrad 小 D VEC_ERROR。Ascend950PR、CANN 9.3.0。documented。
10. **INF-1478** [NEW] [ops-nn#6605](https://gitcode.com/cann/ops-nn/issues/6605)（c2026-10-13，本轮窗口内更新复核收录）— aclnnDynamicQuant x 空张量早退返回 SUCCESS 且不写非空 scaleOut（[M]）：调用方拿到未初始化 NaN scale、反量化污染且随显存复用不可复现；GE tiling 路径同场景拒绝（两路径不一致）。documented。
11. **INF-1479** [NEW] [Ascend/pytorch#5526](https://gitcode.com/Ascend/pytorch/issues/5526) + 修复 PR [!48172](https://gitcode.com/Ascend/pytorch/merge_requests/48172)（open）— bmm tiling config BLOCK_N 未对齐 16 致精度劣化；**按 issue-PR 对应关系收录，issue 正文仅一行需求、偏差量级未量化（证据边界弱，已在 remarks 注明）**。
12. **INF-1480** [NEW] [vllm-ascend#19119](https://github.com/vllm-project/vllm-ascend/issues/19119)（closed，回溯型跟踪单）— level-2 sleep 丢弃 weights pool 后唤醒更新省略 preserve 参数则原值不恢复、推理静默错值；修复 #18838 已合入（发布未核实）。

### NPU 留观与拒收（主要项）

- **留观（下轮优先）**：op-plugin#599（embedding_bag 2D+include_last_offset 末包 sum/mean 错值+梯度污染——机制清晰正文已读，A3 挖掘系列量大本轮合并量控制留观下轮优先）；op-plugin#604（pack→pad 往返 pad 区残留原输入脏数据——同上系列，变长训练污染为主）；op-plugin#603（multinomial 非法权重不走主机校验崩核 507018——fail-loud 崩核域，随 #589/#592 崩核族化评估）；ops-math#3675（aclnnAddN 同进程二次调用踩残留状态 N 变化 coredump/dtype 变化 SIGSEGV——fail-loud 为主，与 INF-1474 同算子不同机制，已在其 remarks 关联）；ops-math#3677（CdistGrad NaN 处理逻辑——正文为空模板仅标题，无法判定留观）；ops-nn#6891+PR !12179/!12180（conv3d_backprop_input_v2 vec 兜底 FP32 短行累加精度——issue 正文一行、PR 描述为编译期分派改运行期分支改坏 Kahan 补偿链，训练反传域为主留观）；hccl#827（单 rank 遗漏 V 算子位移——10-16 留观维持，本轮 u2026-10-19 无实质新证据）；ops-nn PR !12257（GeluV2 approximate 值域校验——防御域修复）；ops-nn PR !12168（rms_norm bf16 按 int16 处理精度——arch 门控宏缺陷，标题明确机制但未见错误量级正文，留观）；Ascend/pytorch PR !48113（inductor 精度工具覆盖模板 GEMM——维测工具域）；torchair issue #874/PR !3859（AUTOTUNE_TOPN 精度风险文档——文档域，已 merged/closed）；vllm-ascend#19172（ScoreEncoderCacheManager 缓存嵌入数校验绕过——多模态缓存错配候选，64→256 报错 fail-loud、256→64 可静默完成，正文已读，留观下轮优先）；vllm-ascend#19176（静态 kernel cache 重启不复用——性能/构建域 closed）；vllm-ascend#19116（dspark 高并发 TTFT 增——性能域）；vllm-ascend#19173（kimi-k3 PD 重复高精度图 prefill 卡死——挂起域）；vllm-ascend#19207/#19208（PD 分离精度测试脚本报错——fail-loud 测试域）；vllm-ascend#19191（Mooncake ZMQ bind 失败被吞——启动域）；msprobe#638（对比精度 error 行假阳性——观测工具误报域，不转述为计算故障）；msprobe PR !1114（nan_check eager 回调优化——工具域）；msmodelslim issue #389（量化实践 feature closed）；hcomm PR !6105/!6112（空闲核越界访问修复——修复单无 issue 细节随 PR 观察）；ge PR !5454（precision mode 配置——标题无细节留观）。
- **拒收/不占号**：ops-nn/ops-math/ops-transformer InferShape 整改族（!12236 GeGluV2 奇数维拒绝、!12237 max_pool3d_grad INT64 判据、!12239 SoftmaxGrad dtype 一致性、!6508 sparse_bincount 依赖、!6511 space_to_batch 尾块越界、!6470 AddN InferDataType、#3646 26 算子图模式交付件等——防御/推导域批量任务）；ops-math#3652 已收录；ops-math PR !6219（logaddexp 等值无穷——边界语义修复）；runtime #1145/#1147（HostFunc/日志读取——日志域）；mindspore #29606/#29697（CPU softplus 精度/修复聚合——CPU 域+训练域混合，29697 closed 修复单）；mindformers PR 8814（布局参数键名——训练域）；sgl-kernel-npu PR #857（SwiGLU tile A3 UB overflow 修复——修复单无 issue）；triton-ascend#2530 已收录（10-18 INF-1441 维持）。

### 沿用判定（NPU）

- op-plugin#589/#591/#593/#602 前序（10-18 已收录 INF-1434/1435/1436，本轮无状态变化）；ops-nn#6841（INF-1439 维持 open）；ops-nn PR !11513（INF-1440）；triton-ascend#2531（INF-1375，见回访）；vllm-ascend#19065（INF-1429/INF-1398 线维持）；#19080（enable_sparse_li_c8 零层选择静默——性能配置域，10-17 未编目，本轮正文确认无数值错误证据维持不收录）；ops-nn#6708/#6816+PR !12196（INF-1420 线维持）。

## 三、回访已收录来源（update）

- **INF-1437**（randperm 重复值）：ops-math#3576 已 **closed**（u2026-10-19T07:32，关闭原因/修复未注明，修复 PR !6411 状态未复核）→ update note。
- **INF-1353**（nvfp4 KV stride）：vllm#61225 仍 open（core REST 核验）→ update note。
- **INF-1375**（triton-ascend 特化污染）：#2531 仍 open（u2026-10-19T01:45 正文无修复进展）→ update note。
- **INF-1379**（vllm-ascend）：#18754 维持 open（u2026-10-14 后无活动）→ update note。
- **INF-1428**（ops-transformer epsilon）：#6119 open、修复 PR !13800 仍 open（u2026-10-19T08:37）→ update note。
- 轮巡（≥3）：上述 INF-1375/INF-1379/INF-1428 即本轮 3 个更早 open 条目回访（经共享批次本地快照，零 API 成本）；INF-1430（vllm-ascend#19073 merged，10-18 已记录）与 INF-1435（修复 PR !6052 仍 open，u2026-10-19T06:18）状态复核无变化。

## 四、arXiv

6 组查询直连（首轮 ax2/ax3 超时重试后全部 200 有效）。窗口内新论文：ax1 首条 10-13 Cauchy-Schwarz SDC（10-14 已判非故障域）；ax2 最新 10-06 BARE-AI（10-12 已判）；ax3 无窗口内新文；ax4/ax5/ax6 无窗口内相关新文。**无新收录**。

## 五、API 账本与未覆盖范围

| 组 | 来源 | 状态 |
|---|---|---|
| 1-14 | GitHub Search（7 仓 × 2 类查询，created:>=2026-10-12，per_page=100，≥9s 间隔） | 14 组全部 200 有效（q01 vllm bug 68、q02 vllm 词 67、q03 sglang 42、q04 transformers 2、q05 TRT-LLM 9、q06 triton 13、q07 tt-metal 149、q08 flashinfer 17、q09 vllm 96、q10 vllm silent 42、q11 sglang bug 1、q12 transformers bug 13、q13 tt-metal bug 44、q14 TRT-LLM bug 5），无截断 |
| 15-19 | GitHub core REST（#61225、sglang #44997/#44758/#44999、tt-metal #61268 状态） | 有效（直连） |
| 20-25 | arXiv 6 组（ax1-ax6） | 首轮 ax2/ax3 超时 0 字节，重试后全部 200 有效 |
| 26 | NPU 共享批次 | 22/22 complete、0 failures、candidates=11687 |

**未覆盖/限制**：(1) **仅按索引/标题判定的候选**：INF-1479（Ascend/pytorch#5526——issue 正文仅一行需求，按 issue-修复 PR 对应关系收录，偏差量级未核实，remarks 已注明）；ops-math#3677（CdistGrad——正文空模板）；ops-nn#6891（conv3d dx 精度——issue 一行标题，机制经修复 PR !12179 描述补证后仍留观）；vllm#62356、triton#12370、transformers#49734/#49578（标题/摘要判定留观，正文未深读）。(2) NPU 增量 2,437 项中约 1,859 项未命中强信号关键词、按标题分类（文档/CI/InferShape 整改/错误码/性能族为主），未逐条读正文——标题与正文严重不符的候选可能漏判。(3) GitHub 评论级深查未做（无 GITHUB_TOKEN、网页端评论抓取成本高），#61681、#6741、#6484 等留观候选的后续讨论未读取。(4) 旧 open 长期来源评论级复查 3 个（INF-1375/1379/1428）经共享批次快照完成，未做逐条评论拉取。(5) arXiv ax5（wrong output+LLM inference）返回 0 条为真实结果（该词组窗口内无新文），非请求失败。

## 六、编目变更汇总

- 新增 25 条（全部 pending_audit）：**INF-1457 ~ INF-1481**
  - vLLM 主仓：INF-1457（DSpark 思考输出损坏）、INF-1458（packed NCCL 权重同步缓冲复用）、INF-1459（KV offload 跨 TP 坏 KV）、INF-1481（structural tag 覆盖 response_format）、INF-1460（FP8 per-tensor padding amax）
  - SGLang：INF-1461（NVFP4 Marlin 小 scale 全乱码）
  - triton：INF-1462（归约外提符号翻转）、INF-1463（warp_specialize 偏移叠加）、INF-1464（persistent matmul 未 padding 输入）
  - tt-metal：INF-1465（SDPA 滑窗部分块）、INF-1466（move/reallocate 搬坏）、INF-1467（sum/mean 8192 封顶）、INF-1468（moreh 反传 keepdim=False）
  - NPU：INF-1469（SDPA bool mask 行忽略）、INF-1470（masked_scatter 越界垃圾）、INF-1471（cross_entropy 越界 target 吞）、INF-1472（norm 族 fp16 ULP）、INF-1473（cumsum fp16 误差增长）、INF-1474（AddN 混 dtype）、INF-1475（PFA FP16 归一化前溢出）、INF-1476（WeightQuantBmmV2 预取越界竞态）、INF-1477（Loss/Norm arch35 同步+精度）、INF-1478（DynamicQuant 空 x 漏写 scale）、INF-1479（bmm BLOCK_N 未对齐，索引级收录）、INF-1480（level-2 sleep 参数丢失）
- update：INF-1437（issue closed 回访）、INF-1353、INF-1375、INF-1379、INF-1428（回访 note）。
- 跨库查重：本轮 11 个 NPU 候选 URL 已在训练库 `catalog_tool.py find` 查重全部 0 命中，无 cross 关系新增；无 var 关系新增（各新条目机制与既有条目均不同机制，不强行族化）。
- `catalog_tool.py build` + `validate`：通过（ok: true，0 errors，90 warnings 为存量；pending_audit 25）。

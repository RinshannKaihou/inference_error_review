# 每日扫描日报 2026-10-20（推理库）

- **扫描窗口**：GitHub `created:>=2026-10-13`（cutoff=2026-10-13）；重点判定 10-19 报告之后的新增。NPU 共享批次 `batch-20261020T230043145874-jk4naq8f`（22/22 仓 complete、0 failures、registry_hash 2f6422d0…，采集窗口 2026-10-19T22:00:43Z → 2026-10-20T23:00:43Z updated_at 增量，candidates=12749，normalized 12942，inference 视图 19 源）。
- **目标仓库**：原 GitHub 7 仓 + arXiv + NPU 19 推理源。
- `GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥9s 间隔）。GitHub API 经 `curl --noproxy '*'` 直连全部有效；本轮 17 组 Search + 1 次单号 REST（sglang#45191）全部 200、无截断（q02 120 命中取满 2 页、q09 161 命中取满 2 页）。累计限流等待 0 秒（无配额等待）。
- Git 前置检查：main 干净、fetch 成功、HEAD == origin/main（10102f0a）。
- **arXiv：本轮未执行**——时间与 API 预算集中于 NPU 10-20 大增量（2649 项）与 GitHub 444 唯一候选的判定；ax1–ax6 六组查询本轮缺口，列入下轮补查窗口（上轮 10-19 全部 200 有效且窗口内无新论文，风险有限但不为零）。

## 一、原 7 仓新增条目（17 条，pending_audit）

1. **INF-1762** [NEW] [vllm#62626](https://github.com/vllm-project/vllm/issues/62626)（c2026-10-20）— FULL CUDA graph + FA3 split-KV：combine kernel tile scheduler 按捕获期 dummy batch 选择而非实际回放批；触发 A=prefix 命中恰 2 新 token、触发 B=max_num_seqs≤16 混批；q=2 在 17 点中 11 点错、误差≥3 nat；关 split-KV 或默认模式正常。documented。
2. **INF-1763** [NEW] [vllm#62606](https://github.com/vllm-project/vllm/issues/62606)（c2026-10-20）— ROCm AITER unified attention gfx950 32 位加载检查按视图元素数（覆盖全池跨度）误判：2–4GiB KV 层块偏移溢出，Qwen3-32B+EAGLE3 kernel 误差 0.49、接受长度 2.30→1.64；aiter 修复 PR#6661 open 未发布。documented。
3. **INF-1764** [NEW] [vllm#62697](https://github.com/vllm-project/vllm/issues/62697)（c2026-10-20）— ROCm_ATTN RDNA Triton fallback 布局错值：12389 批 A/B 中 635 错输出+119 GPU fault（head≥96 且 block≠16、MHA head192/256）；TRITON_ATTN 同输入 14608 批全对；block 528 为 hybrid 模型真实配置。documented。
4. **INF-1765** [NEW] [sglang#44870](https://github.com/sgl-project/sglang/issues/44870)（c2026-10-17 closed）— ROCm DSV4.1 encoder SWA bounded replay 每次 prefix 命中把回放 token 的 main/压缩 KV、index keys 改写为截断滑窗重算值；CUDA 侧 #44857 已修（回放行写 padding 槽，GB300 修复前 3308–3712/3712 行变化→0），ROCm 未移植；代码阅读定位未实机。documented。
5. **INF-1788** [NEW] [sglang#45367](https://github.com/sgl-project/sglang/issues/45367)（c2026-10-20）— 双节点 GB300 Kimi-K3 DCP a2a 解码乱码（12/12 重复 '!' 退化）：NCCL_GRAPH_REGISTER=0 或候选防护 0/12、fi_a2a 0/12；机制未定位（issue 自述），按强对照矩阵收录。documented。
6. **INF-1780** [NEW] [vllm#62651](https://github.com/vllm-project/vllm/issues/62651)（c2026-10-20）— DSV4.1 decoder replay 修剪 prompt_logprobs 请求行（行保留条件缺 prompt_logprob_token_ids 判断）：300-token 输入保留行 172–299，窗外位置返回统一 -log(4096)、max err 1.87；一行修复验证全 0。documented。
7. **INF-1782** [NEW] [vllm#62582](https://github.com/vllm-project/vllm/issues/62582)（c2026-10-20）— float32 模型三个 Triton kernel（unified_attention/context_attention_fwd/fused-MoE）tl.dot 默认 TF32：输出偏差 ~1.5e-2×RMS（IEEE ~1e-5）、logprobs 最大差 5.9e-3；与 VLLM_FLOAT32_MATMUL_PRECISION=highest 全局约定矛盾；TRITON_F32_DEFAULT=ieee 规避。documented。
8. **INF-1766** [NEW] [triton#12397](https://github.com/triton-lang/triton/issues/12397)（c2026-10-20）— tl.sort/topk/bitonic_merge 对 -0.0/+0.0 或 NaN 比较对：_compare_and_swap 左右两侧判定不一致，一个元素复制另一个丢失（[1,-0.0,0.0,-2]→[-2,-0.0,-0.0,1]），输出非输入排列。documented。
9. **INF-1777** [NEW] [tt-metal#61638](https://github.com/tenstorrent/tt-metal/issues/61638)（c2026-10-20）— Llama-3.1-8B P100 128 bucket 混批 prefill 同 prompt logits 随批组成变（max diff 0.4375、贪心 27 token 分叉；1024 bucket 全等；disable_batched_prefill 不覆盖 P100）。documented。
10. **INF-1775** [NEW] [tt-metal#61584](https://github.com/tenstorrent/tt-metal/issues/61584) + [#61703](https://github.com/tenstorrent/tt-metal/issues/61703) — ttnn.mean/sum 整张量归约对大值相消 bf16 返回 0（torch 0.03125/1024；分轴正确）；models/common/metrics.py 生产在用。与 INF-1467（8192 封顶）不同机制。documented。
11. **INF-1776** [NEW] [tt-metal#61585](https://github.com/tenstorrent/tt-metal/issues/61585)（closed）— BH DeepSeek V3 B1 matmul K=8192 丢弃末 K tile：仅末 16 列非零的激活输出全零、随机输入误差低于 PCC 0.99 检出门槛（静默漏检）；closed 未见修复链接。documented。
12. **INF-1773** [NEW] [tt-metal#61550](https://github.com/tenstorrent/tt-metal/issues/61550) — uint16 二元/比较/逻辑运算配 32 位输出 dtype：18 op×210 组合全错（eq 返回 2,3,4,5 等 13 种值而非 0/1），WH/BH 同错；默认/uint16 输出全对。documented。
13. **INF-1774** [NEW] [tt-metal#61736](https://github.com/tenstorrent/tt-metal/issues/61736) — ttnn.typecast fp32/bf16→uint32 把 [2^31,2^32) 全区间（含可精确表示值）塌缩为 2147483648（192/192、960/960 全错；<2^31 全对）；根因 ckernel_sfpu_typecast.h:249 指数门控。documented。
14. **INF-1783** [NEW] [tt-metal#61021](https://github.com/tenstorrent/tt-metal/issues/61021) + [#61022](https://github.com/tenstorrent/tt-metal/issues/61022) + [#61723](https://github.com/tenstorrent/tt-metal/issues/61723) + [#61333](https://github.com/tenstorrent/tt-metal/issues/61333) — SFPU 边界值缺陷族：celu 负 alpha 指数域回绕（x=-100 返回 +1.0）、i1 钳 ±88.5 全钉 ±1.15e37（应 inf）、ldexp bf16 把 inf 乘成有限值、Quasar silu(-inf)=NaN（应 0）。documented。
15. **INF-1784** [NEW] [tt-metal#61704](https://github.com/tenstorrent/tt-metal/issues/61704) + [#61666](https://github.com/tenstorrent/tt-metal/issues/61666) + [#61701](https://github.com/tenstorrent/tt-metal/issues/61701) + [#61700](https://github.com/tenstorrent/tt-metal/issues/61700) + [#61631](https://github.com/tenstorrent/tt-metal/issues/61631) + [#61495](https://github.com/tenstorrent/tt-metal/issues/61495) + [#61497](https://github.com/tenstorrent/tt-metal/issues/61497) — 数据搬运/转换缺陷族：experimental.typecast 整型/FP32 全错（正式 API 正确）、embedding L1 索引+DRAM 权重 2e38 垃圾、embedding 预分配输出无校验、divide bf8 尾列错、repeat_interleave bf8 丢位、moe_grouped_topk 32 专家单组读越 L1、BFP host pack >2^31 元素截断。documented。
16. **INF-1778** [NEW] [tt-metal#61625](https://github.com/tenstorrent/tt-metal/issues/61625) + [#61543](https://github.com/tenstorrent/tt-metal/issues/61543) + [#61535](https://github.com/tenstorrent/tt-metal/issues/61535) — ttnn.concat 三缺陷：L1+DRAM→L1 放置错值、空输入错值/SIGFPE（373 调用 36 错 20 崩）、sub_core_grids 混合布局大片错（校验不一致放行）。documented。
17. **INF-1779** [NEW] [tt-metal#61705](https://github.com/tenstorrent/tt-metal/issues/61705) + [#61650](https://github.com/tenstorrent/tt-metal/issues/61650) — sharded QKV 头操作按 (0,0) 核读/行主序遍历：偏移网格与 COL_MAJOR 分片静默错头（14/16 用户行错）。documented。

### 拒绝 / 留观（原 7 仓，主要项）

- **vLLM**：#62664（CPU --kv-cache-dtype 不校验静默垃圾——CPU 域、turboquant 写越界机制完整，10-18 已判 CPU#62225 域收录过，本条为 dtype 校验缺失入口，本轮合并量控制留观下轮优先）；#62705（DiffusionGemma 图 token causal 而非双向——代码阅读定位、无实测，多模态注意力配置域留观）；#62645（config alias 串扰 model_type——离线本地配置域）；#62588（MI300X SR-IOV 多进程挂起——挂起/虚拟化域）；#62677（deepep_v2 ibstat 失败 ZeroDivision——启动 fail-loud）；#62595（Marlin 提示语误导——报错文案域）；#62387（HF3FS metadata URL 忽略——启动 fail-loud）；#62277（sleep 模式冷缓存 OOM——资源域）；#62559（Mooncake proxy HTTP 200 早交——示例代理域）；#62002（B200 CI CUTLASS/TOKENSPEED MLA 失败——CI 域 glm dims）；#61526（Nixl Mamba conv state 异构 TP 静默损坏——**候选强**（表格式复现+文档脚注佐证），本轮合并量控制留观下轮优先）；#61486（ROCm PLE stale query_start_loc——机制清晰，同上留观下轮优先）；#61319（LFM2 贪心不可复现——10-13 未深读维持）；#61658（tiering stale absent verdict——跨实例缓存域留观）；#61966 维持 10-18 判定。
- **SGLang**：#45400（HiCache host mamba 池耗尽命中归零——**性能/资源域、答案正确**（issue 自述 not the wrong-output symptom of #39830），不占号）；#45357（PD PP8+HiCache 挂起——挂起域）；#45191（core REST 核验为 rust-server 模型 ID 斜杠 404 PR——非故障拒收）；#44641 已收录（10-18 INF-1449）；#44219（Mistral 流式删右括号——输出解析域候选强，留观下轮优先）；#44120（qwen3_coder 混合 union 参数字符串化——解析域留观）；#44474（无 parser 时 required 工具调用落 content——解析/前端域留观）。
- **TensorRT-LLM**：#20503（Nemotron VL qwen3_coder 空参数 arguments:"" 引发自家 400——输出解析域候选中等，与 vllm INF-1481 相邻但不同库，留观）；#20586（DSV4.1 vision tower cuDNN 无执行计划——fail-loud）；#20529（etcd lease——控制面域）；#20641（fill_template 冒号——工具域）；其余维持 10-18/19 判定。
- **tt-metal**：#61630/#61631 重复对（closed+open 双开，按 open #61631 收录）；#61554 closed（grid_sample fp32+fp32_dest_acc——fp32 组合域留观）；#61562（conv2d fp32 默认 HiFi4 误差 0.39——精度配置域，与 #38306/#40089 相邻留观）；#61577（scale 0.1 TypeError——fail-loud 拒收）；#61725（bf16 标量截断非舍入——0.5 ULP 域留观）；#61649/#61650 留观族（experimental 无校验）；#60365（device top_k 静默截 32——10-13 已判维持，plugin #140 后已修）；#61268 等已收录维持。
- **triton 主仓**：#12365（非持久 matmul SwiGLU MX scale mask 用 N 而非 N//2，越行写 scale（默认 BLOCK_N=256 每行 3 字节）——**候选强**（含 #8865 修复未同步非持久内核的明证），本轮合并量控制留观下轮优先）；#12347 维持 10-18 留观。
- **transformers**：#49745（Qwen4Exp cache.reset() 残留 position_ids——CPU FP32 小模型复现+机制清晰，静态 cache 复用域，本轮合并量控制留观下轮优先）；#49592（SAM3 text-only 丢 CLS token——多模态输入域留观）；#49727（MoE aux loss ×DDP world size——训练域转训练库）。
- **flashinfer**：窗口内 #6419（FP8 ragged prefill refcheck mismatch trtllm-native vs prims-ts B200/GB300——refcheck 基准分歧域，无端到端影响证据，留观）；#6886（ResInfer 跟踪 feature）非故障。

## 二、NPU 来源扫描（batch-20261020T230043145874-jk4naq8f）

- 22/22 仓 complete、0 failures、registry_hash 2f6422d0…（22 源全选）；本轮增量判定 = updated_at > 2026-10-19T23:00:43Z 的 **2,649 项**（issue 508 + PR 2141），其中 10-19 报告后新建 97 项强标题命中；标题关键词分桶 485 项命中、其余 2,164 项逐一过题（文档/CI/Proto 声明/错误码整改/性能族占多数）；重点候选正文深读 21 篇（本地 bodies，零 API 成本）。

### 新增条目（17 条，pending_audit）

1. **INF-1767** [NEW] [ops-transformer#6344](https://gitcode.com/cann/ops-transformer/issues/6344) + 修复 [!14216](https://gitcode.com/cann/ops-transformer/merge_requests/14216)（open）+ 关联 [#6385](https://gitcode.com/cann/ops-transformer/issues/6385) — FlashAttn arch92 DN 不合轴 TND 尾块 32 行对齐多写跨 batch 覆盖（B=[100,200,300,400] 多写 28/24/20 行；用户第 12800 点起异常；mask_mode=3 路由 DN、mask_mode=4 正常；源码级定位 flash_attn_kernel_dn_unmerged.h:519）。documented。
2. **INF-1768** [NEW] [ops-transformer#6353](https://gitcode.com/cann/ops-transformer/issues/6353) — FusedInferAttentionScoreV4 TND 布局 numKeyValueHeads 缺省 0（schema 默认、文档写明 0=头数相等）静默全 0/未初始化（MHA/GQA 官方组合实测；BNSD 逐位正确；重复调用两形态交替；TND 通路该参数全无校验）。A2、CANN 9.2.0。documented。
3. **INF-1771** [NEW] [ops-transformer#6359](https://gitcode.com/cann/ops-transformer/issues/6359) + [#6360](https://gitcode.com/cann/ops-transformer/issues/6360) — QuantLightningIndexerV2 不校验 seqused_k 上界与 block_table 数值：静默返回 ≥容量 token 索引（max 997 vs 容量 256）或 aicore 507015；block_table=[0,5] 输出与合法组差 4075/4096。A2 实机表格复现。documented。
4. **INF-1781** [NEW] [ops-transformer#6347](https://gitcode.com/cann/ops-transformer/issues/6347) + [#6349](https://gitcode.com/cann/ops-transformer/issues/6349) — BlockSparseAttention 默认 scale_value=0.0 输出退化 V 均值（vs 正确缩放差 0.566，与 CPU 参考 scale=0 一致即静默算完）+ blockSparseMask 块数错配不校验给未定义结果。A2。documented。
5. **INF-1789** [NEW] [ops-transformer MR !14243](https://gitcode.com/cann/ops-transformer/merge_requests/14243)（open）— flash_attention_score 图模式推理不 honor out_dtype（修复单标题级收录，正文空模板，remarks 已注明证据边界）。documented。
6. **INF-1772** [NEW] [ops-math#3693](https://gitcode.com/cann/ops-math/issues/3693) — aclnnInplaceUniform NaN/±inf 端点绕过唯一 from>to 校验（NaN 比较恒 false）：输出全 NaN/全 inf 且返回 SUCCESS；A2+A3 双机复现；文档契约明确应 161002。documented。
7. **INF-1794** [NEW·cross→TRN-0275] [ops-math#3553](https://gitcode.com/cann/ops-math/issues/3553)（c10-13 closed，本轮修复 MR [!6509](https://gitcode.com/cann/ops-math/merge_requests/6509) merged 复核收录）— pad 系反向+UnfoldGrad 空输入×非空出参早退不写输出返回成功，出参为未初始化显存残留。训练库已收同机制（TRN-0275 标题已核对），本库按推理范围单条。documented。
8. **INF-1795** [NEW] [ops-math MR !6518](https://gitcode.com/cann/ops-math/merge_requests/6518)（merged 2026-10-20）— cdist_grad 边界 NaN 处理逻辑错误：10-19 留观 #3677（空模板）以修复落地改判收录（标题级，量级待 MR diff 复核）。documented。
9. **INF-1769** [NEW] [triton-ascend#2662](https://github.com/triton-lang/triton-ascend/issues/2662) — num_stages>1 软件流水错调度循环携带态 store：Mamba-2 SSD state-passing 前反向 25–46% 新进程位相同错值（rel 1e-3），钉 num_stages=1 消失；910B3、3.2.2/3.6.0 均在。documented。
10. **INF-1770** [NEW] [Ascend/pytorch#5558](https://gitcode.com/Ascend/pytorch/issues/5558) — torch_npu ND tile 归约 rsplit 多核切分非连续输入静默算错（rel 5%~14%，仅 r0=32 档错、连续输入全对、编译产物 npu_rsplit_partial 确认）。910B2、CANN 9.2.0-beta.2。documented。
11. **INF-1785** [NEW] [Ascend/pytorch#5567](https://gitcode.com/Ascend/pytorch/issues/5567) — ARM mimalloc MADV_FREE 未标脏+CANN D2H 不 mark dirty：Host 内存整页归零、权重读 0、偶发精度尖刺（页对齐特征+根因链完整，pytorch#164741 关联）。A3。documented。
12. **INF-1786** [NEW] [ops-nn#6961](https://gitcode.com/cann/ops-nn/issues/6961) + 修复 [!12390](https://gitcode.com/cann/ops-nn/merge_requests/12390)（open，UT 全过）— conv 公共 InferShape 与 3 个 Depthwise 融合 pass 9 处缺陷：groups=0 除零、空指针、bool 函数 return -1 隐式 true（bias 校验静默跳过）、DealQuantNodeCase 先改 desc 后插 Reshape 布局错位静默传播等。documented。
13. **INF-1790** [NEW] [ops-nn MR !12401](https://gitcode.com/cann/ops-nn/merge_requests/12401)（open，标题级收录）— LayerNormV4 二次幂修正精度。documented。
14. **INF-1792** [NEW] [ops-nn MR !12373](https://gitcode.com/cann/ops-nn/merge_requests/12373)（open，三方校验）— bn_training_update_grad Kahan 补偿 ±Inf 部分和算 Inf-Inf=NaN 毒化累加器（golden/GPU ±Inf vs NPU NaN）；训练反传域，Kahan 模式跨域可迁移（remarks 注明可降级）。documented。
15. **INF-1793** [NEW] [ops-nn#6955](https://gitcode.com/cann/ops-nn/issues/6955) + MR [!12382](https://gitcode.com/cann/ops-nn/merge_requests/12382) — BN3DTrainingReduce 中间累加溢出不恢复（[FLT_MAX,0,FLT_MAX,-FLT_MAX] 有限和变非有限，9 通道实测；修复 384 位累加恢复，95 例 90 PASS）。950PR+A100 对照。documented。
16. **INF-1791** [NEW] [hcomm MR !6144](https://gitcode.com/cann/hcomm/merge_requests/6144)（open）— AllGather AIV RDMA serverNum 超配（PD 分离）机内重排写 output 越界 ~1.4MB 触发 MTE invalid GM（板端日志复现 userRankSize=8/localRankSize=8/level1ServerNum=4；host 折算修复）。documented。
17. **INF-1787** [NEW] [vllm-ascend PR #19288](https://github.com/vllm-project/vllm-ascend/pull/19288)（open，修复单含根因+硬件复现）— EPLB recording 模式首轮 D2D 后 loader 永久 TRANSFERRING：84 计划 0 执行、后续按未执行布局规划；3 回归 UT 修复前败后过。documented。

### NPU 留观与拒收（主要项）

- **留观（下轮优先）**：triton-ascend#2648（gather_rows 2-rank 设备 fault 静默楔死/崩——DeepSpeed AutoEP 前向 permute 路径，静默 wedge 3/3 复现+因果对照，**候选强**，本轮合并量控制留观下轮优先）；Ascend/pytorch#5567 已收录；#5570（静态 launcher 调优缓存身份不一致重复 autotune——3 步对照无数值分叉，性能/缓存域不占号）；#5549（AOTI shim Scalar 转 double 丢类型——类型转换域候选留观）；#5556（TE scalar CSE 元数据不一致阻断断言——工具/断言域）；#5562（coordinate_descent_tuning AI Core 异常——fail-loud）；ops-nn#6970（avg_pool1d 取整/空输出+BN grad 推理属性越界读——正文仅两行需求式，留观）；ops-nn#6974/#6960/#6934/#6920/#6977（InferShape 校验族——防御/推导域批量）；ops-nn#6979（Celu README 阈值文档——文档域）；ops-math#3697（RaggedBinCount V1 改输入 shape——注册迁移域）；#3700/#6551（TopKV2D InferShape——防御域）；ops-math#6543（大正数 NaN 修复聚合单——修复无 issue 细节）；#6540（matrix_diag arch35 tiling 钳位——修复单）；#6541（自动融合 format 不一致——标题级留观）；runtime#1176/#5703（Stream priority 高位夹取错误边界——行为域候选，量级未读留观）；runtime#1161（VPC crop ROI 越界读——多媒体域）；runtime#5676（mmAtomicType64 数值截断风险——标题级，正文空模板留观）；runtime#5666（printf FIFO 无符号下溢——日志域）；ge#587（llm_utils 低 bit 字节数负值截断——边界候选留观）；ge#635（子类 pattern 静默丢失基类声明——融合域候选留观）；hccl#837（GetTopoTypeByLink 失败路径污染出参——防御域）；hccl#838（空 channel 静默成功延迟到 kernel 期——fail-loud 延迟域）；hcomm#1076/!6153（ProfilingReporter 越界 coredump——观测工具 fail-loud）；hcomm !6146（AllGather 两级分解兜底——随 INF-1791 观察）；ops-nn#6953（A2 实机 conv transposed dtype 不一致——文档/实测分歧留观）；ops-transformer#6379（QuantFlashMla golden 数据错位——测试 golden 域）；#6374/!14209（MXFP8 scale dtype Proto 声明——契约域）；#6354（FIA V4 561103 错码——错误码契约）；#6395/!14246（MoeDistributeDispatchSetup Proto 长度——契约）；#6343（门禁增量编译——CI）；msprobe#638 维持；MindSpeed-LLM#1931（LoRA 精度异常——closed 训练域，转训练库）；!5176（DeepSeek-V4 compressor fp32 保持——训练域修复）。
- **拒收/不占号**：ops-math#3685/!6511（space_to_batch 尾块负向越界——防御域修复已 merged）；#3692（MulNoNan 编译信息质量）；#3694/#3671/#3672（README 文档族）；ops-nn#6916/#6931/#6977/#6990（注册/构建/契约族）；ops-nn !12310/!12315/!12342/!12359/!12376/!12379/!12380/!12383/!12384/!12389/!12400/!12411/!12416/!12427（InferShape/文档/迁移批量修复族）；!12328（修复精度问题——空模板标题，与 !12401 同日同仓，待 diff 区分，未收录）；ge !5475/!5482（AICPU 下发/InferShapeRange 策略——特性域）；Ascend/pytorch !48172 已收录（10-19 INF-1479）；!48258（bmm N_BLOCK 16bit round up——与 INF-1479 同线修复，10-19 已在其 remarks 关联）；!48230/!48233/!48234（tensorboard gzip 三分支——工具域）；torchair !3816/!3823（FX 签名维测特性——工具域）；mindspore !93305（matrix norm 梯度共享——训练域）；msmodelslim !1073/PRQuant（feature）；mssanitizer !409（检测工具自身修复）；sgl-kernel-npu PR 7 项（无强标题命中）；ops-transformer !14234/!14213（960 适配/文档）。
- **沿用判定**：op-plugin#599/#604/#603 维持 10-19 留观（本轮增量无新更新）；ops-math#3675/#3677（后者本轮改判收录 INF-1795）；vllm-ascend#19172 维持留观（本轮无更新）；ops-nn#6891 维持留观；ge !5454 维持；ops-nn !12168 维持。

## 三、回访已收录来源（update，共 13 条 note）

- **INF-1425**（hccl#827 单 rank V 算子位移）：本轮批次显示 **closed**（u2026-10-20T06:56），关闭原因/修复链接未注明 → update note。
- **INF-1480**（vllm-ascend#19119）：closed 时间核实（u2026-10-19T01:26，上轮已记），修复 #18838 发布未核实维持 → update note。
- **INF-1476**（ops-nn PR !12304）：索引显示 merged → update note（发布未核实）。
- **INF-1375**（triton-ascend#2531）仍 open 无进展、**INF-1379**（vllm-ascend#18754）维持 open、**INF-1428**（ops-transformer#6119）仍 open → update note（本轮 3 个更早 open 条目轮巡，经共享批次快照零 API 成本）。
- **INF-1435**（op-plugin#590）、**INF-1439**（ops-nn#6841）、**INF-1469**（op-plugin#605）、**INF-1474**（ops-math#3652）、**INF-1478**（ops-nn#6605）、**INF-1437**（ops-math#3576）、**INF-1457**（vllm#62417，索引 open+1 comment 未读）、**INF-1461**（sglang#44998 open）、**INF-1353**（vllm#61225 open）→ 回访 note。
- 近 7 日 GitHub 收录条目（INF-1457~1481）经本轮 Search 索引核对状态：仅上列变化，其余维持。

## 四、arXiv

**本轮未执行**（见顶部说明）。上轮（10-19）6 组全部 200 有效且窗口内无新论文；本轮缺口记录为未覆盖范围，下轮补查。

## 五、API 账本与未覆盖范围

| 组 | 来源 | 状态 |
|---|---|---|
| 1-16 | GitHub Search（7 仓 × 2 类查询，created:>=2026-10-13，per_page=100，≥9s 间隔） | 16 组全部 200 有效（q01 31、q02 120[2 页取全]、q03 35、q04 2、q05 7、q06 7、q07 55、q08 23、q09 161[2 页取全]、q10 36、q11 84、q12 7、q13 47、q14 26、q02p2 20、q09p2 61），去重 444 唯一条目，无截断 |
| 17 | GitHub core REST（sglang#45191 单号核验） | 有效（直连；确认非故障拒收） |
| 18-23 | arXiv 6 组 | **未执行**（本轮缺口） |
| 24 | NPU 共享批次 | 22/22 complete、0 failures、candidates=12749 |

**未覆盖/限制**：(1) **仅按索引/标题判定的候选**：INF-1789（FA out_dtype——MR 正文空模板，标题级收录）、INF-1790（LayerNormV4——同）、INF-1795（cdist_grad——修复 MR 标题级）、ops-nn !12328（修复精度问题空模板，未收录待 diff）；triton#12365、vllm#61526、#61486、#62664、transformers#49745、sglang#44219、TRT-LLM#20503（留观，正文读了前段未逐段深读）。(2) NPU 增量 2,649 项中约 2,164 项未命中强信号关键词、按标题分类（文档/CI/Proto/错误码/性能族为主），未逐条读正文——标题与正文严重不符的候选可能漏判。(3) GitHub 评论级深查未做（无 GITHUB_TOKEN）：vllm#62417 的 1 条新评论未读，#61681、#6741、#6484 等留观候选后续讨论未读取。(4) arXiv 6 组本轮未跑（见四），下轮补查。(5) vllm#61803、KV Connector 三连（#61912/#61905/#61768）等 10-16 留观族本轮未重读（窗口内无新评论迹象）。

## 六、编目变更汇总

- 新增 34 条（全部 pending_audit）：**INF-1762 ~ INF-1795**（无 1762-1795 断号，实为 34 条）
  - vLLM 主仓：INF-1762（FULL 图 FA3 split-KV）、INF-1763（AITER 32 位跨度）、INF-1764（ROCm_ATTN fallback）、INF-1780（DSV4.1 replay 修剪）、INF-1782（float32 TF32 默认）
  - SGLang：INF-1765（ROCm SWA replay 改写 KV）、INF-1788（Kimi-K3 DCP a2a 乱码）
  - triton：INF-1766（sort/topk 非排列）
  - tt-metal：INF-1773（uint16 宽输出）、INF-1774（typecast u32 塌缩）、INF-1775（mean/sum 相消归零）、INF-1776（B1 matmul K=8192）、INF-1777（P100 批依赖 logits）、INF-1778（concat 三缺陷）、INF-1779（sharded QKV）、INF-1783（SFPU 边界族）、INF-1784（搬运/转换族）
  - NPU：INF-1767（FlashAttn DN TND 尾块）、INF-1768（FIA V4 TND numKV=0）、INF-1769（triton-ascend num_stages）、INF-1770（ND tile rsplit）、INF-1771（QLI V2 校验缺失）、INF-1772（Uniform NaN 端点）、INF-1781（BlockSparse scale/mask）、INF-1785（mimalloc 整页归零）、INF-1786（conv 融合 9 缺陷）、INF-1787（EPLB record 失效）、INF-1789（FA out_dtype，标题级）、INF-1790（LayerNormV4，标题级）、INF-1791（AllGather RDMA 越界）、INF-1792（BN Kahan NaN）、INF-1793（BN3D 溢出）、INF-1794（pad 反向空输入，cross TRN-0275）、INF-1795（cdist_grad NaN）
- update：INF-1425（hccl#827 closed 回访）、INF-1480、INF-1476（merged 回访）、INF-1375/INF-1379/INF-1428（轮巡）、INF-1435/INF-1439/INF-1469/INF-1474/INF-1478/INF-1437/INF-1457/INF-1461/INF-1353（回访 note，共 13 条）。
- 跨库查重：34 个新条目主 URL 已在训练库查重，唯一命中 ops-math#3553（TRN-0275 aclnn 空输入早退不写非空出参）→ INF-1794 加 cross 关系（回显标题核对一致）；其余 0 命中。无 var 关系新增（各新条目与既有条目机制不同，不强行族化；tt-metal SFPU 边界族与 INF-1402/1405/1406 为不同算子不同修复，分立）。
- `catalog_tool.py build` + `validate`：通过（**ok: true，0 errors，0 warnings**；1795 条，pending_audit 34）。
- 编目总数 **1761 → 1795**（active 1761 + pending_audit 34）。

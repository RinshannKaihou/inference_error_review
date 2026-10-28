# 推理故障每日扫描 2026-10-27（UTC）

- 窗口：GitHub/arXiv `created:>=2026-10-20`（7 日）；NPU 共享批次 `batch-20261027T230043148191-vocxzpm3`（采集 since 2026-10-26T22:00:43Z / until 2026-10-27T23:00:43Z，22 仓全覆盖、0 failures，candidates 18615）。
- 编目：本轮新增 **29 条**（INF-2207 ~ INF-2235，pending_audit），总数 **2235**；validate `"ok": true`（n_errors=0，warnings=51 为既有模式待定项增量）。
- Git：前置检查通过（main 干净、HEAD=origin/main=5a69588e）；本轮末提交并推送。

## 一、原 GitHub 7 仓 + arXiv 覆盖账本

- GitHub 未认证 Search API（约 7-8s 间隔，GITHUB_TOKEN 未配置），12 组查询全部 `created:>=2026-10-20` 且全部 200 有效（vllm×2、sglang×2、transformers×1、TensorRT-LLM×2、triton×2、tt-metal×2、flashinfer×1；中途 rem 到 0 一次，前台分段 sleep 45s 后自然回补，累计等待 45 秒）。候选正文全部由 search item 自带 body 覆盖，未消耗 core API。
- 核对：vllm#63480 维持 10-24/10-25 判定；triton#12397=INF-1766、#12425 维持 10-24 判定（interpreter 调试域）、#12395 维持、#12394/#12392/#12374 已判；fi#7179=INF-2166、#7021 维持 10-22 判定；tt#62133=INF-2131（closed 维持）、#62134 维持 10-22、#62153/#62016=INF-2122/#62019 维持；sg#46259=INF-2179、#46446=INF-2186、#45755/#45864 维持 10-14~10-22 判定；tf#49873=INF-2134、#49810 维持 10-21 判定。
- **新增收录（GitHub 域 16 条）**：vllm#64117/#63759/#63653/#63944/#64048、vllm-ascend#19948/#19929、sglang#46572/#46752/#46476/#46615/#46806、triton#12525/#12429、tt-metal#62719/#62720/#62506/#62451（NPU 域条目见下节，其中 2 条 vllm-ascend）。
- arXiv：4 组查询直连（https 重试一次成功）全部 200 有效。窗口内新论文：ax1 首条 10-24 云架构讨论（非故障域）；ax2 最新 10-22 Rowhammer LLM 推理攻击（外部攻击域，未入故障编号）；Cauchy-Schwarz SDC（10-14 已判非故障域）、BARE-AI（10-12 已判）维持。**无新收录**。

### 新增判定摘要（GitHub 域，正文均深读）

1. **INF-2208** [NEW] [vllm#64117](https://github.com/vllm-project/vllm/issues/64117) — Mooncake BHLNC：head 注册把整块步长当传输长度，每 4KB segment 传 196KB 越写相邻块；8 并发 600 题 0→54-60 错（大量来自旁观请求），跨请求 KV 污染。
2. **INF-2233** [NEW] [vllm#63759](https://github.com/vllm-project/vllm/issues/63759) — Qwen3.5 三个 Triton kernel int32 偏移回绕：209717/174763/262144 token 阈值分别启动失败/非法访存/静默错 qkv；单 GPU 最小复现。
3. **INF-2223** [NEW] [vllm#63653](https://github.com/vllm-project/vllm/issues/63653)（closed）— OffloadingConnector 滑窗层 load 从未写入的 CPU slot 读 KV，attend 全零/陈旧 KV；CPU 单测+e2e 双复现，设备无关。
4. **INF-2224** [NEW] [vllm#63944](https://github.com/vllm-project/vllm/issues/63944) — int8_per_token_head scale view fill_(1.0) 覆写共享物理页的 GDN 状态：引擎首个请求第二 token 起垃圾、之后全对（冒烟测试盲区）。
5. **INF-2232** [NEW] [vllm#64048](https://github.com/vllm-project/vllm/issues/64048) — GH200 KV 基址 4GiB 对齐时 NixlConnector dma-buf MR 回绕返回垃圾；UCX_CUDA_COPY_DMABUF=no 规避（单变量对照）。
6. **INF-2234** [NEW] [sglang#46572](https://github.com/sgl-project/sglang/issues/46572) — Inkling prefill 图 padding 行（out_cache_loc=0）stale 数据写 KV slot 0，页<128 滑窗读后 0×NaN=NaN：默认配置静默 token 0+NaN logprob，关 Gumbel 则每 rank crash。
7. **INF-2210** [NEW] [sglang#46752](https://github.com/sgl-project/sglang/issues/46752) — DSPARK/DFLASH 从不累积惩罚器状态：repetition_penalty 1.0/2.0 输出逐字节相同，参数到达 SamplingParams（400 拒绝证明）但不生效；EAGLE 路径正确。
8. **INF-2211** [NEW] [sglang#46476](https://github.com/sgl-project/sglang/issues/46476) — sample_from_anchor=False 时 draft 每请求 γ+1 行、FA backend 固定推导 γ，cu_seqlens_q 错位；batch>1 接受率显著劣化、无报错；修复 PR #46478（未核实）。
9. **INF-2212** [NEW] [sglang#46615](https://github.com/sgl-project/sglang/issues/46615)（closed）— mixed-chunk+SWA bounded-replay 把 mixed batch 解码行晚层窗口截为 1 token：240 题算术 0.6→0.48-0.55 五臂定量；closed 修复细节未核实。
10. **INF-2231** [NEW] [sglang#46806](https://github.com/sgl-project/sglang/issues/46806) — update_weights_from_disk 无 selector 把目标权重灌进 DFLASH draft：调用成功、接受长度掉 1.0、比关推测更慢。
11. **INF-2226** [NEW] [triton#12525](https://github.com/triton-lang/triton/issues/12525) — 空 tl.range 展开 epilogue 多执行迭代：(-3,-1,-2) 期望 0 得 1（factor2）/9（factor4）；interpreter 正确。
12. **INF-2225** [NEW] [triton#12429](https://github.com/triton-lang/triton/issues/12429) — 磁盘缓存键漏全局变量值等信息：改值重跑仍执行旧二进制，跨进程/进程内双复现（与 #12428 mangle 缺失为相邻独立两层，#12428 本轮留观：一半 fail-loud 编译错、namedtuple 半静默，下轮族化评估）。
13. **INF-2227** [NEW] [tt-metal#62719](https://github.com/tenstorrent/tt-metal/issues/62719) — moreh layer/group_norm 均值≥5 丢方差：50 均值相对误差 123%，fp32 累加不救。
14. **INF-2228** [NEW] [tt-metal#62720](https://github.com/tenstorrent/tt-metal/issues/62720) — 行主序 eltwise 写 WIDTH/BLOCK sharded 输出错值、部分组合挂起；TILE/HEIGHT 正常。
15. **INF-2229** [NEW] [tt-metal#62506](https://github.com/tenstorrent/tt-metal/issues/62506) — softmax 加 -inf 掩 padding 但 NaN+(-inf)=NaN：隐式 padding 残留 NaN 时行被毒化；BEVFormer 4/8 PCC 失败、NaN 数与坏行数 270↔270 精确对应；诊断帖已自更正 #61609 归因。
16. **INF-2230** [NEW] [tt-metal#62451](https://github.com/tenstorrent/tt-metal/issues/62451) — DFB 程序本地 L1 分配只看自身 range waterline，共享核上两 DFB 完全重叠互相覆写；校验不查 DFB 互相重叠。

### 拒绝 / 留观（GitHub 域）

- **vllm#64005**（MRV2 Qwen3-32B TP2 只吐 token 0；MRV1 正常）— 现象强（全 token id 0）但环境不完整（GPU/驱动/CUDA 未知、nightly 未测）、机制未定位；留观下轮。
- **vllm#64106**（MI355X fp8 KV GSM8K 0.975→0.305）— 任务级精度崩塌实证强，但 ROCm/MI355X 平台且根因未定位（quant kernel 未指认）；留观下轮（平台数值差异 vs 缺陷边界同 10-14 #18755 处理）。
- **vllm#63811**（HiSparse+NIXL 长解码 all-NaN logits）— 现象+gumbel 采样器 NaN→1023 机制链清晰，但首个 NaN 来源未定位（RDMA 假设未证实、单节点不可复现）；留观，与 INF-2224/2234 的 NaN 传播族后续并案评估。
- **vllm#63630**（GPTQ dynamic 三名字空间静默不匹配）— mixed dense/quantized checkpoint 加载失败域（fail-loud AttributeError），静默面为"忽略不生效"；与 10-22 #5108/#5328 配置静默不生效族观察，不占号。
- **triton#12428**（helper mangle 缺 const/namedtuple 字段名）— 见上，半 fail-loud 半静默，留观。
- **vllm#63480/#64090** 维持既有判定；**tt-metal quasar 系列**（#62605/#62602/#62613/#62465/#62103/#61938）实验性 API 域（quasar 标签、parent #58407 批量拆单），本轮不逐条占号，若 quasar 转正再评估；**sg#46314**（diffusion 量化层 11 连报）multimodal_gen 子系统非 LLM 推理主路径，其中静默项（#2 smooth-quant scale 丢载）证据完整，留观下轮按需收录。

## 二、NPU 来源扫描（batch-20261027T230043148191-vocxzpm3）

批次摘要：22/22 仓 complete、0 failures、18615 inference 候选（关键词命中 10621，其中窗口内 created/updated 3357）。GitCode 系 issues+pulls 分页全取、GitHub 系 3 仓全页校验。批次缓存含逐条 body（2984 份），本轮判定未额外消耗平台 API。

### 筛选过程

标题强信号+关键词双层筛选后 634 条高分候选；对照近 14 天日报完整 URL 与 `repo#号` 双键查重（63 条既往已判维持）后余 474 条；按仓库优先级深读正文 22 条（vllm-ascend#19948/#19929/#19636/#18509、ops-transformer#6812/#6814/#6655/#6656/#6675/#6626/#6785、ops-nn#7357/#7009/#7397/#7333、ops-math#3849/#3850/#3846/#4123/#4140、op-plugin#651/#652、pytorch#5371/#5238/#5004/#2409、mssanitizer#239、triton-ascend#2797）。

### 新增条目（13 条 NPU 域 + 上节 2 条 vllm-ascend 已列）

1. **INF-2207** [NEW] [vllm-ascend#19948](https://github.com/vllm-project/vllm-ascend/issues/19948) — 见上节 6。
2. **INF-2209** [NEW] [vllm-ascend#19929](https://github.com/vllm-project/vllm-ascend/issues/19929) — 见上节 7。
3. **INF-2213** [NEW] [ops-transformer#6812](https://gitcode.com/cann/ops-transformer/issues/6812) — IndexerQuantCache 负 slotMapping（-2）静默踩写 slot 63 行内偏移 120-123（规律 cache末尾−|slot|×4B、3/3 一致）；var → INF-2162（GatherPaKvCache 索引缺校验族，共享 host/tiling 索引值域校验缺失机制，触发算子与平台不同；关系回显已核对标题）。
4. **INF-2214** [NEW] [ops-transformer#6814](https://gitcode.com/cann/ops-transformer/issues/6814) — moe_token_unpermute/permute_grad/moe_index_copy sortedIndices 无下界：-1 是上游 moe_init_routing_v2/v3 合法无效标记，负偏移 GM 越界读静默错数据或 507035；三文件行号锚点+built-in/源码双基线。
5. **INF-2215** [NEW] [ops-nn#7357](https://gitcode.com/cann/ops-nn/issues/7357)（closed）— WQBMV2 arch35 tiling 三处 L1 预算缺陷（预载越界竞态/无符号下溢/低估占用）致同输入多次执行结果不一致（rel_l2 0.2-1.4）；修复 PR !12304 后逐 bit 确定；cross → TRN-0457（训练库同 issue 已收录，回显标题『WeightQuantBatchMatmul tiling L1 缺陷非确定错值』已核对）。
6. **INF-2216** [NEW] [pytorch#5371](https://gitcode.com/Ascend/pytorch/issues/5371) — rsplit partial kernel 工作区 store 无 rsplit_id 上界，真实并发 1548800 超核内 48 时越界改写相邻掩码张量（越界量逐字可复算 99120128、放大工作区恢复正确）；squeezenet 反向 64/64 全错。
7. **INF-2217** [NEW] [ops-math#3849](https://gitcode.com/cann/ops-math/issues/3849) + [#3850](https://gitcode.com/cann/ops-math/issues/3850)（原地入口）— ClampMax 标量越界按补码截断（int8 配 1000 → -24×5、配 -200 原样返回语义颠倒）；cross → TRN-0394（训练库同 issue 同机制，回显标题已核对）。
8. **INF-2218** [NEW] [ops-nn#7009](https://gitcode.com/cann/ops-nn/issues/7009)（closed）— InplaceSub GE 图转换丢负索引/环绕索引取模语义 + 跨架构编译目标误跳过；修复 PR !12445 双平台 CI 通过。
9. **INF-2219** [NEW] [op-plugin#651](https://gitcode.com/Ascend/op-plugin/issues/651) — 越界索引族无 IndexError：写侧 index_put_ 静默丢数据（哨兵 untouched 20/20）、读侧 embedding/gather/take 静默污染、2.7.1 index_select 设备死；语法糖对照三芯干净抛——API 路径选择性缺检查。
10. **INF-2220** [NEW] [op-plugin#652](https://gitcode.com/Ascend/op-plugin/issues/652) — max_pool indices dtype 三芯分裂（int64/int32/int8）：2.7.1 int8 超 127 环绕为负（64×64 输入 6.2% 索引损坏、双芯逐字一致）。
11. **INF-2221** [NEW] [ops-transformer#6785](https://gitcode.com/cann/ops-transformer/issues/6785) — BlockAttnResPrepare eps 为 fp32 次正规数（1.1e-38~1e-45，资料"有限正数"合法）三输出全 NaN 两段返回 0；8 档分界实测。
12. **INF-2222** [NEW] [mssanitizer#239](https://gitcode.com/Ascend/mssanitizer/issues/239) — A5 SIMT 混合场景越界写 DCache 工具漏检（检测范围只到 248KB 物理上限）；观测/评测域监控盲区记录，不转述为计算损坏。
13. **INF-2235** [NEW] [ops-transformer#6626](https://gitcode.com/cann/ops-transformer/issues/6626)（closed）— GenericBlockSparseAttention 官方 example 零改动 memcheck 稳定 2 笔 GM 越界写（随 topK 浮动）；initcheck 66 笔按工具自述未插桩排除。
14. **INF-2218 补充来源**：修复链已入条目。另 [pytorch#5238](https://gitcode.com/Ascend/pytorch/issues/5238)=INF-1518、[#5004](https://gitcode.com/Ascend/pytorch/issues/5004)=INF-1520、[#2409](https://gitcode.com/Ascend/pytorch/issues/2409)=INF-1526（近端复核确认在库，不重收）。

### 拒绝 / 留观（NPU 主要项）

- **校验缺失/契约大族增量**（ops-nn#7397/#7333/#7410/#7417/#7418、ops-math#4123/#4140/#4111/#4104/#3846/#3954/#3955、op-plugin#649/#654 等）— 与 10-22~10-26 已判定「非法入参静默接受」「错误码契约」「文档声明」大族同型，按族留观维持。亮点单列：#7397（epsilon=NaN 全 NaN 输出）与 #7333（rstdOut 越界写 28B）证据完整，若出现端到端影响再收录；#4123（整型÷浮点标量降精度 8.9e7 偏差）数值面真实但属 FloorDivides 类型推导域。
- **op-plugin#649**（stft compile 失败+精度备忘）— issue 正文为 2.7.1 旧版本线 compile fail-loud 域（BackendCompilerFailed 响亮报错），eager 数值 72/72 PASS；精度面（gelu/stft 链路 maxabs 2.67e-05）在预算内；留观。
- **vllm-ascend#19636** 维持 10-23 判定（挂起域、上游 shm_broadcast 源码闭环已定位）；**#18509**（kv-pool lookup 假命中）closed 但机制叙述以现网 workaround 为主、root cause 未上游修复，留观维持；**#18743**（No-Triton GDN fallback 不可达）自述"无服务失败"维护性报告不占号；**#18710/#17391/#18197** 维持 10-14 判定（机制未定位/量化混合归因）。
- **triton-ascend#2797**（启动器 workspace 32 位乘法溢出）— fail-loud 报错域（申请失败直接报错），且主要影响训练侧 DSA kernel；留观转训练扫描判定。
- **msprobe#643**（dump 范围配置 TypeError）、**mssanitizer 其余**、**hccl#849**（uninstall 脚本日志）— 工具/构建域维持。

### 回访（状态变化）

- **INF-2162**：GatherPaKvCache 族 #6655/#6656/#6675 三 issue 均 **closed**（u10-27）；关闭原因与修复发布未核实。update 已记录。
- **INF-2116**：issue closed 维持、修复 PR #19401 仍 open（u10-27）；merge/发布未核实。update 已记录。
- 近 7 日条目来源快照：INF-2187~2206 来源全部仍 open（ops-math#3974 closed 已在 10-26 记录、其修复 MR !6660 merged 维持）；msprobe 修复 MR !1125 仍 open。
- 轮巡（≥3 个更早 open 条目）：INF-1470（op-plugin#602 仍 open u10-19 无更新）、INF-1469（#605 同）、INF-1471（#598 同）；INF-2179（sglang#46259 不在本批 scope，GitHub 侧无新判定信号）。

## 三、API 账本与未覆盖范围

| 组 | 请求 | 结果 |
|---|---|---|
| GitHub search ×12 | 7 仓 correctness/silent/KV/quant/NaN/mismatch/numerics 等，全部 `created:>=2026-10-20` | 全部 200 有效；1 次 rem=0 分段等待 45s 后回补 |
| arXiv ×4 | SDC/bit-flip/KV corruption + inference | 全部 200 有效（http 首轮失败转 https 成功） |
| NPU | 前置脚本共享批次 | 22/22 仓 complete，0 failures，未额外调用平台 API |

- 仅按索引/标题判定的候选：NPU 校验缺失族约 40 条（与已判大族同型）、ops-transformer/ops-nn PR 修复流正文未逐条读、vllm-ascend PR 流（BugFix PR 正文未读）——标题与正文严重不符的候选可能漏判。
- GitHub 评论级深查未做（无 GITHUB_TOKEN）：本轮新收录 29 条正文自含机制证据（复现脚本/源码锚点/对照实验），评论级补充留待每日审计。
- arXiv 未覆盖：仅 4 组常规查询，long-tail 关键词（如 KV eviction corruption、speculative decoding correctness）未展开。
- 累计等待限流 45 秒（上限 300 秒内）。

## 四、编目与视图

- `catalog_tool.py new`×29 全部 ok（两轮长度修正后通过）；update×3（INF-2217 cross→TRN-0394、INF-2162 回访 closed、INF-2116 回访维持+补充、INF-2213 var 关系备注）；`build` 重新生成 views/；`validate` ok（n_errors=0，warnings=51 为既有模式待定项）；stats：2235 条（active 2206 + pending_audit 29）。

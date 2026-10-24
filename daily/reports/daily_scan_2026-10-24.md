# 推理故障每日扫描 2026-10-24（UTC）

- 窗口：GitHub/arXiv `created:>=2026-10-17`（7 日）；NPU 共享批次 `batch-20261024T230043100293-evtxjwh2`（采集 since 2026-10-23T22:00:43Z / until 2026-10-24T23:00:43Z，22 仓全覆盖、0 failures，candidates 16454）。
- 编目：本轮新增 **10 条**（INF-2159 ~ INF-2168，pending_audit），总数 **2168**；validate `"ok": true`。
- Git：前置检查通过（main 干净、HEAD=origin/main=edf1b97）；本轮末提交并推送。

## 一、原 GitHub 7 仓 + arXiv 覆盖账本

- GitHub 未认证 Search API（约 8s 间隔，GITHUB_TOKEN 未配置，未认证限额 core 60/h、search 10/min），12 组查询全部 `created:>=2026-10-17`：vllm×4（8/4/6/5/9）、sglang×2（7/16）、transformers×1（4）、TensorRT-LLM×1（0）、triton×1（15）、tt-metal×2（50/32）、flashinfer×1（2）。core API 中途 403（60/h 用尽），等待 2 分钟后经 search API 补齐剩余正文（fi#7179、triton#12425、vllm#63250、tt#61903/#61700、fi#6977、sg#45697、tt#62015）。
- 核对：tt-metal 50+32 条全取（单页 ≤100）；本轮 tt-metal 查询返回的多为 10-21/10-22 已判定项（LLK/SFPU 边界族已按族留观）。
- arXiv：3 组查询（SDC、bit-flip、KV corruption）有效返回；近 7 日无新推理故障机制论文（最新相关 10-22 Rowhammer latency attack，10-20 LEAN/Radiation-Hardened 为容错设计类非故障编目对象）。

### 新增条目（4 条，pending_audit）

1. **INF-2168** [NEW] [vllm#63480](https://github.com/vllm-project/vllm/issues/63480)（c10-24，open）— Responses API（DeepSeek-V4 Python tokenizer 路径）在 reasoning 开启且无 tools 时把首条 developer 消息从 prompt 整体删除，HTTP 200；usage.input_tokens 不随策略文本增长定量证实。系统提示静默丢失。
2. **INF-2165** [NEW] [vllm#63514](https://github.com/vllm-project/vllm/issues/63514)（c10-24，open）— `--kv-cache-dtype int4_per_token_head` 量化前 hadacore_transform 奇数行错值（N=1..4352 全扫 1101 个奇数 N 错、相对误差 ~1e36 垃圾 K/V 入 cache）或间歇 illegal memory access；L4 sm_89/vLLM 0.30.0。
3. **INF-2166** [NEW] [flashinfer#7179](https://github.com/flashinfer-ai/flashinfer/issues/7179)（c10-24，open）— trtllm-gen fused MoE `fused_moe_120` 在 SM120 静默输出垃圾（同权重单变量 A/B：DEEPGEMM 全对、flashinfer_cutlass 首 token 起错）；运行时无架构门控，测试矩阵自声明仅保证 SM100/SM103。
4. 以下 tt-metal 新命中（#61903/#61700/#62364/#62383/#61497 等）均为 10-20~10-22 已判定项：#61700/#61497 已并入 INF-1784；#61903 属 #61901 NaN 语义边界族（10-22 留观维持）；#62364/#62383/#62385 为 Quasar/LLK 域留观族新成员，正文未读、维持族留观。

### 重复判定（沿用近期判定，不新建）

- vllm#62225=INF-1409 域（10-22 已判）；#62122=INF-1409；#63043=INF-2133；#62664=INF-2142；#62811=INF-2114；#62651=INF-1780；#62945=INF-2117；#63309=INF-2147；triton#12397=INF-1766、#12348=INF-1462、#12392/#12394=INF-2148/2149、#12347/#12349/#12351/#12365/#12366/#12369/#12370 维持 10-21/10-22 判定；flashinfer#7021=INF-2132；sglang#45755=INF-2141、#45622/#45563/#45481/#45697/#45000/#44999 维持 10-18~10-22 判定；transformers#49888=INF-2158、#49810=INF-2157、#49897/#49902 维持 10-23 判定；tt-metal#62052=INF-2156、#61905=INF-2155、#61942=INF-2112、#61540=INF-2115、#62024=INF-2129、#62016=INF-2122、#62124=INF-2119、#62015/fi#6977 维持 10-22 留观（本轮补读正文：fi#6977 fail-loud ValueError 不占号、sg#45697 流式缓冲输出解析域留观维持、tt#62015 挂起域留观维持）。
- triton#12425（interpreter debug 开关失效）— 测试基建域（TRITON_INTERPRET 调试模式），不影响生产推理数值，不占号。
- vllm#63250（DSV4.1-Flash U+FFFD）— 机制未定位（正文仅环境段），沿用 10-23 同型留观。

## 二、NPU 来源扫描（batch-20261024T230043100293-evtxjwh2）

批次摘要：22/22 仓 complete、0 failures、16454 inference 候选。采集窗口 2026-10-23T22:00Z→2026-10-24T23:00Z。GitCode 系 issues+pulls 分页全取（ops-transformer 110+93 两页 issues、ops-nn 32+208 三页 pulls 等），GitHub 系 3 仓全页校验。批次缓存含逐条 body，本轮判定未额外消耗平台 API。

### 筛选过程

标题强信号命中 833 条；对照近 14 天日报完整 URL 查重后 681 条未在旧日报出现（46 条为窗口内新建、635 条为老条目新更新）；逐仓分桶后优先深读静默/延迟可见正确性候选，本轮深读正文约 35 条。

### 新增条目（6 条，pending_audit）

1. **INF-2159** [NEW] [Ascend/pytorch#5632](https://gitcode.com/Ascend/pytorch/issues/5632)（c10-24）— Inductor（torch_npu master/2.13、950）两处：channels-last pixel_shuffle 拆轴取模被消去（54758/73728 超差）；动态 persistent 归约块按首次尺寸 hint 选择，长度 31→33 漏算。torch.compile 静默错值。
2. **INF-2160** [NEW] [ops-math#3894](https://gitcode.com/cann/ops-math/issues/3894)（c10-24）— aclnnAddcmul/InplaceAddcmul 的 IsEqualToOne 用 `!(v>1||v<1)` 判等，NaN 被判为 1 走 AxpyV2 快路径：value=NaN 输出有限值而非全 NaN；源码行号定位（aclnn_addcmul.cpp:203-209/285）。
3. **INF-2161** [NEW] [ops-transformer#6639](https://gitcode.com/cann/ops-transformer/issues/6639)（c10-24）— aclnnMsaIndexScore key N2>1 时所有 query head 只读第 0 个 KV head：只改 head1 输出 bitwise 不变（diff=0）、改 head0 diff=12672；GQA 多 KV head 注意力静默错值。
4. **INF-2167** [NEW] [ops-transformer#6625](https://gitcode.com/cann/ops-transformer/issues/6625)（c10-24）— aclnnSparseFlashMlaMetadata 在 A2 整条丢弃 sequsedQOptional：sequsedQ=0（应全空）仍产出满长 metadata（指纹逐字节同不传）；对照 sequsedOriKv 生效，证明单参数未接线；下游满长多算。
5. **INF-2162** [NEW] [ops-transformer#6675](https://gitcode.com/cann/ops-transformer/issues/6675) + [#6655](https://gitcode.com/cann/ops-transformer/issues/6655) + [#6656](https://gitcode.com/cann/ops-transformer/issues/6656) + [#6676](https://gitcode.com/cann/ops-transformer/issues/6676) — GatherPaKvCache 同算子三型校验缺失族：blockId 越界读 keyCache 静默返 0 值、sum(seqLens)>keyRef dim0 越界写缓冲外（保护带定量）、seqOffset 折算 tableOffset 越界读 blockTables 外 100KB；修复 !14535（closed）!14536（open）。与 INF-1373（32B 取整丢尾，#6133 已 closed）同算子不同机制。
6. **INF-2163** [NEW] [ops-transformer#6658](https://gitcode.com/cann/ops-transformer/issues/6658)（c10-24）— aclnnMhcPre alpha=(2) 且 hRes 传非空张量时一个字节不写（canary -777 全残留），下游消费未初始化显存；alpha=(3) 对照立即写满，读回路径有效。
7. **INF-2164** [NEW] [op-plugin#618](https://gitcode.com/Ascend/op-plugin/issues/618) + [#619](https://gitcode.com/Ascend/op-plugin/issues/619) + [#637](https://gitcode.com/Ascend/op-plugin/issues/637) — torch_npu 语义不一致三连（10-21 留观"下轮优先"，本轮按承诺收录）：div floor/trunc ±inf 除法错值+丢 -0.0 符号；空张量 max() 静默返 tensor(0.)（CPU 报错）；cummin 不传播 NaN（950PR，cummax 正确）。三份最小复现齐全。

### 拒绝 / 留观（NPU 主要项）

- **校验缺失族增量**（ops-transformer #6690/#6691/#6692/#6695/#6696/#6670/#6672、ops-nn#7291、ops-math#3870/#3882/#3846 等）— 与 10-23 已判定的「非法入参静默接受」大族同型（N=1/负值/非单调/空指针/NaN 传入均 rc=0）；本轮已收录该族中任务影响最直接的 GatherPaKvCache 族与 Addcmul NaN；其余族成员按 10-23「族留观防编号膨胀」原则维持，亮点单列：#6695（A8W8 bias 静默忽略，输出逐字节同不传）与 #6696（groupList 非法值延迟到 sync 才 507015）证据完整，若后续出现端到端影响再收录。
- **ops-transformer#6666**（NsaCompressGrad compressBlockSize×headNum≥256 时 weightGradOut 静默算错，inputGradOut 正确，49 组扫描无一例外、错误值为别的 (row,head) 归约值）— 候选强；但 NsaCompressGrad 为 NSA 反传算子，推理路径不直接消费，训练库视角为主，本轮转训练扫描线索（训练库未收录时可由训练任务评估），本库留观。
- **ops-transformer#6671**（NsaCompressGrad 合法配置 memcheck 4428 条 GM 越界写、数值正常）— sanitizer 观测域：数值输出正确、越界写属潜在内存污染，与 #6666 同算子同域处理，留观。
- **ops-math#3871/#6661**（Tan |x|≥1e7 显式 guard 置 NaN，修复 PR 删除 guard）— 大参数三角函数边界，报告源码级定位完整（LARGE_THRESHOLD 残留），但 torch.compile 推理主路径极少触达 1e7 级 tan 输入；与 10-21 #6841 gelu 族相邻的「语义漂移」域，留观下轮并条评估。
- **ops-math!6660**（concat >512 拆树树形 desc 推导缺失真机静默计算错误）— 修复 PR 正文为空模板，机制细节不可判，留观待 issue 关联或描述补充。
- **cann/ge!5577**（allow_hf32 覆盖 op_precision_mode）— 精度配置优先级缺陷候选强，但正文无用户影响路径（配置失效≠静默错值证据），且属 GE 精度模式基础设施；留观，若出现下游错值报告再评。
- **op-plugin#637 域外**：HBM 坏块#386、多版本缓存冲突#383、paged_attention 变长 causal#372、splitfuse_v2#276 等老 open 条目（2026-08）本轮仅在批次索引出现，正文未重读，维持原状态不判。
- **vllm-ascend**：#19713（DSV3.2 MRV2 精度对齐测试报告——测试跟踪域不占号）；#19624 已 closed（curl 重复循环，关闭原因未见注释，机制未定位维持留观）；#19080（enable_sparse_li_c8 零层选中无告警——优化配置静默不生效域，正文已读，性能/资源类留观）。
- **msprobe#653**（load/dump 解耦报错）、mssanitizer#225（越界误报）— 观测工具域，监测效果未经实测不转述为计算故障，不占号。
- **hccl#795/#3196 等通信精度修复 PR**（PCIE AllReduce 64bit、对称内存 allreduce/reducescatter）— 2026-09 已合入的老修复，无 issue 级用户现象描述，不溯源立目。
- **mindformers#8882/#8705/#8706**（Muon NaN 防护、均衡加载续训精度）— 训练域，转训练扫描视角。

### 回访（状态变化）

- **INF-2138**（ops-nn#7225 swiglu_mode 静默 mode0）：修复 MR **!12598 已 merged**（10-24T01:13Z），issue closed；版本发布未核实。update 已记录。
- **INF-2107**（ops-math#3709 tril 进程级 tiling 残留）：issue 已 closed（10-24 updated），关闭原因未见注释。update 已记录。
- **INF-2099**（ops-math#3747 Tril 掩码半写）：issue 已 closed（10-22）。update 已记录。
- **INF-1373**（GatherPaKvCache 32B 取整丢尾）：issue #6133 已 closed（10-24）；同算子新族条目 INF-2162 已建，update 已记录关联。
- **INF-2116**（vllm-ascend#19357 hc_pre 精度）：issue 与修复 PR #19401 均仍 open（PR 10-24 有更新，内容未读）。update 已记录。
- 其余 INF-2098/2100/2101/2102/2103/2104/2105/2106/2110/2111/2120/2126/2127/2128/2136/2139/2143/2144/2146/2151/2152/2153 在批次中状态无变化（open）。

## 三、API 账本与未覆盖范围

| 组 | 请求 | 结果 |
|---|---|---|
| GitHub search ×12 | vllm/sglang/transformers/trtllm/triton/tt-metal/flashinfer，全部 `created:>=2026-10-17` | 全部 200 有效；core 60/h 用尽后 2 次 403，等待 2 分钟经 search 补齐 |
| arXiv ×3 | SDC/bit-flip/KV corruption | 有效返回，近 7 日无新机制 |
| NPU | 前置脚本共享批次 | 22/22 仓 complete，0 failures，未额外调用平台 API |

- 仅按索引/正文前段判定的候选：ops-transformer 校验缺失族约 15 条（见留观节）、tt-metal Quasar/LLK 域 #62364/#62383/#62385、mindformers 训练域 3 条、hccl/hcomm/runtime 白盒扫描修复 PR 族约 20 条——标题与正文严重不符的候选可能漏判。
- GitHub 评论级深查未做（无 GITHUB_TOKEN）：本轮新收录 10 条正文自含机制证据（复现/源码定位/对照实验），评论级补充留待每日审计。
- 累计等待限流 2 分钟（未超 5 分钟上限）；core API 未再等待小时级重置，剩余正文经 search API 取回。

## 四、编目与视图

- `catalog_tool.py new`×10 全部 ok（2 轮字段长度修正）；update×5（INF-2099/2107/2138/1373/2116 回访）；`build` 重新生成 views/；`validate` ok（n_errors=0，warnings 为既有模式待定项）；stats：2168 条（新增 pending_audit 10）。

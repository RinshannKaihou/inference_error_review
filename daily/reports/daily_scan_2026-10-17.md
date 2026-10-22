# 每日扫描日报 2026-10-17（推理库）

- **扫描窗口**：GitHub `created:>=2026-10-10`（cutoff 实际计算 `2026-10-10`）；重点判定 10-16 报告之后的新增与上轮留观队列。NPU 共享批次 `batch-20261017T230043152926-gwo3drx1`（22/22 仓 complete，0 failures，采集窗口 2026-10-16T22:00:43Z → 2026-10-17T23:00:43Z updated_at 增量，candidates=10482，inference 视图含全部 19 个推理适用源）。
- **目标仓库**：原 GitHub 7 仓 + arXiv + NPU 19 推理源（GitCode/AtomGit 16 + GitHub 3）。
- `GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥8s 间隔、core REST 60/h）。GitHub Search 14 组全部有效（vllm 23+13、sglang 17+8、transformers 2+0、TRT-LLM 3+2、triton 0+0、tt-metal 12+23、flashinfer 22+7，全部 incomplete=false，无 >100 截断）。core REST 2 次单号详情（sglang#44504、vllm#61803），未触限流等待。**首轮 14 组 Search 因保存脚本截断 JSON 全部作废重取**（重取后逐组校验 total/items/incomplete）。arXiv 首轮 429/503 限流，前台分段 sleep（45s+55s+20s+15s…累计约 3 分钟）后 6 组查询全部 200 有效。
- Git 前置检查：main 干净、fetch 成功、HEAD == origin/main（f6bf96b2）。

## 一、原 7 仓新增条目（12 条，pending_audit）

1. **INF-1408** [NEW] [vllm#62225](https://github.com/vllm-project/vllm/issues/62225) — CPU 后端 `--enable-lora` 下 base+LoRA 混批：base 请求被施加 adapter（随机 adapter 35/40 输出与 adapter 逐字节相同、0/40 匹配正确输出）；纯 base/纯 LoRA 批正确。documented。
2. **INF-1409** [NEW] [vllm#62122](https://github.com/vllm-project/vllm/issues/62122) — `--kv-sharing-fast-prefill` 拦截只查 truthy：`prompt_logprobs=0`（及 echo=true 隐式路径）绕过；V2 runner HTTP 200 返回偏差数十 nat 的静默错 prompt logprob，V1 runner EngineCore 崩溃全体失服。documented。
3. **INF-1433** [NEW] [vllm#61453](https://github.com/vllm-project/vllm/issues/61453) — DP>1 无 EP 的 MoE expert-LoRA：token 与 LoRA-slot 映射一起 all-gather 但 slot 表 rank 本地，各副本把本机 LoRA delta 施加到全部 token；副本态不同步时 logits 静默错（对端空转半份 delta、异 adapter 混 delta）。源码行号级+4×H100 实测。documented。
4. **INF-1424** [NEW] [vllm KV Connector 三连](https://github.com/vllm-project/vllm/issues/61768)（[#61768](https://github.com/vllm-project/vllm/issues/61768) abort 留坏块 / [#61905](https://github.com/vllm-project/vllm/issues/61905) 块 ID 复用误杀 / [#61912](https://github.com/vllm-project/vllm/issues/61912) preempt 变体）— `_handle_invalid_blocks` 扫描集缺 preempted/aborted owner：加载失败块留前缀缓存被复用 decode 出坏 KV，或迟到上报命中复用块误杀无辜请求；三种触发均有 CPU 探针复现。上轮留观 → 本轮深读正文后收录。documented。
5. **INF-1414** [NEW] [flashinfer#6479](https://github.com/flashinfer-ai/flashinfer/issues/6479) — `trtllm_ragged_attention` launcher 从不设置 qStrideTokens/qStrideHeads，非连续 Q 按 packed 读取静默错；GLM-4.7-Flash gsm8k 83.6%→0.8%；vLLM 已以 `q.contiguous()` 规避（vllm#61634），FlashInfer 侧未修。documented。
6. **INF-1410** [NEW] [tt-metal#61158](https://github.com/tenstorrent/tt-metal/issues/61158) — 多项式 bf16 tanh（softcap/situ_glu/logit_softcap 同路径）把 NaN 洗成 ±1.0、±Inf→±1：254 种 bf16 NaN 位型全测。documented。
7. **INF-1423** [NEW] [tt-metal#61212](https://github.com/tenstorrent/tt-metal/issues/61212) + [#61206](https://github.com/tenstorrent/tt-metal/issues/61206)（同机制合并）— SFPU replay 体溢出覆写 FPU MOP 槽：accurate bf16 exp 体 20-22 条跨槽 16、dequant 体 15..19 跨界，matmul/reduce 重放被污染程序静默错（tile0 对、tile1+ 错；reduce 行和 -26 vs 40）。documented。
8. **INF-1411** [NEW] [tt-metal#61153](https://github.com/tenstorrent/tt-metal/issues/61153) — 分片行主序 transpose 窄行 pack_untilize 四类形状（行宽非 16B 对齐/17-31 datum/H>32/W%32）静默错（BH 4 datum 行步进 0、32 行叠同 16 字节）。documented。
9. **INF-1412** [NEW] [tt-metal#61110](https://github.com/tenstorrent/tt-metal/issues/61110) — Blackhole fast_untilize 谓词只认 Bfp8_b/Bfp4_b，Bfp2_b 及 A-exponent BFP 走稠密路径不能跳 64B 指数段：tile1 起静默错（36/36 变体失败）。documented。
10. **INF-1415** [NEW] [tt-metal#60365](https://github.com/tenstorrent/tt-metal/issues/60365) — Gemma4 族未声明 max_device_top_k，vLLM TT 设备采样静默把 top_k>32 钳到 32（plugin#140 修复只对声明者生效）。documented。
11. **INF-1413** [NEW] [sglang#44504](https://github.com/sgl-project/sglang/issues/44504)（上轮留观→本轮收录）— `/update_weights_from_disk` 未知视觉权重名静默跳过仍报 success（`.bias` 连日志都没有）；RL 权重更新后视觉塔保留旧权重。源码行号+复现脚本。documented。
12. flashinfer#6316（b12x W4A16 E8M0 scale 次正规）→ **沿用 2026-10-11 判定已编目 INF-0781**，本轮 update note 回访补充机制细节（121/114 双阈值、MiMo-V2.6-Pro 层级 scale 分布、聚合余弦 0.99999 骗过精度门）。

### 拒绝 / 留观（原 7 仓）

- **vLLM**：#61157（GLM hybrid-SSM+MTP 并发请求触发 agent 会话退化——现象真实且任务影响严重，但 #58454/#58368 后残留缺口未定位、无法合成复现，法证报告型，留观跟踪上游）；#62144（NixlConnector TTL 逐出后 60s 阻塞——可用性/挂起域）；#61970（native KV offload crash——fail-loud）；#61608（tower connector LoRA punica 尺寸——startup crash）；#61318（return_sampling_mask 两个前置条件逐个报错——报错语义域非静默）；#61229（Nixl 租约到期块释放 vs 远端 READ——候选与 #61225 相邻，本轮未深读留观）；#61150 已判（10-12）；#61850/#61806 等沿用 10-16 判定。transformers#49497（ZeRO-3 每 rank 全量读 checkpoint——训练资源域，非本库）。
- **SGLang**：#44789（MiMoV2Processor 硬编码 CLIP mean/std——输入预处理错值候选，机制清晰但未深读影响面，留观）；#44070（PD prefill 早发 cached-prefix 在 HiCache 回读完成前传 KV——与 10-13 判定同一"恢复未完成即消费"主线，stale KV 域，本轮正文确认存在但未逐条深读，留观下轮优先）；#43882（bootstrap_room Rust/Python i64 环回差异——latent，报告者自判当前无害）；#43888（usage 计数 u32 回绕——计费域）；#44895（input_embeds 崩溃——fail-loud）。
- **TensorRT-LLM**：#20375 已收录（10-16 INF-1404）；#20138（多模态时间戳 token 丢失——视频输入正确性候选，本轮未深读，留观）；#20083（MPI 1-rank communicator——启动域）。
- **tt-metal**（#58402/#58405/#58407 静态审计族持续产出，本日收录 5 条后其余留观）：#61214（FP32 dest-acc toggle 与 packer 重配置互相打架——实测有 gtest，机制相邻 61212/61206 同 replay/init 域，留观族化评估）；#61196（block_ct_dim 环绕 MOP 循环计数字段——k_chunk_size=4096 时静默错注意力输出，候选强，本轮合并量已大留观下轮优先）；#61056（custom_mm kt_dim=256 面计数 8bit 截断读错面）；#61087（rmsnorm bcast-scalar 两个 init 缺口——当前 latent）；#61084（blocked bcast-col MUL 恒 LoFi）；#61144（sdpa transpose_v 标志被丢弃——无树内调用者）；#61167（docstring 参数序错——文档域但 addcmul 按文档写会静默错，留观）；#61189（fast_tilize 16x32 BH 无路径——静默错+CB 越界，下轮优先）；#60745（Quasar pack DFB 目标错——softmax PCC 候选）；#60849（Quasar tilize init 改行距破坏在飞——下轮优先）。
- **flashinfer**：#6419（FP8 ragged prefill trtllm-native vs prims-ts refcheck 失配——报告者自述未定位哪边错，留观）；#6391（fast_decode_plan 采纳调用者元数据——host 指针非法访问，fail-loud 域）；#6636（cake_fmha LSE 未写——deslop 重构跟踪内含 bug 修复项，随该 issue 跟踪）。
- **triton**（主仓）：0+0 无新 issue。

## 二、NPU 来源扫描（batch-20261017T230043152926）

- 22/22 仓 complete、0 failures、registry_hash 2f6422d0…（22 源全选）；本轮增量判定 = updated_at > 2026-10-16T23:00:43Z 的 1070 项（created>10-14 者其中 573），关键词分桶后 135 项标题命中，逐条标题判定 + 26 项正文深读。上一报告（10-16，覆盖至 10-16T23:00:43Z）之前创建且无新更新的条目不重复判定。

### 新增条目（14 条，pending_audit）

1. **INF-1425** [NEW] [hccl#827](https://gitcode.com/cann/hccl/issues/827)（10-16 留观→范围变更收录）— 单 rank AllGatherV/ReduceScatterV/AlltoAllV 忽略 displacement 错位读写（应写下标 3、4 实写 0、1，无报错）；源码级定位 op_common.cc:3537。**cross TRN-0299**（训练库同机制，标题已核对）。
2. **INF-1431** [NEW] [Ascend/pytorch#4941](https://gitcode.com/Ascend/pytorch/issues/4941)（10-16 留观拟改判→本轮收录，范围变更）— triton_experimental 后端 blanket fallback 物化 9 个 view/mutation 原语，视图 copy_ 被 DCE 静默丢弃；issue closed 且验收口径完整（夜跑 36/0 全绿、default/eager 零变化）。
3. **INF-1426** [NEW] [op-plugin#381](https://gitcode.com/Ascend/op-plugin/issues/381) — max/min 空 tensor 返回 0.0、size=0 维度带 dim 归约返回全 0+全 0 索引（CPU/CUDA 报错）；indices 全 0 使下游 gather 全指向位置 0；PR !4922 只修无 dim 场景。
4. **INF-1416** [NEW] [pytorch PR !48132](https://gitcode.com/Ascend/pytorch/merge_requests/48132) — copy_ 写 conj()/mH 视图不取共轭：9+3j 写入 base.conj() 底层存 9+3j（应 9-3j）；基线 56/100 子用例失败、修复全过（910B3 实测，PR open）。
5. **INF-1427** [NEW] [ops-nn PR !12168](https://gitcode.com/cann/ops-nn/merge_requests/12168) — rms_norm_base.h 宏漏 `__CCE_AICORE__ != 920`：Ascend960 上 bf16 按 int16 处理，rms_norm/add_rms_norm 系精度错；源码级宏定位（无数值实测）。
6. **INF-1417** [NEW] [cann/ge PR !5135](https://gitcode.com/cann/ge/merge_requests/5135) — GE Eager/HostCpu 自定义算子动态 shape 增大时输出缓冲不重分配：kernel 越界写 16384B 缓冲（RT2 真机 4096→8192 复现），仅拷出阶段拦截；`TensorData::Free` 不重置 size_ 二次坑。
7. **INF-1428** [NEW] [ops-transformer#6028](https://gitcode.com/cann/ops-transformer/issues/6028) / [#6119](https://gitcode.com/cann/ops-transformer/issues/6119)（同源双报合并）— SwinTransformerLnQKV epsilon 属性不下发、kernel 硬编码 1e-5（接口默认 1e-7）；固定 commit 静态对照，master 未修。
8. **INF-1418** [NEW] [ops-transformer#5742](https://gitcode.com/cann/ops-transformer/issues/5742) — RingAttentionUpdate 维数校验漏 return：非 3-D 输入第一段放行、(8,2,64,1) 两段全 0 产出未定义语义结果（910_93/CANN 9.2.0 实机 A~F 矩阵）；注意 9.2.0 发布版无此检查。
9. **INF-1429** [NEW·var→INF-1398] [vllm-ascend#19065](https://github.com/vllm-project/vllm-ascend/issues/19065) — Qwen3.5-397B C8 静态 V-scale 部分异步 PD 导入后损坏：长 GPQA ~1536 token 起连续 '!'、MTP 接受率归零；几何 1536/2560 页、块比 3/5（与 35B 不同）；K-scale 修复已装仍复现。var 关联已核对目标标题。
10. **INF-1419** [NEW·var→INF-1394] [vllm-ascend PR #19069](https://github.com/vllm-project/vllm-ascend/pull/19069) — 非 layerwise AscendStore 块跨步 MLA 缓存传输：区间重叠+漏每逻辑块尾 5 物理槽（640 token），17280-token 池恢复后首 MLA 层历史 KV 不同输出不同（Kimi-K3 A3 比 45 实测）；修复与旧池对象不兼容需新 cache_salt。
11. **INF-1430** [NEW] [vllm-ascend PR #19073](https://github.com/vllm-project/vllm-ascend/pull/19073) — GLM KPool 空 work 初始化越界写 TND 输出分配：无关缓存权重 NaN、输出重复 '!' 或非有限 logprobs（DSA-CP+MTP+graph，A3）；修复=哨兵请求+钳空 history，11 用例通过。
12. **INF-1420** [NEW] [ops-nn#6796](https://gitcode.com/cann/ops-nn/issues/6796) — InplaceApplyAdaMax FP32 先除后乘中间量溢出：lr=0 冻结更新返回 NaN、|lr_adj|<1 假溢出 -inf、±inf var 相减 NaN（kernel_l1 5 例 FAIL 实测+修复验证）；推理侧可达面=在线优化器路径。
13. **INF-1421** [NEW] [ops-nn PR !12137](https://gitcode.com/cann/ops-nn/merge_requests/12137) — GatherElementsV3 展平乘积 uint32 回绕：合法 shape [1,2,65536,65536] xPost=2^32→0 读错位置（整数探针复现，NPU 实测未跑）。
14. **INF-1432** [NEW] [triton-ascend#2457](https://github.com/triton-lang/triton-ascend/issues/2457) — masked vector store 漏写合法输出槽位（保留哨兵 -777）：vLLM DFlash 映射写路径、eager 即错、标量循环 workaround 通过；另 [triton-ascend#1202](https://github.com/triton-lang/triton-ascend/issues/1202)（floor(cos) BLOCK=2 小值 fp32 返回 1.0）**并入本条备注**：同仓同日双数值缺陷，机制不同（store lowering vs 向量数学），#1202 细节记录于 INF-1432 备注。

### NPU 留观与拒收（主要项）

- **留观（下轮优先）**：ops-math#3596（SquareSumAll FP32 末级合并三方超差 RMSE 3.77——上轮已判留观维持，修复 PR 6386 已有、属算子精度容差治理域）；ops-math#3604（RadixTopK WS 三缺陷——上轮判定维持，sortLen>2^31 触发面窄+PR #6396 在途）；vllm-ascend#18891（DSV4 KV dtype 声明不被读——auto 解析成不可写 dtype 深入算子才死，fail-loud 但声明 inert 域，随 PR 观察）；ge#5135 已收录但修复未合；ops-nn#6816（InplaceApplyAdaMax 指针防御缺失+废弃码——异常路径防御域非计算错）；ops-transformer#6005（swin ffn 域 17 处内存安全汇总——崩溃级 1 处属 tiling OPTIONAL 解引用，fail-loud 域为主，含 #6028/#6119 epsilon 已单独收录）；ops-transformer#6066（RGDR TTK 比较器 0.5% 不匹配点吞 NaN/Inf——**观测工具漏判域候选强**，修复仅 Host 侧 CPU 回归，随下轮评估是否入观测类）；ops-nn#6809（AddRmsNormQuantV2 RegBase 历史量化 dtype 兼容缺失 161002——fail-loud 报错域）；mindformers PR 8705/8706（均衡加载续训优化器状态——训练域不收）。
- **拒收/不占号**：msmodelslim#624/PR#1100/PR#1058（DeepSeek-V4.1-Flash W4A8/W8A8 量化支持——feature）；ops-transformer PR 14034（LightningIndexerV2 return_value 按 bool 读整数属性——修的是 InferShape 读取类型，契约域）；ops-nn PR 12180/12179/12084（conv3d_backprop FP32 短行累加精度——训练反传域，三方精度治理相邻 #3596）；hccl PR 3642（SK 满核空闲核同步失配 MTE invalid——fail-loud 报错域修复）；ops-math PR 6429（space_to_batch 整除截断——一句话 PR 无实测，留观下轮找关联 issue）；vllm-ascend PR 18897/18845（Kimi A5 draft KV+FP8 MLA 修复聚合——修复型 PR，其内含的 FP8 输出错值作为 #18914 后续跟踪）；vllm-ascend PR 17048（Kimi K3 MlaPrologV3 启用——feature，其 padded Q 权重物化修复记录在案）；PR 19096/19079（packed SFA KV cache 规格选择——graph capture 失败 fail-loud 域修复）；torchair PR 3823/3816（SIGTRACE/traced_graph 精度维测工具——方法域，喂零静默错误作为精度对拍失效面记录，不占故障号）；PR 48113（inductor 精度工具——观测域）；PR 11853（csv 序列化——工具域）；Ascend/pytorch#5501（NpuFusedAdamW unscale_ storage 越界——训练优化器 crash 域）；issue 4445/PR 45697（API 一致性任务）。
- **沿用判定**：ops-transformer#6119 与 #6028 同源（10-12/13 双报，本轮合并收录）；vllm-ascend#18746/18754/18755、ops-nn#6516/#6482、ops-transformer#6114、vllm-ascend#17476（msmodelslim expert 集错位，closed 无新评论）、#18769/#18794/#18898 均按 10-16 判定/回访记录维持。

## 三、回访已收录来源（批次内状态核对 + update）

- **torchair#868（INF-1399）**：closed 10-15，PR #3844/#3845 merged → update note（dlclose 修复已合 master，发布未核实）。
- **vllm-ascend MXFP8 scale 族（INF-1398）**：#19064/#19017 closed、#19016 暂缓、#19065 揭示 397B V-scale 缺口 → add-evidence #19065 + note；397B 变体另立 INF-1429（var）。
- **vllm-ascend Mamba KVPool（INF-1394）**：backport #19060 closed、main #18901 open → add-evidence + note。
- **AscendStore 上报（INF-1393）**：修复方向演进 #19101/#19090（隔离传输失败）→ add-evidence #19101 + note。
- **维持无实质变化**（update note 记录）：op-plugin#586（INF-1400）、Ascend/pytorch#5451+PR#48029（INF-1396，均 open）、ops-math#3616（INF-1395）、ops-transformer#6221+#13938（INF-1401）、ops-nn#6783+#12124（INF-1403）、vllm-ascend#18754（INF-1379）、triton-ascend#2531（INF-1375）、#18746（INF-1374，closed 无修复维持）。
- **flashinfer#6316（INF-0781）**：10-11 已收录，本轮 note 回访补机制细节（阈值 121/114、MiMo 层级 scale 分布、聚合余弦 0.99999 骗过精度门）。

## 四、arXiv

6 组查询（silent data corruption+inference / bit flip+LLM / KV cache+corruption / soft error+GPU / numerical correctness+inference / wrong output+LLM inference）经限流分段等待后全部 200 有效。窗口内新论文：2610.18557（CSNB SDC 检测，10-14 已判非故障域）、2610.08739（BARE-AI，10-12~14 已判）；2610.03117（剪枝×容错权衡）、2609.31796（激活探针数值非确定性）经摘要判定非本库故障域。**无新收录**。

## 五、API 账本与未覆盖范围

| 组 | 来源 | 状态 |
|---|---|---|
| 1-14 | GitHub Search（7 仓 × 2 词） | 全部有效（首轮脚本截断作废重取，二轮逐组校验） |
| 15-16 | GitHub core REST（#44504、#61803 详情） | 有效 |
| 17-22 | arXiv 6 组 | 首轮 429/503，分段等待约 3 分钟后全部有效 |
| 23 | NPU 共享批次 | 22/22 complete、0 failures |

**未覆盖/限制**：(1) GitHub 未认证 Search 每词一页 per_page=100，本轮各组 total 均 ≤23 无翻页需求；(2) vllm#61157（GLM SSM 并发退化）与 sglang#44070（PD 早发 stale KV）为强任务影响候选但根因未定位/未逐条深读，留观已列；(3) tt-metal#58402 审计族本日仅收录静默错值类 5 条，其余 init/replay 域条目留观族化；(4) NPU 批次按 updated 增量判定，10-16T23:00Z 前创建且其后无更新的条目不再重判（由 10-13~16 日报覆盖）；(5) triton 主仓本轮 0 新 issue。

## 六、编目变更汇总

- 新增 26 条（全部 pending_audit）：**INF-1408 ~ INF-1433**
  - vllm 主仓：INF-1408（CPU LoRA 混批泄漏）、INF-1409（fast-prefill logprob 绕过）、INF-1433（DP MoE-LoRA 跨副本）、INF-1424（KV Connector 三连）
  - flashinfer：INF-1414（ragged attention Q 步长）
  - tt-metal：INF-1410（tanh NaN 洗白）、INF-1423（replay 体跨界）、INF-1411（transpose 窄行）、INF-1412（fast_untilize BFP）、INF-1415（top_k 钳 32）
  - sglang：INF-1413（权重更新静默跳过）
  - NPU：INF-1425（HCCL 单 rank 位移，cross TRN-0299）、INF-1431（TE 后端 DCE）、INF-1426（max/min 空 tensor）、INF-1416（copy_ 共轭）、INF-1427（rms_norm 920）、INF-1417（GE 动态 shape 越界写）、INF-1428（SwinLnQKV epsilon）、INF-1418（RingAttention 漏 return）、INF-1429（397B V-scale，var INF-1398）、INF-1419（块跨步传输，var INF-1394）、INF-1430（KPool 空 work）、INF-1420（AdaMax 溢出）、INF-1421（GatherElements 回绕）、INF-1432（masked store + floor(cos)）
- update：INF-0781（回访 note）、INF-1393/1394/1396/1398/1399/1401/1403/1395/1374/1375/1379（回访 note/evidence，见第三节）。
- 关系核对：INF-1429→INF-1398（vllm-ascend C8 MXFP8 KV scale 损坏输出错）、INF-1419→INF-1394（vllm-ascend Mamba KVPool 恢复前缀后输出乱码）、INF-1425→TRN-0299（HCCL 单 rank 路径 V 算子忽略位移错位读写）——回显标题均与预期一致。
- `catalog_tool.py build` + `validate`：全部通过（ok: true）。

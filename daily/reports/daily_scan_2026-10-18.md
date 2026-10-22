# 每日扫描日报 2026-10-18（推理库）

- **扫描窗口**：GitHub `created:>=2026-10-11`（cutoff 实际计算 `2026-10-11`）；重点判定 10-17 报告之后的新增。NPU 共享批次 `batch-20261018T230043096100-fk9ec6pn`（22/22 仓 complete，0 failures，采集窗口 2026-10-17T22:00:43Z → 2026-10-18T23:00:43Z updated_at 增量，candidates=10816，inference 视图 19 源）。
- **目标仓库**：原 GitHub 7 仓 + arXiv + NPU 19 推理源。
- `GITHUB_TOKEN` 未配置，按未认证限额执行（Search ≥9s 间隔）。**网络事件**：terminal 注入代理与 mihomo（127.0.0.1:7890）访问 api.github.com 均超时不可达（25s/15s 0 字节，等待 45s+60s 重试仍 000；github.com 网页亦不可达）；改 `curl --noproxy '*'` 直连后 api.github.com 恢复 200，14 组 Search 全部直连完成。arXiv 直连（--noproxy）6 组全部 200 有效。core REST 3 次单号详情（PR#19073/#19101、issue#61225 状态核验），未触限流等待。累计限流等待 ~105 秒（未达 5 分钟上限）。
- Git 前置检查：main 干净、fetch 成功、HEAD == origin/main（f450da84）。

## 一、原 7 仓新增条目（12 条，pending_audit）

1. **INF-1445** [NEW] [vllm#61053](https://github.com/vllm-project/vllm/issues/61053) — TP 各 rank Inductor 归约独立 autotune（实测同 kernel 4096 vs 8192 config）→ router 输入低位分叉 → 近并列分数下同 token 各 rank 路由不同专家、all-reduce 返回两专家各半之和；三臂差分（双 rank 钉 config 0/0/0 分叉）归因闭合；~1e-3 token 受影响。documented。
2. **INF-1447** [NEW·var→INF-1353] [vllm#61225](https://github.com/vllm-project/vllm/issues/61225)（10-12 主库已编目 INF-1353/8.100，本轮窗口覆盖其创建日、按推理库完整正文复核为 var 收录）— nvfp4 KV 写内核 head-major 内部公式 × reader 实际 stride，QSA 默认 BLNHC 下全上下文输出垃圾、253k needle 0/N、自建 buffer 测试结构性失明。var 目标标题已核对。
3. **INF-1442** [NEW] [vllm#62304](https://github.com/vllm-project/vllm/issues/62304) — n-gram 投机部分接受步 GDN 循环状态漏写已接受 token（probec 探针 twin oracle 回放证明）；k=1 即触发、无投机 12 配置 0 错、draft 全接受 0 错。documented。
4. **INF-1446** [NEW·var→INF-0127] [vllm#62116](https://github.com/vllm-project/vllm/issues/62116) — GDN chunk/prefill 内核把 fp32 循环状态降为激活 dtype（chunk_delta_h/chunk_o 缓冲+tl.dot+API 返回转换三处），fp16 激活+~1e6 状态输出 100% 非有限；--mamba-ssm-cache-dtype float32 不防护；#54146（decode 侧 INF-0127）明示的 chunk 后续。var 目标标题已核对。
5. **INF-1451** [NEW] [vllm#61116](https://github.com/vllm-project/vllm/issues/61116) — thinking_token_budget + 投机：自然 </think> 在接受 draft 内且预算同窗耗尽时二次强制注入，双 </think>、greedy 投机≠非投机。documented。
6. **INF-1455** [NEW] [vllm#61231](https://github.com/vllm-project/vllm/issues/61231) — EAGLE/EAGLE-3 drafter 在 draft config 无 rope 键时双层默认 base 10000，target 值无路径到达 drafter；配置解析级无 GPU 复现。documented。
7. **INF-1456** [NEW] [vllm#62307](https://github.com/vllm-project/vllm/issues/62307) — kv_transfer_params.prompt_token_ids 在无 KV connector 的任何 server 上替换渲染 prompt（跳过 chat template/system prompt/分词），0.27.0 起 main 仍缺 connector/端点检查。documented。
8. **INF-1444** [NEW] [sglang#43725](https://github.com/sgl-project/sglang/issues/43725) — NGRAM 零填充 draft 兄弟逐个累加 p(0)、拒绝残差只扣一次：token 0 过采样 11 倍（0.0485→0.538）。documented。
9. **INF-1448** [NEW] [sglang#44997](https://github.com/sgl-project/sglang/issues/44997) — NGRAM + overlap scheduler 尾步 verify 写 KV 落入/覆盖 radix 前缀块，stop/EOS 结尾 3/192 greedy 错、关 overlap 0/5 精确；KV 污染跨请求扩散。documented。
10. **INF-1449** [NEW] [sglang#44641](https://github.com/sgl-project/sglang/issues/44641) — native /generate 的 mm_hashes 客户端可控缓存身份：声明同 hash 换图即命中前图 KV、按前图回答。documented。
11. **INF-1452** [NEW] [sglang#44758](https://github.com/sgl-project/sglang/issues/44758) — Kimi-K3 nvidia_kda prefill 单头 35-55 token 十倍漂移至 NaN、写入 KV/状态跨请求继承；Triton 同批 0 坏。documented。
12. **INF-1454** [NEW] [sglang#44999](https://github.com/sgl-project/sglang/issues/44999) + [#45000](https://github.com/sgl-project/sglang/issues/45000)（同日同方法论、不同量化内核，分立两条——#45000 为 **INF-1453**）— Marlin MoE（AWQ W4A16）与 FP8 Marlin 在 SM86 间歇错 token：单请求逐次仍错、GEMM 隔离精确、同 checkpoint vLLM/Triton/bf16 全精确；FP8 条关 overlap scheduler 0/5 精确。
13. **INF-1443** [NEW] [tt-metal#61268](https://github.com/tenstorrent/tt-metal/issues/61268) — ttnn.where 张量重载接受 BF16 条件 × FP32 值不报错返回 2 的幂垃圾（~1e34）/denormal；标量重载会拒绝、张量重载漏检；生产命中 FP32 MoE router masked-fill。documented。
14. **INF-1450** [NEW] [TRT-LLM#20504](https://github.com/NVIDIA/TensorRT-LLM/issues/20504) — breakable prefill CUDA graph 共享输出缓冲归错 bucket，部分 bucket 回放输出损坏。**按 search item 标题/短语收录**（正文逐段深读未完成，见未覆盖）。

### 拒绝 / 留观（原 7 仓，主要项）

- **vLLM**：#61738（宽 verify≥7 EOS 早停——10-16 已判留观维持，本轮正文确认 prompt 变体敏感、绑定混合 GDN 宽 verify，仍留观）；#61231 已收录（本轮）；#61117（CPU rejection sampler -1——10-12 已判强留观维持，CPU 域）；#61172 已编目 INF-1359（10-12）；#62225（CPU LoRA 混批——**10-17 已收录 INF-1408**，本轮窗口重叠不重判）；#62122（prompt_logprobs=0——10-17 已收录 INF-1409）；#61453（DP MoE-LoRA——10-17 INF-1433）；#61768/#61905/#61912（KV Connector 三连——10-17 INF-1424）；#62051（DSV4.1 DSML 解析器截断/畸形处理——输出解析域候选，机制清晰表格完备，本轮合并量控制留观下轮）；#61966（MinTokensLogitsProcessor 全局 id 索引 TP 分片 logits 越界——CPU IndexError/GPU 越界 assert 崩溃为主，实测在 vllm-ascend 910B2C TP8，fail-loud 域但触发面宽，留观）；#61383（XGrammar 间歇终止 V2 DFlash——间歇 500 域）；#61681（GLM-5.3-Flash mm-encoder-tp-mode data 图像混淆——多模态 TP 候选强，本轮未深读留观下轮优先）；#61806（DSpark draft 数虚增——10-16 留观维持，性能域）；#62307/#61231/#61116 本轮已收录。
- **SGLang**：#44070（PD 早发 stale KV——10-13 判定维持留观）；#43997（GLM-5.3-Flash hybrid KDA+DSA host-tier 恢复错输出——closed 未详查，与 10-16 INF-1392 同域相邻，留观）；#45034（XPU DSpark TP=2 draft 图回放分叉——XPU 域留观）；#43956（Qwen3.8-27B 分数依赖 batch shape——10-13 判定维持）；#43857（mm token-layout u32 回绕——>2^32 token 才触发，面极窄留观）；#43738（10-12 判定维持）。
- **TensorRT-LLM**：#20375 已收录（10-16 INF-1404）；#20138（10-17 留观维持）；#20509/#20506/#20376/#20382/#20377/#20510/#20379（测试跳过/workspace 尺寸/注册命名/JIT 失败/性能——非静默正确性域）。
- **tt-metal**（#58402 静态审计族持续）：本轮窗口内 #61268 已收录；#61247（div_no_nan_bw host 标量 0 测试——subnormal 梯度错域，留观族化）；#61242（logiteps_bw eps<0 域外行为——训练反传域）；#61194（max_reduce_with_indices accumulate 被忽略——留观）；#61192（approximate exp_tile 忽略 scale——留观）；#61189/61146/#61144/#61128/#61126/#61056/#61078/#61080/#61069/#61091/#61093/#61099/#61121/#61173/#61036 等（10-17 留观族维持，本轮无新证据不重判）；#60938/#60927（generic_op 运行时参数/网格重叠——留观）；#61210（matmul_block 奇数维挂起——挂起域）；#61214（10-17 留观维持）。
- **flashinfer**：#6741（CUTLASS MLA 接受池外页索引——越界读候选，本轮未深读留观下轮）；#6484（sparse MLA decode 小 KV 越界读——同上留观）；#6473（SIGFPE 除零——fail-loud）；#6434（_WEIGHT_CACHE 增长——资源泄漏域）；#6399（top_k 非确定错误索引——间歇采样域候选，留观）；#6479（10-17 INF-1414）；#6316（10-11 INF-0781）。
- **triton 主仓**：#12347（for 循环边界符号静默加宽——语言语义域候选，正文未深读留观）。
- **transformers**：#49647（GGUF tokenizer 忽略 add_bos_token 默认丢 BOS——输入编码域候选，BOS 缺失对下游是静默输入错位，本轮未深读留观下轮优先）；#49564（SDPA enable_gqa 静默回退 math kernel——性能回退域）；#49499 已 closed（HunYuan-VL beam search 错图——batched 图错配候选，closed 未详查留观）；#49498（SynthID CPU/MPS 采样表不同——平台差异域）；#49698（input_projections 未注册 module——缺参数错误域）。

## 二、NPU 来源扫描（batch-20261018T230043096100）

- 22/22 仓 complete、0 failures、registry_hash 2f6422d0…（22 源全选）；本轮增量判定 = updated_at > 2026-10-17T23:00:43Z 的 **894 项**（issue 119 + PR 775），标题关键词分桶 205 项命中、其余 689 项逐一过题（文档/CI/风格/Proto 声明类占绝对多数）；34 项正文深读。上一报告（10-17，覆盖至 10-17T23:00:43Z）之前创建且无新更新的条目不重复判定。

### 新增条目（10 条，pending_audit）

1. **INF-1434** [NEW] [op-plugin#593](https://gitcode.com/Ascend/op-plugin/issues/593) — fp32 conv 全族（conv1d/conv2d/conv_transpose1d/2d）静默 HF32 类降精度 rel 6.0e-4~1.8e-3、梯度同污染；high_precision 选项废弃无效；matmul/bmm 同机正常可排除平台性；双芯逐位相同。documented。
2. **INF-1435** [NEW] [op-plugin#590](https://gitcode.com/Ascend/op-plugin/issues/590)（修复 PR [!6052](https://gitcode.com/Ascend/op-plugin/merge_requests/6052) open）— max_pool2d/1d/3d(return_indices=True) indices 四种异形（int8×8 末维/多一行/多一维/int32），转换层未折叠 int64 视图未裁行尾填充；内核数值正确、MaxUnpool2d 下游不可用。documented。
3. **INF-1436** [NEW] [op-plugin#591](https://gitcode.com/Ascend/op-plugin/issues/591) + 同机同根因 [ops-math#3446](https://gitcode.com/cann/ops-math/issues/3446)（10-14 留观→**范围变更+新证据改判合并收录**：#591 提供 round 通路闭源定位与 arch35 保符号对照）— Round/Trunc/Ceil/Floor/Sinh 负零符号位丢失家族（round(-0.5)=+0.0），arch35 显式保符号而 910 路径缺失。documented。
4. **INF-1439** [NEW] [ops-nn#6841](https://gitcode.com/cann/ops-nn/issues/6841) — torch_npu ATen gelu 分发 tanh-only aclnnGelu：approximate='none' 与 'tanh' 逐位相同、偏差 4.7e-4；npu_gelu 带 attr 正确（CANN 包内 erf 路径存在仅分发选错）；源码级定位。documented。
5. **INF-1440** [NEW] [ops-nn PR !11513](https://gitcode.com/cann/ops-nn/merge_requests/11513)（u2026-10-18）— quantize_add_layer_norm 950（arch35）量化前缺低精度回转（fp32→fp16/bf16 RTNE→fp32）：22 用例 1,867 个 ±1、CPU 双路复算证明设备算法自洽而规格要求回转；910 系有该步、950 是家族唯一缺失分支；修复后 950 真机 22/22 逐位恢复。documented。
6. **INF-1437** [NEW] [ops-math#3576](https://gitcode.com/cann/ops-math/issues/3576) + 姊妹 [#3575](https://gitcode.com/cann/ops-math/issues/3575)（修复 PR [!6411](https://gitcode.com/cann/ops-math/merge_requests/6411) open）— StatelessRandperm 浮点输出 dtype 重复值破坏排列合同（FP16 n=2051 仅 2050 个不同值、BF16 n=259 仅 258）；Host 只按最大有限值校验未按连续整数精确表示范围；#3575 整数边界差一误拒同 PR 修复。documented。
7. **INF-1441** [NEW] [triton-ascend#2530](https://github.com/triton-lang/triton-ascend/issues/2530) — 归约轴 split（R0_BLOCK<rnumel）时 index 归约丢 chunk 偏移：value 对、indices 返回局部下标；单 chunk 0 mismatch 反证；与 INF-1375 同仓不同机制（特化污染 vs 拼接丢偏移）。documented。
8. **INF-1438** [NEW] [vllm-ascend#18859](https://github.com/vllm-project/vllm-ascend/issues/18859)（c2026-10-15 报告，本轮窗口内发现）+ 修复 PR [#19107](https://github.com/vllm-project/vllm-ascend/pull/19107)（c2026-10-18 open）— MLAPO 对 Q/KV RMSNorm 硬编码 1e-6 不读模型 rms_norm_eps（GLM-5.2 W8A8=1e-5）；修复 PR 910B3 实测 baseline 16 failed→24 passed。documented。
9. （同上第 8 条合并计入 INF-1438。）
10. NPU 其余新 PR/issue 按下列判定处理（见留观/拒收）。

### NPU 留观与拒收（主要项）

- **留观（下轮优先）**：op-plugin#592（index_select 越界→GatherV3 设备 assert 507035 毒化上下文——fail-loud 崩核域，但**主机校验缺失使非法输入直达设备**的防御面与 #589 同族，批量测试一个崩核级联污染整批，随 #589 族化评估）；op-plugin#589（F.pad replicate 负 padding 向量核 UB 越界崩核+设备上下文报废——同上崩核域，reflect 报错无文案为附带）；ops-transformer#6275（MsaIndexScore TND InferShape 解引用设备侧序列长度——Host 非法读致调用失败，防御域）；ops-nn#6846（Threshold arch 门控跳过 dtype 校验返回错码——错误码契约域）；ops-nn#6295（quant_flash_attn 枚举 4 值错位少 1 成员——实验算子文档/头文件两套编号，契约域）；ops-transformer#6293/#14064（AlltoAllMatmul 910B scale rank 校验缺失——防御域修复）；#6283/#14054（MoeFinalizeRoutingV2 k=0 除零——host tiling 拒绝域）；cann/runtime#1141（多设备饱和模式串用——设备配置状态域，对推理数值无直接影响证据）；ops-nn PR !12196（InplaceApplyAdaMax 聚合加固——10-17 已收录 INF-1420，本轮聚合 PR 为其修复实现，update 已在 10-17 记录）；ge PR !5423（GEIR DataType 枚举映射错——10-16 留观维持）；ops-nn PR !12073（ScatterNdUpdate 非连续回退丢失回写目标 561103——非连续防御域+修复完备，候选中等留观）；ops-nn PR !12124（PadV3 小数填充误融合——10-16 已收录 INF-1403 修复 PR，维持）；ops-transformer PR !14026（Compressor stride int 窄化——大 state_cache 边界域）。
- **拒收/不占号**：Ascend/pytorch#5501-#5510 系列（API 一致性任务）；torchair PR 3854/3857（inductor 特性/缓存序列化修复——工具域；3857 的 GraphModule 序列化原地破坏为确定性 TypeError，fail-loud）；torchair PR 3795（benchmark eager_fn 参考语义——维测工具域，PR 自述不影响真实编译路径）；torchair PR 3789（dequant dtype 默认值对齐——converter 默认值契约，未伴数值实测）；ops-nn 6821/6818/6819/6823/6840/6843-6845/6833-6839（Proto/README 声明不一致文档族 ~20 条）；ops-nn 6820/12186（NonZeroWithValue transpose 默认值——tiling 拒绝域）；ops-nn 6824/12190（EuclideanNorm 标量 keep_dims rank——shape 推导域）；ops-nn 6829/12195（MultilabelMarginLoss target 形状——防御域）；ops-nn 6842/12210（SoftplusGrad 广播推导——推导失败域）；ops-nn 12150（HingeLossGrad 软门衰减——训练反传域）；ops-nn 12086（fqmm swiglu 16core fail——修复聚合单无细节）；ops-nn 12162（max_pool3d_grad Gather 分发——修复单无 issue 细节）；ops-nn 12108（avg_pool1d 负被除数取整——修复单已验证，下轮复核是否需独立条目）；ops-transformer 6268-6298 文档族（README shape/公式/参数说明 ~25 条）；ops-transformer PR 14053（PagedAttention 空 actualSeqLengths 拒绝——防御域）；ops-math PR 6005（AngleV2 复数零符号相位——负零族相邻但复数域，修复 PR 已有留观下轮族化评估是否并入 INF-1436）；ops-math PR 6034（Round AICPU FP64 近半整数——同 Round 族相邻，DOUBLE 域，随 INF-1436 观察修复是否覆盖）；ops-math PR 6411 附属（#3575 整数边界误拒——误拒即报错，防御域，已并入 INF-1437 备注）；runtime 1131/1147/5630（错误日志读取未写输出——日志域）；runtime 5485（aclrtMemsetD32 不支持误返成功——防御域强候选，影响面 memset 场景，留观下轮）；hccl PR 3655/3650/3645（性能/特性）；hcomm PR 6100（RpingDeinit 悬空指针——fail-loud 域）；msmodelslim PR 1100/1095/1099（量化特性/QAT——feature）；vllm-ascend #19113/#19114（RFC HiSparse——设计文档）、PR 19109（HiSparse 实现 draft）、PR 19112/#19041（KDA prefill 补齐修复——Refs #15665，本轮未见对应 issue 细节，随 PR 观察）、PR 19110（MXFP8 MLA draft 权重准备 backport——#18893 修复线）、PR 19108（SFA dummy O-proj 集合通信 join——挂起域修复）、PR 19098（HIXL EPLB 缓冲 fence——竞态修复线 #17681/#18896）、PR 19107（**已收录 INF-1438**）；mssanitizer PR 427（性能优化）；mindspore PR 93192（Dense 混 dtype——训练域）；mindformers 8708/8699/8727（训练/工具域）。
- **沿用判定**：ops-transformer#6119/#6028（epsilon——10-17 INF-1428；本轮 PR !13800 仍 open 维持）；ge#5135（INF-1417 维持 open）；ops-nn#6796（INF-1420）；ops-nn#6708/#6816（10-16/10-17 判定维持，聚合 PR !12196 为修复实现）；triton-ascend#2531（INF-1375，u2026-10-18 无实质变化）；vllm-ascend#18754（INF-1379 维持）。

## 三、回访已收录来源（update）

- **INF-1430**（GLM KPool 空 work）：修复 PR vllm-ascend#19073 closed **merged=True**、v0.30.0 backport #19084 亦 merged → update note（发布版本未核实）。
- **INF-1393**（AscendStore 上报）：#19101 closed merged（main）、backport #19090 merged → add-evidence #19101 + note。
- **INF-1353**（nvfp4 KV stride，vllm#61225）：issue 仍 open、上游修复未见；本轮窗口完整正文复核 → note（新增推理库 var INF-1447，本条保持主体）。
- **轮巡（≥3）**：INF-1003（vllm#58824）、INF-0343（vllm#58973）——本轮批次与 Search 窗口均无新活动记录，update note 写明；INF-0654（sgl#41617）维持 10-16 记录（无新活动）。

## 四、arXiv

6 组查询（silent data corruption+inference / bit flip+LLM / KV cache+corruption / soft error+GPU / wrong output+LLM inference / numerical correctness）直连全部 200 有效。窗口内新论文：10-13 Cauchy-Schwarz SDC 检测（10-14 已判非故障域）；其余均为 10-06 前旧文（BARE-AI 10-12 已判等）。**无新收录**。

## 五、API 账本与未覆盖范围

| 组 | 来源 | 状态 |
|---|---|---|
| 1-14 | GitHub Search（7 仓 × 2 词，created:>=2026-10-11，per_page=100） | 首批因代理超时 0 落盘；改 --noproxy 直连后 14 组全部 200（vllm 81+131、sglang 40+58、transformers 16+17、TRT-LLM 11+27、triton 3+0、tt-metal 117+41、flashinfer 37+54），全部 incomplete=false，无 >100 截断 |
| 15-17 | GitHub core REST（PR 19073/19101、issue 61225 状态） | 有效（直连） |
| 18-23 | arXiv 6 组 | 直连全部 200 有效 |
| 24 | NPU 共享批次 | 22/22 complete、0 failures |
| - | 代理路径（mihomo 7890 / 注入代理） | api.github.com 与 github.com 网页均不可达（超时 0 字节），未用于任何数据获取 |

**未覆盖/限制**：(1) GitHub 网页端（github.com）本轮全程不可达，仅 API 可用——不影响 Search/状态核验，但无法抓取 issue 评论网页视图；评论级深查未做，涉及 #61681、#6741、#6484 等留观候选的后续讨论未读取；(2) **仅按索引/标题判定的候选**：TRT-LLM#20504（INF-1450，标题+短语收录、正文未逐段深读）、tt-metal 留观族（10-17 判定维持未重读正文）、transformers#49647（标题判定留观）、triton#12347（标题判定留观）；(3) NPU 批次 689 项未命中关键词条目按标题过题（文档/CI/Proto 声明族占 ~85%），未逐条读正文——若标题与正文严重不符的候选可能漏判；(4) vllm#61157（GLM SSM 并发退化，10-16 留观）本轮 Search 窗口无新评论迹象，未重读；sglang#44070 维持 10-17 判定；(5) 旧 open 长期来源评论级复查仅 2 个（INF-1003/INF-0343 经批次+search 快照双确认无活动），第三个（INF-0654）沿用 10-16 回访记录，本轮 core REST 额度大部分用于新候选正文与修复状态核验。

## 六、编目变更汇总

- 新增 23 条（全部 pending_audit）：**INF-1434 ~ INF-1456**
  - vLLM 主仓：INF-1445（TP autotune 专家分叉）、INF-1447（nvfp4 KV stride，var INF-1353）、INF-1442（ngram GDN 态漏写）、INF-1446（GDN chunk fp16 降型，var INF-0127）、INF-1451（双 </think>）、INF-1455（EAGLE RoPE base）、INF-1456（prompt_token_ids 替换）
  - SGLang：INF-1444（NGRAM token0 过采样）、INF-1448（radix KV 污染）、INF-1449（mm_hashes 错图）、INF-1452（nvidia_kda NaN）、INF-1453（FP8 Marlin）、INF-1454（Marlin MoE）
  - tt-metal：INF-1443（where BF16×FP32 垃圾）
  - TensorRT-LLM：INF-1450（breakable prefill 缓冲，按索引收录）
  - NPU：INF-1434（fp32 conv HF32）、INF-1435（maxpool indices 契约）、INF-1436（负零符号族，#3446 改判合并）、INF-1439（gelu tanh 分发）、INF-1440（量化前回转缺失）、INF-1437（randperm 重复值）、INF-1441（split 归约 index 偏移）、INF-1438（MLAPO epsilon）
- update：INF-1430（修复 merged）、INF-1393（修复 merged + evidence）、INF-1353（回访 note）、INF-1003/INF-0343（轮巡 note）。
- 关系核对：INF-1447→INF-1353（nvfp4 KV 写读 stride 公式分裂，BLNHC 布局全上下文垃圾）、INF-1446→INF-0127（vLLM GDN 读出按 fp16 物化，溢出致重复 token）——工具回显与 find 复核目标标题均一致。
- `catalog_tool.py build` + `validate`：全部通过（ok: true，0 errors；pending_audit 23）。

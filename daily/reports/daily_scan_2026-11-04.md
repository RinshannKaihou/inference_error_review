# 每日扫描日报 2026-11-04（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（1f310982ef9b4380edacc3193f270ea6390f5212）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete**，无 failures。窗口 2026-11-03T22:00:43Z → 2026-11-04T23:00:43Z（25 小时）。registry_hash 2f6422d0…（22 源）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261104T230043198033-f598q1fg`，generated 2026-11-04T23:00:43Z。inference scope 候选 25289 条（updated 滚动窗口全量），覆盖 19 个 inference 适用仓。
- 窗口内新建（created ≥ 2026-11-03T22:00:43Z）949 条：ops-nn 215、vllm-ascend 161、ops-math 121、pytorch 109、ops-transformer 112、runtime 45、hcomm 43、mindspore 27、hccl 19、torchair 12、triton-ascend 11、op-plugin 6、mindformers 6、msprobe 8、msmodelslim 9、mssanitizer 4、sgl-kernel-npu 8、MindIE-LLM 1（ge 快照 fetched 62 条内窗口新建并入上列）。
- 筛选流程：中英文关键词正则（精度/静默/错值/越界/NaN/wrong/mismatch/silent 等）标题层命中 71 条全部过目 → 逐条深读正文 28 条（含修复 PR 关联与源码定位核对）；vllm-ascend 窗口 31 条 issue 标题全过目 + 5 条深读；小仓（runtime/ge/hccl/hcomm/op-plugin/msmodelslim/triton-ascend/sgl-kernel-npu/mssanitizer/MindIE-LLM/mindformers）窗口标题全部人工过目。
- 近 14 天日报完整 URL 查重：71 条命中全部无既往判定（昨日批次窗口与本批重叠仅 1 小时）；编目 `find` 查重：本轮拟收 16 个来源 URL 中仅关联族 #6814=INF-2214、#6812=INF-2213（作为 var 关系目标，不重复占号），其余全部 0 命中。

### 新增条目（NPU/GitHub-NPU 来源 13 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2386 | [vllm-ascend#20860](https://github.com/vllm-project/vllm-ascend/issues/20860) | GLM-5.3-Flash 原生 FP8 检查点 modules_to_not_convert 用扁平前缀，嵌套包装模型真实前缀多一层，exact 匹配全部失败：约 1150 个排除层（若干无 scale 张量）被静默 FP8 量化；11 层索引器 query 向量实测恒为零，AIME2026 40%→93.3%（修复 #20861 后）；无告警 |
| INF-2387 | [vllm-ascend#20904](https://github.com/vllm-project/vllm-ascend/issues/20904) | sparse KV offload + fused overlap 以 Adler32 做请求身份，32 位碰撞时新请求复用旧请求物理行、跳过失效、旧前缀 KV 记为常驻命中；受控实验改常驻槽 ±8 同时改变注意力输出与生成 token |
| INF-2388 | [ops-transformer#7088](https://gitcode.com/cann/ops-transformer/issues/7088) | ScatterPaKvCacheWithKScale SIMT 路径位宽只看输入元素数不看 cache 偏移上界；cache>2^32 元素时 uint32 写偏移回绕：合法 slot 目标位置零写入、数据落到 cache 中部其他 block，返回 0/0/0 |
| INF-2389 | [ops-nn#8102](https://gitcode.com/cann/ops-nn/issues/8102) | IndexAddV2 空 index（合法退化输入）两段返回 0 但 out 一字节未写（0xA5 哨兵 48/48 残留），读回未初始化脏数据；V1 接口同输入正确；四种 dtype/mode/dim 组合复现 |
| INF-2390 | [ops-nn#8096](https://gitcode.com/cann/ops-nn/issues/8096) | RmsNormDynamicMxQuant 接口声明 epsilon=double、原型落 Float 属性，0<eps<1.4e-45 静默归零：全零 x 下 eps=1e-46 返 rstd=inf（期望 1e23，在 fp32 值域内） |
| INF-2391 | [ops-math#4460](https://gitcode.com/cann/ops-math/issues/4460) | ChunkCat 空列表/全空 tensor 两条早退排在参数校验前：out 空指针/shape 校验失效且不写数据，预填 NaN 12/12 残留仍返回 0 |
| INF-2392 | [vllm-ascend#20844](https://github.com/vllm-project/vllm-ascend/pull/20844) | V1 runner+PP>1+MTP 三重缺陷（draft 不达调度器/非末级接受计数陈旧/batch queue 超前）：输出 token 静默污染；drafts=0 时 PP2=PP1「不可见地正确」；Fixes #14207，async 组合仍错由启动守卫拒绝 |
| INF-2393 | [vllm-ascend#20972](https://github.com/vllm-project/vllm-ascend/issues/20972) | CaMem sleep/wake 忽略 ACL memcpy 非零返回：失败 sleep 提交坏备份并释放有效设备数据；失败 wake 标已恢复、清空唯一 CPU 备份；AST 加载生产代码+mock 复现（source-level） |
| INF-2394 | [vllm-ascend#20973](https://github.com/vllm-project/vllm-ascend/issues/20973) | compressed-tensors W4A16 MoE 工厂丢弃每层 group_size，scheme 读无关顶层配置默认 32：group64/128 检查点 scale 形状错（(2,512,16) vs (2,512,4)）无法匹配加载 |
| INF-2395 | [ops-transformer#7089](https://gitcode.com/cann/ops-transformer/issues/7089) | StemOamPrepPagedKv 不校验 kFlat 末维=stemStride*D：kflat_dim 传 1024 时 kernel 每行仍写 2048 bf16，越界写 2048 字节（canary 2048/4096 被踩），返回 0；var→INF-2162 |
| INF-2396 | [torchair#939](https://gitcode.com/Ascend/torchair/issues/939) | aten.addmm(二维 bias) lowering 拆成 MM+Add，bias 来自 BF16 Mul 时生产者内联进 FP32 epilogue 丢失舍入边界：首轮 200252/1048576 元素差 3.05e-5，五轮放大超 83 判据；恢复 10 处 BF16 舍入后逐 bit 一致（因果闭合） |
| INF-2397 | [mindformers#2624](https://atomgit.com/mindspore/mindformers/issues/2624) | 普通 MLA+TND+EOD 压缩时 DSA 标志误判提前返回空 attention mask，MLA FA V4 读整段序列：首 token 与因果前缀参考差 0.90（应仅 3.4e-5），输出与梯度均偏；修复 MR !8893 open |
| INF-2398 | [vllm-ascend#20924](https://github.com/vllm-project/vllm-ascend/issues/20924) | Ascend RoPE 返回旋转张量而 DFlash 假定就地语义，返回值丢弃后未旋转上下文 key 入 draft KV cache：DSpark/DFlash 接受率实测仅 50% 出头；修复 #20002 open |

### NPU 留观（窗口内代表）

- **ops-transformer #7085/#7086/#7087**（StemOamPrepPagedKv kvIndices 值域不校验→507035 fail-loud；ScatterPaKvCacheWithKScale value.dim0≠numTokens 越界读、cacheLayoutOptional 任意字符串静默按 BNBD）——真机复现+源码定位完整，但 #7085 为 fail-loud（设备侧错误码可见），#7086/#7087 属非法入参静默接受域；与已收 INF-2213/2214 同族（host/tiling 入参校验缺失）。本族代表值最高的 #7088/#7089 已收，其余按族留观。
- **ops-nn #8104**（FusedCrossEntropyLossWithMaxSum 1-D 输入硬编码 fp32 读，非法 dtype 下 2 倍 GM 越界读 567 条，仅 mssanitizer 可见）——机制真实但触发需绕过 op_def dtype 匹配（TTK kernel 直测路径），正常运行链路有 dtype 门；留观。
- **ops-nn #8121/#8122/#8123**（UnsortedSegmentProd/Max/Min rank 下溢 host 越界写、LinearIndex 负 axis 越界读、ReluV2 动态 C 维静默零维 mask）——InferShape 契约域，配修复 PR !13648/!13649/!13650 同日挂出；与 10-27 已判 infershape 整改族同型，按族留观。
- **ops-nn #8093/#8097**（BNInference FP32 中间溢出/倒数舍入/累积误差 + L2 异常测例预期校准）——测试资产与测例校准域为主（#8097 明确为测例问题），精度问题部分随修复 !13598 open 未合入，留观修复合入。
- **ops-math #4447**（PR#6507 SIMD VF 整改后 13 算子精度异常，已回退 #7079）——回退已执行，机制（AddrReg 寻址步进/尾块零迭代）与 #7100（kv_rms_norm_rope_cache arch35 精度，closed）同族，属仓内整改过程缺陷且已从 master 撤回；留观重整方向。
- **ops-math #4476/#4497**（div_no_nan/mul_no_nan UT 失败聚合）——UT 级触发，触发原因为 opbase 依赖滚动，无静默任务影响证据；留观。
- **ops-nn #8118**（ApplyCamePart4 参数声明与二进制配置不一致）、**#8125**（fake_quant 舍入边界+scale 下溢 Inf、fused_mul 尾块越界读/标量静默截断）、**#8101**（DeformableOffsets golden 精度）、**#8085**（Threshold golden 收窄）——优化器/量化工具/golden 参考域，推理库范围内非计算路径，留观（8125 fake_quant 部分如后续有推理侧实测再评）。
- **ops-transformer #7116**（SparseFlashMlaGrad D=511/513 触发 507015、D=1024 静默接受）——grad 算子主属训练域，D=1024 静默接受面待训练侧判定；#7112（FIA OOM 包插桩误判越界）观测工具域。
- **pytorch #49041/!49042**（Inductor online softmax looped 分支缺失静默回落两遍形式+NaN 传播语义偏差）——数值近似正确（回落形式），NaN argmax 语义偏差属 triton_helpers 包装域；#5865（torchbench 未关 kv cache 致误报）测试 harness 域。均留观。
- **vllm-ascend #20922**（MegaMoe shared-expert 融合不可行组合静默降级 =3→=2，仅加告警）——配置可观测性改进，性能/资源域；#20935（MiniMax FA3 scheduler API mismatch fail-loud）；#20928/#20908 CI/发布域。留观。
- **vllm-ascend #20925/#20957/#20963/#20948**（AscendStore 读写保留门不对称浪费池、TPOT 回归、图模式 host 开销、性能）性能域；**#20967~#20976**（14:32-14:35 批量提交的 source-level mock 复现族：num_compute_units classmethod、SFA/DSA 投影池残留、Mooncake 关闭残留、KV 池初始化等待、offload num_cpu_blocks）——多为资源管理/可用性域或纯源码级无静默计算污染证据，与已收 #20972（CaMem 权重丢弃，权重域入收）区别对待；**#20969**（DSpark 取消请求进度残留）报告者自述不声称 tensor 泄漏或端到端失败。均留观。
- **hcomm #1205**（Scatter RoCE 链路 CQE 只进不出打满 CQ 致下发卡死 120s 自救无效）——集合通信域挂起（fail-visible 卡死而非静默错值），根因分析完整，修复方向已给（非 fence 场景去 signaled）；留观修复合入，如后续证实静默数据损坏再评。
- **msprobe #669/!1138**（eager nan_check 反向梯度方向漏存误存）观测工具域（监测≠计算故障）；**msmodelslim #1148**（GLM4-MOE-LITE 专家数字段落回退）工具域；**mindformers #2625** 特性回合跟踪帖。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走代理出口共享配额，请求间隔 ≥8 秒。21 组查询（7 仓 × correctness/silent wrong output/NaN，`is:issue created:>=2026-10-28`）全部 HTTP 200、无限流等待（累计等待 0）。
- 与既往判定比对：vllm#65325/#65324（=INF-2383）、#65385（=INF-2385）、#64696（=INF-2322）、#64704/#64855/#64326/#64517/#64239、sglang#48230/#47508/#47245、transformers#50216/#50204/#50224/#50195/#50212/#50051、flashinfer#8570/#8443/#8119/#8284/#7980/#7687、TRT-LLM#21519/#21385/#21571、triton#12555/#12506、tt-metal#64021（=INF-2384）、#63965/#63910/#63924/#63683/#62655/#62654 等 tt-robot/quasar 族均沿用 10-24~11-03 判定，无新证据不改判。
- 本轮新收 tt-metal#63940（昨日日报明确「下轮候选优先」），如约执行。

### 新增条目（GitHub 常规来源 4 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2399 | [flashinfer#8607](https://github.com/flashinfer-ai/flashinfer/issues/8607) | GB200 上 trtllm_fp8_block_scale_routed_moe PDL（默认开）下读到未写完 scratch 返回错行：2~550 行/次成组、\|diff\| 至 3.5；0xFF 预填放大 2.6~3.1%/万次，普通分配 0.01~2%随内存布局；PDL off 32000/96000 次全 0 错 |
| INF-2400 | [tt-metal#63940](https://github.com/tenstorrent/tt-metal/issues/63940) | ttnn.permute MultiCoreTiledGeneric 以 uint32 累乘 tile 填充体积：n=2047（0.999*2^32）位级正确，n=2048（恰 2^32）20,971,520/20,971,520 全错（垃圾值非 NaN），源码行号定位 main 未修 |
| INF-2401 | [transformers#50336](https://github.com/huggingface/transformers/issues/50336) | DeepSeek-V4 HCA/CSA 压缩状态不随 beam 重排（reorder_cache 只重排 keys/values）：beam=3+cache 与 use_cache=False 候选不一致，补 batch 选择即恢复（CPU FP32 复现） |
| INF-2402 | [vllm#65680](https://github.com/vllm-project/vllm/issues/65680) | Kimi-K3 hybrid（69 GDN 层）TP16+FLASHMLA 长 prompt（~9.7k tok）部分请求 @ token 墙至 max_tokens；前缀 CJK 括号标记 24/24 确定性触发、串行单请求可复现；连接器 GET/驱动/采样已逐一排除，根因未定位（留观向维护者求证线性注意力状态页） |

### GitHub 留观（本轮判定）

- **transformers#50334**（MiniMax-M3-VL 视觉 attention 跨样本混批）——多模态 pipeline 域，与 11-03 已判 #50315/#50328 同口径（CV/音频管线非 LLM 推理核心路径）留观。
- **transformers#50212**（Trainer 并行下 floor-divide 静默错缩梯度，closed）——训练域，训练库范围。
- **vllm#65734**（ROCm dense attention torch.empty 空分配 + CUDA-graph padding 行污染）——ROCm 特定 + padding 行语义边界，正文待深读，下轮。
- **vllm#65564**（Qwen3.5-9B 工具选择慢 11x）性能域；**#65513**（Gemma4 全注意力层跑整 prefill）性能域。
- **flashinfer#8607 收录而 #8570 维持留观**：#8570（nvfp4 oracle vs deep_gemm 分歧）报告者自述可能是两条 FP4 数值路径预期差异，沿用 11-03 判定。
- tt-metal tt_dit 新批（#64150/#64127/#64113 等）DiT/视频管线域，非 LLM 推理路径，按既往口径不收。

### arXiv

- 2 组查询（"silent data corruption"+LLM 8 entries、"bit flip"+inference 8 entries，sortBy=submittedDate）HTTP 200 有效。无新论文：SDC 组 8 篇与 bit-flip 组 8 篇全部已在库或已判（10-12~11-03 各轮；bit-flip 组为攻击与容错硬件域，按既往口径不收）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **INF-2373~2382、INF-2383/2385**（昨日 13 条）：批次快照下全部源 issue/修复 PR 状态无实质变化（#20827/#20828/#20829/#20830/#20725/#8006/#8007/#8012/#8013/#7063/#29710/#65325 均仍 open；**#7051 closed**——修复 !15196 已 merged，与昨日记录一致，发布版本携带仍未核实），已逐条 update --note。
- **INF-2366/2367/2372/2361/2353/2354/2331**（更早日条目）：#20653（11-04T04:38 更新）、#20677、#20595、#20546、#20550、!13327 均仍 open；#20540 维持 closed（原因未核实）。已注记。
- **INF-2259/2260**（轮巡旧源）：#6910 仍连续多轮在滚动窗口外状态未核（下轮优先 API 单点核验）；#7640 批次快照 open 无变化。
- 昨日待办兑现：**tt-metal#63940 本轮已收**（INF-2400，昨日「下轮择重（#63940 候选优先）」）；#63965 维持留观（源码审读未上机，证据面弱于 #63940）。

## 4. 编目与视图

- 新写入 **17 条**（全部 pending_audit）：INF-2386 ~ INF-2402，编号连续。映射：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2386 | vllm-ascend#20860 | GLM 嵌套前缀失配 FP8 排除层静默量化 |
| INF-2387 | vllm-ascend#20904 | Adler32 请求 ID 碰撞残留 KV 复用 |
| INF-2388 | ops-transformer#7088 | ScatterPaKvCache 2^32 元素写偏移回绕 |
| INF-2389 | ops-nn#8102 | IndexAddV2 空 index 输出未写入 |
| INF-2390 | ops-nn#8096 | RmsNormDynamicMxQuant epsilon 静默归零 |
| INF-2391 | ops-math#4460 | ChunkCat 空输入早退跳过校验返回脏数据 |
| INF-2392 | vllm-ascend#20844 | V1 runner PP+MTP 三重缺陷静默错 token |
| INF-2393 | vllm-ascend#20972 | CaMem sleep/wake 忽略 memcpy 失败丢权重 |
| INF-2394 | vllm-ascend#20973 | W4A16 MoE 忽略 group_size 静默按 32 |
| INF-2395 | ops-transformer#7089 | StemOamPrepPagedKv kFlat 越界写（var→INF-2162） |
| INF-2396 | torchair#939 | torchair AddMM 拆分丢 bias 舍入边界 |
| INF-2397 | mindformers#2624 | MLA TND 因 DSA 标志误判丢因果掩码 |
| INF-2398 | vllm-ascend#20924 | DSpark draft KV 未旋转接受率跌半 |
| INF-2399 | flashinfer#8607 | PDL 路由 FP8 MoE 错行 |
| INF-2400 | tt-metal#63940 | ttnn.permute uint32 填充体积溢出全错 |
| INF-2401 | transformers#50336 | DeepSeek-V4 beam 压缩缓存未重排 |
| INF-2402 | vllm#65680 | Kimi-K3 长 prompt @ token 墙退化 |

- var 关系：INF-2395 → INF-2162（GatherPaKvCache 索引/长度缺校验族，关系回显标题已核对）。
- 回访 update --note：INF-2373~2383/2385、INF-2366/2367/2372/2361/2353/2354/2331、INF-2259/2260 共 20 条。
- 训练库跨库查重：17 个拟收 URL 在 training_error_review `find` 全部 0 命中，本轮无 cross 关系。
- `catalog_tool.py stats`：2402 条（active 2385 + pending_audit 17）；`build` 完成；`validate` ok:true（n_errors 0；n_warnings 62——均为 pending_audit 常规 warning，无存量新错误）。

## 5. 未覆盖范围与 API 账本

- **GitCode/AtomGit issue 评论线程未拉取**（无低成本批量入口），全部按正文判定；本轮收录的 NPU 候选正文均含真机自包含复现或源码定位，依赖评论改判概率低。
- **vllm-ascend 窗口 130 条 PR**（非 issue）按标题过滤 + 8 条重点正文深读（#20844/#20861/#20922/#20935 等），未逐条深读全部修复流正文。
- **GitHub 未认证 search**：21 组全 200 无限流（累计等待 0 秒，未动用 5 分钟等待预算）；本轮正文单点补拉 6 条（#63940/#8607/#65680/#50336 已读，#65513/#65734 标题级留观）。评论线程未拉。
- **arXiv** 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高）。
- **#65680（INF-2402）根因未定位**：报告者已排除连接器/驱动/采样/并发，向维护者求证中；本条按「已观察症状 + 已排除项」收录，置信度 documented，机制字段明确写根因未定位。
- 本轮无任何候选因限流/配额只按索引判定；全部收录/留观判定均基于已读正文或批次缓存正文。

## 6. 结论

17 条新机制入库（待审计），20 条既有条目回访注记。本轮覆盖：NPU 22/22 源完整批次（949 条窗口内新建、71 条强信号全筛、约 34 条深读）+ GitHub 7 仓 3 关键词（21 组全 200、约 150 unique、4 新收）+ arXiv 2 组（无新论文）。重点新面：GLM 原生 FP8 排除层前缀失配静默量化（INF-2386，基准塌缩 40%→93.3% 修复对照）、Adler32 请求身份碰撞跨请求 KV 复用（INF-2387）、超大 KV cache uint32 写偏移回绕（INF-2388）、PDL 默认开下 FP8 MoE 静默错行（INF-2399，GB200 生产默认配置直接受影响）、ttnn.permute 2^32 全错（INF-2400）。vllm-ascend 量化域本轮密集（2386/2394 + 留观 20922），与前几日 block-FP8/QuaRot 线索连贯。

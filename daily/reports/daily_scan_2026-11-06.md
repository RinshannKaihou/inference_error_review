# 每日扫描日报 2026-11-06（推理库）

## 0. Git 前置检查

- 分支 `main`，上游 `origin/main`；`git status --porcelain -uall` 为空；`git fetch origin main` 成功；HEAD 与 origin/main 一致（62d6fc2283f7b4d434f48cd9175b84dc122a42aa）。通过，开始扫描。
- Cron 前置脚本 NPU 批次完整：**22/22 源全部 complete**，无 failures。窗口 2026-11-05T22:01:00Z → 2026-11-06T23:01:00Z。registry_hash 2f6422d0…（22 源，与昨日一致）。

## 1. NPU 新增来源（共享批次）

- 批次：`/root/.hermes/profiles/third/cache/fault_scan_sources/batch-20261106T230100738495-kht3f5ld`，generated 2026-11-06T23:01:00Z。inference scope 候选 27085 条（updated 滚动窗口全量），覆盖 19 个 inference 适用仓。
- **本轮新项判定窗口**：created ≥ 2026-11-05T23:01:00Z（昨日批次截止），共 704 条（vllm-ascend 120、ops-nn 178、ops-transformer 119、pytorch 93、ops-math 47、hcomm 28、ge 17、hccl 14、runtime 20、triton-ascend 27、msmodelslim 8、mssanitizer 3、op-plugin 3、torchair 11、mindspore 4、sgl-kernel-npu 9、msprobe 3）。254 条 issue 标题全部人工过目；~26 条重点候选全文深读（含昨日留观转收 2 条）。
- 近 14 天日报完整 URL 查重：本轮拟收 22 个来源 URL 全部 0 既往判定命中；编目 `find` 查重 0 命中；训练库跨库查重 22 URL 全部 0 命中（无 cross 关系）。
- 机制级比对：#5924（argmax NaN 索引）与已收 INF-2412（triton-ascend#2981）同为 argmax/argmin 索引域但缺陷位置不同（torch_npu triton_helpers NaN 匹配语义 vs lowering 通道损坏），分立两号；#8790（flashinfer MLA rescale FMA）为 INF-2108（#6907）同机制残留路径，按规则 update 补证据未占新号；#8346 与 INF-2408（#8260）不同算子不同机制；#7210 与 INF-2171（#6636）不同算子。

### 新增条目（NPU/GitHub-NPU 来源 20 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2426 | [pytorch#5930](https://gitcode.com/Ascend/pytorch/issues/5930) | Inductor dynamic=True 归约：initialize_range_tree 用首次长度（4096）替换符号长度并固化进调用参数，后续 2048 输入返回 8386560（=4096 之和）应 2096128，512 得 4208640；CPU FakeTensor 代码生成器+Triton 解释器双重验证 kernel 把逻辑输入外元素计入求和；910C/CANN 9.1.0/torch-npu 2.10.0.post6，现场全局关 torch.compile 规避 |
| INF-2430 | [pytorch#5920](https://gitcode.com/Ascend/pytorch/issues/5920) | static Triton launcher 在 Ascend950（soc≥250 one-program-per-tile）grid/参数处理错误致 in-place pointwise kernel 重复执行：最小复现差 5.7~6.0、AlbertForMaskedLM 1.56、107 社区模型 74 个 fail_accuracy；坏区 compiled/raw≈0.0119（0.125 幂）；关 launcher 即修复（0.0）；910B group-dispatch 免疫 |
| INF-2427 | [pytorch#5911](https://gitcode.com/Ascend/pytorch/issues/5911) | TE 标量累加器四族缺陷：loop-carry init 旧秩（编译断言）、int64 累加器降级 int32、-inf 哨兵 fptoui 变 255 毒化归约（u8 amax 全 255 静默错值）、窄整型 sum 标量携带被 vendor 编译器误编译（u8 sum 差 183）；A2/triton-ascend 3.6.0，75 例 12 红 |
| INF-2428 | [pytorch#5924](https://gitcode.com/Ascend/pytorch/issues/5924) | triton_helpers 极值归约索引阶段只判 value==极值，NaN 自比较为假致全匹配失败返回填充最大索引：全 NaN 列 argmax 返回 31 应 0（2/32 错，相对差 inf）；修复 !49185 open（极值阶段传播 NaN+索引阶段补 NaN 匹配，6 项回归 PASS） |
| INF-2429 | [pytorch#5925](https://gitcode.com/Ascend/pytorch/issues/5925) | CppWrapperNpu.generate_args_decl 默认整数分支先于 signature 分支，i64 符号参数按 int32 打包：前两字段少 8 字节、Y0BLOCK 偏移 48→40，拼接输出第二分块 76800/307200（25%）静默错；修复 !49186 open（分支重排 +8/-8，8 组全 PASS） |
| INF-2435 | [ops-math#4577](https://gitcode.com/cann/ops-math/issues/4577) | SliceLastDim arch35 多域缺陷：srcStride 经 uint32 窄化丢高 32 位（行距≥2^32 静默错数据）、infershape 缺 stride/rank 校验（负 stride 负长度维度返回成功、[-2] 未知秩 0 字节输出）、tiling 除零 SIGFPE、空输出 O(outerSize) 空转；950PR/CANN 9.3.0 |
| INF-2431 | [ops-math#4595](https://gitcode.com/cann/ops-math/issues/4595) | ReplicationPad2d/3dBackward 资料 DOUBLE、def 未注册却放行：DOUBLE 输入 rc=0 走未定义路径，gradInput=[0,2.25,0,2.25]（FLOAT [4,4,4,4]），不符合任何 fp64 语义；对照算子正确返回 161002；A2/CANN 9.3.0 |
| INF-2436 | [ops-nn#8338](https://gitcode.com/cann/ops-nn/issues/8338) | ConvertWeightToINT4Pack 组间偏移 idxG*k*n 用未填充 ND 尺寸：K=17 时后组首字节落 136（应 256），与前组重叠 120B 前组 [136,256) 被覆盖；A2/A5 双平台一致，合法输入（校验不约束 K 对齐） |
| INF-2432 | [ops-nn#8346](https://gitcode.com/cann/ops-nn/issues/8346) | WeightQuantBatchMatmul A16W8 PerGroup ND 路径静默算错：文档合规输入 NRMS 0.1935（PerChannel 0.00166 对照），M 缩到 256 仍 0.1806 排除尾块；910B3/CANN 9.3.0/torch_npu 2.14，公开 API 最小脚本 |
| INF-2433 | [ops-nn#8322](https://gitcode.com/cann/ops-nn/issues/8322) | PadConv2dFusionPass 融合后在 dn2nz 缺失平台仍写 padding=SAME，InferConv2DPadsWithPadding 重算覆盖融合 pads：Conv 输出 257→256、MaxPool 128→127 无报错；修复 !13841 open（EXPLICIT+NOTSET，19 用例过） |
| INF-2434 | [ops-nn#8304](https://gitcode.com/cann/ops-nn/issues/8304)（同型 [#8309](https://gitcode.com/cann/ops-nn/issues/8309)/[#8310](https://gitcode.com/cann/ops-nn/issues/8310)） | LayerNorm/FastLayerNorm/BatchNorm 三算子对 eps<0 放行（rc=0）静默输出全 NaN；同族 AddRmsNorm 同条件正确返回 161002，同仓行为分裂 |
| INF-2440 | [ops-nn#8180](https://gitcode.com/cann/ops-nn/issues/8180) | AdaptiveMaxPool2d NHWC C>1：ConvertNHWCtoNCHW 只转输入，结果直接 Reshape 回 NHWC 缺反向 transpose——恒等池化 126/128 元素错位（首差 0.51→0.58），两段 rc=0；NCHW/C=1 对照全对（11-05 留观转收，本轮深读） |
| INF-2441 | [ops-nn#8209](https://gitcode.com/cann/ops-nn/issues/8209) | GroupNormSiluQuant rstd 整块 NaN：E[x²]−E[x]²+eps 直接进 sqrt，|mean|≫扰动时灾难性抵消为负；eps=1 恢复（反推 var≈−0.875）+channel 交替 NaN 精确按组跟随 128/128 双判据；非量化族已在 51ed92dd3 钳位 15 处但不含 quant 入口；A2/A3 双机逐位一致（11-05 留观转收，本轮深读） |
| INF-2439 | [ops-transformer#7210](https://gitcode.com/cann/ops-transformer/issues/7210) | ScatterPaKvCache PA_NZ 模板（tilingKey 500/501）只过滤负 slot 未过滤上界：slot=39168(=306×128) 两段 0/0，越界写 cache 外 32 字节（尾部哨兵实测改写）；#6571 修复覆盖 normal/ND 与 A2/A3 遗漏 PA_NZ；950PR 实测 |
| INF-2437 | [ops-transformer#7222](https://gitcode.com/cann/ops-transformer/issues/7222) | MoeGatingTopKSoftmax CheckOutShape 只比维数/dim0/3D-dim1 漏末维==k：yOut [8,1]（k=2）ret=0 被接受，memcheck 观测按 k 写越界（4B 连续跨 aiv4-7，参数不可追溯仅量级佐证）；三输出共用该函数 |
| INF-2445 | [ops-transformer#7172](https://gitcode.com/cann/ops-transformer/issues/7172) | FA arch35 hasRope：s1BaseSize>=s1Size 时 query rope 偏移多乘 s1BaseSize/s1Size 与 Q/keyRope 的 goIdx 不一致，GQA（gSize>1）读错 rope 数据精度异常；修复 !15422（issue closed，合入未核实） |
| INF-2442 | [ops-transformer#7188](https://gitcode.com/cann/ops-transformer/issues/7188) | arch38 FA DN：第二段 DataCopy 源 gap 用 vec1S2CopyLenDn 且 L1 pitch 未按 s2RealSize 64 对齐、halfS1RealSize 误取 align32——S2 尾块非 64 对齐用例精度异常；修复 !15443（closed，合入未核实；源码定位无实测） |
| INF-2438 | [vllm-ascend#21157](https://github.com/vllm-project/vllm-ascend/issues/21157) | 原生 FP8 checkpoint 在 A2/A3 静默以 BF16 运行：is_950() 决定 mxfp8_method，非 950 反量化整块权重走非量化 GEMM；显存翻倍破坏 27B FP8 单卡属性、算术口径改变，唯一痕迹一条 info 日志，文档写成 intended |
| INF-2443 | [vllm-ascend#21164](https://github.com/vllm-project/vllm-ascend/issues/21164) | AscendStore decode_adaptor_prefill_pp 硬编码 @pp_rank:0 替换，pp_rank>0 的 worker 不命中：N 个分区写同键互相覆盖只留一份 KV、all() 聚合假报完整命中、缺失分区永久跳过；静态检查+确定性路径 |
| INF-2444 | [vllm-ascend#21166](https://github.com/vllm-project/vllm-ascend/issues/21166) | mooncake hybrid/layerwise 连接器 prompt 截断后未重建 block_hashes（#20213 只修两处）：前缀缓存/KV 复用决策仍指向截断前 prompt，hybrid 截断对齐查找的设计目标未达成；补两行即修 |

### NPU 留观（本轮判定）

- **ops-nn #8334/#8342**（DS V4.1 W4A8 FP32 Linear 触发 507035 向量核异常；910B prefill 与 A5 各一单）——fail-loud 崩溃域（aclnnIndex/aclnnDynamicQuantV4 507035），无静默证据；与 vllm-ascend#21180（同为 DSA compressor.wgate aclnnIndex 507035 prefill 崩溃）疑似同族，跨仓证据未闭合，随崩溃域留观。
- **ops-nn #8345**（QueryBank src_len>UINT32_MAX 持锁返回，后续 InitAtcOpBank 死等）——挂起域（bank 锁泄漏），触发面为超大查询；留观。
- **ops-nn #8365**（Conv2dBaseTiling 并发读写 convTilingParseInfo_ 无锁数据竞争）——静态检查报告（ThreadSanitizer 风格），无实测错误输出证据；留观待实测。
- **ops-nn #8308**（matmul fusion pass 平台门控读 soc_info.arch_type 真机拿不到，3510 融合静默不生效）——融合不生效是性能/功能面（走非融合路径结果仍正确），非静默错值；留观。
- **ops-nn #8298**（Ascend960 fixp 优化部分 case 性能劣化）、**#8290**（batch_norm_grad_v3 arch35 fp32 dgamma 精度提升需求）、**#8291**（ApplyCamePart1 长归约精度）——性能/精度优化流（训练侧），不占推理编号。
- **ops-nn #8216/#7218**（MhcPreSinkhornPremixBackward 空 batch 早退不写参数梯度）——「空输入早退+输出未写」族新批，与已收代表（INF-2417/2420/2422）同型；按族留观。
- **ops-nn #8218**（BlockSparseAttention 全 0 mask 两段返回 0 但 attentionOut 完全未写，哨兵 7.0 全残留）——同上空态族，正文未深读；下轮优先（证据密度可能足单独占号）。
- **ops-nn #8319**（自定义算子包 libcust_opsproto_rt2.0.so NEEDED 缺 acl_rt 安装必现中止）、**#8321**（AddOpTestCase 未注册 UT 不运行）——构建/测试域。
- **ops-nn 文档族**（#8266/#8270/#8272/#8274-8287 等约 30 单 Foreach/Inplace 上限未披露、#8297/#8307/#8318/#8325 支持 false 与实现不符、#8317/#8330/#8332 错误码/约束资料）——纯文档缺口，不占编号。
- **ops-nn UT 失败族**（#8347-8364 共 16 单）——CI 测试基建，不占编号。
- **ops-math #4584**（Ndtri 边界精度+infershape 缺失）、**#4585**（grouped_bias_add_grad 等 EZ9999 非法错误码）、**#4590**（concat_v2 infershape canndev 迁移不对等）、**#4598**（BatchNormStats workspace 偏大）、**#4599/#4600**（空指针列表 SIGSEGV rc=139）——错误码/构建/fail-loud 崩溃域或信息不足；留观。
- **ops-math #4601**（aclnnCat 全空输入早退跳过 out 校验，非法 out 静默成功保持未初始化）——空态族变体，按族留观。
- **ops-transformer #7192/#7193/#7194**（GroupedMatmul 族 mssanitizer illegal read 256/1768）——观测工具报越界读，未见数值错误证据；工具正确报告与算子真实缺陷未区分，留观（监测域 vs 计算故障分开）。
- **ops-transformer #7189**（CompressorGrad 把资料要求必空的参数当必选，A3 误报 161001）、**#7169**（RingAttentionUpdateV2 空指针 SIGSEGV）、**#7170**（SparseFA Grad nullptr 两段成功后 507057）、**#7212**（QuantFlashAttnMetadata 空指针 abort rc=134）、**#7213**（MlaProlog cacheIndex 越界 507015 两段返回 0）——fail-loud/错误码域；#7213 两段 0 但设备侧拦截，非静默。
- **ops-transformer #7197**（AllGatherMatmulV2 ascend910_93 INT4 binary 缺 2 条）、**#7220**（CANN 9.2.1 编译 megamoe 失败）——交付/构建域。
- **vllm-ascend #21134**（MinerU2.5-Pro 输出格式异常）——第三方业务侧管线问题（mineru_vl_utils 解析），vllm-ascend 侧无定位证据；留观。
- **vllm-ascend #21140**（0.30.0 GLM5.3 Prefix Cache 命中不了）——现象级报告无根因定位；留观（若后续定位到 hash/块管理缺陷再评）。
- **vllm-ascend #21159**（decode recompute+engram dp_shared_memory DP 级 AllGather 死锁、HCCL watchdog 204s 级联）——挂起/可用性域，plog 证据充分但非静默错值；留观（与 #21077 同 DP/通信交互域）。
- **vllm-ascend #21180**（DSV4.1-Flash w4a8 prefill aclnnIndex 507035 崩溃）——fail-loud 崩溃域，4p4d 生产部署 prefill 反复重启；与 ops-nn #8334/#8342 疑似同族，留观。
- **vllm-ascend #21122**（HCCL Error 19 loopback 配置解决）、**#21156/#21161**（Nightly 测试失败）、**#21171/#21181/#21182/#21189/#21192/#21202/#21208**（文档反馈）——环境配置/CI/文档域。
- **vllm-ascend #21165**（sleep() 无条件注销 Mooncake buffer，weights-only wake 后不重注册，P2P 传输持续失败）——机制清晰（静态检查）但后果为传输失败 fail-loud（后续 transfer 报错），非静默数值损坏；留观。与已收 #20972（CaMem sleep/wake）同 sleep/wake 域。
- **msmodelslim #685**（DSV4.1-Flash FP8→MXFP4/MXFP8 转换后 vllm-ascend 拉起报错）——转换产物服务化失败（fail-loud），且 MXFP4「部分层反量化为 BF16」细节未拆分；留观，与 #680 同模型域。
- **msmodelslim #684**（msagent --version 版本号错误）、**#681/#682**（Kimi 适配 Feature）——功能/适配域。
- **op-plugin #688**（test_atb_flash_attention_v2 bf16 alibi compress 用例 FAIL 1e-05）——单测失败无定位（alibi+compress 组合精度差 1），正文无根因；留观深读。
- **pytorch #5916**（OOM 快照回调判空崩溃）、**#5922**（29.3GiB 必现 OOM）、**#5933**（多流 replay 地址异常挂死）、**#5931**（recordStream 同流短路缺失致 Active>Allocated）——崩溃/资源/挂起域；#5931 内存统计泄漏域留观。
- **pytorch #5909/#5910/#5913/#5915/#5917**（硬件解耦屏蔽批：算子 dtype 不支持）——适配任务清单，非缺陷。
- **pytorch #5912**（online softmax 适配套件断言/eager 参考按 default 写，TE 后端误报）——测试套件域，随 INF-2427 跟踪。
- **pytorch #5919**（310B1 首次用 NPU 固定占 5GB）——资源占用咨询。
- **pytorch #5921**（非 950 平台 pointwise 特殊函数 simt_only 编译失败）、**#5923**（Fast GELU approximate 不接受编译失败）、**#5926**（bmm Triton 模板覆盖不全性能低）、**#5928**（codegen_comment kernel name）、**#5929**（hcclDeterministic A5 用例失败+comm config 日志）——fail-loud 编译失败/性能/日志域。
- **pytorch #5927**（黑名单瘦身批 F：asin/asinh/atan 21 条）——正面整改流，随 11-05 B/C/D 批跟踪。
- **torchair #945/#947**（npu_chunk_gated_delta_rule/histc 不支持图模式）、**ops-transformer #7190**（同 chunk_gated_delta_rule op_graph 漏注册）——功能缺口（fail-loud 图编译失败），#947 与 INF-2418（histc 数值缺陷）同算子不同面。
- **hcomm #1236-#1241**（检视意见跟踪/文档/UT 补充）、**#1242**（回退标卡识别变更）——流程域。
- **runtime #1341**（queue_schedule 异常路径清理/校验/日志）、**#1342**（不依赖 protobuf 打包）——加固/构建域。
- **mindspore #29728/#29729**（GPU+PYNATIVE GRU 崩溃 CUDA 719、CUDA11.8/WSL2 支持口径）——GPU 域/文档；#29728 非本库范围（GPU），文档域。
- **mssanitizer #247/#248/#249**、**msprobe MR**、**ge/hccl 本轮窗口内 PR**——工具功能/构建/文档域。
- **triton-ascend #3029**（flex attention UB overflow+bishengir SIGABRT 编译失败）——fail-loud；**本轮 26 条 PR**（窗口内）为修复/迁移流，未逐条深读。

## 2. GitHub 常规 7 仓 + arXiv

- 认证状态：环境无 GITHUB_TOKEN（未认证）；search 走共享配额，请求间隔 ≥7 秒（实测 6-8 秒）。21 组查询（7 仓 × correctness/NaN/silent wrong output，`is:issue created:>=2026-10-30`）：**21/21 组 HTTP 200**（昨日 HTTP 000 的词组组本轮改用 URL 编码后全部成功）。无限流等待（累计 0 秒）。
- 与既往判定比对：vllm#65324/#65325/#65680/#65564/#65513/#65734/#64696/#65337/#65131、sglang#48660（=INF-2425）、#47508/#47501、transformers#50336（=INF-2401，已 closed）、#50315/#50334/#50212/#50216/#50440、TRT-LLM#21519/#21385/#21571、flashinfer#8570（=INF-2399）/#8443/#8284/#8607、tt-metal tt_dit 族（#64150 权重缓存/#64151 grid 固化/#64103 T5 bf16/#64113 SP padding/#64144 0*inf）与 tt-robot 族（#63619/#63629/#63910/#63924 等）均沿用既往判定，无新证据不改判。
- **flashinfer#8790**（MLA history rescale exp2 FMA 收缩，同 #6907 机制在 mla.cuh:522/:557、mla_hopper.cuh:610 残留，#6908 只修 FA2 prefill）——已按同机制 update 补证据进 INF-2108，未占新号。

### 新增条目（GitHub 常规来源 2 条）

| 编号 | 来源 | 机制摘要 |
|---|---|---|
| INF-2446 | [vllm#66141](https://github.com/vllm-project/vllm/issues/66141) | ROCM_AITER_MLA gqa_ratio=16+fp8+decode_qlen 3/4 派发非 persistent kernel 静默返回不相关输出：cos_diff 0.84-0.90（门限 0.03，qlen2→3 是 805 倍悬崖）；三组单变量对照过、确定性；gqa_ratio=32 同组合被既有检查大声拒绝——guard 存在但漏 16 |
| INF-2447 | [sglang#48964](https://github.com/sgl-project/sglang/issues/48964) | GLM-5.3-Flash（mamba/KDA）处理 ~1M token 请求（约 127 chunk）后引擎状态永久损坏：后续请求全垃圾、MTP 接受率 0.00（正常 0.6-0.8）、json_schema HTTP 500，flush_cache 无效仅重启；v0.5.22 干净/v0.5.23+nightly 复现；嫌疑 #44098 删 KDA 池 NextN 层（35→34）而 prefill 模式 draft 仍读写，根因未定位 |

### GitHub 留观（本轮判定）

- **vllm#66144**（EPLB async worker 异常静默卡住 EPLB）——异步容错域标题强信号，正文未深读；下轮优先。
- **tt-metal #64678**（moe_fused_swiglu FUSE_BIAS 下竞争）——并发竞争标题强信号，正文未深读；下轮优先。
- **tt-metal #64648/#64637**（lgamma_bw/multigammaln_bw 有限垃圾值、polygamma_bw NaN 语义）——backward 数学函数域，tt-robot 风格报告；随既往 tt-metal 数学函数族留观。
- **sglang#48967**（AMD ragged batch MLA decode NaN/wrong rows）——AMD Triton 域，正文未深读；下轮优先。
- 其余 RFC/Feature/Perf/CI 不占编号。

### arXiv

- 2 组查询（"silent data corruption"+LLM 8 篇、"bit flip"+inference 8 篇，sortBy=submittedDate）HTTP 200 有效。无新论文：16 篇全部已在库或已判（SDC 组：TrainSDC/SCOUT/LLM-PRISM 已收、Cauchy-Schwarz SDC 检测与 Syndrome Decoding 量化整数 SDC 为既往已判推理域论文无新增证据；bit-flip 组为攻击与容错硬件域 Rowhammer/JITterFlip/BARE-AI 按既往口径不收）。

## 3. 回访（近 7 日条目 + 轮巡旧源）

- **昨日 23 条（INF-2403~2425）批次快照核验**：全部仍 open 无状态变化；#4574（INF-2403）、#8260（INF-2408）、#21103（INF-2414）、#7164（INF-2417）11-06 有更新但状态未变；hccl **修复 MR !3919（INF-2419）仍 open 未合入**（11-06T04:11 更新）；已对 8 条代表性条目 update --note（INF-2403/2408/2414/2417/2419/2412/2382/2259）。
- **11-04 批次条目**（#20904/#20972/#20973/#20924/#20844/#7088/#7089）：全部仍 open 无状态变化；#20861 修复 PR 仍 closed 状态（11-05 已注记）。
- **轮巡旧源**：INF-2259（#6910）快照 open，最后更新 2026-10-30 连续多轮窗口外（!14811 未核，已注记继续留观）；INF-2382（#7063）仍 open，修复 MR !15405 合入未核实（已注记）；**INF-2108**（flashinfer#6907）：发现同机制 MLA 路径残留 #8790，已 update --add-evidence + --note。
- flashinfer#8607、tt-metal#63940、transformers#50336（已 closed，昨日已注记）、vllm#65680、sglang#939/#2624、op-plugin#8096/#8102/#4460：批次/搜索窗口内无新增活动，状态沿用昨日记录。

## 4. 编目与视图

- 新写入 **22 条**（全部 pending_audit）：INF-2426 ~ INF-2447，编号连续。映射：

| 编号 | 来源 | 标题 |
|---|---|---|
| INF-2426 | pytorch#5930 | Inductor dynamic 归约长度固化首次值静默算错 |
| INF-2427 | pytorch#5911 | TE 标量累加器四族代码生成缺陷 |
| INF-2428 | pytorch#5924 | NPU argmax/argmin 全 NaN 归约返回错误索引 |
| INF-2429 | pytorch#5925 | AOT i64 参数按 int32 打包参数块错位 |
| INF-2430 | pytorch#5920 | A5 static launcher 重复执行 in-place kernel |
| INF-2431 | ops-math#4595 | ReplicationPad 反传 DOUBLE 放行结果错 |
| INF-2432 | ops-nn#8346 | WQBMM PerGroup ND 路径静默算错 |
| INF-2433 | ops-nn#8322 | PadConv2dFusionPass 融合后 SAME 覆盖显式 pads |
| INF-2434 | ops-nn#8304/+8309/+8310 | LayerNorm 族 eps<0 放行静默全 NaN |
| INF-2435 | ops-math#4577 | SliceLastDim srcStride 窄化静默错数据 |
| INF-2436 | ops-nn#8338 | ConvertWeightToINT4Pack 后组覆盖前组 |
| INF-2437 | ops-transformer#7222 | MoeGatingTopKSoftmax 输出校验漏末维==k |
| INF-2438 | vllm-ascend#21157 | 原生 FP8 在 A2/A3 静默以 BF16 运行 |
| INF-2439 | ops-transformer#7210 | ScatterPaKvCache PA_NZ 越界写 |
| INF-2440 | ops-nn#8180 | AdaptiveMaxPool2d NHWC 输出错位 |
| INF-2441 | ops-nn#8209 | GroupNormSiluQuant rstd 整块 NaN |
| INF-2442 | ops-transformer#7188 | arch38 FA DN pitch 与 halfS1 计算错误 |
| INF-2443 | vllm-ascend#21164 | AscendStore decode PP 键扩展静默不生效 |
| INF-2444 | vllm-ascend#21166 | hybrid/layerwise 截断后 block_hashes 未重建 |
| INF-2445 | ops-transformer#7172 | FA arch35 hasRope 偏移错误精度异常 |
| INF-2446 | vllm#66141 | ROCM_AITER_MLA gqa16 fp8 qlen3-4 静默算错 |
| INF-2447 | sglang#48964 | sglang 长请求后引擎状态损坏后续全退化 |

- var/cross 关系：本轮无（同机制判定：#8790→INF-2108 走 update 补证据；#5924 与 INF-2412 缺陷位置不同分立两号）。
- 回访 update：INF-2108（补证据 #8790+注记）、INF-2403/2408/2412/2414/2417/2419/2382/2259 共 9 条注记。
- 训练库跨库查重：22 个拟收 URL 在 training_error_review `find` 全部 0 命中，无 cross 关系。
- `catalog_tool.py stats`：**2447 条**（active 2425 + pending_audit 22）；`build` 完成；`validate` ok:true（n_errors 0；n_warnings 62——均为 pending_audit 常规 warning，无存量新错误）。

## 5. 未覆盖范围与 API 账本

- **GitCode/AtomGit issue 评论线程未拉取**（无低成本批量入口），全部按正文判定；本轮收录候选正文均含自包含复现/源码定位/对照实验，依赖评论改判概率低。
- **vllm-ascend 窗口 ~102 条 PR、ops-nn ~78 条 PR、pytorch ~66 条 PR** 等按标题过滤，修复流正文未逐条深读。
- **GitHub 未认证 search**：21/21 组 200（含昨日失败的 silent wrong output 词组组，本轮 URL 编码修复）；正文单点补拉 5 条（#66141/#48964/#8790 detail+2 组查询条目）。评论线程未拉。
- **留观未深读正文项**（下轮优先）：vllm#66144（EPLB 静默卡住）、tt-metal #64678（moe_fused_swiglu 竞争）、sglang#48967（AMD MLA NaN）、ops-nn #8218（BlockSparseAttention 全 0 mask 未写输出）、op-plugin #688（atb FA bf16 alibi compress FAIL）。
- **arXiv** 仅 2 组主题词（SDC/bit-flip），KV corruption/数值正确性主题词未单独跑（与既往结果集重叠度高）。
- 本轮无任何候选因限流/配额只按索引判定；全部收录/留观判定均基于已读正文或批次缓存正文。

## 6. 结论

22 条新机制入库（待审计），10 条既有条目回访注记（含 INF-2108 补 MLA 残留路径证据）。本轮覆盖：NPU 22/22 源完整批次（704 条窗口内新建、254 条 issue 标题全过目、~26 条深读）+ GitHub 7 仓 3 关键词（21/21 组 200、2 新收）+ arXiv 2 组（无新论文）。重点新面：**torch_npu 编译器静默错值密集爆发**（dynamic 归约长度固化、A5 static launcher 重复缩放、TE 标量累加器四族、argmax NaN 索引、AOT i64 打包错位——五条全部静默错值域）；**量化链路三连**（WQBMM PerGroup ND NRMS 0.19、INT4Pack 组间覆盖、FP8 静默 BF16 降级）；**KV/状态污染两条**（AscendStore PP 键扩展失效、mooncake stale block_hashes）；昨日留观转收 2 条（AdaptiveMaxPool2d NHWC 错位、GroupNormSiluQuant rstd NaN，正文深读后证据密度足）。CANN「空输入早退+输出未写」族与文档缺口族持续高发，继续按族留观。

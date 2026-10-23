# 推理故障每日扫描 2026-10-23（UTC）

- 窗口：GitHub/arXiv `created:>=2026-10-16`（7 日）；NPU 共享批次 `batch-20261023T230043156300-b8ti3f0c`（采集 since 2026-10-22T22:00:43Z / until 2026-10-23T23:00:43Z，22 仓全覆盖、0 failures）。
- 编目：本轮新增 **17 条**（INF-2142 ~ INF-2158，pending_audit），总数 **2158**；validate `"ok": true`（45 warnings 均为既有模式 `待定` 项）。
- Git：前置检查通过（main 干净、HEAD=origin/main）；本轮末提交并推送。

## 一、原 GitHub 7 仓 + arXiv 覆盖账本

- GitHub 未认证 Search API（约 7s 间隔，GITHUB_TOKEN 未配置），15 组查询全部 `created:>=2026-10-16`：vllm×3（82/69/141）、sglang×3（53/33/60）、transformers×2（23/5）、TensorRT-LLM×2（6/6）、triton×2（19/7）、tt-metal×2+续页（193=100+93）、flashinfer×2（35/35）；去重后 586 唯一条目，关键词打分 ≥3 的 171 个强候选逐一比对近 14 天判定。
- 核对：vllm_3 p2 返回 0 条（total 141 与首页 100+41 合计相符，GitHub 搜索去重行为）；tt-metal 193 条全部取回（2 页）。
- arXiv：3 组查询（SDC/bit-flip/KV-corruption/numerical-correctness）共 42 篇，近 7 日无新推理故障机制论文（最新相关为 10-23 Rowhammer latency attack、10-23 Revisiting Cauchy-Schwarz SDC detection——均非新故障编目对象）。

### 新增条目（9 条，pending_audit）

1. **INF-2142** [NEW] [vllm#62664](https://github.com/vllm-project/vllm/issues/62664)（closed）— CPU 后端不校验 `--kv-cache-dtype`：量化格式按 uint8 分配缓存、写核按 bf16 字节步长越界写；~100-token prompt 从首 token 起垃圾输出，无报错；最小复现直接测出 512 字节越界写。
2. **INF-2150** [NEW] [vllm#62969](https://github.com/vllm-project/vllm/issues/62969) — EVS 视频剪枝 chunked prefill 边界落进视频占位符内部时 `recompute_mrope_positions` 写错 M-RoPE 位置，`mrope_position_delta` 漂移全部解码 token；CPU-only 最小复现，与 #62937/#62959/#48833 均不同缺陷。
3. **INF-2147** [NEW] [vllm#63309](https://github.com/vllm-project/vllm/issues/63309) — Hermes 工具解析器流式路径遇正文字面 `<tool_call>` 后静默丢弃全部后续 content（非流式完整）；解析器级 CPU 复现，两条路径结果分歧。
4. **INF-2148** [NEW] [triton#12392](https://github.com/triton-lang/triton/issues/12392)（closed）— sm_89 把 fp8 `acc += tl.dot(a,b)` 静默重写为 `tl.dot(a,b,acc)` 低精度累加：K=20000 全 1 得 16384（-18%）、K=8192 随机偏差至 15%；`max_num_imprecise_acc` 控制组正确。
5. **INF-2149** [NEW] [triton#12394](https://github.com/triton-lang/triton/issues/12394)（closed）— 批量 3D `tl.dot` N==1 且 B>num_warps 时 MMAv2 批间寄存器步进减半、相邻 batch 共享累加器：全 1 输入应 16 处出现 0/32。
6. **INF-2154** [NEW] [flashinfer#6860](https://github.com/flashinfer-ai/flashinfer/issues/6860) + [#6861](https://github.com/flashinfer-ai/flashinfer/issues/6861) — CUTLASS/CuTe monolithic MLA 把末页尾与 padding 页存储 NaN 传入有效输出行（B300 单请求最多 540,672 非有限元素；同输入 FA2 正常）；疑似 0×NaN 机制报告者自明未证实。
7. **INF-2155** [NEW] [tt-metal#61905](https://github.com/tenstorrent/tt-metal/issues/61905) — bfloat8_b max/min 归约末维非 16 倍数时 padding ±inf 污染共享指数块：[1,1,1,1]=6.1875 的 max 返 0；every dim、always。
8. **INF-2156** [NEW] [tt-metal#62052](https://github.com/tenstorrent/tt-metal/issues/62052) — depthwise conv1d（groups=C、无 bias）reader 忽略 dilation：对照 torch 精确参考 30-40% 元素错；GDN 形状 C1024/K4/L12 命中；dilation=1 或带 bias 通用路径正确。
9. **INF-2157** [NEW] [transformers#49810](https://github.com/huggingface/transformers/issues/49810)（closed）+ [transformers#49888](https://github.com/huggingface/transformers/issues/49888)（同批两库两机制）— Gemma2 sdpa（默认）静默丢 `50*tanh(x/50)` soft-capping（#40744 移除警告后无任何提示）。另 **INF-2158** [transformers#49888](https://github.com/huggingface/transformers/issues/49888) — Qwen4-Exp eager QSA 左 padding 批次有效 token NaN logits（随机小模型 CPU 复现，SDPA 正常）。

### 重复判定（沿用近期判定，不新建）

- vllm#62554=INF-1458、#62417=INF-1457（10-22 前已编目）；sglang#44997=INF-1448、#44735=INF-1392、#45000=INF-1453；tt-metal#61268=INF-1443；gitcode Ascend/pytorch#5613=INF-2139、ops-transformer#6347=INF-1781（今日批次复核状态，见回访）。

### 拒绝 / 留观（主要项）

- **tt-metal** #61052/#61078/#61091/#60984/#60978/#60963（LLK 深层契约缺口族，正文为源码级审计式报告、非用户故障路径——LLK 内部 API 家族留观，量控制）；#61698（unary 小 tile 静默 32x32 重塑——与 #61625 族相邻，正文简短留观）；#61796（Quasar binary_ng 8-tile DST 批次硬编码——experimental quasar 域留观）；#61791（transpose 偏移网格错——10-21 已留观维持）；#61649（rotary fused_qk head_dim>256 不拒绝——experimental 域留观）；#61488（TT-NOP injector race——测试基建不占号）。
- **flashinfer** #6995/#7087（CI 数值断言失败，dlqa-triage 机器人报：单一测试参数化失败、1-3 元素差、正在排查——留观等根因）。
- **triton** #12349（两个 jit helper 仅捕获值不同静默跑第一个——调度缓存键缺陷候选强，正文未含用户影响路径，留观）；#12351/#12395（warp_specialize 数据分区丢 store——10-20 已留观族）；#12347（for 循环按循环变量符号扩宽边界——正文简短留观）；#12370（triton_kernels 批量 split-K flexpoint scales——triton_kernels 子项目域留观）。
- **transformers** #49897（default_data_collator NumPy int 标签 cast float32——训练数据管线域）；#49902（GOT-OCR2 NumPy 图像 (h,w)/(w,h) 颠倒——多模态预处理确定性 bug，影响框坐标，候选中等留观下轮）。
- **TensorRT-LLM** #20634（Perf 域）；本轮 12 条搜索命中里 11 条已在 10-18~10-22 判定过。
- **sglang** #45481（TP1/PP2+NEXTN 并发贪心分歧——determinism 配置语义候选，与已收录并发分歧族相邻，正文含对照表留观下轮）；#45622（context-length 静默取模型配置——配置域留观）；#45563（HiCache NIXL DSA sidecar 0 字节——与 #44735 相邻留观）；#45697（Kimi K3 工具参数流式缓冲——输出解析域留观）。
- **vllm** #63276（DSV4.1 Flash1017 数字复述漂移——仅用户对话截图、无配置/复现，机制不明留观；19624 vllm-ascend 同型留观）；#63363（RFC 不占号）；#62703（PNG 损坏 500——fail-loud 前端）；#62080（NIXL TCP 回退——性能域）；#63037（KV offload 元数据丢失候选中等，与 offload 族相邻留观）；#62865（compile mode 2 Transformers 后端 kernel 错——torch.compile 域候选中等留观）；#62340（LoRA+MTP 启动崩溃——fail-loud）。
- **vllm-ascend** #19650（200 并发 5h 后请求挂起——挂起域，无输出错误证据）；#19636（>16MB 多模态 payload 楔死——挂起域且源码级闭环定位为上游 shm_broadcast，作为可用性域留观）；#19624（closed，curl 输出重复循环——正文仅环境信息+截图，机制未定位）。

## 二、NPU 来源扫描（batch-20261023T230043156300-b8ti3f0c）

批次摘要：22/22 仓 complete、0 failures、16090 候选（inference scope）。采集窗口 2026-10-22T22:00Z→2026-10-23T23:00Z。GitCode 系按 issues+pulls 端点分页全取（如 ops-nn 498 条 issues 2 页+pulls 4 页、pytorch 249 条 issues 1 页+pulls 2 页），GitHub 系 3 仓（vllm-ascend 356、triton-ascend 93、sgl-kernel-npu 9）全页校验。批次缓存含逐条 body（bodies/*.md），本轮判定未额外消耗平台 API。

### 筛选过程

采集器 keyword 标记 8978 条；标题强信号（精度/静默/错值/NaN/mismatch 等）1180 条；对照近 14 天日报 URL 查重后 1101 条未判定（436 条创建于近 7 日）；按主题分布（数值/精度回归 586、sanitizer 修护 193、量化 105、NaN/Inf 100、KV/attention 81、随机/seed 65）逐仓抽读正文，优先静默/延迟可见正确性缺陷。

### 新增条目（8 条，pending_audit）

1. **INF-2143** [NEW] [op-plugin#620](https://gitcode.com/Ascend/op-plugin/issues/620) — gather/scatter/embedding/index_select 越界索引静默接受：越界读返回数据、越界写静默丢弃、index_select 错误路由 MaskedSelect 报 507035；CPU 全部报错。A3/CANN 9.0.0。
2. **INF-2151** [NEW] [op-plugin#628](https://gitcode.com/Ascend/op-plugin/issues/628) — 整数 div 除零静默返回 INT64 极值垃圾、remainder 原样返回被除数、fmod 整数除零触发设备级 507018（910B3 进程崩溃 ERR99999，A3 毒化上下文级联 507035）；双机双版本一致复现。
3. **INF-2144** [NEW] [ops-math#3859](https://gitcode.com/cann/ops-math/issues/3859) — aclnnDiv 整型一律提升 FP32 计算：|x|>2^24 静默丢位（1073741825/1=1073741824），int64 大数误差 1e8 量级；源码级定位 PromoteIntegerInputsToFloat。
4. **INF-2152** [NEW] [ops-nn#7234](https://gitcode.com/cann/ops-nn/issues/7234) — aclnnEinsum "abcd,abced->abce" 整型 Cast FP16 计算：>2048 失精、>65504 溢出饱和 INT64_MAX；声明支持 INT16~UINT64。
5. **INF-2145** [NEW] [triton-ascend#2662](https://gitcode.com/triton-lang/triton-ascend/issues/2662) — num_stages>1 软件流水错排 loop-carried store：Mamba-2 SSD 前向/反向 25-46% 新进程 bit 级相同错值；进程内恒定、跨进程翻转；pin num_stages=1 全消失。
6. **INF-2146** [NEW] [ops-nn#7007](https://gitcode.com/cann/ops-nn/issues/7007) + 修复 [!12590](https://gitcode.com/cann/ops-nn/merge_requests/12590)（open）— ThnnFusedLstmCell host 侧全局可变状态并发互覆：输出失配/561113/段错误，单线程正常；与 INF-2098 同属 host 全局状态家族但机制不同（并发互覆 vs 跨调用残留），var 关联。
7. **INF-2153** [NEW] [ops-transformer#6533](https://gitcode.com/cann/ops-transformer/issues/6533) — MoeDistributeDispatch V2/V3/V4 不校验 commAlg：枚举外取值与不支持组合静默按默认算法执行，2-rank 探针复现。
8. 以上 NPU 条目均在训练库查重无同 URL 命中；Ascend/pytorch#5613（npu_format_cast_ aclnn 误用 self）本轮判定为已编目 INF-2139（10-22），沿用。

### 拒绝 / 留观（NPU 主要项）

- **ops-nn 校验缺失族**：#7047（NLLLoss target 越界静默 -0.0）、#7039（DeepNormGrad mean/rstd 形状不校验）、#7051/#7053（aclnnMv 空 self dtype/USE_HF32+BF16 不报 161002）、#7167（KlDivBackward dtype 不一致+reduction 越界回退 none）、#7173（SwiGluGrad dim 非法/yGrad 未减半）、#7237（Matmul BF16×FP16 混 dtype 不报）——均为「非法入参静默接受」同型，与已编目 #6609（KeyPool rotaryMode）等构成大族；本轮收录其中任务影响最直接的 4 条机制（索引越界读、除零垃圾、整型精度、einsum 饱和），其余族成员留观防编号膨胀，后续按族并条评估。
- **#7177/#7178**（MaxUnpool2d/3dBackward 越界 indices 读显存外部、同输入三次结果不同）——训练反传域为主（gradOutput），推理库视角降级留观；机制证据强，训练库如未收录可转评。
- **ops-math#3846**（Multinomial 空 probs+replacement 静默 SUCCESS 不写 out）——与已收录 INF-2126（AdaptiveAvgPool2d 空输入同型）机制同族：early-return 空输入不写非空 out；按「同机制优先补证据」用 update 补入 INF-2126 证据。
- **#6841**（gelu approximate='none' 被按 tanh 计算）——与已编目 INF-2101（op-plugin#616 gelu 丢 approximate）同一用户可见症状但方向相反（那个 erf 被算成 tanh 的镜像：tanh 被算成 erf）、不同仓不同层（ops-nn kernel 无 erf 分支 vs op-plugin 分发丢 attr）；两者共用 gelu 语义漂移主题但机制独立，今日先补证据留观，待审计定是否 var。
- **vllm-ascend #19636/#19624/#19650**：见上节留观理由（挂起域/机制未定位）。
- **hccl#838/#3692、runtime/ge uninstall.sh 系、ops-nn#12390 infershape pass 系、msprobe/mssanitizer 工具报**：工具链/构建/测试基建域不占号。

### 回访（状态变化）

- **INF-1767**（ops-transformer#6344 FlashAttn DN 多写）：修复 MR !14216 已 **merged**（10-21）；issue 关闭原因未见注释。
- **INF-2116**（vllm-ascend#19357 Sinksink epsilon）：issue 与修复 PR #19401 均 open，无变化。
- **INF-2098**（ops-nn#7006 AddRmsNormQuant 全局残留）：issue 已 closed（关闭原因未见注释）。
- **INF-2110**（ops-nn#7143 Median radix NaN 误计）：issue 已 closed（修复 PR 合入状态未核实）。
- **INF-1781**（ops-transformer#6347 BlockSparse scale=0）：issue 已 closed（10-22）。
- **INF-2138**（ops-nn#7225 swiglu_mode 静默 mode0）：修复 MR !12598 仍 open。
- 其余 10-21/10-22 NPU 条目（INF-2099/2100/2101/2102/2103/2104/2105/2106/2107/2111/2120/2126/2136）在批次中状态均无变化（open）。

## 三、未覆盖范围

- NPU 批次 1101 条未判定候选中，本轮深读约 40 条正文（静默信号优先）；其余 1000+ 为 sanitizer 修护/构建/文档/perf/fail-loud 类标题，未逐条读正文——若后续批次出现同型新机制候选，按本日报「族留观」清单对照。
- GitHub 未认证 core API 本轮用尽 60/60 取详情（36 issue 全文）+search 10/10；未再取评论线程（拟收录条目正文自含机制证据，评论级复核留待每日审计）。
- arXiv 42 篇标题级筛完，无新机制；摘要级未逐篇读（标题+查询词已限定 SDC/数值正确性域）。

## 四、编目与视图

- `catalog_tool.py new`×17 全部 ok（3 轮字段长度修正后）；`build` 重新生成 views/（CATALOG.zh.md、fault_catalog.csv、fault_catalog.xlsx）；`validate` ok（n_errors=0，45 warnings 为既有模式标题待定项）；stats：2158 条（故障 2142、结论 8、方法 7、指针 1；active 2141 + pending_audit 17）。

# LLM 推理故障每日扫描报告 — 2026-09-03

- **扫描日期**：2026-09-03（滚动窗口 created:>=2026-09-02，与昨日报告重叠部分已按主文档 issue 编号查重剔除）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 856 条（第 0 层 17 + 第 1 层 725 + 第 2 层 114）
- **结果**：新发现并合并 **26 条**（21 个新子签名 + 5 个已知类别变体）；当日 arxiv 命中均为已编目论文（TrainSDC、SCOUT、LLM-PRISM、Understanding SDC、Exploring SDC、JITterFlip、Bit-Position Sensitivity/UEP、RangeGuard、Rotated Robustness、Targeted Bit-Flip on Agents、BitFlipScope、When Latent Agents Lie、Addressable Memory、Compression-Aware Abstention）或既往已判定不收录的方法论文（Moment-KV、Nexus、Runtime Observability、Shared-Prefix KV for Diffusion LM 等），无新编目。
- **API 事件记录**：GitHub Search API 当日 **10 组查询全部一次成功**（无 Validation Failed、无限流 403；每组间隔 ≥7s）；core REST issue 详情调用 23 次全部成功（全程 `--http1.1`）；arxiv 5 组查询全部成功（A5 GPU soft error 命中 0）。

---

## 一、新编目条目（26 条，已合并进主文档）

### 第 1 类 · 硬件 / SDC（数值内核原语）（+16）

1. **tt-metal #55406** — Quasar: Completion queue wraparound uses the wrong address basis — https://github.com/tenstorrent/tt-metal/issues/55406
   `[新类别/新签名]`（→ 1.146）host 完成队列读指针回写把 dispatch-core L1 地址经 `% completion_tlb_size` 折算 TLB 窗口偏移——多 CQ 下地址按 CQ zone stride 平移，modulo 折叠进错误 L1 地址；设备见陈旧指针、误认为队列仍满（wraparound 附近尤甚）。「地址基换算错误」新子签名。
2. **tt-metal #55391** — ttnn.cbrt (FLOAT32) is wrong for every subnormal input, returning 0 or a value up to 6306x low — https://github.com/tenstorrent/tt-metal/issues/55391
   `[新类别/新签名]`（→ 1.147）Moroz magic-constant Newton 种子假设正常指数域；SFPMAD flush subnormal 时返回 ±0、未 flush 时种子系统性错位（29% subnormal 偏差 >1%）；正常输入 ≤3 ULP——纯域间隙。
3. **tt-metal #55356** — ttnn.lgamma is 474,555 fp32 ULP off at x=0.6 … Stirling's series evaluated at the input with no argument shift — https://github.com/tenstorrent/tt-metal/issues/55356
   `[新类别/新签名]`（→ 1.148）Stirling 渐近级数直接在输入处求值、无参数移位：0.2<x<2 误差数十万 ULP（x=0.6 处 474,555）；误差是 (x−0.5) 的偶函数（诊断指纹）；注释自称 0–3 ULP。
4. **tt-metal #55351** — ttnn.add/subtract/multiply silently return wrong values at rank >= 7 when an operand is broadcast at dim -6 — https://github.com/tenstorrent/tt-metal/issues/55351
   `[新类别/新签名]`（→ 1.149）`extract_nD_dims` 把 -6 及更外维度折叠成单标量计数器——rank≥7 + dim -6 广播时读指针结构被压平、4/6 tile 读过缓冲末端；同 shape 降一秩正确、无广播正确、dim -7 被 TT_FATAL 拒（验证自己的许可规则）。
5. **tt-metal #55347** — ttnn.rms_norm on a rank-0 tensor computes x / |x|: epsilon is dropped — https://github.com/tenstorrent/tt-metal/issues/55347
   `[新类别/新签名]`（→ 1.150）rank-0 显式分支 `x/|x|`：epsilon 从未被读取——x=0 NaN、|x|<sqrt(eps) 恰 ±1；「参数被声明但某分支不消费」族。
6. **tt-metal #55346** — ttnn.argmax: FLOAT32 never selects a NaN and BFLOAT16 lets a later +0 beat an earlier -0 — https://github.com/tenstorrent/tt-metal/issues/55346
   `[新类别/新签名]`（→ 1.151）两个位型比较器 NaN/±0 语义互相不一致且都异于 torch：fp32 永不选中 NaN、bf16 符号位优先使 +0>−0——**同输入两 dtype 返回不同索引**；四个 reader kernel 全部继承。
7. **tt-metal #55320** — ttnn.xlogy(0, y) returns NaN for y <= 0 where the convention is 0 — https://github.com/tenstorrent/tt-metal/issues/55320
   `[新类别/新签名]`（→ 1.152）NaN 守卫只检查第二操作数——`x==0` 永无短路：`xlogy(0,0)` 走 `log(0)=-inf`×0=NaN、`xlogy(0,-1)` 直接命中 NaN 分支；单元素把整个 loss 变 NaN。
8. **tt-metal #55316** — ttnn.xlogy_bw and ttnn.addcdiv_bw hard-code the sign of the infinity at a zero divisor — https://github.com/tenstorrent/tt-metal/issues/55316
   `[新类别/新签名]`（→ 1.153）零除数处写常量符号 ±inf（忽略完整乘积符号、无 0×inf NaN 臂）；同文件 `div_bw` 已正确——**修复未对齐到同族兄弟 op**（1.122 的同文件镜像）。
9. **tt-metal #55294** — ttnn.div and div_no_nan return NaN for every subnormal denominator; div(x,x) is NaN instead of 1 — https://github.com/tenstorrent/tt-metal/issues/55294
   `[新类别/新签名]`（→ 1.154）subnormal 分母一律 NaN（`div_no_nan` 违反名字契约）；#52094（大端返回 0）的小指数镜像，PR #52340 只修大端且只修 WH；边界精确落在 2^-126/2^-127。
10. **tt-metal #55291** — ttnn.hardswish_bw takes the wrong branch at exactly x = ±3 and uses 0.3333 for 1/3 — https://github.com/tenstorrent/tt-metal/issues/55291
    `[新类别/新签名]`（→ 1.155）两处边界比较相对 torch 全部反向（−3 用 `<` 应 `<=`、+3 用 `<=` 应 `<`）——x=−3 梯度符号都错；斜率常数 0.3333f 全域损 336–2097 ULP；nightly uniform+PCC 盲区放行。
11. **tt-metal #55290** — ttnn.atanh_bw returns NaN for every element whose upstream gradient is exactly zero — https://github.com/tenstorrent/tt-metal/issues/55290
    `[新类别/新签名]`（→ 1.156）NaN 注入守卫只看 grad 不看极点条件——开区间 0<|x|<1 内 grad=0 一律 NaN；零上游梯度（mask/dropout/padding）是常客，一个零毒化整个 backward。
12. **tt-metal #55289** — ttnn.rpow_bw computes the gradient of x**base instead of base**x; NaN for every negative input — https://github.com/tenstorrent/tt-metal/issues/55289
    `[新类别/新签名]`（→ 1.157）对 x 求了导（应 `base**x*ln(base)`）+ 负输入全域 NaN；**注册 backward golden 复刻同一误读**——错误 oracle，对拍测试原理上无法发现（测试失效的更深层形态）。
13. **tt-metal #55298** — ttnn.unary_chain silently drops every op after the first when the head op uses a specialised kernel — https://github.com/tenstorrent/tt-metal/issues/55298
    `[新类别/新签名]`（→ 1.158）kernel 选择只看链头；六个特化 kernel 无 `SFPU_OP_CHAIN_0` 展开点——`hardswish→square` 返回未平方结果（负值）；「分派路径截断算子链」新子签名。
14. **tt-metal #55304 / #55305 / #55311** — dtype= silently ignored（div 全 dispatch nullopt；global_avg_pool2d 快路径；pow/maximum/minimum/gcd/lcm/bias_gelu 九签名） — https://github.com/tenstorrent/tt-metal/issues/55304
    `[新类别/新签名]`（→ 1.159，三 issue 合并）九个签名接受 `dtype=` 静默丢弃（含刚通过 TT_FATAL 校验该参数的分支）；同文件 add/subtract/multiply 正确转发；global_avg_pool2d 连显式支持的 bfloat8_b 也丢而普通路径同参数大声拒——「参数被接受但不被消费」跨 op 族。
15. **tt-metal #55414** — eltwise/binary_ng: program-cache key incorrectly omits Alignment and Tile — https://github.com/tenstorrent/tt-metal/issues/55414
    `[新类别/新签名]`（→ 1.160）hash 键漏 Alignment/Tile——两个仅 layout 不同的张量共享 cache 条目；且 relaxation 从不解放 layout。与 6.141（vLLM compile cache key 漏 spec tokens）跨栈同构：cache key 未覆盖全部语义输入。
16. **tt-metal #55184** — [TT-NOP Injector][BH] mismatch race on hadamard — https://github.com/tenstorrent/tt-metal/issues/55184
    `[已知类别]`（→ 1.161，var:1.99）NOP 注入器时序扰动在 BH hadamard h128 触发 **mismatch（错数据）**：同一 unpack:2 STALLWAIT 跨 fidelity/normalize/dest_sync/dest_acc 全参数面 15 组合失败——竞态族第五成员、首次系统性参数面扫描、显形面从 hang 扩展到错数据。

### 第 6 类 · 服务栈 / 缓存正确性（+5）

17. **vLLM #54914** — [Bug]: KV Cache Offloading AssertionError after several days of uptime with DeepSeek-V4-Pro on 3×8 H100 (TP8+DP3) — https://github.com/vllm-project/vllm/issues/54914
    `[已知类别]`（→ 6.184，var:6.157）数天 uptime 后 offload `prepare_load` 的 CPU cache 查不到 block key——assert 失败 → EngineDeadError、DP rank 服务死亡；24×H100 生产环境（fp8 KV + dspark k=7 + OffloadingConnector 512 GiB）。
18. **vLLM #55221（缺陷 2）** — GLM-5.3-Flash: the indexer prefill workspace is sized in tokens, but its KV is pool-granular — https://github.com/vllm-project/vllm/issues/55221
    `[已知类别]`（→ 6.185，var:6.173）indexer prefill workspace 按 token 数申请而 KV 是 pool 粒度（compress_ratio==index_kpool）——deepseek_v4 同调用点已除、GLM 路径漏除：1M 上下文 per-GPU 多锁 ~2.6 GiB。
19. **vLLM #55132** — [Bug][ROCm] Sparse-MLA indexer reserves a 32 GiB decode-logits workspace during profiling — https://github.com/vllm-project/vllm/issues/55132
    `[已知类别]`（→ 6.186，var:6.173）profiling run 按 `max_num_batched_tokens`（prefill 上界）申请 decode-logits workspace 并被 `WorkspaceManager.lock()` 终身冻结——decode 实际只需 256 行却保留 8192（32×）：32 GiB/GPU、21% KV cache。
20. **SGLang #37834** — --enable-linear-replayssm forces no_buffer, which degrades mamba prefix caching and inflates TTFT up to 4.7x — https://github.com/sgl-project/sglang/issues/37834
    `[新类别/新签名]`（→ 6.187）ReplaySSM 强制 no_buffer 策略把 mamba checkpoint 存到后续请求够不到的 radix 深度：rag 前缀 hit 88.2%→65.9%、TTFT +369%；fewshot hit →0.0%、TTFT +92%；吞吐不变——「特性互斥静默降级」性能侧新签名。
21. **SGLang #37852** — sglang:kv_cache_memory_usage_gb reports 0 for DeepSeek-V4 pools — https://github.com/sgl-project/sglang/issues/37852
    `[新类别/新签名]`（→ 6.188）DSV4 族 KV 池从不调 `_finalize_allocation_log()`、顶层不聚合子池——`mem_usage` 继承初始 0 直达指标：KV 内存监控/告警/容量规划全部失明。「遥测对新型池拓扑失效」新子签名。

### 第 8 类 · KV 缓存 / 注意力机制故障（+2）

22. **vLLM #55221（缺陷 1）** — --kv-cache-dtype fp8 cannot start on FLASHINFER_MLA_SPARSE_SM90（storage dtype uint8 传给 plan()） — https://github.com/vllm-project/vllm/issues/55221
    `[新类别/新签名]`（→ 8.62）fp8 KV 以 uint8 storage 分配、builder 把 storage dtype 传 wrapper——`ValueError: MLA kv_data_type torch.uint8 is not supported` 启动即死；放弃 fp8 KV 损失 1.78× 容量。
23. **vLLM #55048** — encoder attention rejects quantized KV cache only at request time — https://github.com/vllm-project/vllm/issues/55048
    `[新类别/新签名]`（→ 8.63）embedding 模型 + fp8 KV：服务成功启动、首个请求才 NotImplementedError——能力组合检查不在 preflight；伴随 #55046（后端选择期 ValueError）/#55049（构建期 assert）构成「错误显形位置随机分布」光谱。

### 第 10 类 · 分词器 / 对话模板故障（+2）

24. **vLLM #55080** — [Bug]: tool_choice='none' silently deletes tool-call-shaped content — https://github.com/vllm-project/vllm/issues/55080
    `[已知类别]`（→ 10.46，var:10.42）`DelegatingParser` engine-based 分支在 tool_choice=none 时返回解析存活物而非原文——tool-call 形状 content 变 None；9 个 parser 受影响（全部 `engine_based_streaming=True`）；CPU tokenizer stub 可复现。
25. **SGLang #37707** — pattern + minLength: xgrammar silently drops the length bound — https://github.com/sgl-project/sglang/issues/37707
    `[已知类别]`（→ 10.47，var:10.40）xgrammar 下 pattern+minLength 同时出现丢长度界（接受违反 minLength 的串）；outlines 后端丢 pattern+数值界——两个后端各丢不同约束子集；fuzzer 发现。

### 第 12 类 · 权重 / 适配器 / 检查点加载故障（+1）

26. **vLLM #55193** — [Bug]: LoRA adapter targeting no configured modules silently serves the base model — https://github.com/vllm-project/vllm/issues/55193
    `[已知类别]`（→ 12.30，var:12.28）校验（worker 侧不看 target_modules）与应用（manager 侧加 is_in_target_modules）用不同过滤器——MLP-only adapter 在 `--lora-target-modules qkv_proj` 下通过校验、激活成功、广告为 served model，但 set_lora 零层命中：每请求 HTTP 200 返回**基模型原样输出**。fuzzer 发现。

### 补充来源（不占新编号）

- **vLLM #55008** — CUTLASS FP8 linear kernel 在 Ampere 上被自动选择而 SM80 dispatch 是 INT8-only（`cutlass_scaled_mm_sm80_epilogue` RuntimeError）——7.53（2026-08-22 编目 `vllm_cutlass_fp8_kernel_pre_sm89_ungated_autoselect`）的 **SM80 实测显形 + 32 卡 TP16+PP2 大规模部署面**：7.53 当时编目的是 pre-SM89 门槛缺失与 PR #53376 修复，本条给出 A100 上 deterministic crash 的完整栈与 marlin workaround——作为 7.53 的补充来源引用（缺陷同一：`is_supported()` 不探测算力）。
- **tt-metal #55394** — ccl reduce_scatter_minimal_async line-topology perf 测试 300s 超时（comp_pcc 32-46s）——性能 CI 超时、无正确性机制细节，不编目。

---

## 二、查重后跳过/仅记录不编目的条目

- **SGLang #37606**（窗口内命中）— 已编目（→ 6.183，2026-09-02 合并），跳过。
- **tt-metal #55265 / #55266 / #55269 / #55272 / #55273 / #55276 / #55227 / #55130**（窗口内命中）— 已编目（→ 1.138 / 1.139 / 1.141 / 1.144 / 1.140 / 1.142 / 1.143 / 1.137，2026-09-02 合并），跳过。
- **tt-metal #55180**（窗口内命中）— 已编目（→ 1.145），跳过。
- **triton #11519**（7 天窗口命中）— 已编目（→ 1.127，2026-08-30 合并），跳过。
- **vLLM #55196** — fp8 KV cache 对 Mamba/GDN hybrid 模型内存收益甚微（RFC 性质讨论，非正确性故障）——不编目。
- **vLLM #55125** — `--cc.<path>=<value>` 静默拒绝只有 `-cc` 短别名可用（CLI 参数解析便利性：dotted-rewrite 只注册短别名，`--cc` 报 unrecognized arguments 而非提示 `-cc`）——fail-loud 且纯 DX 问题，不编目。
- **vLLM #55189** — VLLM_USE_MODELSCOPE 路径转换用旧缓存布局 → 多模态 processor 加载 HFValidationError crash——fail-loud crash、单机加载路径（无静默面），信息价值低，不编目；若后续出现静默形态再编。
- **TensorRT-LLM（r7）** — correctness 关键词命中 0 条。
- **transformers（r4）** — 窗口命中 0 条。
- **arxiv 全部命中** — 已编目论文（见报告头）或不收录的方法论文，无新增。

---

## 三、重点新发现摘要

1. **tt-metal backward 算子群发（1.153–1.157，五连）** — 当日最重要集群：xlogy_bw/addcdiv_bw 常量符号 inf、hardswish_bw 边界反置+舍入常数、atanh_bw 零梯度全域 NaN、rpow_bw 对错误变量求导——**错误全部位于 backward 复合算子**且三例共享同一测试盲区（uniform 采样 + PCC 忽略 NaN lane / 边界值推不动 PCC）。其中 **rpow_bw 的注册 golden 复刻同一误读**（1.157）：错误 oracle 使对拍测试原理上失效——比 PCC 盲区更深层的测试基础设施失效形态，编目首例。
2. **「参数被接受但不被消费」跨 op 族成型（1.150 + 1.159）** — rms_norm rank-0 丢 epsilon 与九签名丢 dtype= 同日出现：参数被声明、部分分支被校验、无一分支被应用（div 在**刚通过 TT_FATAL 校验 dtype 的分支**传 nullopt）。epsilon/dtype 两参数、eltwise/binary/pool/norm 四域——系统性的签名-实现脱节。
3. **subnormal 域间隙双子（1.147 + 1.154）** — cbrt magic-seed 与 div reciprocal 同日在 subnormal 域失效（返回 0/6306× 低与 NaN）；div 是 #52094 大端缺陷的小指数镜像（修复只修了大端且只修 WH）——**修复的域覆盖不完整**作为缺陷源。
4. **vLLM LoRA 静默服务基模（12.30）** — 校验与应用过滤器脱节使 adapter「成功加载、零层命中」：HTTP 200 返回基模型输出，与基模型逐字相同——对依赖 adapter 定制的生产部署是**无信号正确性黑洞**（fuzzer 发现，A/B 对照中 B 臂与基模型 identical）。
5. **指标遥测黑洞（6.188）** — DSV4 KV 池 mem_usage 恒 0：不 crash、推理正确，但监控/告警/容量规划全部失明——「检测器看不见故障」之外的第三类盲区：**检测器本身对新型拓扑失效**（对依赖该指标做 SDC/泄漏检测的部署是结构性盲点）。
6. **能力协商时序光谱（8.62 + 8.63）** — 量化 KV 与后端能力错配的显形位置横跨 启动期（fp8 KV sparse MLA，好）→ 构建期（#55049）→ 选择期（#55046）→ 请求期（encoder 量化 KV，差）——同域缺陷的 fail-loud 位置决定用户代价。
7. **indexer workspace 容量错配双栈（6.185 + 6.186）** — NVIDIA 侧单位错（token vs pool granule，姊妹模型路径已有修复未对齐）+ AMD 侧 bound 错（prefill 上界申请 decode workspace 且被 lock 冻结，32 GiB/GPU、21% KV）——同一模型族（sparse-MLA indexer）在两栈上各错一次。

---

## 四、API 调用记录

| # | 查询 | 命中 | 状态 |
|---|---|---|---|
| 1 | vllm correctness/wrong/corruption/silent/NaN/garbage in:title（created:>=2026-09-02） | 3 | OK |
| 2 | vllm quantization/fp8/deterministic/speculative in:title | 3 | OK |
| 3 | sglang correctness/wrong/garbage/NaN in:title | 1 | OK |
| 4 | transformers wrong/NaN/rope/tokenizer/incorrect/corruption in:title | 0 | OK |
| 5 | tt-metal numerical/pcc/wrong/NaN/corruption in:title | 18 | OK |
| 6 | tt-metal incorrect/silent/bitexact/mismatch in:title | 13 | OK |
| 7 | TensorRT-LLM correctness/wrong/NaN/corruption in:title | 0 | OK |
| 8 | triton wrong/incorrect/NaN/miscompile in:title（7 天窗口） | 1 | OK |
| 9 | vllm cache/hang/deadlock/prefix in:title | 9 | OK |
| 10 | sglang silent/corruption/incorrect/corrupted/cache in:title | 3 | OK |
| A1–A5 | arxiv: SDC+LLM / bit flip+LLM inference / KV cache+corruption / LLM inference+correctness / GPU soft error+learning | 5/5/5/5/0 | OK |

core REST issue 详情调用 23 次全部成功（`--http1.1`）；无 rate-limit 403、无 Validation Failed 事件。Search API 用量 10/10（当日限额内）。

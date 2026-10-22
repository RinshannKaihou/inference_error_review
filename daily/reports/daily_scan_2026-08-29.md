# LLM 推理故障每日扫描报告 — 2026-08-29

- **扫描日期**：2026-08-29（滚动窗口 created:>=2026-08-28，与昨日报告重叠部分已按主文档 issue 编号查重剔除）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 764 条（第 0 层 17 + 第 1 层 633 + 第 2 层 114）
- **结果**：新发现并合并 **25 条**（20 个新子签名 + 5 个已知类别变体）；昨日已编目条目（#54193/#36886/#36889/#54189/#36822/#36807/#54206/#54252/#54273/#54197/#48371）今日再现均查重跳过；#36840 维持昨日跳过结论。arxiv 无新编目论文。

---

## 一、新编目条目（25 条，已合并进主文档）

### 第 1 类 · 硬件 / SDC（数值内核原语）（+7）

1. **tt-metal #54813** — KDA `prepare_chunk_recurrence` returns a wrong `t_inv` at real decay magnitudes, breaking 68 of 69 Kimi-K3 KDA layers — https://github.com/tenstorrent/tt-metal/issues/54813
   `[新类别/新签名]`（→ 1.118）UT 变换求逆从 Horner 换成倍增展开后，中间量显式幂 N²..N¹⁶ 在真实 Kimi-K3 chunk（‖N‖∞ 中位 2.4、达 17）上 O(1e2) 而待求逆 ≤1——灾难性相消吃掉全部有效位：68/69 KDA 层输出与自身 torch 参考**不相关**（PCC −0.005），唯一幸存的 layer 0 恰是 N 下溢为 0 的层。fp32 精确恒等式在 bf16 舍位中间量上误差放大 4 个数量级；设备端无 compute config 可救。同日 #54839 记录让该缺陷穿透全部既有测试的 5 个覆盖盲区。
2. **tt-metal #54798** — MatmulMultiCoreReuse silently drops output blocks when num_output_blocks > num_cores — https://github.com/tenstorrent/tt-metal/issues/54798
   `[新类别/新签名]`（→ 1.119）每核一块的隐式契约 × 过订阅无校验：128 输出块 vs 110 核 → 过剩块**永不被计算**，输出尾部保持 buffer 原有内容，无错误无警告；校验函数对 interleaved 输入提前 return。**更险**：先跑过的 matmul 把正确值留在 buffer 里使过订阅 config 后跑"通过"——失败被运行顺序掩盖成间歇假象。
3. **tt-metal #54770** — ttnn.celu: exponent-field wraparound returns finite garbage (and the wrong sign) where the golden is -inf — https://github.com/tenstorrent/tt-metal/issues/54770
   `[已知类别]`（→ 1.120，var:1.114）expm1 单向 clamp + setexp 只写指数域低 8 位：负 alpha 时返回 +1.0 / −6.24e9 而 golden 为 −inf——**符号相反的有限垃圾**。与 1.114（exp2 上溢回绕）同一「单向 clamp + setexp 截断」构造模式第三次出现。
4. **tt-metal #54708** — sdpa_decode: mask seed tiles are scalar stores, then NoC-read with no store drain — https://github.com/tenstorrent/tt-metal/issues/54708
   `[已知类别]`（→ 1.121，var:1.9）非零 mask seed 的 512 次/tile 标量 store 后立即 NoC 回读、无 drain——RISC 核与 NoC 无程序序保证；零路径有 barrier、非零路径没有。与上游 #50373→PR #50374 已修的同构竞态完全一致（correctness-adjacent）。
5. **tt-metal #54857** — Quasar softplus still takes the linear branch at beta*x == threshold; the Wormhole/Blackhole fix did not propagate — https://github.com/tenstorrent/tt-metal/issues/54857
   `[新类别/新签名]`（→ 1.122）**修复未跨 arch 后端传播**：WH/BH 已修的 threshold 边界分支，Quasar 移植未同步——已修复缺陷经后端移植缺口复活。
6. **tt-metal #54847** — ttnn.floor_div off by one on exactly-divisible inputs (host reciprocal rounds down before floor) — https://github.com/tenstorrent/tt-metal/issues/54847
   `[新类别/新签名]`（→ 1.123）host 预计算 1.0f/alpha 舍入方向使精确商越过整数边界——**恰好整除输入上静默差一**（索引/分桶最坏情形：边界必错、非边界正确）。
7. **TensorRT-LLM #18338** — Blackwell CuTe DSL GVR top-k decode kernel corrupts output / crashes (barrier-divergence race) on all-zeros pre_idx and tie-heavy rows — https://github.com/NVIDIA/TensorRT-LLM/issues/18338
   `[新类别/新签名]`（→ 1.124）hint 依赖内核的**冷启动零态**（GVR 先验 buffer 首步全零）与 tie 密集行触发 barrier 分歧竞态；完美 hint 也复现（tie 行 gathered hint 值逐位碰撞）。SM100/SM103，cutlass-dsl 三版本均复现。生产首个 decode step 即可达。

### 第 4 类 · 数值 / 精度 / 确定性（+1）

8. **vLLM #54243** — batch-invariant M buckets collapse under `torch.compile` (0.28.0) — https://github.com/vllm-project/vllm/issues/54243
   `[新类别/新签名]`（→ 4.14）`VLLM_BATCH_INVARIANT=1` 的 (arch,N,K,M-bucket) 查表在 Dynamo trace 期被**常量折叠烘焙为 constexpr**，M 仍是运行时 i32——每个编译调用点全区间一个静态 tile，decode M=8 / prefill M=2048 无法各取表项；无 graph break（trace"成功"恰因查找已被折叠）。该模式对一切"按运行时 shape 查表"的配置系统普遍成立。

### 第 5 类 · 分布式 / 并行推理（+2）

9. **SGLang #36879** — GLM-5.3-Flash --enable-dp-attention: MHC communicator missing two should_use_dp_reduce_scatterv() branches — https://github.com/sgl-project/sglang/issues/36879
   `[新类别/新签名]`（→ 5.24）`communicator_mhc.py` 从早于 PR #24785 的快照改编未回映——**顺序依赖缺陷对**：(1) SUM_LEN 轮 MoE partial 永不 combine（每 token 只带 1/4 routed experts → "A: A: A:" 重复坍缩）；(2) 只修 (1) 则 dense MLP 双重归约精确 ×4（流畅输出但 tool-call schema 49/63 vs 372/372）。"只移植一半修复"的静默回归路径已在报告者身上发生。
10. **SGLang #36943** — --enable-symm-mem: all TP schedulers deadlock silently in ncclCommWindowRegister (no timeout) when the symmetric-memory pool grows at runtime — https://github.com/sgl-project/sglang/issues/36943
    `[新类别/新签名]`（→ 5.25）对称内存池**运行时增长**触发 NCCL window 注册路径阻塞 UDS recvmsg（无超时）——全部 TP scheduler 静默 wedge，无 traceback 无 watchdog。活性类 C.6 盲区族。

### 第 6 类 · 服务栈 / 缓存正确性（+9）

11. **vLLM #54317** — GLM-5.3-Flash: recurring CUDA illegal memory access on 4xB200, surfacing in three unrelated kernels — https://github.com/vllm-project/vllm/issues/54317
    `[新类别/新签名]`（→ 6.159）6 小时 5 次 IMA，**同一故障在三个无关内核族显形**（Triton KDA / TileLang MHC / TRT-LLM fused MoE），同 crash 不同 rank 报不同内核——单点上游 corrupter 多内核随机显形；先前 15.5h 无 crash（负载/状态依赖）。stack trace 的显形点≠根因。
12. **vLLM #54225** — FlashInfer attention backend CUDA illegal memory access on sm_120 with NVFP4 + fp8 KV cache; crashes on a 16-token request — https://github.com/vllm-project/vllm/issues/54225
    `[新类别/新签名]`（→ 6.160）单 flag A/B（FLASHINFER 3/3 crash vs TRITON_ATTN 0/2、413 请求 1h45m 干净）；**首个 16-token 请求第一步 decode 即死**；两次 crash 报不同 CUDA 错误码（内存损坏变面显形）。
13. **vLLM #54215** — AsyncLLM streaming session appends later chunk tokens to the caller's prompt list in place — https://github.com/vllm-project/vllm/issues/54215
    `[新类别/新签名]`（→ 6.161）客户端流式处理**原地修改调用方 prompt list**：64-token 会话后变 76；复用同一 list 的后续请求静默携带污染 token（未发送的 888 出现在输出）。跨请求污染发生在引擎边界之外，服务端检查原理上不可见。无权重最小复现。
14. **vLLM #54184** — Engine liveness monitor can SIGTERM a healthy EngineCore after spurious sentinel readiness — https://github.com/vllm-project/vllm/issues/54184
    `[新类别/新签名]`（→ 6.162）看门狗把 sentinel readiness 当死亡证据（不复核 exitcode/心跳）——**健康引擎被自己的监控器杀死**，后续请求 EngineDeadError 且表象与真死亡不可区分。监控面假阳性致死。
15. **SGLang #36859** — Kimi-K3 output abnormal when high pressure — https://github.com/sgl-project/sglang/issues/36859
    `[已知类别]`（→ 6.163，var:6.6 radix_kvcache_cross_contamination）高并发多轮对话下后续轮次**回答其他请求的问题**（任务 000045 答成 000030/000011）；两套正交配置（DCP+hierarchical / EP+marlin，均 fp8 KV）均命中、禁 DCP 仍复现——根因开放，公共层可疑。
16. **vLLM #54281** — VLLM_DEEPEP_V2_ALLOW_HYBRID_MODE declares a default of True but resolves to False — https://github.com/vllm-project/vllm/issues/54281
    `[已知类别]`（→ 6.164，var:6.19）TYPE_CHECKING 声明默认 True、运行时 lambda 读 "0" 实际 False——**声明默认与运行时默认分裂**，多机 EP 静默跑 direct mode 偏离 DeepEP 上游默认；类型检查层审计全部失效。
17. **SGLang #36880** — KDA Triton backend remaps padding slot -1 to the last mamba pool slot, which is allocatable — https://github.com/sgl-project/sglang/issues/36880
    `[新类别/新签名]`（→ 6.165）重映射目标 `shape[0]−1` 注释假设是保留哨兵——实际保留槽在**另一端**（slot 0），尾槽可分配：padded 行读写**活请求的 SSM 状态**，INPLACE_UPDATE 的 `index≥0` 守卫被重映射前置击穿。静默跨请求状态损坏。
18. **SGLang #36935** — [Prefix Cache / Hybrid] Branch reuse silently collapses under LRU pressure — https://github.com/sgl-project/sglang/issues/36935
    `[新类别/新签名]`（→ 6.166）`kv_pool_tokens / chunked_prefill_size > max_mamba_cache_size` 时 mamba 槽 LRU 驱逐破坏 branch 链——多轮/分支命中**静默坍塌**，无日志，被当作预期缓存行为。容量派生约束静默生效族（同 5.23）。
19. **SGLang #36941** — [GB10] Long prefill (>40k tokens) exhausts unified memory and silently kills the worker rank — https://github.com/sgl-project/sglang/issues/36941
    `[新类别/新签名]`（→ 6.167）GB10 统一内存长 prefill **静默杀死 worker——无 traceback、无 OOM 记录**（跨栈对照 54k 通过）。OOM 失去 OOM 签名；与 6.15 构成 GB10「静默死亡」家族第二实例。

### 第 7 类 · 量化（+1）

20. **vLLM #54150** — ModelOpt NVFP4 checkpoints emit invalid UTF-8 byte tokens on SM120; compressed-tensors NVFP4 of the same model is clean — https://github.com/vllm-project/vllm/issues/54150
    `[新类别/新签名]`（→ 7.64）同模型同 kernel 下 ModelOpt NVFP4 发射**无效 UTF-8 token id**（韩文 U+FFFD 62–94/6 vs CT 版 0/6）；`return_token_ids` 离线解码复现排除 detokenizer，greedy 下指向 logits；逐项排除上下文/采样/MTP/PDL/abliteration/ignore。checkpoint 格式是唯一判别变量——与 7.60 同「ModelOpt NVFP4 元数据链」域、CT 同精度干净。

### 第 8 类 · KV 缓存 / 注意力（+2）

21. **vLLM #54359** — [ROCm] GLM-5.3-Flash kpool indexer overwrites its own KV cache: backend advertises a kernel block size the kernels do not use — https://github.com/vllm-project/vllm/issues/54359
    `[新类别/新签名]`（→ 8.56）ROCm 后端声明任意 16 倍数块 → 无 kernel 粒度 block table，但内核按 32-state 页寻址——5× 过订阅使索引器**静默覆写自己的 KV**：4096 pool 条目写进 864 slot（79% 键错误），文本仍流畅、temp=0 不可复现。文档化的 `storage_kernel_block_size` 拆分意图在树中无处实现。
22. **tt-metal #54833** — SDPA reads zero-padded mask columns when Sk % 32 != 0 — https://github.com/tenstorrent/tt-metal/issues/54833
    `[新类别/新签名]`（→ 8.57）显式 attn_mask 经 `ttnn.from_torch` 默认零填充，key 长度非 32 倍数时 SDPA **读取物理 padding 的零值列**（0 被当"允许注意力"而非 −inf）——仅改 `pad_value=-inf` 即 bit-for-bit 恢复。逻辑 mask 值域无关、tile 对齐长度无关。

### 第 9 类 · 位置编码 / RoPE（+2）

23. **SGLang #36762** — Qwen3.5-VL grounding/bbox offset ~90px: fused_qk_gemma_rmsnorm_rope_gate mishandles partial + interleaved MRoPE — https://github.com/sgl-project/sglang/issues/36762
    `[已知类别]`（→ 9.17，var:9.16）逐组件 bisect 到 CUDA 融合内核对 **partial_rotary_factor=0.25 × mrope_interleaved=True** 组合的处理缺陷（位置 id 逐位精确、独立 rope helper 一致、首个 full_attention 层发散跳 10×）；标准 rotary 路径恢复 81.7%/0.5px。**为 9.16（NVFP4 grounding 178px）提供非量化解释**。
24. **transformers #48392** — Nested per-layer `rope_parameters` is misparsed when `layer_types` omits one of its keys — https://github.com/huggingface/transformers/issues/48392
    `[新类别/新签名]`（→ 9.18）短层 config（Gemma3/Olmo3 2 层无 full_attention 层）下嵌套 dict 被当扁平全局 dict：注入 `rope_type`/`rope_theta=None` 且 **save_pretrained 持久化污染 checkpoint**、真实 per-layer yarn/linear 参数**静默从未被校验**；当前推理仍正确。解析结果依赖层数奇偶。

### 第 11 类 · 解码 / 采样（+1）

25. **SGLang #36938** — Prefill input logprobs are served from the wrong request when a batch member is retracted or finished — https://github.com/sgl-project/sglang/issues/36938
    `[新类别/新签名]`（→ 11.56）skip 分支不推进 `logprob_pt` 游标——同 batch 后续请求读到**移位切片 = 别人的 prompt logprobs**；断言只查长度不查 offset、移位恒向后永不越界——无异常。数值域完全合法，蒸馏/评测管道不可检；overlap 调度默认开 + retraction 常态即触发。

## 二、重点新发现摘要

今日最值得注意的五条：

1. **1.118（#54813）倍增求逆的灾难性相消**——数学等价的重构（Horner→doubling）在真实数据幅值（‖N‖∞≈17）下把 bf16 中间量误差放大 4 个数量级，68/69 层输出与参考**不相关**（PCC≈0，比 NaN 更难被有限性检查捕捉）。"fp32 下精确"不是低精度硬件上的正确性证据。
2. **8.56（#54359）能力声明与寻址粒度失配的自我覆写**——后端声明任意 16 倍数块大小 → 跳过 kernel 粒度 block table → 内核按页寻址 5× 过订阅覆写自身 KV。修复判据是 slot 去重计数而非输出质量；文本流畅 + temp=0 不可复现是唯一外部签名。
3. **6.165（#36880）哨兵假设端点错误**——注释假设尾槽保留、分配器实际保留 slot 0：padded 行经 `-1→size` 重映射读写活请求 SSM 状态，且击穿内核 `index≥0` 守卫（重映射发生在守卫之前）。静态代码审计级根因 + 未在生产观测 = 典型"潜伏地雷"。
4. **5.24（#36879）镜像文件快照漂移的顺序依赖缺陷对**——衍生文件未回映上游修复对，第一缺陷（缺 combine）掩蔽第二缺陷（双重归约 ×4）；只修第一个反而引入更隐蔽的流畅但语义错输出。
5. **4.14（#54243）Dynamo 常量折叠吞掉运行时查表**——M-bucket 调参系统对编译路径整体失效且 trace"成功"（无 graph break）。该失败模式对一切"按运行时 shape 查表选配置"的系统普遍成立，值得作为通用反模式编目。

## 三、已编目条目重复出现（未新增）

| 条目 | 对应主文档条目 |
|---|---|
| vLLM #54193 KV offload 未写入末槽投毒（08-28） | 6.157（昨日编目） |
| vLLM #54189 NVFP4 未初始化 activation scale（08-28） | 7.60（昨日编目） |
| SGLang #36886 DCP index-K OOB（08-28） | 6.158（昨日编目） |
| SGLang #36889 DFLASH mamba 并发钳制（08-28） | 5.23（昨日编目） |
| SGLang #36822 GPTQ Marlin MoE NaN（08-28） | 7.62（昨日编目） |
| SGLang #36807 fast_topk_v2 丢候选（08-28） | 8.55（昨日编目） |
| SGLang #36840 TBO unreachable（08-28） | 昨日已评估跳过 |

## 四、跳过（未编目）及理由

- **vLLM #54300 / #54173 / #54179 / #54178 / #54318 / #54219 / #54126 / #54125 / #54170 / #54153 / #54237 / #54199** — fail-loud crash / 启动报错 / checkpoint 加载失败（crash 类为主，无静默损坏窗口）；#54199 为 RETRACTED 重复。
- **SGLang #37052 / #36906 / #37059 / #37022 / #36894 / #36802 / #36855 / #36876 / #36877 / #36830 / #36829 / #36853** — loud crash / hang / 功能不可达 / 性能类。其中 #36876（zombie requests after disconnect，静默容量泄漏）与 #36877（rolling cache_control 使前缀缓存失效，静默抖动）为边界案例，记此备查；#36829（EAGLE3/MTP bring-up 连环 crash，workaround 后 acceptance≈1.0）为 loud bring-up 缺口。
- **triton #11497**（gfx12 fp8 upcast 落软件路径——HW 指令未用，数值正确，纯性能）、**#11494**（autotuner 并发 crash，fail-loud）、**#11512**（NFS 共享 cache 目录并发 crash，infra）。
- **TensorRT-LLM #18333–18337 / #18376 / #18377**（Rubin/CuTe 修复追踪 issue，一行正文，机制细节不可编目；#18335 "sC reuse 前同步 bulk-async 归约"提示同类竞态待观察）。
- **tt-metal #54839**（KDA t_inv 缺陷的 5 个测试覆盖盲区——作为 1.118 的伴生证据在条目内引用，不单列）、**#54842**（mesh poisoning after kill -9 等测试环境陷阱）、**#54835**（Ring 拓扑 deadlock，hang）、**#54697**（L1 放不下 TT_FATAL，loud）、CI 失败追踪类（#54710/#54809/#54815/#54831/#54684）、测试重构类（#54763/#54761/#54757/#54715/#54705）。
- **transformers #48400**（audio-flamingo 加载 crash，loud）、serge triage 系列（#48413/#48403/#48384）。
- **vLLM #54363**（filesystem KV offload 完整性 RFC——设计提案非故障）。

## 五、arxiv 搜索结果（无新编目）

关键词：`silent data corruption LLM` / `bit flip LLM inference` / `KV cache corruption` / `LLM inference correctness` / `GPU soft error inference`。命中（SCOUT 2608.11034、LLM-PRISM 2604.10390、Addressable Memory for Video World Models 2608.07408、Runtime Observability 2608.05863、When Latent Agents Lie 2606.28958、指令级 GPU 软错误 FI 2512.xxxx 等）均已在先前扫描编目或引用；`bit flip LLM inference` 与 `LLM inference correctness` 两组零命中。

## 六、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GH Search: vllm label:bug created:>=2026-08-28 | 13 hits |
| 2 | GH Search: sglang issues created:>=2026-08-28 | 34 hits |
| 3 | GH Search: TensorRT-LLM issues created:>=2026-08-28 | 9 hits |
| 4 | GH Search: triton issues created:>=2026-08-28 | 4 hits |
| 5 | GH Search: tt-metal +pcc created:>=2026-08-28 | 14 hits |
| 6 | GH Search: transformers issues created:>=2026-08-28 | 6 hits |
| 7 | GH Search: tt-metal +numerics | 1 hit |
| 8 | GH Search: vllm +wrong | 13 hits（含 RFC） |
| 9 | GH Search: tt-metal +wrong result | 10 hits |
| 10-31 | GH Issues API 单条详情（#54304/#54243/#54215/#54184/#54281/#54359/#54317/#54225/#54150/#36938/#36762/#36859/#48392/#54813/#54798/#54770/#54833/#18338/#11497/#54708/#36879/#36880/#36829） | OK（间隔 ≥2s，未触发 403） |
| 32-36 | arxiv API × 5 组关键词（HTTPS，-L） | 无新编目论文 |

Search API 用量 9/10（首查 rate_limit 不计数），全程未触发限流。

## 七、合并后状态

- 主文档：764 → **789 条**（第 1 层 633 → 658；第 0 层 17、第 2 层 114 不变）
- 类别分布变化：第 1 类 117→124 · 第 4 类 13→14 · 第 5 类 23→25 · 第 6 类 158→167 · 第 7 类 63→64 · 第 8 类 55→57 · 第 9 类 16→18 · 第 11 类 55→56
- 轴分布变化：infra_off 611→635 · ambiguous 78→79 · config 166→170 · other_stack 147→154 · partial 168→171 · hw_specific 143→153 · yes 95→96 · documented 547→560 · verified 211→223 · NEW 651→671 · var:x 110→115
- 导出：fault_catalog.xlsx / fault_catalog.csv 预期 789 − 17 = **772 行**

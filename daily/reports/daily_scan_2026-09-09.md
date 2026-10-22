# 每日扫描报告 — 2026-09-09

- 扫描日期：2026-09-09（滚动窗口 `created:>2026-09-08`，覆盖 2026-09-08T00:00Z 之后创建的 issue）
- 目标仓库：vllm-project/vllm、sgl-project/sglang、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、huggingface/transformers、flashinfer-ai/flashinfer
- 主文档基线：1007 条（截至 2026-09-08 合并后）
- 去重方法：对每条候选在主文档全文检索 issue 编号（`#NNNNN`），全部确认未编目后方进入合并。注意：`#5042` 初查命中 2 处为 vLLM **#50425/#50427** 的子串撞号、`#5064` 命中 1 处为 vLLM **#50649** 撞号——flashinfer #5042/#5064 实际均未编目。

---

## 一、扫描结果总览

- GitHub Search API：10 组查询全部一次成功（vLLM×4、SGLang×1、TRT-LLM×1、triton×1、tt-metal×1、transformers×1、flashinfer×1）；无限流 403、无 Validation Failed
- core REST 详情：28 次全部成功（`--http1.1`）
- arxiv：5 组关键词查询，命中论文均为此前已编目（TrainSDC/SCOUT/LLM-PRISM/JITterFlip/GBFA/ROBBIN/2606.28958 等）或不收录的方法论文（Sample-Guided Top-K、MetaKV、Signed Rescue Routing、2601.19912 已编目）；无新编目论文
- **独立候选 25 条 → 合并 18 条**（8 个新子签名 + 10 个已知类别变体；7 条不收录见「三」）

## 二、已合并条目（18 条）

### A. 新子签名（NEW，8 条）

| # | 标题 | 来源 | 归类 |
|---|---|---|---|
| 1 | ttnn.typecast(float→int32) 正溢出饱和到 INT32_MIN——单一饱和常量 0x80000000 无正路径（−INT_MIN==INT_MIN），正侧 2^31 以上 100% 符号反转（NaN 也返 INT32_MIN），覆盖 19.34% float32 域 | tt-metal #55933 | 第 1 类 → 1.190 |
| 2 | ttnn.matmul/linear 传 core_grid 或 program_config 时静默把 bf16 math fidelity 从 HiFi2 降到 LoFi（`increase_fidelity` 单行规则），实测 95.0% 输出 word 改变且快 14%；仓库 shipped 代码 146 处调用中招 | tt-metal #55889 | 第 1 类 → 1.191 |
| 3 | moreh_adam/adamw 把全部 5 个超参截断到 bf16 再运算：beta2=0.999 执行为 0.99609（V-horizon 1000→256 步）、0.9999 与 0.999 位相同轨迹；adamw EMA 与 bias correction 用不同 beta，轨迹偏离 torch 34–44%；PCC 对 rescale 盲 | tt-metal #55935 | 第 1 类 → 1.194 |
| 4 | SGLang chunked prefill 把截断点切进 encoder-decoder 请求的 encoder placeholder 区：`prepare_encoder_info_extend` 无条件减整段 encoder_len → `extend_num_tokens` 为负 → 断言炸 scheduler SIGQUIT 整服务器 | SGLang #38627 | 第 6 类 → 6.221 |
| 5 | HYV4 (Hunyuan-V4) + a2a MoE backend：`require_attn_tp_gather` 单凭"a2a backend 已设"推导 num_token_non_padded 为序列分片，但 HYV4 无 LayerCommunicator、每 rank 全宽喂 MoE → topk_ids=-1、每 batch 只算 1/attn_tp_size 的 token，输出流畅但错，吞吐反而 +16.7% | SGLang #38606 | 第 5 类 → 5.36 |
| 6 | flashinfer recurrent_kda decode（cute-dsl，dense T=1 路径）：Python gather/scatter 不检查 ssm_state_indices 符号，-1 wrap-around 覆写 state pool 尾槽（内核级路径全部正确） | flashinfer #5042 | 第 8 类 → 8.79 |
| 7 | GLM-5.3-Flash 经 transformers `save_pretrained` round-trip 后加载 SGLang：`conversion_mapping.py` "glm5_next" 读时单向重命名、存时不回写——524/1609 张量（304B 参数、97% 权重）静默丢弃，服务 200、生成 token 噪声 | SGLang #38618 | 第 12 类 → 12.33 |
| 8 | Qwen4Exp (Qwen3.8-Flash-Next) 无法服务 DFlash/DSpark drafter：5 个独立阻断点（V1 runner 不供 PLE inputs、V2 deref None、aux hidden state 捕获机制被 delattr 删除、返回槽被 hyper-connection 流占用等） | vLLM #56088 + SGLang #38589 | 第 11 类 → 11.73 |

### B. 已知类别新实例（var:，10 条）

| # | 标题 | 来源 | 归类 | var 族 |
|---|---|---|---|---|
| 9 | Triton FuseNestedLoops 对"上界<下界的空动态内层循环"miscompile：flatten 后内层体每轮外层各执行一次（ub=3 返回 4，应为 0）——1.127（非支配 assume）同 pass 第三变体 | triton #11656 | 第 1 类 → 1.192 | var:1.127 |
| 10 | tt-metal log/log2/log10_bw 在 x=−0.0 返回错误符号无穷、polygamma_bw 用硬编码 n 列表选极点符号（n=0 与偶 n≥12 反转）——组合级缺陷，与理想原语无关 | tt-metal #55931 | 第 1 类 → 1.193 | var:1.153 |
| 11 | Triton causal_conv1d_update（SGLang fallback 路径）三错：oversized state 读头部而非尾部窗口、声明 cache_seqlens_ptr 但不读（线性而非循环缓冲）、width=5 被接受但只实现 2–4 | SGLang #38622 | 第 1 类 → 1.195 | var:1.132 |
| 12 | vLLM ngram 推测解码 × qwen3_coder/qwen3_xml/mimo tool parser：工具参数静默变 `{}`、键值折叠（`{"command=ls": ""}`）、函数名损坏、裸 XML 标签泄漏；模型 reasoning 已给出正确参数——11.28（TRT-LLM spec acceptance 拆开控制块）的 vLLM ngram × 增量 parser 对应物 | vLLM #56077 | 第 11 类 → 11.74 | var:11.28 |
| 13 | SGLang PD disagg + DCP>1 decode retraction：retraction CPU 备份/恢复路径把 DCP-widened 逻辑索引直接喂 `get_cpu_copy`（缺 owner-rank localize），越界 → CUDA device-side assert；在界时也拷错行 | SGLang #38645 | 第 6 类 → 6.222 | var:6.157 |
| 14 | MiniMax-H3 FL2VA layerwise offloading 生成可播放但视觉严重损坏的花屏视频（H.264 流合法、帧内容 garbled） | SGLang #38605 | 第 6 类 → 6.223 | var:6.193 |
| 15 | Kimi-K3 strict tool grammar：structural-tag builder 不把具名属性从 additionalProperties matcher 排除——`{"type":"integer"}` 属性按更松 schema 发出仍过 grammar（CPU 确定性复现） | SGLang #38587 | 第 10 类 → 10.59 | var:10.47 |
| 16 | TRT-LLM PyTorch executor 把 multimodal metadata 失败当致命错误：单条 well-formed-but-unsupported 请求 hard-kill 全部 rank（exit 137），客户端重试 × 粘性路由两分钟连杀两个 pod | TRT-LLM #18971 | 第 6 类 → 6.224 | var:6.192 |
| 17 | flashinfer trtllm-gen NVFP4 MoE SiTU (siTuGlu) GEMM1 epilogue 部分配置不写输出 block scale factors | flashinfer #5053 | 第 7 类 → 7.81 | var:7.69 |
| 18 | flashinfer SM107 dense NVFP4 mm_fp4 内核族被静默禁用：父类 import 自 `nvidia_cutlass_dsl.examples`（wheel 根本不暴露），`except ImportError: pass` 吞掉——SM107 永远走非原生路径，无任何警告 | flashinfer #5065 | 第 7 类 → 7.82 | var:7.77 |

## 三、不收录说明（7 条）

1. **vLLM #56128 / #56126 / #56123** — 均为 RFC/设计讨论（PCIe all-reduce、DBO for dense TP、P/D transfer contract），非正确性故障。
2. **vLLM #56009** `Possible response/request cross-talk` — 观察报告，报告者自己声明"not a confirmed vLLM defect"、无 reproducer、无法区分服务端串扰与模型回声；不满足编目置信度。若后续确认再编目（与 6.163/6.6 跨请求污染族相关，留观）。
3. **vLLM #56041** `FixFunctionalizationPass crashes ... control_deps nodes` — torch.compile pass crash，发生在 out-of-tree 平台插件、报告者自述"未建立上游-only reproducer、不知道哪些上游工作负载自然产生该图模式"；根因在 PyTorch inductor 侧而非 vLLM 推理路径，置信度不足。
4. **vLLM #56022** `Assistant message contains both non-empty content and tool_calls` — DSML 流式 parser 的 OpenAI 兼容性问题（首错后 session 内持续），与已编目 10.55（DSML 增量 parser 部分事务逃逸）同域但属协议层格式问题而非推理正确性损坏；留观。
5. **vLLM #55986** `Multi-channel target audio ignores channel_reduction` — RFC 且自认 **latent**（所有 in-tree 模型 target_channels=1，路径未被行使）；代码注释已标注 provisional。
6. **SGLang #38745** `DUAL_STREAM_TOKEN_THRESHOLD makes DSA dual-stream gate unsatisfiable on ROCm` — 死代码/性能特性门控矛盾（enable_dual_stream 恒 False），无数值正确性影响；#14337 TODO 的跟进。记录备查。
7. **SGLang #38695 / #38676** — Feature request（MoE router GEMM 统一、多模态预处理指标），非故障。**flashinfer #5069 / #5051** — graduation 跟踪 issue；**#5058** top_k_page_table_transform SM120 cudaErrorNotSupported 为 fail-loud 启动 crash（SGLang --dsa-topk-backend flashinfer 启动即死），与 8.76（Ascend page16 tiling 静默垃圾）相反为显性失败、且未见静默损坏，暂不收录。**#5064** JIT link `cannot find -lcudart` 为构建环境问题（cpp_ext.py 硬编码 lib64 vs cu13 lib/）。**tt-metal #55957**（QB2 pinned-memory warm cache hang）、**#55904**（dual-phase matmul_tiles deterministic stall）为 hang/性能类（fail-stuck），非静默数值损坏——但 #55904 的"非阻塞相邻 op 之间线程死亡"模式留观。**#56020**（llama4 query-scale cache 24GB/设备 @1M ctx）为内存占用设计问题。**#55844**（AI-Gen 内核 bfloat8_b PCC drop awareness）为跟踪 issue。**#55879**（exact-accumulation reference offer）为提议。

## 四、重点新发现摘要

1. **save/load 单向重命名丢 97% 权重（#38618）**：transformers 读 checkpoint 时经 `conversion_mapping.py` 重命名张量、`save_pretrained` 不回写——任何 fine-tune/merge/dtype 重导出的 GLM-5.3-Flash 在 SGLang 加载时静默丢弃 524/1609 张量（含全部 routed MoE experts、mHC、KDA forget gate），服务器健康 200、输出 token 噪声。「序列化格式单向转换」是新子签名，且是**最大的单条静默权重损坏面**（97% 参数）。
2. **a2a backend 门控推导 token 数 → MoE 只算 1/8（#38606）**：`require_attn_tp_gather()` 单凭"a2a backend 已设置"推导 `num_token_non_padded` 为序列分片，但 HYV4 的模型实现没有 LayerCommunicator、每 rank 全宽喂 MoE——超出 shard 宽度的行全部 `topk_ids=-1`。**输出流畅、服务健康、吞吐反而 +16.7%**，只有 logprob 比较能发现——「错误配置更快」的静默损坏范式。
3. **饱和常量单符号 → 19.34% fp32 域符号反转（#55933）**：typecast 的 SFPU 内核只加载一个 `0x80000000` 饱和常量，正溢出没有通往 INT32_MAX 的路径（−INT_MIN==INT_MIN）。与 1.185（from_torch cast UB）同域但机制不同——本条在设备侧内核、1.185 在宿主侧 UB。
4. **kernel 参数静默降精度（#55889）**：传 `core_grid`/`program_config`（用户为了 sharding/融合/网格）就静默降 LoFi——"A user who passes a config to speed a matmul up is measuring the fidelity, not the config"。shipped 代码 146 处调用中招，与 7.77（MoE provider 先于 per-layer quant config）同族的"可选参数改变数值语义"。
5. **-1 wrap-around 覆写 state pool 尾槽（#5042）**：文档化契约"负 slot 是 padding"只在内核级路径实现，Python 层 gather/scatter 用负索引直接 wrap 到末槽——与 6.165（KDA padding 重映射到可分配尾槽）互为镜像（一个 Triton 后端、一个 cute-dsl 后端）。

## 五、API 调用记录

| # | 类型 | 查询/目标 | 结果 |
|---|---|---|---|
| 1 | search | vllm correctness in:title | 0 命中 |
| 2 | search | vllm quantization wrong result | 0 命中 |
| 3 | search | vllm corrupt OR NaN OR silently | 9 命中（5 有效） |
| 4 | search | sglang wrong OR corrupt OR NaN OR silent | 11 命中（9 有效） |
| 5 | search | TensorRT-LLM 全部新 issue | 1 命中 |
| 6 | search | triton wrong OR NaN OR miscompile | 1 命中 |
| 7 | search | tt-metal numerical OR PCC OR NaN | 9 命中（6 有效） |
| 8 | search | transformers wrong result OR NaN OR corrupt | 0 命中 |
| 9 | search | flashinfer 全部新 issue | 7 命中（3 有效） |
| 10 | search | vllm label:bug KV cache | 1 命中（与 #3 重复） |
| 11–38 | core REST | 28 个 issue 详情 | 全部 200 |
| 32–36 | arxiv | SDC+LLM / bitflip / KV cache / correctness / GPU soft error | 全部已编目或不收录 |

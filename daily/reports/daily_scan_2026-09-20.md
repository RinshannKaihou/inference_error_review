# LLM 推理故障每日扫描报告 — 2026-09-20

- **扫描日期**：2026-09-20（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-18`（与 09-19 扫描窗口重叠两日，已逐条去重）；arxiv 按提交时间倒序取最近 5 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1149 条（2026-09-19 更新后）；编号编目 1035 条 + K* 扩展 114 条

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-18 wrong OR incorrect OR corruption OR NaN` | 20 命中 |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-18` | 17 命中 |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-18 wrong OR incorrect OR corruption OR NaN` | 9 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-18 wrong output OR incorrect OR NaN` | 2 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-18 wrong OR incorrect OR corruption` | 0 命中 |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-18 wrong OR miscompile OR NaN` | 0 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-18 numerical OR wrong OR pcc OR silent` | 22 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-18 wrong OR incorrect OR NaN` | 3 命中 |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-18 quantization OR "KV cache" OR deterministic` | 32 命中 |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-18 silent OR "bit flip" OR "weight corruption"` | 18 命中 |
| 11–15 | arxiv | `"silent data corruption" AND LLM` / `"bit flip" AND "LLM inference"` / `"KV cache" AND corruption` / `"weight corruption" AND inference` / `"GPU soft error" AND "machine learning"` | 5/1/5/2/0 命中 |

GitHub Search API 10 组查询全部有效（**修复 09-19 q09 缺陷**：全局 SDC/bit-flip 查询改为 repo:vllm-project/vllm 限定的 silent/bit flip/weight corruption 查询，18 命中全部有效）、无限流 403、无 Validation Failed；core REST 详情调用 13 次全部成功（`--http1.1`）。arxiv 5 组查询全部成功（1 组 0 命中，无 429）。

## 已编目/已判条目（本次跳过）

以下命中条目经编号核对已在主文档中（或 09-19 扫描已明确判定），本次不重复添加：

- vLLM #57720（→4.25）、#57719（→6.245）、#57716（→8.91）、#57661（→6.243）、#57727（→6.244）、#57726（→10.73）、#57723（→10.74）、#57721（09-19 已判：fail-loud 启动 assert 不收录）、#57725（09-19 已判：错误报告质量留观）、#57614（→8.90）、#57562（→6.242）、#57578/#57532（09-18 已作 8.84 补充来源）、#57580（09-18 已判：synthetic 回归追踪留观）、#57493（→4.24）、#57490（→10.72）、#57486（09-18 已编目为 7.95 同族语境）、#57649/#57608/#57479/#57738/#57794（RFC/Feature）、#57555/#57521（性能）、#57650/#57671/#57771（资源治理/工具/Docs）、#57672/#57566/#57530（RFC）、#57680（性能）
- SGLang #40285（→12.41）、#40156（→11.81）、#40144（→11.82）、#40127（→12.40）、#40084（→7.94）、#40305（Feature）、#40370（误投已删）
- flashinfer #5339（→11.83）、#5334（→1.230）、#5318（09-19 已判：release 回移缺失不收录）
- transformers #48947（→6.246）、#48954（serge 集成 triage，closed）
- tt-metal #57088（→1.232）、#57090（09-19 已判：CI 文件治理）、#57086（测试增强 tracking）、#56986（→9.27）、#56955（→1.231）、#56993/#56945/#56958/#56949/#57004/#57023/#57022（09-18/09-19 已判留观或语境补充）、#56976（性能）、#56954（文档缺陷）
- TRT-LLM / triton 窗口 0 命中

## 🆕 新发现（本次合并，共 9 条）

### A. 全新子签名（3 条）

1. **[新类别/新签名] vLLM #57740 — 用户正文字面 `<|image_pad|>` 与模板媒体占位符 token 值碰撞，多模态绑定错位、视觉安全裁决翻转** — vLLM 通用 `PromptReplacement` 在 chat 渲染+分词后的平坦 token 序列上**按 token 值匹配**占位符，不保留匹配 token 来源（模板 vs 用户正文）。Qwen2.5-VL 的 `<|image_pad|>` 是用户可拼写的特殊 token（id 151655）：用户文本出现更早时吃掉排队的图像更新——256-token 图像特征 span 被插进用户正文、真占位槽只剩 1 个 raw pad token；两布局总长与 token 多重集完全相同（292 tokens）。端到端：Qwen2.5-VL-3B/7B 视觉安全裁决 YES/UNSAFE→NO/SAFE、BLOCK→ALLOW；describe 探针仍识别图中武器（可感知但结构性 grounding 错位）。候选受影响面含 Qwen2-VL/Qwen3-VL/LLaVA/InternVL 等 `<image>`-占位符 adapter 族。→ 归入第 10 类（分词器/模板），Cov NEW（绑定键退化到 token 值、缺 provenance）。
2. **[新类别/新签名] vLLM #57791 — Responses API 一轮多个 built-in tool call 只 dispatch 最后一个，其余静默漏执行** — 非 Harmony Responses 路径下模型 emit 多个 built-in 调用（web_search_preview + code_interpreter）时只有最后一个被执行，更早的留在 `response_messages` 里从未 dispatch；与默认启用的 `parallel_tool_calls` 矛盾。CPU-only pytest+AsyncMock 复现。→ 归入第 10 类，Cov NEW（built-in 调度循环缺迭代）。
3. **[新类别/新签名] tt-metal #57066 — Quasar TRISC3 mailbox 返回 incorrect results** — Quasar bring-up 硬件原语缺陷（bare report）；与 1.201（copy_via_memmove）、1.204（p_unpacr flavour）同平台第三处独立缺陷位点。→ 归入第 1 类（硬件/SDC），Cov NEW。

### B. 已知类别新变体（6 条）

4. **[已知类别] vLLM #57822 — SM120 + compressed-tensors per-tensor static FP8（W8A8）生产流量下 Xid 13 三连崩，v0.24.0 同机同 checkpoint 完全稳定** — v0.28.0 官方镜像 RTX PRO 6000：3 小时 2×IMA + 1×illegal instruction（async_utils.py:168 异步浮出），主机日志 580 条 Xid 13 全落三次崩溃 burst 内；崩溃时调度器近乎空闲（单请求、KV 0.066–0.094%），三次均为部分前缀命中+非对齐剩余 chunk 的单请求 prefill 步。非 block-scaled 变体（#47436/#51884 那类加载期失败）——干净加载、服务约 1 小时后才崩。→ 归入第 6 类，Cov var:6.210（SM120 持续负载 Xid 13 IMA 族；本条附加 v0.24↔v0.28 版本二分证据）。
5. **[已知类别] SGLang #40459（re-file #21168）— embedding 超长请求错误分支漏 `set_finish_with_abort`：全长度入队、客户端挂到超时** — `handle_embedding_request` 的 `validate_input_length` 分支（scheduler.py:3516-3523）漏调 abort，generate 路径（:3044-3052）与同函数其余三个错误分支都调；预期 HTTP 400 永不送达、超长请求占用全长度调度资源。原修复 PR #21169 被不活跃机器人自动关闭，缺陷自 #21168 存续于 main。→ 归入第 6 类，Cov var:6.192（错误输入处理路径失效族）。
6. **[已知类别] vLLM #57699 + #57827/#57826 — 四族 tool arg converter（qwen3_coder/minimax_m2/glm47_moe/deepseek_v4）在模型省略末参数闭合标签时静默丢末参数** — 尾部值提取只在 `if partial:` 分支运行：非流式 `arguments` 少一个键；流式 partial 已把值流出、flush 产出更短 JSON——客户端停在 `{"body": `。BFCL live_simple 温度 0 实测触发；SGLang 同名 detector 在 `</function>` 处正确保留。#57707 只修了 qwen3，三族仍在 main。→ 归入第 10 类，Cov var:10.42（流式 parser 事务性丢失族）。
7. **[已知类别] tt-metal #57116 — `ttnn.selu` 注册 golden 忽略 op 自己的 `scale=`/`alpha=` kwargs，返回标准 SELU：最高 3416% 偏差而 PCC>0.99 恒过** — `torch.nn.functional.selu` 无该二参数，调用方传参落 `**kwargs` 被丢弃；参数真实穿入内核（`DEFINE_UNARY_OP_WITH_TWO_FLOAT_PARAMS`）。golden 对 `scale=3.0,alpha=2.0` 与 `scale=0.5,alpha=0.1` 返回逐位相同张量。→ 归入第 1 类（验证面失效族），Cov var:1.180。
8. **[已知类别] tt-metal #57115 — ttnn.sum/mean/max/min/std/var 六归约 golden 静默丢弃 `scalar=` kwarg：参考最高偏差 300% 而 PCC 恒 1.0** — `_create_golden_function` 工厂签名 `def golden_function(input_tensor, dim=None, keepdim=False, **_)` 吸收 scalar 从不转发；`scalar` 为活的文档化参数（nanobind `nb::arg("scalar")=1.0f`，对照 `correction` 才是显式弃用的）。一处工厂签名缺陷放大成六 op 参考错误。→ 归入第 1 类，Cov var:1.180（与 #57116 构成 golden kwargs 吞噬姊妹对）。
9. **[已知类别] tt-metal #57095 — streaming SDPA prefill 开 `fp32_dest_acc_en` 后行和仍 bf16 打包，逐 K chunk 舍入复合** — 融合 rescale `salad_correct_fused` 把 online-softmax 行和与 bf16 输出累加器共享同一 pack 格式；S=131072 时 rmse 0.0097（legacy 内核 0.0053，劣化 1.83×）。#56989 以 >512 chunks 回退 legacy 止损；正解需 fp32 sum CB + pack/unpack 双侧改（单改 factory 返回 rmse inf）。→ 归入第 1 类，Cov var:1.106（SDPA 中间格式精度失配族）。

### C. 评估后不收录（记录备查）

- **vLLM #57759**（FULL_AND_PIECEWISE capture + Engram PLE cpu_offload 负载下间歇 cudaErrorNotPermitted）：fail-loud、无根因。参照 #57354 先例留观。
- **vLLM #57748**（GLM-5.3-Flash-NVFP4 on Hopper）：正文为请求稳定回退路径的 Feature 请求，非缺陷报告。不收录。
- **vLLM #57838**（RDNA4 gfx1201 RowWiseTorchFP8ScaledMMLinearKernel 选择代价 5-24% decode）：纯性能。不收录。
- **vLLM #57755**：空标题占位 issue。**#57738/#57794**：RFC。**#57771**：Docs。不收录。
- **SGLang #40417**（humming MoE runner EP 下调优形状 4× 虚胀）：报告者自证 correctness 不受影响（launcher 按真实形状校验）、纯调优选择问题，且"显而易见的修复"实测回归 -5.1%。性能域不收录。
- **tt-metal #57109**（Kimi-K3 AttnRes plane count 三处派生 + BLOCK_SIZE 双拼写）：代码重复治理，无错误输出证据。**#57135**（traced prefill KDA 单 slot 阻塞 perf sweep）：工具链/性能。**#56967**（LO16 定义与 tt-isa-documentation 分歧）：文档语义分歧留观。不收录。

### D. arxiv

5 组查询全部成功（1 组 0 命中），命中均已在编目或已判方法论文：

- `"silent data corruption" AND LLM`：Syndrome Decoding for SDC in Quantized Integer GPU Arithmetic（SProbe，09-18 已编 §3.1 补充来源）、TrainSDC、SCOUT、LLM-PRISM、Exploring SDC（均已编目）
- `"bit flip" AND "LLM inference"`：JITterFlip（已编目 3.18）
- `"KV cache" AND corruption`：Contiguity, Not Importance（2609.17983，09-17 已记录）、Addressable Memory（2608.07408，已编目 14.7）、Runtime Observability（2608.05863，已编目）、Nexus（方法论文不收录）、When Latent Agents Lie（2606.28958，已编目）
- `"weight corruption" AND inference`：BLINK（已编目）、Recover-LoRA（量化恢复方法论文不收录）
- `"GPU soft error" AND "machine learning"`：0 命中

无新编目论文。

## 重点突出的新发现

1. **10.76 的"绑定键退化到 token 值"**（vLLM #57740）：多模态媒体绑定在平坦 token 序列上按值匹配、丢失来源信息——用户可拼写的占位符 token 使"第一个匹配消费媒体更新"的顺序假设被打破。两条布局**总长与 token 多重集完全相同**，长度/词法层面审计全部失明；安全裁决可被单条用户文本翻转（UNSAFE→SAFE）。这是多模态输入管线上第一个**用户可控的 grounding 错位**签名。
2. **1.233/1.234 的"PCC 门禁对仿射幅度差结构性失明"姊妹对**（tt-metal #57116/#57115）：3416% 与 300% 的输出偏差分别在 PCC>0.99 / PCC=1.0 下恒过——相关性门禁对"正确形状+错误幅度"的整体缩放类参考错误原理上不可见。与 1.180（golden 调不存在的 API 从未跑过）构成验证层失效谱系的两端：**炸不出声 vs 恒绿**，后者更隐蔽。
3. **10.75 首次成族**（vLLM #57699 + #57827/#57826）：同一 `if partial:` 结构被复制进四个 converter（qwen3_coder/minimax_m2/glm47_moe/deepseek_v4），#57707 只修了 qwen3——"修复未覆盖复制体"使单点设计缺陷扩散为四族静默丢参；流式/非流式语义分裂（flush 产物比已流出发送的内容更短）是新的失效方向。

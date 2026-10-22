# LLM 推理故障每日扫描报告 — 2026-09-22

- **扫描日期**：2026-09-22（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-20`（与 09-21 扫描窗口重叠两日，已逐条去重）；arxiv 按提交时间倒序取最近 5 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1171 条（2026-09-21 更新后）；编号编目 1057 条 + K* 扩展 114 条

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-20 wrong OR incorrect OR corruption OR NaN` | 14 命中 |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-20` | 28 命中 |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-20 wrong OR incorrect OR corruption OR NaN` | 7 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-20 wrong output OR incorrect OR NaN` | 0 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-20 wrong OR incorrect OR corruption` | 1 命中（RFC，09-21 已判） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-20 wrong OR miscompile OR NaN` | 0 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-20 numerical OR wrong OR pcc OR silent` | 42 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-20 wrong OR incorrect OR NaN` | 3 命中 |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-20 quantization OR "KV cache" OR deterministic` | 33 命中 |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-20 silent OR "bit flip" OR "weight corruption"` | 29 命中 |
| 11–15 | arxiv | `"silent data corruption" AND LLM` / `"bit flip" AND "LLM weights"` / `"KV cache" AND corruption` / `"LLM inference correctness"` / `"GPU soft error" AND "machine learning"` | 5/1/5/1/0 命中 |

GitHub Search API 10 组查询全部有效、无限流 403、无 Validation Failed；core REST 详情调用 17 次全部成功（`--http1.1`）。arxiv 5 组查询全部成功（1 组 0 命中、无 429）。

## 已编目/已判条目（本次跳过）

以下命中条目经编号核对已在主文档中（或 09-21 扫描已明确判定），本次不重复添加：

- vLLM #57822（→6.248）、#57740（→10.76）、#58029（→10.78）、#58020（→6.250）、#57938（→6.251）、#57986（→11.84）、#57950（→10.79）
- vLLM #57999/#57998/#57997（09-21 已判：API 语义/错误信封/元数据质量）、#57996（echo 元数据留观）、#57974（资源治理）、#57971（性能）、#57941（fail-loud IMA 留观）、#58034（CPU 平台不可达 fail-loud assert，09-21 窗口命中留观）
- SGLang #40459（→6.249）、#40564（09-21 已判：fail-loud null 包装留观）、#40504（性能）
- TRT-LLM #19501（09-21 已判：RFC）；transformers/triton 窗口 0 命中

## 🆕 新发现（本次合并，共 13 条）

### A. 全新子签名（8 条）

1. **[新类别/新签名] tt-metal #57410 — host BFP4 packer 直接舍入到 3-bit，与设备 typecast（先 BFP8 再截断）在 42.5% 元素上分歧** — 同一 bf16 值经 `ttnn.from_torch(dtype=bfloat4_b)`（host）与 `ttnn.typecast`（设备）得到不同位型：435/1024 元素不同、最坏差一个完整 BFP4 mantissa step（值 ×2），含 host 非零→device 零。根因是 host `convert_u32_to_bfp` 把 24-bit mantissa 直接 RNE 到目标宽度，而 ISA 规定先产 7-bit BFP8（RNE ties-away）再截断——"先舍到 3 位"≠"先舍到 7 位再截"。`clone`/`experimental.typecast` 均与设备一致：host 一侧独错。→ 归入第 1 类，Cov NEW（host/device 同名转换双实现数值契约分裂）。
2. **[新类别/新签名] tt-metal #57408 — unary SFPU 内核 SFPSTORE 截断窄化（round-toward-zero），与 binary 内核 RNE 分裂：`ttnn.square(x)` ≠ `ttnn.mul(x,x)`** — 同一乘积两个入口 440/1024 元素不一致；`square`/`rad2deg`/`deg2rad`/`softsign`/`hardsigmoid`/`prelu`/`leaky_relu` 七内核每个 inexact 结果**单向偏向零至多 1 bf16 ULP（0.39%）、从不反向**（1024 输入无一例外）；`rad2deg(-4.5)=-256.0`（应 -258.0，0.71% 误差）。binary 路径与 golden/FPU 均 RNE。→ 归入第 1 类，Cov NEW（unary/binary 同函数入口舍入模式分裂）。
3. **[新类别/新签名] tt-metal #57382 — `ttnn.slice` 非最内维截断 + 最内维 begin 非对齐时漏跳一行：坏行内容是应丢弃的前一行** — `(5,65,1025)` 输入切片得 6144 元素错（6/320 行），每个坏行恰含输入中**应被跳过的行**（线性偏移 −1025 = 一行 stride）；batch 0/4 全对、受害 batch 仅前几行错后 resync；bf16 时错行数降为 1（错误行数 dtype 依赖）；ROW_MAJOR/TILE 同错；输出形状正确、零错误——静默错数据。bisect 锁定 `70ba57dc3ab`。→ 归入第 1 类，Cov NEW（数据移动游标数学错族——与 1.196/1.226 同族新签名）。
4. **[新类别/新签名] tt-metal #57413 — 近似 tanh 6-entry SFPLUT reland（#57179）使 Llama 3.2-90B-Vision e2e BERTScore 0.722→0.6815 跌破 0.69 门禁** — 相邻 commit bisect 精确定位；per-sample 分数与 scheduled run 逐位一致（非噪声）；视觉路径 `llama_cross_block.py` 等以 `fast_and_approximate_mode=True` tanh 门控 cross-attn/MLP 输出，单元容差内通过但 e2e 语义质量退化 4 个点。→ 归入第 1 类，Cov NEW（近似内核替换越过单元容差、仅 e2e 质量门可见的回归）。
5. **[新类别/新签名] SGLang #40730 — PD retract rebootstrap 重算请求不携带 prefill DP rank 绑定，错误路由后双侧 300s 超时请求失败** — #39357 修复的 `routed_dp_rank` 字段取自客户端提示（通常 None）→ 重算请求被普通 load balancer 派发；而 decode 侧 `CommonKVReceiver.init(prefill_dp_rank)` 是键控**绑定**不重解析——重算落到其它 DP rank 时 sender 找不到 decode 注册、decode 等永不开始的传输，双侧超时（默认 300s）后失败；错误路由的 sender 还会**覆写 bootstrap 表的 room→dp_rank 映射**。→ 归入第 5 类，Cov NEW（重算路由键与 KV 接收绑定键断裂）。
6. **[新类别/新签名] flashinfer #5438 — SM90 W4A16 MXFP4 CUTLASS fused MoE（0.6.10–0.6.15.post1）TMA 描述符原地更新与在途 scale 加载竞态，偶发写错 tile** — 2000 相同调用 15 次错（行级 rel err 最高 0.152、跨相同调用非确定）；正交对照证明错 tile 用的是**别的专家的 scale**（权重/激活无误）；pin tactic 时 19/28 tactic 出错；decode（M≤256）全对；0.6.18（#3738 重写）清洁。→ 归入第 7 类，Cov NEW（TMA 描述符原地更新 × 在途加载竞态）。
7. **[新类别/新签名] vLLM #58105 — 文本 grammar 放行普通 token 拼出的分隔符时，Qwen 流式 parser 静默丢失整个 tool call** — grammar 允许普通词表 token 序列拼出 `</think>`/`<tool_call>`/`</tool_call>` 字符串（如 `[27,14,26003,29]` vs 专属 ID 248069）时，parser 不认这些普通 token 边界——调用整体留在 reasoning、无结构化输出；**仅替换三处分隔符为专属 ID（解码文本逐字不变）即恢复调用**。EBNF 字符串约束 ≠ parser token ID 约束的边界契约缺口。→ 归入第 10 类，Cov NEW（语法字符串匹配 vs parser 专属 token ID 匹配的契约缺口）。
8. **[新类别/新签名] vLLM #58138 — cross-encoder `padding="max_length"` 把 padding 的 token_type_ids 填成最后真实 token 的段 ID（双重缺陷：压缩表示也编码不了正确形状）** — host 侧用 `token_type_ids[-1]` 而非 `pad_token_type_id=0` 补 pad；worker 侧压缩只存一个 0→1 边界——**能编码 `0*1*` 却编码不了正确的 `0*1*0*`**，即使修 host 填充、压缩重建仍回退全 1。`/score`/`/rerank` 静默跑在错误 segment 输入上，可改变分数/排序/阈值决策。→ 归入第 10 类，Cov NEW（padding 段 ID 填充 + 有损压缩表示双位点）。

### B. 已知类别新变体（5 条）

9. **[已知类别] SGLang #40735 — MiniMax-M3 遗漏 `is_ascend_fuseep()` 检查误走普通 MoE 路径：fuseep 已全局合并的输出再过 TP all-reduce（TP8 即 ×8）+ shared expert 错误切分 + `ep_size`/`top_k` 未初始化** — 三重效应叠加（数值放大 + 残缺 + 属性缺失），确定性劣化、无 crash/NaN。→ 归入第 5 类，Cov var:5.26（GPT-OSS DP attention 双重归约族）。
10. **[已知类别] vLLM #58031 — GB300（SM103）上 FlashInfer `trtllm_fp4_block_scale_moe` autotune 无限 wedge：trtllm_gemm cubin 仅 sm_100a、无 PTX、autotuner 无超时** — 启动 boot 永不完成（40+ 分钟 100% GPU、零日志推进、autotune_configs.json 从未写入），无 error/traceback。→ 归入第 5 类，Cov var:5.40（FlashInfer autotune 栈活性失效族）。
   - **2026-09-23 事后更正（保留 09-22 原扫描判读以供追溯）**：上句关于 `trtllm_gemm.so` 无 PTX/sm_100a-only 导致该 MoE 候选不能在 SM103 执行的**根因归属已撤回**；MoE 符号实际位于含 sm_103a cubin 的 `fused_moe_trtllm_sm100.so`。40+ 分钟高 GPU、无进度日志、cache 未写入是报告的症状，**不足以证明永久挂死**：调优中途不记录 bucket 进度，cache 仅在完成时写出；极慢调优是可能解释但尚未实证。主文档 5.44 已降为 `documented`，改按 SM103 启动调优长期无进度现象编目；与 5.40 仅症状关联，不能推定共根因。原 issue [EDIT](https://github.com/vllm-project/vllm/issues/58031) / [报告者确认](https://github.com/vllm-project/vllm/issues/58031#issuecomment-5788562400)。
11. **[已知类别] vLLM #58119 — 部分块（partial block）KV 事件引用从未上报的 parent hash，外部消费者前缀链不可重建** — 1800 token 请求只发 `H1536` 与 `H1792` 事件，中间 `H1664` 从不发出，`BlockStored(H1792).parent=H1664` 悬空；单元级复现完整。→ 归入第 6 类，Cov var:6.220（KV 事件对外契约缺陷族——6.220 载荷不可解码，本条链结构不完整）。
12. **[已知类别] vLLM #58127 — Hunyuan A13B 流式把 `</answer>` 闭合标记泄漏进 content（非流式干净）** — 同一 token IDs 流式 `content="answer\n</answer>"` vs 非流式 `"answer"`；repro 经真实引擎+HTTP+生产 parser（logits processor 强制序列，无需 150GiB 权重）。→ 归入第 10 类，Cov var:10.79（流式/非流式双解码路径分歧族第三成员）。
13. **[已知类别] vLLM #58080 — MTP draft 不继承 dict 形式 `--hf-overrides`：长度门超界后全批 dummy-0 提案，MTP 接受率静默坍塌至 0** — draft 保持原生 `max_position_embeddings`；`_input_fits_in_drafter` 越界时跳过提案、以普通 token 0 填充（rejection sampler 只认 -1 占位 → 接受率 ≈0%）；target 输出保持正确、零日志，每步仍付 verify 成本。Qwen3.8-27B-FP8 + 2×H20 生产实测。→ 归入第 11 类，Cov var:11.63（MTP 接受率静默坍塌族——配置继承断链成因）。

### C. 评估后不收录（记录备查）

- **vLLM #58203**（Qwen3.5/3.8 多模态 × 原生 MTP profile_run crash `'NoneType' object has no attribute 'size'`）：fail-loud 启动崩溃，两特性单独均正常。参照 #57721 fail-loud 先例不收录、留观（若修复时发现静默面再编目）。
- **vLLM #58035**（DiffusionGemma post_norm shape mismatch 257 vs 256 杀引擎）：结果面 fail-loud（RuntimeError→EngineDeadError）。**静默 clamp 子机制**（`sc_embeds[slot,:257]` 对 256 长维度静默截断）有诊断价值，但触发面窄（固定尺寸 per-slot buffer × 超 canvas 长度请求）且崩溃可见。留观。
- **vLLM #58214**（PR #56323 后 6 个 CPU Triton fallback monkey-patch 模块级全局而 dispatcher 持 `self.kernel`——arm64 spec decode 预期 broken）：报告者为静态源码分析（自述未执行），预期 fail-loud TypeError；仅 linux/arm64 生产路径（x86 有 triton、CI 不覆盖）。留观，待运行时 traceback 确认。
- **vLLM #58135**（`/score` Unicode 截断后重分词超 per-input token 预算，CJK/emoji ZWJ）：静默预算超限真实（HTTP 200、超限 1 token），但影响面为资源/成本与边际分数变化、单端点 pooling 路径、客户端可自行约束。参照 #57927 单例留观先例。
- **vLLM #58191**（watermark 每请求 opt-out 绕过服务器强制水印）：安全/策略执行面，非推理正确性。不收录。
- **vLLM #58155**（DeepSeek V4 thinking 模式不忽略 temperature/penalties，与官方 API 行为分歧）：API 兼容域，非静默损坏。不收录。
- **vLLM #58145**（Mistral grammar 路径在主事件循环重建 tool parser + 全词表 regex 编译，每请求）：性能。不收录。
- **vLLM #58144**（`--api-key` 被 `/invocations` 绕过）：安全面。不收录。
- **vLLM #58087**（KVCR secondary tier 工厂传入 `backpressure_detector=None` 致 TypeError 启动失败）：fail-loud 启动崩溃、实验性适配器。不收录。
- **vLLM #58063**（NixlConnector MNNVL 远端引擎状态滞留 67GiB/GPU 致 P/D 替换死锁）：资源生命周期治理，fail-loud（显存检查 ValueError 退出）。不收录。
- **vLLM #58178**（UVA offload：MXFP4 `process_weights_after_loading` 经 `replace_parameter` 静默反卸载层）：Feature/Design 议题 + 资源治理（内存节省静默失效），非正确性。不收录。
- **vLLM #58134**（CI Failure MI355 concat_mla_q fp8）：CI 分拣。不收录。
- **SGLang #40623**（`flashinfer_megamoe` × EAGLE spec：draft 走 unquant Triton fused-MoE IMA 崩溃）：fail-loud IMA；组合未在 validated_architectures 内属未支持配置。参照 #57941 fail-loud IMA 先例留观。
- **tt-metal #57394**（T3K FABRIC_1D_RING CCL 全挂 + Galaxy assert）：fabric bring-up/基础设施。不收录。
- **tt-metal #57370**（upstream BH ccl smoke 仅 loudbox PCC 失败）：CI 环境特异性 flake。不收录。
- **tt-metal #57403**（Prefill↔Decode migration 支持）：feature。不收录。
- **tt-metal #57360–#57367**（8 份静态审计 meta-issue：sdpa/kv_cache/paged_cache/indexer_score、create_qkv_heads、quasar binary/matmul、normalization、data_movement、to_layout 等，共 ~86 项缺陷）：静态审计产物、自述"有小概率误报"、无逐项复现数据。整批暂记留观——待单项缺陷获得运行时证据后逐条编目（#57364 中 `windowed_mask_gen.hpp` 的 mid-loop source race 已显式与 #50974/#51223 界分，是后续优先核实项）。
- **flashinfer #5393**（main CI test_prefill_cudnn_backend rel err 1.259e+00 超容差 5e-03，自动分拣）：CI 信号真实但无根因定位（cudnn prefill 后端数值分歧值得跟踪）。留观。
- **arxiv**：5 组查询命中全部已编目或已判（SProbe/TrainSDC/SCOUT/LLM-PRISM/BitSifter/2609.17983/2608.07408/2608.05863/2606.28958/2604.00726/2604.10390 已编目；Nexus/ReaLM 为已判方法论文不收录）。无新编目论文。

## 合并动作摘要

- 主文档新增 **13 条**（8 个新子签名 + 5 个已知类别变体）：1.241–1.244、5.42–5.44、6.252、7.96、10.82–10.84、11.85
- 总条数 1171 → **1184**；编号编目 1057 → **1070**（K* 扩展 114 不变）
- 分类分布：第 1 类 240→244、第 5 类 41→44、第 6 类 251→252、第 7 类 95→96、第 10 类 81→84、第 11 类 84→85
- 观察：1.241 与 1.242 构成"同一数学函数 host/device 与 unary/binary 双入口数值分歧"对；10.82 首次把 parser 流式分歧根因定到 token-ID 层；11.85 使 MTP 接受率静默坍塌族新增配置传播断链成因；7.96 是量化 MoE 内核首条 TMA 描述符竞态签名。

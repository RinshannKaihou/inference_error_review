# LLM 推理故障每日扫描报告 — 2026-09-21

- **扫描日期**：2026-09-21（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-19`（与 09-20 扫描窗口重叠两日，已逐条去重）；arxiv 按提交时间倒序取最近 5 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1158 条（2026-09-20 更新后）；编号编目 1044 条 + K* 扩展 114 条

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-19 wrong OR incorrect OR corruption OR NaN` | 13 命中 |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-19` | 30 命中 |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-19 wrong OR incorrect OR corruption OR NaN` | 7 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-19 wrong output OR incorrect OR NaN` | 2 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-19 wrong OR incorrect OR corruption` | 1 命中（RFC） |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-19 wrong OR miscompile OR NaN` | 0 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-19 numerical OR wrong OR pcc OR silent` | 19 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-19 wrong OR incorrect OR NaN` | 2 命中 |
| 9 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-19 quantization OR "KV cache" OR deterministic` | 31 命中 |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-19 silent OR "bit flip" OR "weight corruption"` | 25 命中 |
| 11–15 | arxiv | `"silent data corruption" AND LLM` / `"bit flip" AND "LLM weights"` / `"KV cache" AND corruption` / `"LLM inference correctness"` / `"GPU soft error" AND "machine learning"` | 5/1/5/1/0 命中 |

GitHub Search API 10 组查询全部有效、无限流 403、无 Validation Failed；core REST 详情调用 32 次全部成功（`--http1.1`）。arxiv 5 组查询全部成功（1 组解析首访失败后以 `-o` 落盘重发成功、1 组 0 命中，无 429）。

## 已编目/已判条目（本次跳过）

以下命中条目经编号核对已在主文档中（或 09-20 扫描已明确判定），本次不重复添加：

- vLLM #57822（→6.248）、#57740（→10.76）、#57721（09-19 已判：fail-loud 启动 assert 不收录）、#57794/#57649（RFC）、#57759（09-20 已判：fail-loud 无根因留观）、#57748（09-20 已判：Feature 请求）、#57838（09-20 已判：性能）
- SGLang #40459（→6.249）、#40417（09-20 已判：性能域）、#40370（误投已删）、#40305（Feature）
- transformers #48947（→6.246）、#48954（serge 集成 triage，closed）
- tt-metal #57116（→1.233）、#57115（→1.234）、#57095（→1.235）、#57088（→1.232）、#57135（09-20 已判：工具链/性能）、#57109（09-20 已判：代码重复治理）、#57086（测试增强 tracking）、#57090（CI 文件治理）
- flashinfer #5339（→11.83）
- TRT-LLM #19501（RFC）；triton 窗口 0 命中

## 🆕 新发现（本次合并，共 13 条）

### A. 全新子签名（7 条）

1. **[新类别/新签名] tt-metal #57248 — Metal 2.0 基座 `ProgramSpecFactoryConcept` 缓存命中路径静默错绑别名输入张量** — 绑定解析按 `MeshTensor` 地址 `std::find_if` 取**首个**匹配：同一 handle 出现在两个参数槽时两者都解析到 index 0 并**冻结进缓存条目**；后续同 spec 不同张量调用 cache hit 复用陈旧索引表——`concat([x,x])` 预热后 `concat([a,b])` 返回 `concat(a,a)`，wrong results 无错误。concat 有 op-local workaround，但根因在共享 spec adapter，**每个基座概念多输入已移植 op 结构性暴露**（`repeat_interleave` 单 handle 多传、扩散引导 `concat([latents,latents])` 是典型触发）；受是否命中取决于**调用顺序**。CustomProgramSpecFactoryConcept 路径不受影响。→ 归入第 1 类，Cov NEW（程序缓存按地址绑定 × 别名参数折叠子签名）。
2. **[新类别/新签名] tt-metal #57163/#57164 — `nonzero` ROW_MAJOR sharded reader 构造 bank-major 设备页索引喂给要求逻辑页索引的 `TensorAccessor`** — `page_id = bank * pages_per_bank + bank_page_idx` 把 shard 定向编码进页 id，而 `TensorAccessor::get_bank_and_offset` 自身已按 `tensor_shape_in_pages` 做物理映射——**双重映射**。两公式仅在 `grid_w==1` 或 `pages_per_bank==1` 时重合；WIDTH/BLOCK-sharded 且每 shard 多页行时内核读错页、报告来自错误行列的非零坐标。**所有访问都在界内，无 assert——静默错数据**。#57164 为同文 dup（已 closed）。→ 归入第 1 类，Cov NEW（坐标系双重映射子签名；与 1.196 slice bounding-box vs corerange 序分歧、1.224/1.225 gather/scatter 独立折叠同族——分片坐标系约定错位）。
3. **[新类别/新签名] vLLM #58020 — 引擎侧解析的 prefix-cache match unit 不向 worker 传播，混合组下 mamba checkpoint 静默丢弃** — `EngineCore` 把 `hash_block_size` 解析为各组 block size 的 GCD（如 attention 16 / mamba 1600 → 16）但只用于调度器；worker 侧消费者（`mamba/checkpoint.py:86`）用 `prefix_match_unit or block_size` 自行回退到本地 1600。**checkpoint 位置按粗粒度单位计算 floor 到 0、被判无效、builder 发出 `checkpoint=None`——无错误丢弃**。弹性扩容/重启新建 worker 同样无法采用在役引擎已选定的几何。对照 `kv_cache_layout` 已有引擎解析→worker 采纳的管道。→ 归入第 6 类，Cov NEW（解析值传播缺失——配置权威双源子签名）。
4. **[新类别/新签名] vLLM #57890 — HunYuan Dense V1 dynamic RoPE `Tensor.__bool__` 强制 D2H 同步破坏 CUDA graph capture，全 cudagraph 模式启动死** — Transformers 后端 `@dynamic_rope_update` 的 `if seq_len > max_seq_len_cached:` 在 capture 中执行 host 侧同步 → `cudaErrorStreamCaptureInvalidated`。vLLM `max_model_len <= max_position_embeddings` 使整个动态更新实为 no-op——**但检查本身仍同步**。≤0.28 原生 `hunyuan_v1.py` 在 `__init__` 静态计算 NTK-alpha inv_freq 时 CUDA graph 正常（回归）。workaround `cudagraph_mode=NONE`。→ 归入第 9 类，Cov NEW（host-sync 于 capture 段——no-op 检查的副作用子签名）。
5. **[新类别/新签名] vLLM #58029 — Whisper `verbose_json` 在每个 chunk 切点静默丢弃最后一个完整 segment 之后的词** — 长音频按 30s 窗切分，切点落语音中时带时间戳解码的最后一个完整 segment 后、切点前的词从 `segments` 与 `text`（由 segments 拼装）双丢失；`json`/`text` 用 `<|notimestamps|>` 解码保留这些词。非 #32588 时间戳漂移、非 #52927 切点扫描 bug（切点落在语音内即丢）。espeak-ng 合成语音确定性复现：67 词 vs 59 词；977s 无停顿剪辑上损失按比例放大。→ 归入第 10 类，Cov NEW（带时间戳解码的 segment 完整性窗口子签名；与 10.41 Whisper 提前 EOT 静默丢音频同域新方向——丢的是"最后 segment 后的词"而非音频头尾）。
6. **[新类别/新签名] vLLM #57950 — FunctionGemma parser 对连字符工具名的行为随流式/非流式分裂：非流式漏 tool call 并泄漏控制帧** — 声明 `get-weather` 时流式返回结构化 tool call、非流式无 tool call 且把完整 `<start_function_call>` 控制帧作为 assistant content 暴露（含本应被 parser 消费的模型控制 token）。改名 `get_weather` 两模式都正常。应用从流式切非流式时**静默收到普通 content 而非工具调用**。→ 归入第 10 类，Cov NEW（同一输出流式/非流式双解码路径分歧子签名；与 10.75 四族 converter 流式/非流式语义分裂同维度）。
7. **[新类别/新签名] vLLM #57986 — `allowed_token_ids` + `bad_words` 交集为空时采样器静默破防，返回 allowlist 外 token** — allow `[264]` 同时 ban `" a"`（token 264）：两约束各自有效、组合使合法 next token 集为空——vLLM 接受请求返回 200，但生成 token 0 四次（不在 allowlist）。非空 `allowed_token_ids=[]` 请求；两字段各自单独使用均正常，组合不被报为不支持。依赖 allowlist 强制固定词表的应用（标签 ID/路由选择/约束解码）会**在成功响应里接受非法结果**。→ 归入第 11 类，Cov NEW（多约束组合交集空时静默失效子签名；与 10.47 xgrammar pattern+minLength 丢长度界同属"多约束交叉点未验证"，但本条在采样器层且失效方向是**放飞**而非拒收）。

### B. 已知类别新变体（6 条）

8. **[已知类别] tt-metal #57149 — `ttnn.grid_sample` 注册 golden 忽略 `use_precomputed_grid`：文档化路径返回 NaN + 3× 过宽张量** — flag 落 `**_` 被丢弃；golden 无条件按标准 grid 解包，precomputed grid `(N,H,W,6K)` 的 6 元素（int16 位型的 h0/w0 + 4 个 bf16 双线性权重）被当 3 对坐标喂 `F.grid_sample`——NaN 与无关有限值、宽度 3×。fp32 host 实测 max abs diff 与形状对照。→ 归入第 1 类（验证面失效族），Cov var:1.180→1.233（golden 吞 kwargs 姊妹对第三成员：1.233 单 op 手写、1.234 工厂签名批量、本条 golden **语义模式**吞噬——flag 不是数值参数而是路径选择）。
9. **[已知类别] tt-metal #57205 — `topk_large_indices` column split 遗留 Tensix 状态，毒化下一次 `indexer_score` 运行的首 tile（glm5 q_bf16 PCC 0.54）** — 确定性测试序交叉污染：topk_large_indices 之后跑 indexer_score PCC 恒 0.5409，单独跑/复位后/连续第二个 indexer 程序均过。输出面：640×56320 score 平面上 175 个元素错、全部落在每 core column 一个 8×8 块内；值非正确值的排列、沿行逐列约翻倍——落错 tile 的累加特征。两运行间插入 eltwise `ttnn.add` 即恢复、1.1MB/core 清零 sharded tensor 不恢复——**非 L1 数据残留而是计算单元状态**。需 #57077 树。→ 归入第 1 类，Cov var:1.129（ttmetal 跨算子架构状态位泄漏族——1.129 是 zero-flag 状态泄漏，本条扩展到 topk→下游 indexer 的功能性污染，静默错值且测试序依赖）。
10. **[已知类别] vLLM #57938 — LMCache KV load 失败路径 `_update_from_kv_xfer_finished` 断言杀 EngineCore（异构注意力 6 KV 组模型）** — `failure_policy=fail` 下 LMCache 读错误（NVMe/hipFile I/O）触发 `assert req_id in self.requests` AssertionError，EngineCore 致死；期望行为是优雅重算。Gemma 4 31B 混合 SWA+全局注意力、MI325X。→ 归入第 6 类，Cov var:6.139（KV load 拒绝后状态机部分回滚分歧族——回滚路径删除请求后完成事件仍到达的时序洞）。
11. **[已知类别] vLLM #57883 — async EPLB 重排步计数器单调跑负永不停摆：无警告无错误（EP16 跨节点 nixl）** — `step_interval=1000` 应在 1000 触发重排；计数越过 1000/2000/3000 持续为负（-1300/-1400/-1500/-2000），async worker 多 EP rank 停在 `consumed_event.wait()`。EPLB 静默失效=负载不均持续恶化，日志无任何告警。→ 归入第 5 类，Cov var:5.32（DBO 微批早退 ring handoff 挂起族——异步 rearrange 事件握手丢失方向）。
12. **[已知类别] vLLM #57935 — `/v1/chat/completions/batch` 在 `include_reasoning=false` 时经 logprobs 与 token_ids 泄漏隐藏推理链** — 响应移除 `message.reasoning` 但保留完整原始生成的元数据：`<think>PRIVATE_CHAIN</think>` 可从 logprob token 字符串或 `choice.token_ids` 完整恢复。Qwen3-0.6B + 批处理响应组装器确定性探针复现。→ 归入第 10 类，Cov var:10.65（include_reasoning flag 语义失效族——10.65 是 Responses API 忽略 flag，本条是 batch 端点 flag 部分生效：正文剥离但元数据通道全放行）。
13. **[已知类别] vLLM #57929 — `/v1/completions/derender` 丢弃请求的 `prompt_logprobs`：HTTP 200 但字段为 null** — 同一请求 `prompt_logprobs=1` 在 `/v1/completions`、`/v1/completions/render`、`/inference/v1/generate` 均返回，唯独末端 derender 步丢弃；生成文本与 usage 正常，缺失可无声滑过。→ 归入第 10 类，Cov var:11.51（prompt_logprobs 字段失效族——11.51 是 drafter graph buffer 复用覆写（内核层），本条是管线末端 derender 组装丢弃（API 层），同字段双位点）。

### C. 评估后不收录（记录备查）

- **vLLM #57999**（后台 `/v1/responses` 取消已完成响应被拒"Cannot cancel a synchronous response"）：API 语义边缘 case，非推理正确性。不收录。
- **vLLM #57998**（`/v1/responses` output items 恒 `status:"completed"` 即使 incomplete）：item 级状态字段无信息量属 API 元数据完整性，顶层 status 正确。参照 #57725 错误报告质量先例留观。不收录。
- **vLLM #57997**（`/v1/messages` 双错误信封 + Python repr）：错误报告质量/客户端兼容性，非正确性。不收录。
- **vLLM #57996**（`echo:true` 被 `return_token_ids:true` 静默关闭）：输出字段静默丢失、非生成内容损坏；与 10.52/10.65 flag 被忽略族相近但影响面为回显元数据。留观（若后续成族再编目）。
- **vLLM #57974**（V1 InputBatch 残留 prompt_embeds 张量）：有界内存滞留（≤8 GiB host）、无正确性影响，资源治理。不收录。
- **vLLM #57971**（SM8x sparse-MLA prefill 64× 冗余 KV 流量）：纯性能（[Perf] 前缀）。不收录。
- **vLLM #57944**（Batch runner 误分类 diarized 转录为 streaming，已 closed）：转录元数据分类、已修复。不收录。
- **vLLM #57941**（spec decode draft 上下文短于 target 时 IMA）：fail-loud IMA、`--hf-overrides` 扩窗+短窗 drafter 组合；参照 #57721 fail-loud 启动/运行崩溃先例。其静默面（误导向 target forward 的 traceback）有诊断价值但无输出损坏。留观。
- **vLLM #57936**（`--kv-cache-memory` 建议双计 CUDAGraph 内存）：容量建议工具回归、非正确性。不收录。
- **vLLM #57932**（SM120 GLM-5.3-Flash FlashInfer 后端拒启 `MLA not supported`）：fail-loud 启动拒绝（ValueError 明确原因）。参照 #57713 拒绝型先例。不收录。
- **vLLM #57927**（chunked embeddings 非单位归一化破坏点积）：pooling 归一化语义问题——`use_activation=true` 时 chunked 处理返回非归一化向量，点积 0.616 阈值拒真。正确性影响真实（阈值决策翻转）但根因是归一化缺失于 chunk 路径、单例且 workaround 明确（客户端侧 L2 归一化可修）。留观，若后续复现于其它 pooler 再编目。
- **vLLM #57878**（fp8 FlashMLA sparse decode `--enforce-eager` SM90 init crash）：fail-loud 启动 crash（aten::new_empty dispatch failure）。参照 6.248 之外 fail-loud 先例不收录。
- **vLLM #57853**（sleep 中 weight update 段错误杀 EngineCore）：RL weight-update 生命周期治理（sleep→update_weights 窗口无拒绝），crash fail-loud、无静默输出损坏。留观（与 6.149 sleep level=1 wake 静默损坏 LoRA 状态相邻但本条 fail-loud）。
- **vLLM #57847**（DecodeBenchConnector 共享 state 张量重复 KV 填充）：benchmark 工具连接器性能（RPC 超时于 MLU 后端）。不收录。
- **vLLM #57839**（watermark 检测 server 不支持 dual_key_gumbel、alpha 默认分歧 0.2 vs 0.1）：水印工具参考实现缺口；alpha 默认分歧改变 p 值属检测器校准边缘。不收录（工具域）。
- **vLLM #57894**（deepep_low_latency buffer 大小忽略 EPLB 冗余专家、2048 阈值过时）：容量规划文档/阈值陈旧，非正确性。不收录。
- **vLLM #57981**（humming 线性后端 json.loads 造成 Dynamo graph break）：性能/可用性（W8A16-FP8 不可用 torch.compile）。不收录。
- **vLLM #57935**（batch logprobs/token_ids 泄漏隐藏推理链 `include_reasoning=false`）：→ 已编目为 **10.80**（B 节第 12 条）：隐私/安全面正确性（隐藏推理可经 logprobs 恢复），静默且影响真实。
- **vLLM #57929**（derender 丢 `prompt_logprobs`，HTTP 200 但字段 null）：→ 已编目为 **10.81**（B 节第 13 条）：请求字段静默丢弃于管线末端组装步。
- **vLLM #57956**（AMD mirror 声明与 test-amd.yaml 步骤不一致）：CI 配置治理。不收录。
- **SGLang #40564**（MooncakeHostTensorAllocator 包装 null 指针而非在分配点失败）：**null 指针张量远端延迟爆炸**（pinning 时才浮出）——错误表面位移有代表性，但触发条件是 host 内存耗尽（mmap 失败）路径、fail-loud（远端 pin 报错）。参照 #40504 fail-loud 先例。留观。
- **SGLang #40504**（KDA short-conv 硬编码 fp32 造成 2× 带宽浪费）：纯性能。不收录。
- **tt-metal #57212**（comparison mode 从不 grade scalar-first div 及 tensor-tensor 调用）：验证基建覆盖缺口（unsatisfiable 阈值 1.1 探针证明未比较），`xfail(strict=True)` 已在测试中记录。与 1.135（PCC 门 parametrize 不可达从未运行）同族——**1.135 先例是编目了的**。但本条信息已在测试 marker 中固化、且 add/div tensor-tensor 跳过在 unmodified main 即如此属已知豁免。折中：并入 1.239 所在验证面族语境、不占独立编号（同日 golden kwargs 族已 +1，验证面失效三条边界足够）。留观。
- **tt-metal #57211**（整数 dtype 独立 unary op 拒绝但融合为 binary activation 时放行：int32 读出 float 位型 0x3F000000=0.5）：**双路径语义分裂 + 位型重解释**，机制上是 1.221（binary_ng 融合激活混合 dtype unpack 位域旋转）的融合路径绕过 dtype gate 变体。但 standalone 路径 fail-loud（TT_FATAL）、fused 路径静默错值——静默面真实。**未编目原因**：报告者立场是"两答案应统一"（设计歧义而非确定缺陷方向），且 1.221 已覆盖 binary_ng 融合激活 dtype 混淆机制。作 1.221 语境补充引用。留观。
- **tt-metal #57230**（profiler 孤儿 ZONE_END 使通过测试变 TT_FATAL + 95 分钟挂起）：profiler 基建。不收录。
- **tt-metal #57236**（prefill runner launch/reconnect 健壮性）：工具链 bring-up。不收录。
- **tt-metal #57155**（Matt Pocock Skills Reviewer failed）：CI 机器人。不收录。
- **tt-metal #57178**（streaming SDPA fp32 rescale 慢于 legacy 8q/1kv bfp8 +17~30%）：性能；与 1.235（fp32 dest 精度）同内核的性能面。不收录。
- **flashinfer #5389**（Cake GDN 冻结变体查找对 SCALE 精确浮点相等匹配，等价表达式 1 ULP 差 fail-closed）：fail-closed 抛 `CakeGDNUnsupportedError`——静默面是"等价重写禁用整个 decode 路径"但错误可见。参照 #57713 拒绝型先例。留观（`128**-0.5` vs `1/math.sqrt(128)` 的 load-bearing 拼写陷阱已记录）。
- **TRT-LLM #19501**（训练/推理数值一致性 RFC）：RFC。不收录。

### D. arxiv

5 组查询全部成功（1 组 0 命中），命中均已在编目或已判方法论文：

- `"silent data corruption" AND LLM`：SProbe（2609.19743，09-18 已编 §3.1 补充来源）、TrainSDC（已编目）、SCOUT（已编目）、LLM-PRISM（已编目）、Exploring SDC（已编目）
- `"bit flip" AND "LLM weights"`：BitSifter（2510.00490，已编目 3.15）
- `"KV cache" AND corruption`：Contiguity, Not Importance（2609.17983，09-17 已记录）、Addressable Memory（2608.07408，已编目 14.7）、Runtime Observability（2608.05863，已编目）、Nexus（方法论文不收录）、When Latent Agents Lie（2606.28958，已编目）
- `"LLM inference correctness"`：ReaLM（2503.24053，2025-03 ABFT 方法论文不收录）
- `"GPU soft error" AND "machine learning"`：0 命中

无新编目论文。

## 重点突出的新发现

1. **1.237 的"程序缓存按地址绑定 × 别名折叠"**（tt-metal #57248）：缓存键解析用 `std::find_if` 首个地址匹配——别名调用（同 handle 多槽）把两参数解析到同一索引并**冻结进缓存条目**，后续不同张量调用复用陈旧索引表。失效是否出现**取决于历史调用顺序**（先别名后distinct 才触发），使复现呈非确定性伪影；暴露面是"每个基座概念多输入已移植 op"的结构性面。与 6.211（mm uuid 逐字缓存键碰撞）构成跨层缓存键退化对：设备程序缓存 vs 服务栈请求缓存。
2. **6.250 的"配置权威双源"**（vLLM #58020）：引擎解析的 `hash_block_size`（GCD=16）只活在调度器，worker 自行回退本地 `block_size`（1600）——同一配置项两个权威源，mamba checkpoint 位置 floor 到 0 被**无错误丢弃**。与 6.207（SGL_ 前缀重写晚于 deprecation registry）同属"解析时序/权威分裂"，但本条是**引擎↔worker 传播断链**首个条目，弹性扩容 worker 同样受害。
3. **11.84 的"约束组合交集空时静默破防"**（vLLM #57986）：`allowed_token_ids` 与 `bad_words` 各自有效、组合合法 token 集为空——采样器不报错不拒绝，直接生成 allowlist 外 token（实测 token 0×4）。这是约束采样栈"多约束交叉点未验证"的首个**放飞方向**实例（10.47 xgrammar 是拒收方向、10.73 结构化输出 bitmask 是不施加方向）——三方向构成约束栈失效谱系。
4. **1.240 跨运行 Tensix 状态污染的功能性首证**（tt-metal #57205）：topk_large_indices → indexer_score 测试序依赖的 PCC 0.54 毒化，eltwise add 可清、L1 清零不可清——指向计算单元状态而非数据残留；1.129（zero-flag 状态泄漏）的机制假设（架构状态位跨算子存续）首次在**功能输出面**得到复现级证据。

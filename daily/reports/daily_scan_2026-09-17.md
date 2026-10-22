# LLM 推理故障每日扫描报告 — 2026-09-17

- **扫描日期**：2026-09-17（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-15`（与 09-16 扫描窗口重叠两日，已逐条去重）；arxiv 按提交时间倒序取最近 5 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1113 条（2026-09-16 更新后）；编号编目 999 条 + K* 扩展 114 条

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-15 wrong OR incorrect OR corruption OR NaN` | 19 命中 |
| 2 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-15 wrong OR incorrect OR corruption OR NaN` | 10 命中 |
| 3 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-15` | 25 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-15 wrong output OR incorrect OR NaN` | 2 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-15 wrong OR incorrect OR corruption` | 2 命中 |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-15 wrong OR miscompile OR NaN` | 2 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-15 numerical OR wrong OR pcc` | 39 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-15 wrong OR incorrect OR NaN` | 3 命中 |
| 9–25 | GH REST | 17 次 issue 详情抓取（8+9） | 全部成功（`--http1.1`） |
| 26 | arxiv | `"silent data corruption" AND LLM` | 5 命中（全部已编目） |
| 27 | arxiv | `"bit flip" AND LLM inference` | 5 命中（已编目 4 + 1 不相关） |
| 28 | arxiv | `"KV cache" AND corruption` | 5 命中（**1 新** + 已编目 4） |
| 29 | arxiv | `LLM inference correctness` | 空格被拆词为 OR，返回不相关最新提交（记录，无有效命中） |
| 30 | arxiv | `GPU soft error AND machine learning` | 同上拆词问题（记录，无有效命中） |

GitHub Search API 8 组查询全部一次成功，无限流 403；core REST 17 次全部成功。arxiv 3 组有效查询成功，2 组因查询串空格被拆解为 OR 返回无关结果（已记录）。

## 已编目/已判条目（本次跳过）

以下命中条目经编号核对已在主文档中（或 09-16 扫描已明确判定"不收录"），本次不重复添加：

- vLLM #57059（→1.222）、#57087（→6.237）、#57064（→6.238）、#57019/#56889（→6.239）、#57156（→8.89）、#57136（→11.80）、#57032（→11.79）
- SGLang #39830（→6.240）、#39626（→7.91）、#39684（→7.92）、#39836（→10.69）、#39782（→12.38）、#39831（→12.39）、#39708（→13.39）、#39692（09-16 已判：证据不足留观）、#39515（09-15 已判：性能，维持）
- vLLM #57223（09-16 已判：训练侧 mask 消费者，留观）、#57125（fail-loud）、#57138（fail-loud）
- flashinfer #5252（→1.223）、#5260（→8.88）、#5233（09-15 已判维持）
- transformers #48818（→13.38）、#48842（训练侧）
- triton #11795（09-14 已判维持 fail-loud 留观）
- vLLM #57449（MLA workspace OOM 启动失败 fail-loud）、#57440（权重加载内存治理非性能正确性）、#57350（性能）、#57376（API 字段兼容性观察）、#57303（/dev/shm 泄漏资源治理）、#57259（加载失败 fail-loud）、#57276（功能支持问题）
- tt-metal #56908 中 L1 sizing/hang 部分归 #55075 追踪（语境）；#56824（fail-loud assert + 设计限制）；#56856（memory_config 忽略——已 closed、放置语义非性能正确性且 09-17 当日关闭）；#56808（测试编码基准错误）；#56823（测试覆盖任务）；#56791（perf counter 语义）；#56872/#56863/#56770（CI/机器人）
- vLLM #57373（RFC）、#57346（Feature）、#57069（Feature）、TRT-LLM #19232（RFC）
- arxiv：TrainSDC（2608.30769）、SCOUT（2608.11034）、LLM-PRISM（2604.10390）、BLINK（2609.06781，09-10 已判 CNN 域不收录）、JITterFlip（2608.29745→3.18）、GBFA（2608.25276→3.16）、ROBBIN（2608.23774→3.17）均已编目；Nexus/Moment-KV/2606.28958 等方法论文维持不收录

## 🆕 新发现（本次合并，共 12 条）

### A. 全新子签名（6 条）

1. **[新类别/新签名] tt-metal #56880 — `ttnn.gather` rank>4 双操作数独立折叠错位（50% 输出错）** — `silicon wrong data`：`pre_gather_transform_tensor` 对 input 与 index 各自 `squeeze_from_ND_to_4D`（用各自 extents 合并 leading dims 为线性 id），kernel 却把两侧折叠 id 当共享坐标逐 stick 对读；host 校验明文允许 `index_shape[d] < input_shape[d]` 的内部 leading dim 不等（合法输入），此时同一融合 id 在两张量中指向不同元素。Blackhole 实测 `(2,3,4,5,64)`×`(2,2,4,5,64)` dim=-1 → **2556/5120 元素错（~50%）**，全部越界访问不存在、无 assert。→ 归入第 1 类（硬件/SDC，ttnn 数据移动原语静默错值），Cov NEW。
2. **[新类别/新签名] tt-metal #56876 — `ttnn.scatter` 同族独立折叠：行写错位+目标行漏写** — 与 #56880 同根因（`transform_to_4d_tensor` 独立折叠），reader kernel `in_bounds()`/`to_id()` 把 input 折叠坐标当作与 index 共享：input 行 `(0,2)` 被解析为 index 行 `(1,0)`——**源行写进错误位置且原目标行漏写**，5 组形状实测 199–2407 元素错、静默。→ 归入第 1 类，Cov NEW（与 #56880 构成同族对，分别编目为 gather/scatter 两条）。
3. **[新类别/新签名] tt-metal #56827 — block-sharded 宽度 concat 游标按 shard 容量推进错位** — `ConcatBlockShardedProgramFactory` 宽度分支 `cum_w += input_shard_w * shard_grid_w`（容量）而非真实宽度：ragged 末 shard 的 padding 列使后续每个输入错位一列（间隙为陈旧 L1 数据、尾部越出输出被丢弃）。Wormhole 实测 `(1,1,64,11)` 双输入 **704/1408（50%）** 元素错；所有 ragged 配置必错、精确配置全过（11/15/23/31/47 宽度全部 50% 错）。→ 归入第 1 类，Cov NEW。
4. **[新类别/新签名] tt-metal #56853 — `ttnn.pow(int32, k∈{0..3})` 静默返回 float 位模式重解释值** — composite `pow(Tensor, int32_t)` 快速路径（指数 0–3）绕过 dtype policy 调 `power_iterative`（无 dtype 校验）：INT32 输入被按 float32 位模式解释——k=0 返回 `0x3F800000`（1.0f 位型）、k≥2 负输入返回 `0x7FC00000`（quiet NaN 位型）、`pow(x,1)!=x`；声明 dtype 仍 INT32。tensor-exponent 路径同输入正确 TT_FATAL 拒绝（策略本身声明 float_only）。→ 归入第 1 类，Cov NEW。
5. **[新类别/新签名] vLLM #57030 — `StagedWriteTensor.apply_write` int32 行偏移回绕越界写** — `buffer_utils._apply_write_kernel` 中 `row_ptr += row_idx * row_stride`：int32 张量 × host int32 标量在 int32 中乘法，`rows×stride > 2^31` 即回绕为负偏移——1M 上下文 QSA hybrid（乘积 ~2.6e9）首请求即 `Memory access fault`（8 TP worker 毫秒级齐崩）；更危险的是**不 fault 时静默 scribble 到同 allocation 其他位置**。附 10GB 无模型最小复现与 `.to(tl.int64)` 一行修复。→ 归入第 1 类（跨 int32 溢出族 1.167/1.190），Cov NEW。
6. **[新类别/新签名] triton #11836 — `OptimizeThreadLocality` 归约累加器循环内额外使用静默 miscompile** — 归约优化把 `acc = acc + tl.sum(...)` 重写为 thread-local 累积 + 环境外归约时，只检查 reduction/update 的单用途、**不检查 loop-carried 累加器块参数是否有其他使用者**：`createYield` 把原累加器改为恒等携带（初值 0），循环内 `tl.store(OUT, acc)` 快照全部读到 0 而最终值正确——部分正确部分错的最隐蔽 miscompile 形态（LLM 推理中对应"中间 logits/部分和错误"）。sm89 实测。→ 归入第 1 类（Triton 编译器 miscompile 族 1.127/1.192），Cov NEW。

### B. 已知类别新变体（6 条）

7. **[已知类别] vLLM #57267 — hybrid Mamba2 前缀缓存 `mamba_cache_mode="all"` 跨调用非确定** — 同批次同序四次 `generate()`（greedy、同权重、逐次重置前缀缓存）在 token 5 即分叉（16 vs 24）；需"prefill 跨 scheduler step 分块 + cache_mode=all"两条件同时成立，`align`/`none`/不分块均确定；1500 次服务端 fuzz 中 20 对调度完全一致的运行产出不同 token（align 模式 0/700）。两厂商两架构（granite-4.0-h-tiny/Falcon-H1-Tiny dense）复现，排除 MoE。→ 已知类别第 4 类（数值/非确定性，4.12 族），Cov var。
8. **[已知类别] vLLM #57266 — `--mamba-block-size` 显式设置下 `conv_ssm_forward` 越界切片杀 EngineCore** — `num_unaligned_computed_tokens` 用 `num_computed_tokens_p_cpu` 计算的块对齐索引切 `varlen_states`，产出 `[0,48,64,128]` vs `[N,48,64,128]` 广播错误——RuntimeError 在 EngineCore 内抛出，引擎死亡后所有后续请求（含单 token "hello"）HTTP 500 `EngineDeadError`。→ 已知类别第 6 类（服务栈，Mamba 状态管理族 6.204/6.229），Cov var。
9. **[已知类别] vLLM #57429 — gfx942 aiter fused MoE（blockscale FP8 1stage）引擎冻结/无错死亡** — GLM-5.3 FP8 TP8 MI325X：新 batch 宽度触发 `fmoe_bf16_blockscaleFp8_g1u1_vs_ps_silu_64x256E`（kernelName2=nan）分发后引擎停止推进（2000+ tok/s → 0），部分冻结 75s–12.8min 后自愈、部分永不恢复且 8 worker 无任何错误/traceback/HIP error 死亡（6 天 4 次致命死亡）；gfx950 同负载 16026 请求零错误。→ 已知类别第 7 类（量化 ROCm aiter 族 7.68/8.69）——以引擎冻结/静默死亡为主要行为效应，归入第 7 类，Cov var。
10. **[已知类别] vLLM #57423 — FlashInfer autotune config cache 仅 rank0 命中 → 多 rank 启动死锁** — autotuner 只为 rank 0 dump `autotune_config.json`：TP8/EP8 下 rank0 config hit 跳过调优、其余 rank miss 后阻塞在调优同步 all-reduce 上，引擎启动无限挂起（8×H100 NVFP4 MoE）。跨 rank 调优缓存一致性缺陷。→ 已知类别第 5 类（分布式启动挂起族 5.28/5.33），Cov var。
11. **[已知类别] SGLang #39752 — `/v1/messages` 流式 thinking block 缺 signature → Anthropic SDK 冷并发崩溃** — 无原生 thinking signature 的开源推理后端（Qwen 等）流式发出无 signature 的 thinking block；`captured_thinking_signature` 从未被赋值。Anthropic SDK 0.87.0 冷启动并发流初始化时嵌套 `content_block` 保持 raw dict，accumulator 在 `model_dump` 上 AttributeError；首个请求对（主 agent 轮 + 标题生成）同时到达即触发，串行重放不复现（首个请求已预热 type adapter）。→ 已知类别第 10 类（API 模板/流式适配层族 12.38 同层），Cov var。
12. **[已知类别] vLLM #57324 — Claude Code deferred tools 首请求 400 静默重试** — `ENABLE_TOOL_SEARCH=true` 时 Claude Code 首请求只发工具名（无 schema），`AnthropicContentBlock.type` 不接受 `tool_addition` 块 → 400 + ~20KB 错误体；客户端吞掉 400 重发全 schema，会话表面正常——每会话启动静默损失一次往返（一下午 3×400/29×200）。报告者当日自行更正根因定位（非 input_schema 而是消息块类型校验）。→ 已知类别第 10 类（API 适配层静默拒绝/重试，10.42 族语境），Cov var。

### C. 评估后不收录（记录备查）

- **vLLM #57321**（Whisper verbose_json segment id 重复嵌套循环重置）：输出元数据 id 唯一性违反 OpenAI 规范，但**转录文本内容本身不受影响**——非推理正确性、下游以 id 聚合时才出问题。留观。
- **tt-metal #56847**（Kimi 2.7 chunked prefill padded accuracy regression，CI PCC 0.8754<0.88）：仅两条 CI 链接、已 closed、无根因分析。与 6.221（chunked prefill 切进 encoder 区）触发面相近但证据不足，留观（CI 事件类与 8.87 warmup trace 双计同判）。
- **tt-metal #56908**（$3000 bounty：distributed LayerNorm/RMSNorm 2D core grid row-stride 腐败）：bounty 正文描述 tiles_per_core>1 时 reader 线性推进而非全前行 stride——**疑似真实静默数值损坏**，但为 bounty 招标帖（无复现数据、修复导向），scope 与 #55075（L1 sizing + hang）交织。作为语境补充记录于本报告，待复现数据出现再编目。
- **TRT-LLM #19319**（Scaffolding role-filtered messages_to_retain KV 前缀边界语义）：报告者自述"静态检查、不主张推理输出损坏或性能损失"——设计澄清请求非已证实故障。留观。
- **vLLM #57342**（KV offload DMA 路径按页大小选择但按 fragment 拷贝）：[Performance] 标签的性能路径选择问题。不收录。
- **SGLang #39752 之外的 /v1/messages 项**、**transformers #48842**（context parallelism trainer.predict 训练侧）：训练侧/不相关。

### D. arxiv

- **2609.17983 "Contiguity, Not Importance: Budgeted Repair of Stale KV Caches After Document Edits"**（2026-09-16 提交）——文档编辑后陈旧 KV cache 的预算化修复方法论文（contiguity 优先于 importance 的修复策略）。方法论文，不满足编目标准（非故障实例/非 LLM 推理正确性 bug 报告），但与 15.9（KV 压缩证据逐出幻觉弃权）和 K§4（cross-request 缓存键污染）有语境关联。**不收录、记录备查**。
- 其余命中（TrainSDC/SCOUT/LLM-PRISM/JITterFlip/GBFA/ROBBIN/BLINK）均已编目；两组查询因空格拆词失效已记录。

## 重点突出的新发现总结

1. **tt-metal gather/scatter 独立折叠对（#56876/#56880）**：同一根因（`transform_to_4d_tensor` 双操作数独立折叠 + kernel 共享坐标假设）在两个姊妹算子上各产生 ~50% 静默错值——host 校验明文允许的合法形状反而必错，是"API 契约与内核坐标假设脱节"的教科书案例。
2. **vLLM #57030 int32 行偏移回绕**：长上下文（≥~512k）QSA/混合模型的系统性风险——1M 上下文必崩且伪装成"长上下文内核 bug"，更危险的是不崩时静默写坏同 allocation 其他数据；一行 `.to(tl.int64)` 修复已在 6 生产副本验证。
3. **triton #11836 部分正确 miscompile**：编译器优化只保证最终累加器正确、循环内快照全错——对推理栈意味着"最终 logits 对但中间探针/部分和错"，CI 端到端断言无法发现。
4. **vLLM #57267 Mamba2 非确定性**：与 4.12/4.13（SGLang VLM temp=0 非确定、deterministic flag 静默无效）构成三栈证据链——hybrid Mamba + 前缀缓存 `all` 模式下 greedy 批内非确定是系统性模式。

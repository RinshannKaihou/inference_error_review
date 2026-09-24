# LLM 推理故障每日扫描报告 — 2026-09-19

- **扫描日期**：2026-09-19（UTC）
- **扫描窗口**：GitHub `created:>=2026-09-17`（与 09-18 扫描窗口重叠两日，已逐条去重）；arxiv 按提交时间倒序取最近 5 条/查询
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **主文档基线**：1137 条（2026-09-18 更新后）；编号编目 1023 条 + K* 扩展 114 条

## API 调用记录

| # | 类型 | 查询 | 结果 |
|---|------|------|------|
| 1 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-17 wrong OR incorrect OR corruption OR NaN` | 21 命中 |
| 2 | GH Search | `repo:vllm-project/vllm is:issue label:bug created:>=2026-09-17` | 26 命中 |
| 3 | GH Search | `repo:sgl-project/sglang is:issue created:>=2026-09-17 wrong OR incorrect OR corruption OR NaN` | 7 命中 |
| 4 | GH Search | `repo:huggingface/transformers is:issue created:>=2026-09-17 wrong output OR incorrect OR NaN` | 2 命中 |
| 5 | GH Search | `repo:NVIDIA/TensorRT-LLM is:issue created:>=2026-09-17 wrong OR incorrect OR corruption` | 1 命中 |
| 6 | GH Search | `repo:triton-lang/triton is:issue created:>=2026-09-17 wrong OR miscompile OR NaN` | 1 命中 |
| 7 | GH Search | `repo:tenstorrent/tt-metal is:issue created:>=2026-09-17 numerical OR wrong OR pcc OR silent` | 28 命中 |
| 8 | GH Search | `repo:flashinfer-ai/flashinfer is:issue created:>=2026-09-17 wrong OR incorrect OR NaN` | 3 命中 |
| 9 | GH Search | `"silent data corruption" OR "bit flip" LLM created:>=2026-09-17`（全局） | **失效**（46,824 命中全为无关仓库——缺 repo 限定词，查询串被拆解为 OR 全局匹配） |
| 10 | GH Search | `repo:vllm-project/vllm is:issue created:>=2026-09-17 quantization OR "KV cache" OR deterministic` | 39 命中 |
| 11–14 | arxiv | `"silent data corruption" AND LLM` / `"bit flip" AND "LLM inference"` / `"KV cache" AND corruption` / `"weight corruption" AND inference` | 5/1/5/2 命中 |

GitHub Search API 10 组查询 9 组有效（q09 失效已记录）、无限流 403、无 Validation Failed；core REST 详情调用 13 次全部成功（`--http1.1`）。arxiv 4 组查询全部成功（无 429）。

## 已编目/已判条目（本次跳过）

以下命中条目经编号核对已在主文档中（或 09-18 扫描已明确判定），本次不重复添加：

- vLLM #57429（→7.93）、#57423（→5.40）、#57614（→8.90）、#57562（→6.242）、#57493（→4.24）、#57490（→10.72）、#57578/#57532（09-18 已作 8.84 补充来源）、#57580（09-18 已判：synthetic 回归追踪留观）、#57508/#57649/#57479（Feature/RFC）、#57555/#57521/#57342/#57350（性能）、#57449（fail-loud OOM 已判）、#57440（内存治理）、#57376（API 字段兼容已判）、#57354（无根因留观）、#57515（无堆栈不可定位）、#57373/#57346（RFC/Feature）
- vLLM #57680（0.26.0→0.29.0 解码吞吐 3.3× 回归——性能）、#57650（调度初始化失败 worker 不退出——资源治理）、#57671（collect-env MAC——工具）、#57672/#57566/#57530（RFC）
- SGLang #40156（→11.81）、#40144（→11.82）、#40127（→12.40）、#40084（→7.94）、#40305（Feature）、#40370（误投已删）
- flashinfer #5334（→1.230）、#5318（09-18 已判：release 回移缺失不收录）
- triton #11836（→1.229）、TRT-LLM #19319（09-17 已判：设计澄清留观）
- tt-metal #56880/#56876（→1.224/1.225）、#56986（→9.27）、#56955（→1.231）、#56908（09-17 已判：bounty 招标留观）、#56993（09-18 已判：CI 链接留观）、#56945（09-18 已判：潜伏缺陷留观）、#56949（09-18 已判：fail-loud）、#56938（09-18 已作 1.227 语境补充）、#56958（09-18 已作 1.221 语境补充）、#57004/#57023/#57022/#57066（09-18 已判留观）、#56976（性能）、#56954（文档缺陷）、#56873（CI/机器人）
- transformers #48886（09-18 已判：API 契约非推理正确性）

## 🆕 新发现（本次合并，共 12 条）

### A. 全新子签名（8 条）

1. **[新类别/新签名] vLLM #57720 — Mamba-hybrid 单 token prompt 在 full CUDA graph 下每次 greedy 乱答且不确定** — 默认配置（仅设 `max_model_len`）下，恰好 1 token 的 prompt 在 temperature 0 也不确定：granite-4.0-h-tiny 9/9 个单 token prompt 6 次重复各得 4–6 个不同答案、多数与 prompt 无关（乱答）；2–6 token prompt 17/17 全稳定。Falcon-H1 6/6 同形（边界是 token 数不是词）。`--enforce-eager` 与 `cudagraph_mode:PIECEWISE` 均 0/40 失败——锁定 full CUDA graph 捕获内 Mamba/attention 内核在 1-token prefill 形状的执行。→ 归入第 4 类（数值/非确定），Cov NEW。
2. **[新类别/新签名] vLLM #57661 — NIXL P/D 分离按 base_addr 去重传输区域，混合注意力层共享存储基址被折叠吞掉 block_len → 静默花屏** — `layers.2.attn.compressor.state_cache`（block_len=32768）与 `layers.0.attn.swa_cache`（MLA，19008）共享存储基址→折叠进 region 0 不更新 block_len，compressor state 截断+错位传输。DeepSeek-V4.1-Flash 在 2P×TP8+2D×TP8 H200 上 200-token 生成全花屏/空 content，无任何错误；同拓扑 GLM-5.3（纯全注意力）正确。修复（按 `(base_addr, block_len)` 去重）后 0% 花屏。→ 归入第 6 类（服务栈/缓存），Cov NEW。
3. **[新类别/新签名] vLLM #57727 — `POST /reset_prefix_cache` 返回 success 但 CPU offload 层照常服务同一批块** — GPU 前缀缓存被清空、connector 的 CPU 层原样保留：reset 后同 prompt 请求由外部层服务（external hits +1008、GPU 命中 0）。任何"reset 后发送"的冷基线测量都在静默测热路径。→ 归入第 6 类，Cov NEW。
4. **[新类别/新签名] vLLM #57719 — `prompt_embeds` + 任一 penalty 触发 scatter-gather 越界 device assert，EngineCore 死亡全租户 500** — penalty 采样路径假设请求必有 token ids；prompt_embeds 请求无 ids → 索引越界 → CUDA device-side assert → EngineDeadError，服务器永不恢复须重启。16 并发混合长度第一轮必死（10/10），单 repetition_penalty 也死——一次普通请求突发即整服务器 DoS。→ 归入第 6 类，Cov NEW。
5. **[新类别/新签名] vLLM #57723 — outlines 结构化输出后端拒绝自己放行的 EOS：每个自然完成的约束请求返回 500** — guide finished 后 `accepts_tokens([eos])` 恒 False，而 bitmask 侧恰把 EOS 作为唯一合法 token放行；scheduler 把 False 当"grammar rejected tokens"终止请求。180 秒 38,589 请求实测 25,425 个 500（66%）；guidance/xgrammar 0 错；长输出不 emit EOS 是掩盖机制。GPU-free 复现。→ 归入第 10 类，Cov NEW。
6. **[新类别/新签名] flashinfer #5339 — `top_k/min_p_sampling_from_probs` 在 `indices` + per-request 阈值张量时按输出位置而非采样行查阈值 + 越界读** — kernel 用 `top_k_arr[bx]`（输出坐标）而 `top_p`/`top_k_top_p` 自 #923 起用 `row_idx`（数据行）——四入口语义不一致；gather/重排批下系统性行错位，采样分布静默错误；`len(indices)>probs.size(0)` 时读过张量末尾。→ 归入第 11 类（解码/采样），Cov NEW。
7. **[新类别/新签名] SGLang #40285 — glm5_next MTP 跳过守卫测试原始名，`model.language_model.layers.*` 前缀绕过 → MTP 专家权重喂进主模型 loader** — 守卫匹配 `model.layers` 前缀但 checkpoint 命名带 `language_model.` 中缀；混合格式 checkpoint（layer-45 MXFP8 vs 主体 NVFP4）fail-loud shape mismatch，**同格式 checkpoint（nvidia 官方 NVFP4）静默把 MTP 层加载进主模型错误位置**。修复应在 NextN 跳过之前做前缀重命名。→ 归入第 12 类（权重/加载），Cov NEW（与 12.33 GLM 命名处理族互为镜像：丢权重 vs 多加载）。
8. **[新类别/新签名] vLLM #57596 — 跳过外部缓存查找的请求（prompt_logprobs/pooling）仍被计为满额 external miss** — #57269 后 scheduler 正确跳过使用命中，但统计仍记 `num_tokens` 查询、0 命中——外部命中率被系统性稀释，与本地缓存对同类请求的排除规则不一致（并排打印的两个命中率遵循不同规则）。CPU 后台 ExampleConnector 无 GPU 复现。→ 归入第 6 类，Cov NEW。

### B. 已知类别新变体（4 条）

9. **[已知类别] vLLM #57716 — Kimi-K3 fp8 latent cache NaN 字节经 P·V（0×NaN=NaN）掩码旁路引爆 MLA decode** — 并发负载下 verify 步 target logits 全 NaN 且该请求永久中毒；NaN 首现于 self_attn（输入全有限）；池取证：block table 刚跨入页的头几行散布 0x7F/0xFF（e4m3 NaN）。内核级注入 harness 证明 causal mask 把 QK 分数替换为 -1e6 也无用——P·V MMA 把零概率乘以中毒列的字节。源头锁定 `chunk_kda_with_fused_gate` 对混合批 decode 行返回 NaN（#50649 已修 ROCm 副本、NVIDIA 副本缺修）。bf16-KV 对照同引爆。→ 归入第 8 类（KV/注意力），Cov var:6.21（同源内核 + P·V 掩码旁路新传播机制）。
10. **[已知类别] transformers #48947 — `DynamicCache(offloading=True)` 在 hybrid GDN+attention 模型上首次 cache 复用即 logits 错值** — 首次 forward 与 offloading=False 逐位一致（0.0），第一个复用调用 max_abs_diff 2.12（top-1 logit ≈6.2 量级）、确定性复现、逐 greed 步复合放大；linear_attention 层的 recurrent/conv state 经 CPU offload 往返损坏；显存/速度收益方向正常掩盖正确性破坏。→ 归入第 6 类，Cov var:6.240（host 层往返损坏族跨第三栈显形）。
11. **[已知类别] vLLM #57726 — 配置 `--reasoning-parser` 后 `/v1/completions` 的结构化输出静默完全不约束** — grammar 以 `reasoning_ended` 门控，裸 completion prompt 无 think 标记→门控恒 False→bitmask 从不施加：schema `additionalProperties:false` 被违反且对象不闭合，200 无警告；同请求去掉 parser 即受约束；chat 端点不受影响。→ 归入第 10 类，Cov var:10.40（约束静默失效族）。
12. **[已知类别] tt-metal #57088 — disaggregated-prefill KV gate 以字面量 `PREFILL_NUM_LAYERS=61` 定界，深于 Kimi-K2.7 的模型深层从未被评分** — 无 manifest 声明该变量、sc4 臂全按 61 层评分：GLM-5.2（78 层）的 KV PCC sc4 腿只评 0..60，从未检查 61..77——而深层恰是 PCC 最低处（公布 per-rank 最低 0.8621 正来自深端）；缺失层不报 missing（循环从未到达）。→ 归入第 1 类（验证面失效族），Cov var:1.105。

### C. 评估后不收录（记录备查）

- **vLLM #57721**（Mamba-hybrid + spec decode 启动 assert：page padding 未含 speculative 加宽）：fail-loud 启动失败（`num_speculative_tokens=1` 可启动），无静默损坏。参照 #57125/#56949 先例不收录。
- **vLLM #57713**（GLM5.3-Flash fp8 KV cache dtype 在 Hopper 报 `torch.uint8 not supported` RuntimeError）：fail-loud 启动拒绝。不收录。
- **vLLM #57725**（畸形结构化规范返回 500 空消息 + 自引用 `$ref` 过验证后杀请求 mid-generation）：错误报告质量（空 message）为 API 人体工学；自引用 `$ref` 验证缺口与 10.74 同域但为请求级中断、无静默损坏。留观（如后续出现静默损坏实例再编目）。
- **tt-metal #57090**（per-model mesh-graph 描述符逐字节相同）：CI 文件治理（复制粘贴清理），非性能正确性。不收录。
- **tt-metal #57086**（Category-6 unary ops BF16 覆盖工作项）：测试覆盖增强 tracking issue，非缺陷报告。不收录。
- **SGLang #40305**（DEBUG_MEMORY_POOL 适配 page_size=1 allocator）：Feature。**#40370**：误投已删。
- **vLLM #57680/#57521/#57555/#57342/#57350**（性能回归/阈值调优）、**#57650**（资源治理）、**#57671**（工具）、**#57672/#57566/#57530/#57649/#57608/#57479/#57373/#57346**（RFC/Feature）：均非性能正确性。

### D. arxiv

4 组查询全部成功，命中均已在编目或已判方法论文：

- `"silent data corruption" AND LLM`：SProbe（2609.19743，09-18 已编 §3.1 补充来源）、TrainSDC、SCOUT、LLM-PRISM、Exploring SDC（均已编目）
- `"bit flip" AND "LLM inference"`：JITterFlip（已编目 3.18）
- `"KV cache" AND corruption`：Contiguity, Not Importance（2609.17983，09-17 已记录语境关联）、Addressable Memory（2608.07408，已编目 14.7）、Runtime Observability（2608.05863，已编目）、Nexus（方法论文不收录）、When Latent Agents Lie（2606.28958，已编目）
- `"weight corruption" AND inference`：BLINK（已编目）、Recover-LoRA（量化恢复方法论文不收录）

无新编目论文。

## 重点突出的新发现

1. **8.91 的"mask 不保护的 P·V NaN 传播"**（vLLM #57716）：因果 mask 把中毒列的 QK 分数压成 -1e6 后 softmax 概率为 0，但 P·V MMA 仍把 0 乘以 NaN 字节——**任何 latent cache 值损坏只要字节落在被读列上就能绕过 mask 结构传播**。这对 8.x/K§1 全部 value-corruption 类故障是一个通用传播定律级别的观察，且附内核级毒化注入 harness（q_len=8 中招、q_len=1 不中招的 tile 边界证据）。
2. **6.243 的"传输区域去重键不充分"**（vLLM #57661）：NIXL 按 `base_addr` 去重，hybrid 模型不同层共享底层存储但传输几何不同——去重键窄于区域身份的这一模式与 1.224/1.225（gather/scatter 独立折叠坐标系）构成跨层同构（"身份键不足以唯一确定操作语义"）。
3. **10.73/10.74 结构化输出的两个相反失效方向**：reasoning-parser 路径完全不约束（静默放行）vs outlines 后端完成即 500（过度拒绝）——同一 subsystem 在两个后端/配置下朝相反方向失效，部署侧只能靠交叉后端对拍发现。

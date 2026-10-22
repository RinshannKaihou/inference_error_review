# LLM 推理故障每日扫描 — 2026-08-26

- 扫描日期：2026-08-26（UTC）
- 搜索窗口：GitHub `created:>2026-08-24`（与上次扫描 2026-08-25 的窗口 `>2026-08-23` 重叠一天，靠 issue 编号去重）；arxiv 最近提交
- 目标仓库：vllm-project/vllm · sgl-project/sglang · NVIDIA/TensorRT-LLM（0 命中）· triton-lang/triton（0 命中）· tenstorrent/tt-metal · huggingface/transformers
- 主文档合并前条数：717 → 合并后：**731**（新增 14 条：11 个新子签名 + 3 个已知类别变体；另有 2 条补充来源引用）

---

## 一、新合并进主文档的条目（14 条）

### 第 1 类（硬件/SDC 内核原语）+2

1. **[新签名] tt-metal #54439 — `ttnn.pow_bw` 复合公式路径：负底数 log→NaN 在 bf16 pack 坍缩为 +inf，exponent=2.0 时全部 32,639 个负有限 bf16 底数返回 +inf（正确梯度 2a 处处有限）；N300/P300 逐位一致 → 缺陷在复合公式层而非 SFPU** → 1.108
   https://github.com/tenstorrent/tt-metal/issues/54439
2. **[新签名] tt-metal #54447 — tensor-scalar 比较 kernel（2023）是 lt/gt 的逐 lane 补码、无法实现 IEEE NaN 语义；tensor-tensor kernel（2026）已带显式 NaN 拒绝——同一 `ttnn.ge/le` 公共 op 两种重载走两代内核语义分歧，heaviside 根本无法发出 NaN** → 1.109
   https://github.com/tenstorrent/tt-metal/issues/54447

### 第 4 类（数值/精度/不确定性）+2

3. **[已知类别新实例：var fp_reduction_order_nondeterminism] SGLang #36480 — Qwen3.5 VLM 同请求 temp=0：v0.5.12 与 v0.5.17.dev 输出大幅分歧，分叉点 top-2 logprob gap=0.0 完全并列；新 build 自身跨会话 342/342/342 vs 282/134/134 不可复现** → 4.12
   https://github.com/sgl-project/sglang/issues/36480
4. **[新签名] SGLang #36291 — `--enable-deterministic-inference` 对 dense BF16 生效，但对 NVFP4 W4A4 + hybrid-Mamba + DFlash2 静默无效：flag 被接受、resolved ServerArgs 显示 enabled，temp=0 仍不可复现——覆盖缺口表现得与成功完全一致** → 4.13
   https://github.com/sgl-project/sglang/issues/36291

### 第 6 类（服务栈/缓存正确性）+3

5. **[新签名] vLLM #53888 — sleep level=1 wake 静默损坏 LoRA 状态：adapter 停止生效（base/adapter 路由同 token）、两路由均偏离各自 sleep 前固定 seed 基线、一张 GPU residency 少 3GB——全程无错误、HTTP 200** → 6.149
   https://github.com/vllm-project/vllm/issues/53888
6. **[已知类别新实例：var 6.137 offload 契约漂移] vLLM #53951 — SWA store-skip 只存 trailing `tail` 块、eagle lookup 要 `tail+1`（未验证尾块）——合格运行不存在，eagle SWA 组永远 0 命中，KV 无限写入 offload 层永不读回（bytes out 恒增、bytes in 恒零，无警告）** → 6.150
   https://github.com/vllm-project/vllm/issues/53951
7. **[新签名·待证] SGLang #36548 — DFlash2 高并发下最后一条 user message 被拼进他人上下文（编码流中突然回答并发 GPQA 题；模型思维链中察觉异常仍照答）——跨请求 token 序列级串扰，机制未定位** → 6.151（speculative）
   https://github.com/sgl-project/sglang/issues/36548

### 第 7 类（量化）+2

8. **[新签名] transformers #48349 — FineGrainedFP8 的 `modules_to_not_convert` 携带多模态 wrapper 路径，text-only（AutoModelForCausalLM）加载时全部失配 → 厂商指定保 bf16 的 PLE 投影/hyper-connection/linear-attn 被无 scale 静默 cast 到 fp8，数值损坏、加载零报错** → 7.54
   https://github.com/huggingface/transformers/issues/48349
9. **[新签名] SGLang #36423 — ModelSlim W8A8 fused GDN 层（in_proj_qkvz）scheme 为空不被识别 → "静默回退不量化"却把 int8 权重原样拷进 bf16 参数（int8 位型按 bf16 解释），语言路径输出全垃圾；vision 路径 NPU matmul crash** → 7.55
   https://github.com/sgl-project/sglang/issues/36423

### 第 8 类（KV 缓存/注意力）+1

10. **[新签名] SGLang #36390 — DeepSeek-V4-Flash FP8 @ MI300X：输出随 prompt 长度平滑概率性退化（<100 tok 正确、~300 起垃圾、NIAH 0/3、>2048 后叠加非确定）——非窗口边界硬悬崖，符合长度复合数值误差，从不 crash/NaN** → 8.54
    https://github.com/sgl-project/sglang/issues/36390

### 第 10 类（分词器/模板/输入处理）+2

11. **[新签名] vLLM #53831 — DeepSeek-V4-Flash `tool_choice=auto` × 深推理档（xhigh/max）高概率返回 finish_reason=stop + 空 content + 无 tool_calls（模型完成工具选择推理后不再输出）；双变量可分离，区别于 #41132/#51914** → 10.38
    https://github.com/vllm-project/vllm/issues/53831
12. **[新签名] vLLM #53850 — deepseek_vl2/ocr 处理器用裸 assert 校验输入：AssertionError→HTTP 500（应 400）；`python -O` 剥除 assert 后图像计数失配静默通过 zip 错配——文本段丢弃/复制、图文错位，token 序列无声损坏** → 10.39
    https://github.com/vllm-project/vllm/issues/53850

### 第 11 类（解码/采样·推测解码）+1

13. **[已知类别新实例：var 11.41 drafter×grammar 族] vLLM #53777 — DFlash2 + xgrammar json_object：同一 draft token（271）被 FSM 确定性拒绝、单请求最多 7 个重试波；最终输出正确（200/有效 JSON）但延迟尖峰 + ERROR 日志风暴** → 11.53
    https://github.com/vllm-project/vllm/issues/53777

### 第 12 类（权重/加载）+1

14. **[新签名] transformers #48323 — v5 动态加载 `Chunk` op 只校验块数不校验均分：不可整除时 torch.chunk 产出不等长块且块数吻合 → 远端 shape mismatch 或 `ignore_mismatched_sizes` 下静默重初始化权重** → 12.23
    https://github.com/huggingface/transformers/issues/48323

### 补充来源引用（不占新编号）

- **vLLM #53912**（prefix caching + MTP 在 v0.28.0 hybrid Mamba/GDN 上仍损坏输出：部分响应退化为重复标点 `!!!!` 或空 completion；#43559 于 2026-08-06 被关闭但 #51113 调度器修复未覆盖）→ 追加为 **K9.16** 的"缺陷仍开放"证据。
- **vLLM #53868**（SimpleCPUOffloadConnector eager TP=2 夜版 dev1214 wedged at Running:0/Waiting:3（1 capacity + 2 deferred），附 wedged 态 py-spy 栈与 /metrics——#45406 stranding 族的独立生产数据点）→ 追加为 **6.135** 的独立复现引用。

---

## 二、扫描到但未合并的条目（记录备查）

| 条目 | 来源 | 不合并原因 |
|---|---|---|
| GLM-5.3-Flash ROCm：platform guard 遮蔽可修的 `forward_hip()` stale 签名（unexpected keyword `gate_score`） | vLLM #53943 | fail-loud RuntimeError，无静默窗口；常规移植缺口 |
| MTP draft 分配第二份全词表 embedding 使 27B INT4 OOM（基线可容纳） | vLLM #53887 | 资源/OOM 类，非正确性 |
| PR #51726 的 ≥160GiB 内存档误捕 MI300X（191.98GiB）使 max_num_batched_tokens 翻倍破坏 KV cache init | vLLM #53961 | 配置档边界错误，fail-loud（init 失败），机制简单 |
| VLLM_PLE_CPU_OFFLOAD 单卡 warmup 死锁 | vLLM #53960 | 死锁/活性类，无正确性退化 |
| Granite 4.2 FP8 缺 weight_scale 加载失败；--cpu-offload-params 够不到 make_layers 外的 embedding | vLLM #53950 / #53931 | 加载期 fail-fast |
| XPU causal_conv1d 不支持 spec/non-spec 同批调用（RuntimeError） | vLLM #53928 | fail-loud，XPU 平台覆盖项 |
| QSA decode 无 SM121 内核路径（GB10/DGX Spark 上 Qwen3.8-Flash-Next 不可服务） | SGLang #36558 | 平台功能缺口，fail-loud 不可服务，非静默正确性 |
| 条目（续） | 来源 | 不合并原因 |
|---|---|---|
| NEXTN/EAGLE draft embed/lm_head 拷贝在 KV pool sizing 后才释放，压缩 max_total_num_tokens | SGLang #36452 | 容量/perf 类 |
| transformers 连续批办系列（VLM config 读取崩溃 #48298、async start() 空队列崩溃 #48303、hybrid 不支持 #48305、TP shard 缺失 #48306/#48307/#48308、CUDA-graph 捕获崩溃 #48312、WARNING 噪音 #48313） | transformers #48296–#48313 | 多为 fail-loud 新特性（continuous batching）磨合期崩溃；#48296 paged\|sdpa 非因果注意已单独编目候选——因依赖 CB manager（实验特性、训练侧 forward 触发）本轮记备查，不下编 |
| TT_METAL_WATCHER segfault、DispatchCoreConfig 校验拒绝合法 placement、trace prefetcher sub-device 作用域、Qwen3-32B perf 目标过期、DFB self-loop 断言 | tt-metal #54495/#54475/#54444/#54491/#54469 | CI/工具/断言类，非数值正确性 |
| Full Transformer PCC test、CSA/HCA Transformer Block、Functional MOE、qsr auto matmul pcc | tt-metal #54461/#54460/#54459/#54457/#54449 | 测试任务/CI 条目，非缺陷报告 |
| KV Cache Compression API RFC、per-request KV telemetry、auxiliary-GPU offloading、native watermarking RFC、SLiding-window 细粒度命中 | SGLang #36522、vLLM #53926/#53908/#53916/#53786 | RFC/特性请求 |
| WireLogprobs 拒绝 5-field LogprobsTensors（CI）、M_LOG2E 构建配置依赖、TORCHINDUCTOR_COMPILE_THREADS 被覆写、XPU TP>1 吃主机 RAM、DSV4-Pro 双节点启动挂起、DGX Spark+Gemma4 无法启动 | vLLM #53932/#53935/#53891/#53822/#53894/#53836 | CI/环境/启动类，无静默正确性机制 |

## 三、arxiv 检索结果

5 组关键词（SDC+LLM、bit flip+LLM inference、KV cache+corruption、LLM inference+correctness、GPU soft error+ML）。**无新可编目论文**：
- SCOUT（2608.11034）、LLM-PRISM（2604.10390）、2604.00726、2502.12340 —— 均已编目/引用
- 2607.19623（Bit-Position Sensitivity/UEP）、2605.04563（RangeGuard）、2603.16382（Rotated Robustness）、2603.10042（Targeted BFA on Agents）、2512.22174（BitFlipScope）—— 均已在第 3 类引用
- 2606.28958（When Latent Agents Lie）、2608.07408、2608.05863 —— 均已编目
- 2608.20397（Nexus，KV-cache sparsing 系统论文）与推理故障编目主题不符；2608.24534 等为临床/校准主题无关
- `GPU soft error + machine learning` 0 命中

## 四、重点新发现摘要

1. **确定性 flag 静默无效（4.13）**：`--enable-deterministic-inference` 对量化 hybrid + DFlash2 组合被接受且 resolved-args 显示生效、实际无效——「覆盖缺口表现得与成功完全一致」，是确定性族首个**保证声明层**（而非具体泄漏路径层）的签名。
2. **sleep/wake 状态恢复契约不完整（6.149）**：level=1 wake 后 LoRA 静默失效 + 双路由基线漂移、无任何失败信号——状态管理 API 的部分恢复无验证探针，服务以错误权重继续 200。
3. **off-by-one 使 offload 契约永假（6.150）**：store 侧 `tail` vs eagle lookup `tail+1`——数学上不存在合格运行，KV 无限写入永不读回；与 6.137 同族，是 hybrid「(n−1) 边界」缺陷族在 offload 域的再现。
4. **量化排除表/回退路径的双重静默（7.54/7.55）**：排除表命名空间失配 → 无 scale 静默 cast；"回退不量化"承诺被违反 → int8 位型装进 bf16。两条都是「承诺语义（排除/回退）与实现行为脱钩且失配方向恰好静默」。
5. **跨请求 token 级串扰（6.151，speculative）**：若证实，是比 KV 值污染更上游的串扰形态——整段 prompt 归属错位，模型自身在思维链中都能察觉。

## 五、API 调用记录

- GitHub Search API：8 次（Q1 vllm wrong/incorrect/corruption、Q2 vllm NaN/quant/KV、Q3 sglang、Q4 trt-llm 0 命中、Q5 triton 0 命中、Q6 tt-metal PCC/numerical、Q7 transformers、Q8 vllm silent/determinis/bit flip）—— 无 403 限流
- GitHub core API：18 次（逐条拉取 #53912/#53888/#53951/#53850/#53831/#53777/#53868/#36548/#36480/#36390/#36291/#36423/#48349/#48323/#48296/#54439/#54447/#53943 正文）
- arxiv API：6 次（-L 跟随重定向；含 1 次 GPU-soft-error 组重试）

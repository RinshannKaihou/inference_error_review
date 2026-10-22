# LLM 推理故障每日扫描报告 — 2026-08-28

- **扫描日期**：2026-08-28（滚动窗口 created:>=2026-08-27）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 753 条（第 0 层 17 + 第 1 层 622 + 第 2 层 114）
- **结果**：新发现并合并 **11 条**（全部为新子签名，无已知类别变体）；1 条确认为已编目条目重复出现（vLLM #53975 = 10.40）；2 条 tt-metal 命中为昨日已处理条目（#54584 昨日跳过、#54516 = 1.112）。arxiv 无新编目论文。

---

## 一、新编目条目（11 条，已合并进主文档）

### 第 5 类 · 分布式 / 并发资源（+1）

1. **SGLang #36889** — [DFLASH] Mamba state cache silently caps effective concurrency on hybrid-KDA models — https://github.com/sgl-project/sglang/issues/36889
   `[新类别/新签名]`（→ 5.23）DFLASH 推测解码下每请求需 `1+D` 个 Mamba state slot，静态内存紧张时调度器**静默把有效并发从 8 压到 2**，仅一条 info 日志；`/get_server_info` 仍报 8，吞吐曲线平坦被误读为硬件带宽天花板（c12 实测 +75% 吞吐被一个 flag 藏住）。属活性/容量类静默钳制。

### 第 6 类 · 服务栈 / 缓存正确性（+2）

2. **vLLM #54193** — KV offloading connector: finished-request store watermark includes a KV slot never written by any forward pass — https://github.com/vllm-project/vllm/issues/54193
   `[新类别/新签名]`（→ 6.157）请求完成时 store watermark 记 `req.num_tokens`，含**从未被 forward 写入的末 token KV 槽**；`offload_prompt_only=False` + block 边界结束时该块按内容 hash 存为完整——后续多轮请求命中加载到垃圾/stale KV。推测解码下该槽持有**被拒 draft token 的 KV（in-distribution，检查缓存都看不出）**。静默跨请求投毒。
3. **SGLang #36886** — DCP in GLM-5.3-Flash: index-K cache OOB corruption crash past max_total_num_tokens — https://github.com/sgl-project/sglang/issues/36886
   `[新类别/新签名]`（→ 6.158）双层缺陷：index-K cache 按 per-rank 尺寸分配却用 DCP-virtual locs 写读（无 `%dcp` 换算）→ 越界写、随机受害内核 crash（6 种签名）；**尺寸修复后输出质量仍退化**（CoT 膨胀 4–9×、翻译丢实体、reasoning 超预算返回空），仅移除 DCP flag 恢复。

### 第 7 类 · 量化（+4）

4. **vLLM #54189** — NVFP4 MoE silently folds an uninitialised activation scale for weight-only checkpoints — https://github.com/vllm-project/vllm/issues/54189
   `[新类别/新签名]`（→ 7.60）weight-only NVFP4 MoE checkpoint（无 `input_scale`）的 `torch.empty` 未初始化参数被折进反量化 alpha，实测 0.0 → **每个 expert 输出恒零**，模型退化为单 token 重复，无错误无警告。GB10 上同症状曾被误归因为"FLASHINFER_CUTLASS 在 GB10 garbage"——独立 harness 证明 kernel 正确，触发面是 checkpoint 而非 GPU；marlin 不消费 activation scale 故绕开。linear 方法有 guard，MoE 没有。
5. **vLLM #54206** — Quark FSE compatibility check matches `shared_expert_gate`, silently disabling fused shared experts — https://github.com/vllm-project/vllm/issues/54206
   `[新类别/新签名]`（→ 7.61）`is_shared_expert_quant_fse_compatible` 在 exclude 列表搜子串 `"shared_expert"`（无尾点），误命中 BF16 router gate `shared_expert_gate`——**FSE 恰在它为之启用的 checkpoint 上被拒用**：无失败、精度不变、每 MoE 层一条 startup warning，ROCm FSE 加速整体丢失。#51695 重构丢掉旧检查的尾点；新增测试只测复数命名。
6. **SGLang #36822** — GPTQ Marlin MoE fails or produces NaNs with BF16 and tensor parallelism — https://github.com/sgl-project/sglang/issues/36822
   `[新类别/新签名]`（→ 7.62）双缺陷：scales 恒按 fp16 分配不跟随 `params_dtype`（BF16 模型触发内核断言）；`desc_act=True` × TP 下每 rank 需完整 w2 scale 表却加载 TP-sharded 版本 → NaN 传播到 logits（`probability tensor contains either inf, nan...`）。修复 PR #35955。
7. **vLLM #54252** — compressed-tensors config collapses GDN linear-attention token distinctness — https://github.com/vllm-project/vllm/issues/54252
   `[新类别/新签名]`（→ 7.63）hybrid GDN 模型只要**存在** CT quantization config——即使 CT 组只 targeting RoutedExperts、GDN Linear 全部 `UnquantizedLinearMethod`——GDN 层也坍塌 token 区分度（hidden-state self-cosine L1 起 0.40+，norm 膨胀 1e10），fp8 config 同权重正确。九阶段排除证明量化组件 byte-exact；专家置零后 hidden state 仍坍塌。**未量化的层因量化 config 的存在而损坏**——罕见表示层故障形态。

### 第 8 类 · KV 缓存 / 注意力（+1）

8. **SGLang #36807** — fast_topk_v2 can silently return wrong top-k sets when a radix threshold bucket exceeds its 4096-entry candidate buffer — https://github.com/sgl-project/sglang/issues/36807
   `[新类别/新签名]`（→ 8.55）`topk.cu` 的 shared-memory 候选 buffer 固定 4096 条目，长行大 k（64×256K、k=2048）时 threshold bucket 溢出，pass-2 staging `if (pos < SMEM_INPUT_SIZE)` **静默丢弃**超出候选：输出形状正常、恰好 k 个合法索引，但每行 6–42 条不在真 top-k；哪些行错非确定（tie-边界依赖）。

### 第 10 类 · 分词器 / 输出解析（+1）

9. **vLLM #54273** — Kimi-K3 parsers miss channel terminators - tool calls silently dropped — https://github.com/vllm-project/vllm/issues/54273
   `[新类别/新签名]`（→ 10.42）parser 假设 close-think 是 think 通道唯一出口：模型直接开 `<|open|>tools<|sep|>` 时 `is_reasoning_end()` 永假 → tool call 永不被解析（`finish_reason: stop` 零输出 token 或裸 XTML 漏进内容）；流式路径把 `<|close|>message<|sep|>` 漏给客户端而非流式不漏。纯 Python 复现只需 tokenizer。

### 第 11 类 · 解码 / 执行模式（+1）

10. **vLLM #54197** — GlmMoeDsa (V2 model runner) cannot use torch.compile — silent fallback — https://github.com/vllm-project/vllm/issues/54197
    `[新类别/新签名]`（→ 11.55）V2 runner 的 DSA 模型类缺 `@support_torch_compile`：mode=3 打印一行警告后**静默跑 eager**（编译请求变 no-op）。强开后又撞 `lru_cache` C 调用与 `ContextVar.get` 两处 fullgraph 失败；V1 回退死于 `triton_kernel_wrapper_functional` 断言——三路径无一条能编译。输出正确、纯性能静默降级。

### 第 12 类 · 权重加载（+1）

11. **transformers #48371** — NVFP4 MoE experts are dequantized without weight_global_scale, silently — https://github.com/huggingface/transformers/issues/48371
    `[新类别/新签名]`（→ 12.25）`DecompressExperts.convert` 反量化不带 per-tensor `weight_global_scale`（`update_weight_conversions` 收集 tensor 时漏掉）——每专家被错误常数缩放，无异常、fused 参数填满使 `missing_keys` 恒空。CPU standalone 可复现。与 7.60 构成 NVFP4 加载域"scale 元数据链断裂"的栈两侧实例。

## 二、重点新发现摘要

今日最值得注意的三条：

1. **7.63（#54252）"量化 config 的存在性损坏未量化层"**——量化故障编目中罕见形态：GDN 层根本没被量化（ignore 列表内、UnquantizedLinearMethod），却因 CT config 存在而坍塌 token 区分度。检测只能靠 hidden-state self-cosine 扫描，权重对拍无效。
2. **6.157（#54193）KV offload 未写入末槽投毒**——被采样 token 的 KV 槽从未写入即被内容寻址缓存收编；推测解码使投毒值是被拒 draft 的 in-distribution KV，肉眼检查缓存无异常。多轮对话是标准触发路径。
3. **7.60 + 12.25（#54189 + #48371）NVFP4 scale 元数据链断裂的栈两侧**——vLLM 推理栈（未初始化 activation scale 归零专家）与 transformers 加载库（丢 weight_global_scale 错误缩放）同日暴露同族缺陷，且 7.60 展示了典型的硬件误归因陷阱（"GB10 garbage"实为 checkpoint 触发）。

## 三、已编目条目重复出现（未新增）

| 条目 | 对应主文档条目 |
|---|---|
| vLLM #53975 legacy guided_json 静默忽略（2026-08-27 创建） | 10.40（昨日编目） |
| tt-metal #54516 prefill reference rope bf16 | 1.112（昨日编目） |
| tt-metal #54584 kimi27 pcc non-determinism | 昨日已评估跳过（CI flap，信息量不足） |

## 四、跳过（未编目）及理由

- **SGLang #36840** — TBO unreachable for GLM-5.3-Flash：三层问题中第 1 层（`--enable-dp-attention` 被 `dp_size==1` post-process 静默重置、错误提示让用户加被丢弃的 flag）有编目价值，但整体是功能不可达 + 误导报错（loud 失败为主），无静默输出损坏；与 4.13（静默 flag 失效）部分重叠但形态更接近配置 ergonomics。不单列条目，在此记录备查。
- **transformers / TensorRT-LLM / triton 搜索** — 窗口内 0 命中（transformers 仅 #48371 经补充关键词命中，已编目为 12.25）。

## 五、数据质量观察

- 主文档存在一对**历史遗留重名**：6.28 与 6.33 同名 `vllm_prefix_cache_silent_noop_mamba_gdn_hybrid`（非本次扫描引入；今日 11 个新名均唯一）。建议后续维护时差异化命名，本次未改动以避免破坏交叉引用。

## 六、arxiv 搜索结果（无新编目）

关键词：`silent data corruption LLM` / `bit flip LLM inference` / `KV cache corruption` / `LLM inference correctness` / `GPU soft error inference`。命中论文（SCOUT 2608.11034、LLM-PRISM 2604.10390、指令级 GPU 软错误 FI 2512.xxxx、Runtime Observability 2608.05863、When Latent Agents Lie 2606.28958、Addressable Memory for Video World Models 2608.07408 等）均已在先前扫描中编目或作为补充来源引用；无新的推理栈故障案例论文。

## 七、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GH Search: vllm issues（wrong/incorrect/corruption/silent/NaN in:title, created:>=2026-08-27） | 6 hits |
| 2 | GH Search: sglang issues（同上） | 5 hits |
| 3 | GH Search: transformers issues（wrong/incorrect/corruption in:title） | 0 hits |
| 4 | GH Search: TensorRT-LLM issues（同上+NaN） | 0 hits |
| 5 | GH Search: triton issues（wrong/incorrect/NaN/precision in:title） | 0 hits |
| 6 | GH Search: tt-metal issues（numerical/pcc/NaN/incorrect/corruption in:title） | 2 hits（均昨日已处理） |
| 7 | GH Search: vllm+sglang（garbage/deterministic/bit flip in:title） | 1 hit（#54252） |
| 8 | GH Search: transformers（NaN/silent/garbage in:title） | 1 hit（#48371） |
| 9-20 | GH Issues API 单条详情（#54273/#54206/#54197/#54193/#54189/#54252/#36889/#36886/#36840/#36822/#36807/#48371/#53975） | OK（未触发 403 限流；间隔 ≥7s） |
| 21-25 | arxiv API × 5 组关键词（HTTPS，-L） | 无新编目论文 |

注：arxiv API 的 HTTP 端点被本环境安全策略拦截（plain-HTTP 下载），改用 HTTPS 端点等效完成。

## 八、合并后状态

- 主文档：753 → **764 条**（第 1 层 622 → 633；第 0 层 17、第 2 层 114 不变）
- 类别分布变化：第 5 类 22→23 · 第 6 类 156→158 · 第 7 类 59→63 · 第 8 类 54→55 · 第 10 类 41→42 · 第 11 类 54→55 · 第 12 类 24→25
- 轴分布变化：infra_off 601→611 · ambiguous 77→78 · config 162→166 · other_stack 144→147 · partial 164→168 · documented 545→547 · verified 202→211 · NEW 640→651
- 导出：fault_catalog.xlsx / fault_catalog.csv 预期 764 − 17 = **747 行**

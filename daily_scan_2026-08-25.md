# LLM 推理故障每日扫描 — 2026-08-25

- 扫描日期：2026-08-25（UTC）
- 搜索窗口：GitHub `created:>2026-08-23`（上次扫描 2026-08-23 之后的新 issue/PR）；arxiv 最近提交
- 目标仓库：vllm-project/vllm · sgl-project/sglang（0 命中）· NVIDIA/TensorRT-LLM（0 命中）· triton-lang/triton（0 命中）· tenstorrent/tt-metal · huggingface/transformers
- 主文档合并前条数：704 → 合并后：**717**（新增 13 条：12 个新子签名 + 1 个已知类别变体；另有 1 条补充修复引用）

---

## 一、新合并进主文档的条目（13 条）

### 第 1 类（硬件/SDC 内核原语）+4

1. **[新签名] tt-metal PR #54229 — all_broadcast interleaved 路径 writer 地址生成器用输入 buffer 构建、指向输出 buffer，跨存储空间 bank 解析错误（L1→DRAM / DRAM→L1 PCC≈0.03）** → 1.104
   https://github.com/tenstorrent/tt-metal/pull/54229 （closed/merged；sharded 路径同样混淆但被显式留置）
2. **[新签名·度量基建] tt-metal PR #54289 — comp_pcc 常量张量 fallback 用 fp32 级 allclose 容差判 bf16，2-ULP 正确结果被报 PCC 0.0（假阳性方向）** → 1.105
   https://github.com/tenstorrent/tt-metal/pull/54289
3. **[已知类别新实例] tt-metal #54165 — SDPA `sum_df` 从 bf16 切 fp32（PR #49948）反使 SDXL VAE decoder PCC 恶化（#50999，PR #50930 放宽阈值绕过）** → 1.106
   https://github.com/tenstorrent/tt-metal/issues/54165
4. **[新签名·待证] tt-metal #54196 — GELU SFPU LUT declmal 与 lut16 十六进制常量不匹配（lreg4 lo 差异极大，孰对孰错未裁定）** → 1.107（speculative）
   https://github.com/tenstorrent/tt-metal/issues/54196

### 第 6 类（服务栈/缓存正确性）+4

5. **[新签名] vLLM #53706 — Mamba hybrid 全量本地 prefix-cache 命中后 N-1 截断使 num_new_tokens=0，`assert num_new_tokens > 0` 崩溃 prefill EngineCore（重复请求 200→500，K8s 容器重启）** → 6.145
   https://github.com/vllm-project/vllm/issues/53706 （Dynamo 1P1D + NixlConnector push + KimiK3）
6. **[已知类别新实例：缓存活性静默族] vLLM #53670 — EAGLE/MTP last-block drop 在 hybrid GDN 布局（scheduler block=1648 token）上每次命中丢弃整块 1,648 token 重算，batch 吞吐 -30~40%** → 6.146
   https://github.com/vllm-project/vllm/issues/53670 （无 is_eagle_group 注解时保守地把全部 KV 组标 eagle）
7. **[新签名] vLLM PR #53802 — hybrid 三个可复用状态注册点从 n 取 floor 而查找上限为 n−1（+eagle drafter 再移位一单元）→ 块对齐/eagle 激活时 hybrid prefix 命中坍缩为零** → 6.147
   https://github.com/vllm-project/vllm/pull/53802 （与 6.144 的 PR #53479 相邻问题互补合成）
8. **[已知类别变体：var K9.10 cache_salt 隔离缺失] vLLM #53495 + PR #53496 — ExampleConnector 以原始 prompt token 为外部 KV 键，cache_salt/LoRA/prompt_embeds 三维隔离全部失效（跨租户 KV 复用 48 token；embeds 请求得 -16 负计数）** → 6.148
   https://github.com/vllm-project/vllm/issues/53495 （RFC #53194 连接子一致性族；HF3FSKVConnector 同模式 #51748）

### 第 8 类（KV 缓存/注意力）+1

9. **[新签名] vLLM PR #53681（fixes #49012）— NVFP4 KV cache 写内核从 `size(1)` 读 block_size、按 stride 序探测布局；运行时 HND 物理形状 `[num_blocks, num_heads, block_size, dim]` 使 num_heads 被误读为 block_size、布局探测反转，`num_heads%4==0` 时唯一下游检查通过 → cache 静默损坏、decode 输出垃圾** → 8.53
   https://github.com/vllm-project/vllm/pull/53681

### 第 10 类（分词器/模板）+1

10. **[新签名] transformers #48320 — StopStringCriteria 用固定 ASCII 探针前缀验证词表，非拉丁脚本 word-level 词表无法 round-trip → 构造期裸 `ValueError: substring not found`** → 10.37
    https://github.com/huggingface/transformers/issues/48320

### 第 11 类（解码/采样·推测解码）+2

11. **[新签名] vLLM PR #53520（fixes #53488）— padded GPU drafter 在 `_bookkeeping_sync()` 前运行并复用全局 CUDA graph 内存池，覆写 target hidden_states → prompt_logprobs 从 draft 输出计算、静默损坏（≤capture 限长的 prompt 全中，之上全对；generation 不受影响）** → 11.51
    https://github.com/vllm-project/vllm/pull/53520 （RTX 5880 Ada 复现报告者的 GB10 场景）
12. **[新签名] vLLM PR #53613 — async scheduling（max_concurrent_batches>1）+ hybrid + spec decode：`_update_states` 重写 block table/状态缓冲时，上一步的 `postprocess_mamba_align_gpu` 仍在 GPU 上读写同一批缓冲——两者间无任何排序** → 11.52
    https://github.com/vllm-project/vllm/pull/53613 （修复：在 context 顶部等待 num_accepted_tokens_event）

### 第 13 类（模型图/架构）+1

13. **[新签名·待证] transformers #48293 — SwitchTransformers token_priority `torch.cumsum(dim=-3)` 疑应为 -2（应在 seq_len 维累加）；且 router_probs/router_logits 输出恒等** → 13.33（speculative）
    https://github.com/huggingface/transformers/issues/48293

### 补充来源引用（不占新编号）

- **vLLM PR #53730**（Mooncake hybrid DCP 分组外部命中加固：分组外部缓存元数据 + 部分加载失败不留下组间失配状态）→ 追加为 **6.139** 的后续加固引用。

---

## 二、扫描到但未合并的条目（记录备查）

| 条目 | 来源 | 不合并原因 |
|---|---|---|
| test_rms_norm 在 26-SM MIG H200 CI agent 上确定性失败、16-SM 全过（非 flake，与 torch 版本无关） | vLLM #53775 | 根因未定位（SM 数依赖的内核数值/容差行为，值得跟踪）；无机制可编目 |
| `--verify-weights` 检查点损坏检测 flag | vLLM PR #53537 | 检测工具而非故障本体；可作为权重损坏类的监控相关工具引用 |
| KV-cache key partitioning conformance suite（RFC #53194 落地） | vLLM PR #53769 | 一致性测试基建（6.148 的系统性防线），非故障 |
| per-request prefix cache telemetry | vLLM PR #53795 | 可观测性特性，非故障 |
| hybrid sliding-window group 细粒度 prefix hit 特性 | vLLM #53786 | 特性请求 |
| 自定义 encoder cache manager 构造器兼容回归（AttributeError fail-loud） | vLLM PR #53761 | 加载期 fail-loud 回归，无静默窗口，编目价值低 |
| quasar pool op face geometry 错误（experimental，无正文） | tt-metal #54300 | 信息不足（body 为空） |
| Byte-level tokenizer.json + LlamaTokenizer/Fast 类探测 | transformers PR #48317 | 修复性 PR，无故障实例 |
| 两词 pretokenized 文档停止配对 | transformers PR #48279 | 训练侧分词器行为，非推理正确性 |

## 三、arxiv 检索结果

5 组关键词（SDC+LLM、bit flip+LLM inference、KV cache+corruption、LLM inference+correctness、GPU soft error+ML）。**无新可编目论文**：
- SCOUT（2608.11034）、LLM-PRISM（2604.10390）、2604.00726、2502.12340 —— 均已编目/引用
- 2606.28958（When Latent Agents Lie）、2608.05863（Runtime Observability）、2608.07408 —— 均已编目
- 其余命中（REST API 测试、LDPC 解码、MT 性别偏置等）与推理故障无关
- `bit flip + LLM inference`、`GPU soft error + machine learning` 两组 0 命中

## 四、重点新发现摘要

1. **NVFP4 KV cache 布局探测反转（8.53）**是本轮最典型的静默损坏：内核入口的形状假设只对测试用的逻辑视图成立，真实 HND 调用方使布局判断反转且唯一检查被 `num_heads%4==0` 巧合放行——「内核入口形状契约 × 测试视图窄化」新签名。
2. **prompt_logprobs 被 draft 输出覆写（11.51）**：CUDA graph 全局内存池使 bookkeeping 读到 drafter 的 hidden states——损坏边界精确等于 capture 限长，是「共享内存池生命周期 × 采样后读回」的结构性盲点新形态。
3. **Mamba 全命中崩溃（6.145）与注册边界错配（6.147）**与既有 6.143/6.144 构成 hybrid 前缀缓存「(n−1) 边界」缺陷族的四连发——注册侧、查找侧、物化侧、截断侧各错一次。
4. **度量基建反向缺陷（1.105）**：comp_pcc 把 2-ULP 正确的 bf16 结果判为 PCC 0.0——假阳性方向的验证器缺陷，与静默损坏同样危害编排（正确内核被拒）。

## 五、API 调用记录

- GitHub Search API：9 次（Q1 vllm wrong/incorrect/corruption、Q2 vllm NaN/quant/kv、Q3 sglang、Q4 trt-llm、Q5 triton、Q6 tt-metal PCC/numerics、Q7 transformers、Q8 vllm silent/determinis/bit flip；Q6 首次因 URL 空格截断重试 1 次）—— 无 403 限流
- GitHub core API：17 次（逐条拉取 #53706/#53670/#53775/#53495/PR53520/PR53802/#48320/#48293/tt#54300/PR54289/PR54229/#54196/#54165/PR53761/PR53681/PR53730/PR53613 正文）
- arxiv API：5 次（-L 跟随重定向）

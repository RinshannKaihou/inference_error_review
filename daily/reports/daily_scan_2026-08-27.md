# LLM 推理故障每日扫描报告 — 2026-08-27

- **扫描日期**：2026-08-27（滚动窗口 created:>=2026-08-20）
- **搜索源**：GitHub Issues Search API（vllm / sglang / transformers / TensorRT-LLM / triton / tt-metal）+ arxiv API
- **基线**：FAULT_MASTER_REFERENCE.zh.md 合并前 731 条（第 0 层 17 + 第 1 层 600 + 第 2 层 114）
- **结果**：新发现并合并 **22 条**（20 个新子签名 + 2 个已知类别变体）；另有 1 条补充来源（#54046 → 1.48）；14 条确认为已编目条目的重复出现；5 条跳过（非正确性故障或信息量不足）。arxiv 无新编目论文。

---

## 一、新编目条目（22 条，已合并进主文档）

### 第 1 类 · 硬件 / SDC（+8）

1. **vLLM #53480** — [Bug][XPU] Silent persistent output corruption (endless "!" / token 0) under sustained concurrent decode on Arc Pro B70, W4A16 27B head_dim 256 — https://github.com/vllm-project/vllm/issues/53480
   `[新类别/新签名]`（→ 1.110）持续并发 decode 后**长寿命 VRAM 状态（权重/持久 workspace）被原地损坏**：HTTP 200、零 dmesg，负载回落后新请求仍只输出 token 0，直到重启引擎（权重重载）才恢复。两张不同物理卡复现排除单卡故障；head_dim 128 的同族 14B/8B 数周不坏。生产 30 小时命中 4 次。
2. **triton #11406** — `tl.softmax` produces incorrect, non-row-wise results — https://github.com/triton-lang/triton/issues/11406
   `[新类别/新签名]`（→ 1.111）`tl.softmax(dim=1, keep_dims=False)` 归约维丢弃后广播错位：2×2 输入行和 [0.398, 5.449] 而非 [1,1]；编译路径与 INTERPRET 同错（前端语义缺陷），keep_dims=True 正确。已 closed（修复）。
3. **tt-metal #54516** — [model]: prefill reference rope is built in bf16, so PCC decays with position — https://github.com/tenstorrent/tt-metal/issues/54516
   `[新类别/新签名]`（→ 1.112）**验证参考自身的 dtype 精度缺陷**：参考模型 bf16 实例化使 inv_freq 成为 bf16 buffer（0.75 vs 0.7498942），`.float()` 无法恢复低位 → 相位误差随位置线性增长，KVPE PCC 0.9976@1024 → 0.9462@5120。q/k 共享同一错表故 attention 输出 PCC 仍 ~0.99——读起来像设备 rope bug 但不是。
4. **tt-metal #54498** — [model]: Qwen3-VL vision tower fails 8 of 22 T3000 cases — PCC degrades with aspect ratio — https://github.com/tenstorrent/tt-metal/issues/54498
   `[新类别/新签名]`（→ 1.113）PCC 失败按 orientation 成对（93.07/93.08）且**误差随长宽比单调放大**（4:3→98.6%、16:9→96.9%、4:1→93.1%）——系统性而非容差问题；另 2 例 max_load canvas 触发 allocator TT_FATAL。根因未定位。
5. **tt-metal #54054** — ttnn.exp2: fp32 path returns NaN / finite garbage instead of +inf for x >= 128 — https://github.com/tenstorrent/tt-metal/issues/54054
   `[新类别/新签名]`（→ 1.114）指数重构只有下溢 guard 无上溢 guard：x≥128.5 时 setexp 只留低 8 位**回绕成普通有限数**（exp2(256)=1.0）——正确 +inf 上一行已算出又被覆盖。
6. **tt-metal #53975** — sqrt / rsqrt return NaN for -0.0, which is inside the domain — https://github.com/tenstorrent/tt-metal/issues/53975
   `[新类别/新签名]`（→ 1.115）`x < 0.0f` 符号位测试把 −0.0 与真负值归入同一分支：fp32 返回 NaN（应 −0.0/−inf）、bf16 返回 inf；fast-approx 模式直接对 −0.0 位型跑 Newton 迭代返回 7e-20 垃圾。
7. **tt-metal #53922** — ttnn.pow(0, NaN) returns 0 on the binary SFPU path — https://github.com/tenstorrent/tt-metal/issues/53922
   `[新类别/新签名]`（→ 1.116）binary 路径 NaN 指数 × 零底数返回 +0（应 NaN）；#53367 修复把失败模式从 loud（inf）改成 **silent**（0）。零底数（padding/ReLU/新 buffer）× NaN sentinel 是两个 masked 张量的天然碰撞。
8. **tt-metal #53872** — ttnn.xlogy_bw: two wheres on eq(x, nan) that never select — https://github.com/tenstorrent/tt-metal/issues/53872
   `[已知类别]`（→ 1.117，var:1.53 `ttmetal_xlogy_dead_nan_guard`）backward 的 `where(eq(x, nan), nan, ...)` 因 NaN≠NaN 恒不选中——1.53 前向死守卫的 backward 复刻；且 4/19 dispatch 为死代码（性能层）。

### 第 5 类 · 分布式 / PD（+1）

9. **SGLang #36371** — [Bug] Mooncake PD silently accepts mismatched Mamba SSM state dtypes/item lengths — https://github.com/sgl-project/sglang/issues/36371
   `[新类别/新签名]`（→ 5.22）PD 两端各自解析出不同 Mamba SSM state dtype/item length，传输用**源端**长度做拷贝长与目的 slot stride → recurrent state 布局错位，无错误、HTTP 200，长/多轮质量严重退化；显式对齐两端 dtype 即恢复。

### 第 6 类 · 服务栈 / 缓存正确性（+5）

10. **vLLM #53726** — Silent CUDA IMA (exit 0) in hybrid GDN + MTP k=3 + async scheduling on RTX 3090 — https://github.com/vllm-project/vllm/issues/53726
    `[已知类别]`（→ 6.152，var:11.52）混合流量下 CUDA IMA 异步浮出后 **API server 以 exit status 0 退出**——systemd `Restart=on-failure` 视为成功不再拉起：crash 伪装成正常退出。#50021/#45100/#53613 类修复在树仍复现（30–50% 重载启动）。
11. **vLLM #53595** — align-mode prefix caching silently retains only semantic checkpoints — https://github.com/vllm-project/vllm/issues/53595
    `[新类别/新签名]`（→ 6.153）`prefix_cache_retention_interval` 默认 0 = 只保留 semantic checkpoints：其余 boundary state 落盘但**永不 hash → 永不命中**；默认值全树无日志、hit rate 还虚报非零。
12. **SGLang #36716** — qwen38flashnext cookbook image: 4 bugs on single-device SM121 (GB10 Spark) — https://github.com/sgl-project/sglang/issues/36716
    `[新类别/新签名]`（→ 6.154）四缺陷：(1) trtllm-gen decode kernel 无 arch guard 在 SM121 静默 garbage（首 token 对→永久 "!!!!"）；(2) `_compact_kv` 写原始列偏移而读侧按紧凑语义 → FA fallback NaN 洞；(3) TMA-O varlen boot crash；(4) sparse prefill fp8 tl.dot。TP2 参考配方不经过的路径全部中弹。
13. **SGLang #36475** — Streaming session client disconnect can silently drop session context and crash the scheduler — https://github.com/sgl-project/sglang/issues/36475
    `[新类别/新签名]`（→ 6.155）SSE 流中途断开 + 紧接同 session 请求：HTTP 200 但 `prompt_tokens=1, cached_tokens=0`——多轮上下文整体蒸发，无任何错误；后续可触发 scheduler crash。
14. **SGLang #35912** — `uv pip install sglang` with uv < 0.12 silently installs 0.5.9 (6 months old) — https://github.com/sgl-project/sglang/issues/35912
    `[新类别/新签名]`（→ 6.156）uv<0.12 不解析 transitive pre-release specifier（flash-attn-4 仅有 pre-release）→ resolver 无声回退 0.5.9 → 旧版 TP>1 丢 down_proj all-reduce 的已修复 bug 复活：**流畅但错误的输出**，无 crash 无 NaN。

### 第 7 类 · 量化（+4）

15. **vLLM #53211** — [Bug][XPU]: AutoRound int4 models silently emit garbage under concurrent requests — https://github.com/vllm-project/vllm/issues/53211
    `[新类别/新签名]`（→ 7.56）单请求正确、并发≥2 全部响应变多语言 token soup；ARK WOQ backend 自动选中**无退出开关**；双 wheel 版本复现排除单一版本。与 1.110 构成 Battlemage XPU 并发损坏家族（本条瞬态、彼条持久）。
16. **vLLM #53411** — DeepSeek v4 Pro Base - default MoE backend produces incorrect output and inflated logprobs — https://github.com/vllm-project/vllm/issues/53411
    `[新类别/新签名]`（→ 7.57）UE8M0 block-scale checkpoint × 默认自选 FLASHINFER_TRTLLM（trtllm_fp8_block_scale_moe）：**分布级畸变**——英文文本 next-token 分布中瑞典语 token 暴涨（greedy 出 ` böjnings`）、perplexity 虚高使 base model 评测失效；deep_gemm 正常，DSv3/Kimi K2 不复现。
17. **SGLang #36569** — Mxfp4MoEMethod.create_moe_runner silently accepts 9 backend values it has no branch for — https://github.com/sgl-project/sglang/issues/36569
    `[新类别/新签名]`（→ 7.58）dispatch 链尾 `else: pass` 吞掉 9 个无分支的 `--moe-runner-backend` 值：加载期零输出，首个 MoE forward 才 `AttributeError: no attribute 'runner'`——fail-late 且错误与根因无关。
18. **SGLang #36395** — bench_one_batch silently ignores --moe-runner-backend — https://github.com/sgl-project/sglang/issues/36395
    `[新类别/新签名]`（→ 7.59）bench 路径绕过 scheduler-only 的 `initialize_moe_config` → backend 恒 AUTO：mxfp4 权重加载期物化 BF16、跑 triton fallback，日志仍打印请求的 backend——全部量化 MoE 基准测错对象。

### 第 10 类 · 分词器 / 输入处理（+2）

19. **vLLM #53975** — legacy guided_json is silently ignored; request returns 200 with unconstrained free-form text — https://github.com/vllm-project/vllm/issues/53975
    `[新类别/新签名]`（→ 10.40）legacy 顶层 `guided_json` 字段被静默丢弃：200 + 完全无约束文本，无 grammar、无 warning——依赖硬结构保证的存量客户端承诺失效。正确行为应为 4xx。
20. **vLLM #53144** — Whisper transcription silently drops audio when the decoder stops early inside a chunk — https://github.com/vllm-project/vllm/issues/53144
    `[新类别/新签名]`（→ 10.41）decoder 提前 EOT 后 chunk 残余无人转写：29.42s 单 chunk 丢 7.9s（1/4）、双 chunk 丢约 1/3 语音；`duration` 全长、句子看似完整、确定性复现，唯一痕迹是 segment-end 算术。

### 第 11 类 · 解码 / 采样（+1）

21. **vLLM #53180** — TurboQuant k8v4 + MTP speculative decoding silently produces degenerate output on hybrid GDN — https://github.com/vllm-project/vllm/issues/53180
    `[已知类别]`（→ 11.54，var:11.34）stock v0.27.1 复现：prompt 依赖的间歇退化（重复墙/garbage），/health 200、接受率指标正常、吞吐正常——冒烟套件可过、真实流量损坏。#40831/#40880/#52475（TurboQuant KV × spec decode）族在 k8v4 + Ada 上的延续，#51812 修复后仍在。

### 第 12 类 · 权重 / 检查点加载（+1）

22. **SGLang #35797** — Qwen3.5 MTP weights silently dropped on compressed-tensors checkpoints (accept len pinned at 1.0) — https://github.com/sgl-project/sglang/issues/35797
    `[新类别/新签名]`（→ 12.24）compressed-tensors checkpoint 把 MTP linears 逐条列入 quantization ignore，`_mtp_quant_config()` 无 compressed_tensors 分支（#23113 只加了 quark）→ MTP 权重静默丢弃：accept len **恒 1.000 零方差**，吞吐坍塌 8 倍，无警告。

---

## 二、补充来源（不占新编号）

- **tt-metal #54046**（closed，$7000 bounty）— ttnn.xielu exponent-reconstruction overflow（x < −87.68 → ±2.7e38/NaN，21,232 点扫描 ~0.9% 绝对误差>1）→ 追加为 **1.48**（#52607 同缺陷 `_sfpu_neg_exp_f32_`）的补充来源。https://github.com/tenstorrent/tt-metal/issues/54046

## 三、已编目条目（重复出现，跳过）

| Issue | 对应条目 |
|---|---|
| vLLM #53488（经 PR #53520） | 11.51 prompt_logprobs drafter graph buffer 复用 |
| vLLM #53431 | 10.35 Gemma4 tool parser FSM 死态 |
| vLLM #53142 | 6.x hybrid mamba align precopy IMA |
| vLLM #53888 | 6.149 sleep level=1 wake LoRA 损坏 |
| vLLM #53850 | 10.39 deepseek VL/OCR 裸 assert |
| vLLM #53831 | 10.38 DSV4 tool_choice=auto 深推理档空响应 |
| vLLM #53912 | K9.16 仍开放证据 |
| SGLang #36548 | 6.151 DFlash2 并发跨请求串扰 |
| SGLang #35949 | 第 7 类 Qwen3.8 NVFP4 坐标误检 |
| transformers #48349 | 7.54 FineGrainedFP8 wrapper 前缀失配 |
| transformers #48293 | 13.33 SwitchTransformers cumsum |
| tt-metal #54165 | 1.106 SDPA sum_df 精度回归 |
| tt-metal #54006 | 1.101 TTTv2 CI 四缺陷 |
| tt-metal #53883 | 1.100 Kimi-K2.6 dense PCC flap |

## 四、跳过（未编目）及理由

- **vLLM #53891** — TORCHINDUCTOR_COMPILE_THREADS 被静默覆盖：编译**性能**问题（单线程编译慢），无输出正确性影响，不属故障编目范围。
- **tt-metal #54584** — kimi27 pcc non-determinism（kv cache bcc 掉 0.82）：CI flap，正文仅两条 run 链接、无可分析机制，与 1.100 flap 族同类但信息量不足。
- **tt-metal #54461** — "Full Transformer PCC test (flash/pro)"：测试任务 issue，非故障报告。
- **tt-metal #54449** — qsr auto matmul pcc 0.708（closed）：quasar 族已有 15 处覆盖，根因未明、无正文分析。
- **tt-metal #54300** — quasar pool op face geometry（closed）：正文为空。

## 五、arxiv 搜索结果（无新编目）

关键词：`silent data corruption LLM` / `bit flip LLM inference` / `KV cache corruption` / `LLM inference correctness` / `GPU soft error inference`。命中论文（SCOUT 2608.11034、LLM-PRISM 2604.10390、2601.19912 指令级 GPU 软错误 FI、2608.07408 视频 world model 等）均已在先前扫描中编目或作为补充来源引用；KV cache 类新论文（Runtime Observability 2608.05863、When Latent Agents Lie 2606.28958 等）为机制/防御研究，非推理栈故障案例，不编目。

## 六、API 调用记录

| # | 调用 | 结果 |
|---|---|---|
| 1 | GH Search: vllm issues（wrong/incorrect/corruption/silent/NaN in:title, created:>=2026-08-20） | 23 hits |
| 2 | GH Search: sglang issues（同上） | 9 hits |
| 3 | GH Search: transformers issues（wrong/incorrect/corruption in:title） | 2 hits |
| 4 | GH Search: TensorRT-LLM issues（同上） | 0 hits |
| 5 | GH Search: triton issues（wrong/incorrect/NaN/precision in:title） | 1 hit |
| 6 | GH Search: tt-metal issues（numerical/pcc/NaN/incorrect/corruption in:title） | 16 hits |
| 7-18 | GH Issues API 单条详情（#53975/#53726/#53595/#53480/#53411/#53144/#53891/#36716/#36371/#36395/#36475/#36569/#54584/#54516/#54498/#54449/#54300/#54054/#54046/#53975tt/#53922/#53872/#35797/#35912/#53180/#53211） | OK（未触发 403 限流） |
| 19-23 | arxiv API × 5 组关键词（-L 跟随重定向） | 无新编目论文 |

## 七、合并后状态

- 主文档：731 → **753 条**（第 1 层 600 → 622；第 0 层 17、第 2 层 114 不变）
- 类别分布变化：第 1 类 109→117 · 第 5 类 21→22 · 第 6 类 151→156 · 第 7 类 55→59 · 第 10 类 39→41 · 第 11 类 53→54 · 第 12 类 23→24
- 导出：fault_catalog.xlsx / fault_catalog.csv 预期 753 − 17 = **736 行**

# 每日扫描报告 — 2026-08-21

**扫描日期**：2026-08-21（UTC）
**扫描窗口**：GitHub issues `created:>2026-08-14`（7 天滚动）+ arxiv 最近提交
**主文档基线**：672 条（合并前）

---

## 一、GitHub Issues 发现

### 查询执行记录（8 次 Search API + 8 次 issue 详情 API，无 rate limit 错误）

| # | 查询 | 仓库 | 命中 |
|---|---|---|---|
| q1 | `wrong OR incorrect OR corruption OR NaN in:title` | vllm-project/vllm | 10 |
| q2 | `quantization wrong in:title` | vllm-project/vllm | 1 |
| q3 | `wrong OR incorrect OR corruption OR NaN in:title` | sgl-project/sglang | 2 |
| q4 | `wrong OR incorrect OR correctness OR NaN in:title` | NVIDIA/TensorRT-LLM | 1 |
| q5 | `wrong output OR wrong result OR RoPE in:title` | huggingface/transformers | 1 |
| q6 | `incorrect OR numerical OR precision OR mismatch in:title` | triton-lang/triton | 0 |
| q7 | `numerical OR pcc OR mismatch OR incorrect in:title` | tenstorrent/tt-metal | 10 |
| q8 | `KV cache OR prefix cache OR speculative in:title` | vllm-project/vllm | 20 |

> 注：首轮 8 次查询因 `--data-urlencode` 语法错误返回 422/空（已记录，未消耗有效配额）；修正后重跑全部成功。

### 已编目（issue 编号已在主文档，跳过）

- vLLM **#53142**（mamba align precopy IMA）、**#53086**（int4 per-token-head KV quant 错值）、**#53084**（MooncakeStore invalid recurrent states）、**#53029**（All-NaN logits → out-of-vocab id）、**#52924**（find_mm_placeholders 歧义匹配 = 10.32）、**#52627**（Kimi-K3 1P1D 静默损坏 = 6.107）、**#52576**（Triton MoE/block-FP8 K-tile OOB）、**#52568**（hybrid-GDN+LoRA NaN）、**#52469**（MTP 流式损坏）、**#53157**（shm_broadcast 启动 wedge）、**#52860**（AITER sparse PA × spec decoding = 8.49）
- SGLang **#35564**（8 个 tool-call parser 流式错误 = 10.31）、**#35257**（DSA top-k tie-overflow）
- TensorRT-LLM **#17964**（Qwen3.5-2B batch>1 错误输出）
- tt-metal **#53649**（RMSNorm PCC = 1.94 同根）、**#53267**（global_avg_pool2d = 6.x 已编目）、**#53207**（sdpa_fw hang 审计已编目）

### 新发现 — 合并进主文档（8 条）

| # | 条目 | 来源 | 分类 | 判定 |
|---|---|---|---|---|
| N1 | **vLLM #53095** — NixlPushConnector heartbeat 未续期，KV lease 30s 到期释放 block，`Num successful transfers=0`，高并发（Batch=128）下 decode 侧请求永久卡死；prefill 侧打印 "Releasing expired KV blocks ... retrieved by 0 remote workers" | [issue](https://github.com/vllm-project/vllm/issues/53095) | 第 6 类（服务栈） | `[新类别/新签名]` — **lease 生命周期缺陷**（heartbeat 续期路径失效），C.6 活性盲区新子签名 |
| N2 | **vLLM #53128** — PD Multi Tier `OffloadingConnector` 的 `UNBOUND_STORE_TIMEOUT_S=60.0` 固定定时器从 store 时刻起算，与消费者是否仍在排队无关；晚到的 FetchMsg 找不到 parked KV，consumer 烧满 30s `LOAD_TIMEOUT_S` 后整段 prompt 重算 | [issue](https://github.com/vllm-project/vllm/issues/53128) | 第 6 类（服务栈） | `[新类别/新签名]` — **unbound store 固定 reap × 消费者排队时序错配** |
| N3 | **vLLM #53130** — scheduler 在 `running + skipped_waiting`("deferred") 达到 `max_num_seqs` 后**永久停止准入**：引擎健康检查通过、KV cache 大量空闲、不再接收新请求，仅重启可恢复 | [issue](https://github.com/vllm-project/vllm/issues/53130) | 第 6 类（服务栈） | `[新类别/新签名]` — **scheduler 准入状态机 wedge**（健康表象下的僵尸服务），C.6 盲区 |
| N4 | **vLLM #53243** — DeepSeek-V4（TP4 H100）开启 hidden-state 提取（层 40/41/42/43，bf16 4096）后 KV cache 容量估算**高估**，模型加载成功后引擎启动被拒；0.26.0 基线失败、793ca6998 修复版通过 | [issue](https://github.com/vllm-project/vllm/issues/53243) | 第 6 类（服务栈） | `[新类别/新签名]` — **hidden-state 显存记账高估 × KV budget 误拒**（fail-closed 形态） |
| N5 | **tt-metal #53394 系列（#53412/#53418/#53822/#53823）** — TT-NOP 时序扰动注入工具在多个 LLK 内核暴露 **mismatch race**：BH `mul_reduce_scalar`、WH `generalized_moe_gate`、WH `eltwise_binary` dest-reuse（unpack STALLWAIT/SEMGET 点 NOP 注入即 mismatch，两轮 1–100 sweep 确定性命中同 8 个 variants，math 线程无 miss） | [#53823](https://github.com/tenstorrent/tt-metal/issues/53823) · [#53412](https://github.com/tenstorrent/tt-metal/issues/53412) · [#53418](https://github.com/tenstorrent/tt-metal/issues/53418) | 第 1 类（硬件/SDC） | `[新类别/新签名]` — **时序 NOP 注入暴露的 LLK unpack-sync 竞态族**（与 1.93 dest-reuse bank 边界损坏同通路不同机制） |
| N6 | **tt-metal #53883** — Blaze pipeline 上 Kimi-K2.6-1T（dense MLA、无 DSA indexer）padded chunked prefill（15k@5k, no-trace）**间歇性**返回完全不相关 decoder 输出（min PCC −0.0012 ≪ 0.88）；branch diff 证明不可达该图仍 flap，main 控制 run 也 flap 后 retry 恢复 | [issue](https://github.com/tenstorrent/tt-metal/issues/53883) | 第 1 类（硬件/SDC） | `[新类别/新签名]` — **dense 路径非确定性 PCC≈0 flap**（无 trace、跨分支复现、根因未定位） |
| N7 | **tt-metal #54006** — tier-1 CI 迁移（强制 watcher、只读 /mnt/MLPerf、冷 tensor cache）暴露 models/common **四个预存缺陷**：attention decode PCC 回归、`Sampling1DConfig.local_indices` 错误、rmsnorm 容差不足、`load_tensor` SIGBUS；字节级确定性复现（非 flake），其中两个使 TT-Transformers tier-1 门红 | [issue](https://github.com/tenstorrent/tt-metal/issues/54006) | 第 1 类（硬件/SDC） | `[新类别/新签名]` — **CI 环境变化暴露的预存缺陷集**（条件从未在 T3K pipeline 施加过） |
| N8 | **HF transformers #48051**（closed）— `video_utils.convert_to_rgb` 对 RGBA numpy 视频在混合行使用 `video[..., 3, :, :]`（**alpha 通道**）作前景而非 `[..., :3, :, :]` → 输出 shape 错误（`2,2,1,1` vs `2,3,1,1`）+ 像素值错误；alpha 被当作 RGB 数据 | [issue](https://github.com/huggingface/transformers/issues/48051) | 第 10 类（分词器/模板 → 多模态预处理） | `[新类别/新签名]` — **alpha-as-foreground 通道索引错误**（多模态视频预处理静默数据污染） |

### 记录但不编目（非正确性域 / 非缺陷）

- vLLM **#53237** — MooncakeConnector 每 (layer, block) 一个 RDMA descriptor：性能域
- vLLM **#53194** — KV-cache key partitioning conformance suite：RFC 提案
- vLLM **#52837** — 从 prefix-cache 记录派生 offload save sources：RFC 提案
- vLLM **#52913** — NixlConnector PP>1 支持请求：feature request
- vLLM **#53042** — KV offload 1.92× host 内存超配（昨日已判定：资源泄漏域，无正确性影响）
- tt-metal **#53510**（closed）— HCA PCC 测试在 Blackhole e2e 失败（0.9933 < 0.998 floor）：CI 测试目录重组导致 94 个用例被两条 legs 重复收集的环境问题，非内核回归
- HF #48051 之外 q5 无其他命中；Triton q6 零命中

---

## 二、arXiv 论文发现（5 组查询，22 篇返回）

| 查询 | 结果 |
|---|---|
| `silent data corruption LLM` | 4 篇全部已编目（SCOUT 2608.11034 = §3.1 补充、LLM-PRISM 2604.10390、2604.00726、2502.12340） |
| `bit flip LLM inference` | 相关 2 篇均已编目（2607.25227 = 3.7、2607.19623 = 3.8）；其余主题不符 |
| `KV cache corruption` | 2606.28958（When Latent Agents Lie）已编目；2608.05863（Runtime Observability）已引用；其余主题不符 |
| `LLM inference correctness` | 全部为 agentic/benchmark 主题，与推理正确性故障无关 |
| `GPU soft error inference` | 2601.19912 已编目（1.98，2026-08-20）；2104.09455（2021）为经典容错，非新故障 |

**结论：arxiv 无新编目论文。**

---

## 三、API 调用记录

| 时间 (UTC) | 调用 | 状态 |
|---|---|---|
| 23:0x | GitHub Search ×8（首轮，语法错误） | 422/空，未有效消耗配额 |
| 23:0x | arxiv query ×5（`curl -sL`） | 全部成功 |
| 23:1x | GitHub Search ×8（修正 `--data-urlencode "q=..."`，间隔 ≥8s） | 全部成功，无 403 |
| 23:2x | GitHub issue 详情 ×9（tt-metal ×4、vLLM ×4、HF ×1，间隔 3s） | 全部成功 |

---

## 四、合并摘要

- **8 条新条目**合并进 `FAULT_MASTER_REFERENCE.zh.md`：
  - 第 1 类 +3（1.99 `ttmetal_nop_injector_llk_sync_race_family`、1.100 `ttmetal_kimi_k26_dense_path_intermittent_pcc0_flap`、1.101 `ttmetal_tttv2_ci_migration_four_defects`）
  - 第 6 类 +4（6.133 `vllm_nixlpush_kv_lease_expiry_zero_transfer`、6.134 `vllm_tiering_unbound_store_60s_reap_recompute`、6.135 `vllm_scheduler_deferred_wedge_healthy_engine`、6.136 `vllm_hidden_states_kv_budget_overestimate_startup_reject`）
  - 第 10 类 +1（10.33 `hf_convert_to_rgb_alpha_as_foreground`）
- **总计 672 → 680 条**
- 全部为 `[新类别/新签名]`（8 个新子签名），无已知类别新实例（所有重复编号已核对跳过）

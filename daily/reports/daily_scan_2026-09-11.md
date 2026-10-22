# 每日扫描报告 — 2026-09-11

- **扫描窗口**：GitHub `created:>=2026-09-10`，arxiv 最近提交
- **目标仓库**：vllm-project/vllm、sgl-project/sglang、huggingface/transformers、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、flashinfer-ai/flashinfer
- **结果**：新发现并合并 **19 条**（13 个新子签名 `NEW` + 6 个已知类别变体 `var:`），主文档 1039 → 1058
- **昨日已编目条目**（今日再现均查重跳过）：SGLang #38815（6.225）、flashinfer #5095（8.82）、tt-metal #56171（8.81）、#56100（6.226）、#56056（1.197）
- **去重注意（跨仓撞号）**：tt-metal #56283/#56281/#56263 与主文档既有 **vLLM** #56283（6.227 来源）等同号——经核验为不同仓库的不同 issue，tt-metal 侧实际未编目，正常收录

---

## 一、新合并条目（19 条）

### 第 1 类 · 硬件/SDC（+6）

1. **[新类别/新签名]** **ttnn.mse_loss/l1_loss 预分配输出被挪用为中间缓冲**（tt-metal #56295）— SUM/MEAN 归约下 `optional_output_tensor` 被直接转发给 `ttnn::subtract`：标量形状预分配 → 形状失配 crash；全形状预分配 → 输出 tensor 静默持有**未归约中间差值**而返回值另分配新 tensor。双形态均 100% 确定复现。→ **1.200** `NEW`
2. **[新类别/新签名]** **Quasar copy_via_memmove 双重叠加 MEM_L1_UNCACHED_BASE**（tt-metal #56283，#56281 为同触发面简报）— #52769 起 DFB getter 已返回 uncached 别名，老消费者再加一次 → 读 `src + 2*BASE` 落在 L1 窗外：misaligned tiled reshape_view 在 ZEBU RTL 上 **753/1152 元素错、736 零、无 fault**（模拟器才 abort）。main 未修。→ **1.201** `NEW`
3. **[已知类别]** **moreh_norm_backward p=±inf 梯度全零/垃圾**（tt-metal #56248）— forward 特判 ±inf、backward 没有：`static_cast<uint32_t>(±inf)` UB + `inf−inf=NaN`；p=+inf 时 argmax 梯度全部 0.0、部分元素垃圾至 1.5e38/inf。归类 var:1.178（moreh_norm 参数域未守卫族）。→ **1.202**
4. **[新类别/新签名]** **同宽 reshard 双 DM RISC 共享 scratch DFB 互踩**（tt-metal #56226）— 同 core 两 RISC 写同一 `scratch+src_offset` 且偏移归零 → 静默错数据；mainline #53028 修复未带到 Quasar 拷贝（#51217 后续）。→ **1.203** `NEW`
5. **[已知类别]** **Quasar p_unpacr UNPACR_NOP flavour 携带 BH 编码**（tt-metal #56204）— BH 九操作数语义值全合法但含义错位：`llk_unpack_dummy` 请求 UNP_CLRSRC(=1) 实发纯 NOP，**SrcA 清除从未发生**、前置 STALLWAIT 等待不存在的清除。归类 var:1.141（UNPACR_NOP 编码跨 arch 错位族）。→ **1.204**
6. **[新类别/新签名]** **Quasar 批量 pack_tile 忽略 DFB stride 契约**（tt-metal #56194）— llk-pack 每 tile +1 entry vs 消费端除以 stride：N tile 批塌缩到 ceil(N/S) 槽、丢 1−ceil(N/S)/N；单 tile 打包不触发故未被发现。→ **1.205** `NEW`

### 第 5 类 · 分布式/并行（+1）

7. **[新类别/新签名]** **KvChunkAddressTable protobuf 导入按名字重排 config id**（tt-metal #56263）— 跨端点迁移的真实契约是"config 名字排序相同"却无人校验：MiniMax-M3 PD 迁移每个 config 静默错位一格（head h+1 坐 row h、v_h0 落 k_h3），无错误、两端 loopback 自洽全过，仅跨端 golden PCC 抓到；三个 builder 已各自独立绕路。→ **5.37** `NEW`

### 第 6 类 · 服务栈/缓存（+4）

8. **[已知类别]** **MTP PP>1 broadcast_drafts 缺 record_stream → idx_mapping UAF**（vLLM #56470，closed + 修复 commit bf33e54548）— 侧流 gather 消费每步 async_copy 分配的 idx_mapping 未 record_stream → allocator 复用 → indexSelectSmallIndex IMA（A100 TP2 PP3 ~90s 必崩）；D1/D0 判别窗 A/B 证实。归类 var:6.194（侧流消费缺 record_stream 族）。→ **6.228**
9. **[新类别/新签名]** **Mamba state 块 (re)allocation 不清零 → 跨租户毒化**（vLLM PR #56524）— 零化机制三重 mamba 盲区（记录只认 AttentionSpec / 直取 block_pool 绕过 / zeroer 不解 KDA tuple 绑定）；生产 Kimi-K3 abort-retry 后 1–2/36 概率输出无界多语言垃圾直至重启，补丁后 0/72。→ **6.229** `NEW`
10. **[新类别/新签名]** **Mamba 前缀 hash 发布先于提交栅栏 → abort-retry 毒化雪崩**（vLLM PR #56525，与 #56524 姊妹）— schedule 时发布 hash、写入晚于此执行（abort 则永不执行），一步有效期守卫不够：retry 命中"已发布未写入"边界 hash → 垃圾 state 起步，**retry 正常完成后把毒化 hash 提交为有效** → 同前缀后续请求全部中毒直至重启。→ **6.230** `NEW`
11. **[新类别/新签名]** **DSv4.1 DBO 微批共享 sparse index buffer 覆写**（vLLM PR #56476）— top-k/候选块为模型级单份 buffer 跨层持有：微批 1 source 层在微批 0 consumer 层读取前覆写；负控制 48/96 失配、4×GB200 全 rank 复现 179830+/1024 行 top-k 覆写。→ **6.231** `NEW`

### 第 7 类 · 量化（+2）

12. **[新类别/新签名]** **flashinfer vendored TRT-LLM fp8 cast 经 bf16 双舍入（修复未传播）**（flashinfer #5128）— #4167 只修了 header-only twin，vendored `csrc/nv_internal` 副本仍是 pre-fix：2.34% 元素恰差一个可表示步、每元素都在容差内只有聚合计数能抓到；当前路径 dormant 但 vendored 树活着。→ **7.83** `NEW`（1.122"修复未跨副本传播"的仓库内镜像）
13. **[已知类别]** **sm100 fp8 GEMM 忽略 leading strides**（vLLM PR #56480）— (8,256) stride(384,1) 视图按 256 寻址；与 #55537（sm90 standard）、#56248（sm90 blockwise）互不覆盖。归类 var:7.75。→ **7.84**

### 第 8 类 · KV 缓存/注意力（+1）

14. **[已知类别]** **GLM-5.3-Flash ROCm indexer block-table 粒度失配 + KpoolTailSpec 通用 slot-map OOB**（vLLM #56380；修复 PR #56381/#56469）— 128-token 存储页 vs 640-token manager 块：表宽外 gather 写零 scale、表宽内 manager ID 指错物理页 → **长 prompt 尾部召回静默退化**；问题 2 wild physical id → GPU memory fault。归类 var:8.73（GLM-5.3-Flash indexer 寻址族）。→ **8.83**

### 第 9 类 · 位置编码/RoPE（+1）

15. **[新类别/新签名]** **YaRN 别名变体 max_len 二次缩放 → RoPE cache 越界**（vLLM PR #56446，merged）— `_get_and_verify_max_len` 只认精确串 "yarn"：telechat3-yarn/deepseek_yarn 把已缩放长度再乘 factor（131072↔32768、5242880↔131072）；修复对齐 transformers 上游规则。→ **9.24** `NEW`

### 第 11 类 · 解码/采样（+2）

16. **[新类别/新签名]** **ngram spec × hybrid recurrent：spec/非spec 路径状态布局约定分歧**（vLLM PR #56531，fixes #40875）— 非 spec 每请求 1 槽 vs spec 窗口按 num_accepted−1 选权威槽；"接受 draft 的步"紧接"零 draft 的步"（ngram 例行）即读陈旧槽 0 → **间歇性掉字/重复字**；修复为纯分类学（恒走 spec 路径）。另修 num_accepted 初始化 0→1（0 会成 −1 槽读）。→ **11.75** `NEW`
17. **[新类别/新签名]** **extract_hidden_states proposer 只读反馈第 0 列**（vLLM #56417）— num_spec=1 时一步可出 2 个有效 token，helper 报 valid_count=1/next=15（应 2/18）：async scheduling 下数字/标点重复可复现；复用 EAGLE kernel 后精确恢复基线。→ **11.76** `NEW`

### 第 12 类 · 权重/适配器/检查点（+2）

18. **[已知类别]** **量化 DFlash2 draft 静默 ~0% 接受**（SGLang #39087）— compressed-tensors W4A16 draft 正常加载服务，accept 3.71→1.03、无错误无警告、decode 比无 drafter 更慢；同权重 BF16 正常；#36599（modelopt_fp4）是同型的 fail-loud 对偶。归类 var:12.31。→ **12.35**
19. **[新类别/新签名]** **权重缓存 daemon 多消费者共享单 IPC refcount 槽**（vLLM PR #56472）— copy client 释放把计数带零 → 存活 engine 的 zero-copy 映射悬垂（A10 实测 allocation 4096→0 时映射仍 live）→ 生产形态为 IMA/静默权重损坏。→ **12.36** `NEW`

---

## 二、检视后不收录（含理由）

| 条目 | 理由 |
|---|---|
| vLLM #56521（ROCm libhsa heap overflow segfault） | fail-loud segfault、上游未定位（同 #38849 标准） |
| vLLM #56516（prompt_lookup docstring 默认值错） | 纯文档 |
| vLLM #56396（SimpleCPUOffloadConnector assert crash） | 首请求裸 AssertionError，fail-loud crash |
| vLLM PR #56448（DFlash profiling 批超输入 buffer） | fail-loud 启动错误（IndexKernel assert），#53970/#53080 已知族 |
| vLLM PR #56486（KV offload 滑窗 pending 标志） | 调度延迟/性能，非正确性损坏 |
| vLLM PR #56520（Anthropic 内联 system 合并） | 前缀缓存命中率损失=性能，非性能正确性 |
| vLLM PR #56500（QSA logits workspace 有界化） | 内存治理/性能 |
| vLLM PR #56511（测试 activation override） | 测试 |
| vLLM PR #56447（GLM-OCR MTP 位置掩码 MRoPE） | merged 但正文仅 3 行、行为效应证据不足——**留观** |
| SGLang #38981（Kimi-K3 abort → free_segments assert 杀全 rank） | fail-loud invariant crash（同 #38840 标准）；生产崩溃已记录于本表 |
| SGLang #39092（GRAPH_BATCH_CAPTURE trace 命名 cursor） | profiling 工具缺陷（IndexError 启动崩） |
| SGLang #38902/#38846/#38889（Feature） | feature |
| tt-metal #56254（tall-narrow uint8 tilize） | host-side TT_FATAL，fail-loud |
| tt-metal #56242（1x32 tiny tiles hang） | hang |
| tt-metal #56238（typecast quasar 1x3 wrong values） | 仅三行"needs investigation"，无根因——**留观** |
| tt-metal #56298（Fabric Z 方向 DA/FLA 极性失配挂起） | hang/饿死（同 hang 类留观标准） |
| tt-metal #56073/#56068/#56133/#56153 | hang/abort fail-loud/未核验/CI |
| tt-metal #56195（pack stride 修复） | #56194（1.205）的修复侧 |
| transformers #48687（router 梯度） | 训练侧梯度（既往已判不收录，closed） |
| TensorRT-LLM #19063（C++20 编译失败） | 构建失败 fail-loud |
| flashinfer #5157（H100 CI cudnn-frontend） | CI 基础设施 |
| flashinfer #5079（per-token FC2 act scaling） | feature（既往已判） |
| triton 仓库 | 窗口内 0 命中 |

## 三、arxiv

5 组查询（SDC LLM / bit flip LLM inference / KV cache corruption / LLM inference correctness / GPU soft error ML）命中全部为已编目或不收录的方法论文：
- 已编目：TrainSDC 2608.30769、SCOUT 2608.11034、LLM-PRISM 2604.10390、JITterFlip 2608.29745、UEP 2607.19623、RangeGuard 2605.04563、Rotated Robustness 2603.16382、Targeted BFA 2603.10042、When Latent Agents Lie 2606.28958、Addressable Memory 2608.07408、Runtime Observability 2608.05863、Compression-Aware Abstention 2608.29934、指令级 GPU 软错误 FI 2601.19912、Exploring SDC 2604.00726、Understanding SDC 2502.12340
- 不收录（方法论文）：Maverick 2609.10264（可验证推理）、Sample-Guided Top-K 2609.08450、MetaKV 2609.07966、Signed Rescue 2609.07786、Moment-KV 2605.29873、Nexus 2608.20397

**无新编目论文。**

## 四、API 调用记录

| 类别 | 调用 | 结果 |
|---|---|---|
| GitHub Search API | 10 组（vllm bug / vllm wrong / sglang wrong / hf wrong / trtllm wrong / triton wrong / ttm bug / flashinfer wrong / vllm quant / sglang kv），间隔 ≥7s | 全部一次成功，无限流 403 |
| GitHub core REST | issues 19 次 + pulls 13 次（`--http1.1`） | 32/32 成功 |
| arxiv API | 5 组（`curl -sL`） | 全部成功 |

## 五、合并操作

- 主文档新增：1.200–1.205、5.37、6.228–6.231、7.83–7.84、8.83、9.24、11.75–11.76、12.35–12.36
- 滚动汇总/目录/分布数字已按实际条目重新计算：1039 → **1058**（第 1 层 908 → 927）
- 导出：`export_faults.py` → fault_catalog.xlsx + fault_catalog.csv（实际 Parsed = **1058** 行；当前脚本版本将 A1–A17 作为一等条目纳入导出——脚本内注释明确 "A1–A17 are first-class catalog entries... must appear in xlsx/csv"，任务描述中"减 17"的旧规则已被脚本自身更新取代）

# 每日扫描报告 — 2026-09-08

- 扫描日期：2026-09-08（滚动窗口 `created:>2026-09-07`，覆盖 2026-09-07T00:00Z 之后创建的 issue）
- 目标仓库：vllm-project/vllm、sgl-project/sglang、NVIDIA/TensorRT-LLM、triton-lang/triton、tenstorrent/tt-metal、huggingface/transformers、flashinfer-ai/flashinfer
- 主文档基线：989 条（截至 2026-09-07 合并后）
- 去重方法：对每条候选在主文档全文检索 issue 编号（`#NNNNN`），全部确认未编目后方进入合并

---

## 一、扫描结果总览

- GitHub Search API：11 组查询（vLLM×4、SGLang×1、TRT-LLM×1、triton×1、tt-metal×1、transformers×1、flashinfer×1，含 1 组仓库名错写被拒后以正确名 `flashinfer-ai/flashinfer` 重发成功）；无限流 403
- core REST 详情：19 次全部成功（`--http1.1`）
- arxiv：5 组关键词查询，全部命中已编目或不收录的方法论文/非 LLM 域论文
- **独立候选 22 条 → 合并 18 条**（7 条新子签名 + 11 条已知类别新实例；4 条不收录见「三」）

## 二、已合并条目（18 条）

### A. 新子签名（NEW，7 条）

| # | 标题 | 来源 | 归类 |
|---|---|---|---|
| 1 | ttnn.softplus (fp32) beta*x < −8.72e6 时返回 ±inf/NaN/高达 3.28e38（Hacker's-Delight magic 常量在 \|z\|>2^22 域外失效，约 840 万个 bit pattern 受影响） | tt-metal #55798 | 第 1 类 → 1.182 |
| 2 | ttnn.reciprocal (fp32, Blackhole) \|x\|≥2^119 丢弃 Newton 精炼返回原始 7-bit SFPARECIP 种子（0.56% 相对误差 × 7 个完整 binade，1.17 亿 bit pattern） | tt-metal #55797 | 第 1 类 → 1.183 |
| 3 | ttnn unary ops 预分配 output_tensor 形状守卫自己与自己比较（TT_FATAL 永不触发；过大形状越界读、过小静默截断） | tt-metal #55800 | 第 1 类 → 1.184 |
| 4 | ttnn.from_torch 整数 dtype 宿主 static_cast 越界为 UB，x86_64 与 aarch64 结果不同且与设备不一致 | tt-metal #55825 | 第 1 类 → 1.185 |
| 5 | DeepSeek-V4-Flash 深上下文精确检索在 prompt 长度 ≡3 (mod 4) 时确定性错 token（28k 处 needle 末位数字错；四个独立栈同错、一个供应商正确——非权重属性；reasoning 开启则正确） | vLLM #55927 | 第 6 类 → 6.215 |
| 6 | TRT-LLM EAGLE3 dynamic-tree 解码发出 grammar 外 token（flat-mask 路径） | TRT-LLM #18892 | 第 11 类 → 11.70 |
| 7 | TRT-LLM guided decoding 在 KV-cache 重计算后重启 grammar（matcher 历史丢失，已提交 token 的约束进度蒸发） | TRT-LLM #18891 | 第 11 类 → 11.71 |

### B. 已知类别新实例（var:，11 条）

| # | 标题 | 来源 | 归类 | var 族 |
|---|---|---|---|---|
| 8 | ttnn.sigmoid_bw 对 x≥6.25 (bf16)/16.75 (fp32) 返回零梯度（s*(1−s) 在 s 舍入到 1 后灾难性相消） | tt-metal #55724 | 第 1 类 → 1.186 | var:1.171 |
| 9 | ttnn.expm1_bw 对 x≤−6.25 (bf16)/−17 (fp32) 返回零梯度（导数用 expm1(x)+1 构造，−1+1 相消） | tt-metal #55723 | 第 1 类 → 1.187 | var:1.171 |
| 10 | ttnn.sign(NaN) 返回 +1（自家 golden 返回 0），sign(grad)*inf 反向习语继承之——8 个 backward 组合把 NaN 梯度变成 +inf | tt-metal #55827(tt) | 第 1 类 → 1.188 | var:1.151 |
| 11 | ttnn.repeat_bw 三重失效：非 tile 倍数 TT_FATAL / dim-2,3 空梯度列表 / 重复维 extent>1 时整轴求和 | tt-metal #55799 | 第 1 类 → 1.189 | var:1.132 |
| 12 | Hybrid Mamba + MTP：drafter 注意力后端把 batch-reorder 阈值降到 1，decode 行被送进 Mamba prefill 内核（只读写 state slot 0）→ 首解码步起 recurrent state 为垃圾，word salad/空循环（0.2–1% 请求，定向触发 7%） | vLLM #55894 | 第 6 类 → 6.216 | var:6.212 |
| 13 | 两 vLLM 实例同刻启动同一 GPU：互相把对方的模型加载计入自己的 non-torch 内存（KV cache 缩水或 "No available memory for the cache blocks"） | vLLM #55827 | 第 6 类 → 6.217 | var:6.136 |
| 14 | HiCache backup 线程遇任何存储后端异常即死：排队备份永不再写 L3、host 内存永不释放，唯一症状是很久之后的 host 池耗尽 | SGLang #38428 | 第 6 类 → 6.218 | var:6.138 |
| 15 | UnifiedRadixCache：前缀在 host 层驱逐后仅以 backuped stub 存活时 L3 storage 永不被查询 → 明明数据在 L3 却全量重算 | SGLang #38452 | 第 6 类 → 6.219 | var:6.137 |
| 16 | cache_salt 在 BlockStored KV 事件中无法用导出的 msgspec 类型解码（salt 静默丢弃——隔离特性对公共类型消费者不可见） | SGLang #38561 | 第 6 类 → 6.220 | var:6.169 |
| 17 | FlashInfer GQA decode + fp8 KV cache 在 sm75 (T600) 共享内存溢出 crash（smem_md 预算未计 GQA 组） | flashinfer #5030 | 第 8 类 → 8.78 | var:8.62 |
| 18 | PP>1 + spec decode：mamba align postprocess 用 pp_size 步前的陈旧 batch 行索引 block table → recurrent state 拷贝到错误物理块（NaN logits / 静默召回损失） | vLLM #55951 | 第 11 类 → 11.72 | var:11.64 |

> 注：tt-metal #55827 与 vLLM #55827 撞号（不同仓库），已在正文区分标注（1.188 vs 6.217）。

## 三、不收录说明（4 条）

1. **vLLM #55916** `[RFC]: Add an RDNA4 FlyDSL all-reduce backend` — 功能 RFC，非正确性故障。
2. **SGLang #38536** `Rust TreeCore (v0.5.19) e2e regression at concurrency` — 性能回归（+7.6~11.9% p50），非正确性/静默损坏；与本编目主题不符。记录备查。
3. **tt-metal #55745** `Triage first prefil bringup run of Mistral Medium 3.5` — bring-up 裁决清单（recipe_gap/donor_wrong 等），未含已证实的数值正确性缺陷；后续若出具体缺陷条目再编目。
4. **flashinfer #5025 / #5026** — 前者为 unified benchmark 套件自身构造了非法 cache dtype（测试基础设施缺陷），后者为 CI 测试超时（flaky 基础设施）；均非推理服务正确性故障。另 tt-metal #55815 为 #55825 的重复报告（已关）。

## 四、重点新发现摘要

1. **mod-4 prompt 长度门控的确定性错 token（#55927）**：DeepSeek-V4-Flash 精确检索在 prompt ≡3 (mod 4) 时确定性出错且**四个独立服务栈给出同一个错 token**、一个供应商正确——指向某个共享实现层（分块/padding 对齐维）的系统性缺陷而非权重或单栈 bug。这是「块对齐边界 × 检索精度」的全新子签名。
2. **批量守卫自比较（#55800 tt-metal）**：`TT_FATAL(shape == computed_shape)` 实际写成形状与自身比较——守卫永不触发。这是继 1.135（parametrize 不可达）、1.180（golden 调用不存在 API）之后「验证层自身坏死」家族的第三个独立实例，且这次守卫缺失直接导致越界读/静默截断两类运行期损坏。
3. **grammar 约束进度的两条丢失路径（TRT-LLM #18891/#18892）**：guided decoding 在 KV 重计算后重启 grammar、dynamic-tree 解码 flat-mask 路径绕过 grammar——约束解码正确性是第 11 类此前只有 xgrammar bitmask fail-open（11.57）一条的新兴子域。
4. **drafter 后端独立选择压低全局阈值（#55894）**：「Never inherit the attention backend from base」这一独立选择通过 `min()` 聚合把**目标模型的正确性阈值**压垮——跨组件配置聚合的正确性侵蚀新机制。

## 五、API 调用记录

| # | 类型 | 查询/目标 | 结果 |
|---|---|---|---|
| 1 | search | vllm correctness created:>09-07 | 2 命中 |
| 2 | search | vllm quantization wrong | 1 命中 |
| 3 | search | vllm "KV cache" corrupt | 1 命中（与 #1 重复） |
| 4 | search | sglang wrong/corrupt/NaN/silent | 4 命中 |
| 5 | search | TensorRT-LLM 全部新 issue | 2 命中 |
| 6 | search | triton wrong/NaN/miscompile | 0 命中 |
| 7 | search | tt-metal numerical/PCC/NaN | 15 命中（10 有效） |
| 8 | search | transformers wrong/NaN/corrupt | 0 命中 |
| 9 | search | vllm silently | 3 命中 |
| 10 | search | flashinfer（仓库名错写） | Validation Failed → 重发 |
| 11 | search | flashinfer-ai/flashinfer | 3 命中（1 有效 #5030） |
| 12–30 | core REST | 19 个 issue 详情 | 全部 200 |
| 31–35 | arxiv | SDC+LLM / bitflip / KV cache / correctness / GPU soft error | 全部已编目或不收录 |

arxiv 明细：2608.30769（TrainSDC，已为 §3.1 补充来源）、2608.11034（SCOUT）、2604.10390（LLM-PRISM）、2604.00726、2502.12340、2608.29745（JITterFlip=3.18）、2608.25276（GBFA=3.16）、2608.23774（ROBBIN=3.17）、2607.25227（Decision-Level Hijacking，已在编目）、2608.07408（=14.7）、2608.05863、2606.28958、2601.19912（=1.98）均已在主文档；2608.20397（Nexus）/2605.29873（Moment-KV）为方法论文不收录（沿用 2026-08-29/30 判定）；2104.09455（2021 AIGC fault tolerance）为非 LLM 推理故障编目域的经典方法论文，不收录。

## 六、合并动作

- FAULT_MASTER_REFERENCE.zh.md：新增 18 条（第 1 类 +8：1.182–1.189；第 6 类 +6：6.215–6.220；第 8 类 +1：8.78；第 11 类 +3：11.70–11.72），**989 → 1007**
- 滚动汇总/目录/来源表数字同步更新
- export_faults.py 重新导出 xlsx + csv

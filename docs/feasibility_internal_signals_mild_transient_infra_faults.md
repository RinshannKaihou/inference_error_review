# 基于模型内部信号的检测方法用于监控轻度/瞬态基础设施故障:可行性评估

## TL;DR
- **语义级内部信号检测器(hallucination probes、semantic entropy、EigenScore)对"不改变语义的轻度数值扰动"原理上不敏感,且从未在硬件 fault injection 下测过响应曲线——这是明确的文献空白;对轻度瞬态 infra 故障真正有原理覆盖的只有 cheap oracle(recompute/batch-invariance)、per-token 统计流变点检测和聚合层特征空间监测三者的组合。**
- **瞬态单 bit flip 在 LLM 推理中绝大多数被屏蔽(单事件下 masked 占主导,abnormal 约 15–30%),但屏蔽率强烈依赖 bit 位置(指数高位是 SDC 主源)、数据类型和发生位置;真正"轻度且瞬态"的扰动幅度常低于 batch 非确定性噪声底 EPS_BATCH≈2.4e-4(注:该数值为已退役原型环境 MPS/fp32/0.5B 的实测值,目标栈 bf16+连续批处理下需重测),在单前向对比下原理上不可分辨。**
- **在 1e-5~1e-8/request 的极低 base rate 下,即便 AUROC 0.9 的检测器 PPV 也趋近于零,inline per-request 语义检测在经济上不可行;正确架构是 fleet-level 离线筛查(Meta Fleetscanner/Ripple)+ 抽样 cheap oracle + 廉价 per-token 统计监测,而非把 hallucination 检测器搬到生产推理路径上。**

## Key Findings

**1. 瞬态故障屏蔽率(有直接实验证据)。** 经典 CNN 数据(Li et al. SC'17;PRDC'22 arXiv 版)给出 masking 基线:在 CIFAR-100 上对正确分类样本注入单 bit flip,约 98–99% 被 masked,整体 SDC 率约 1–2%(其中 20–50% 演变为 SCM/misclassification)。LLM 专项的 instruction-level 研究(arXiv:2601.19912,基于 NVBitFI,约 300 GPU-hours)表明:单 bit flip 注入时"大多数错误可被 masked",abnormal 率(DUE+SDC)随注入数从 1 增至 8 由约 15%/25%/30% 升至 >75%;且**更大模型 masked 比例更高、更可靠**(参数冗余更多)。bit 位置是决定性因素:该研究指出低位 bit 几乎无害(SDC 率随位不显著),SDC 风险高度集中在高位指数 bit——最高有效指数位(fp32 的 bit 30)是 SDC 主源,而符号位(bit 31)翻转常被后续非线性/归一化吸收;DUE 则相对均匀分布于各 bit。DNN 内存位敏感研究(arXiv:2607.19623,16 个 workload 含 transformer,870+ 注入 run)发现存在 data-type-specific 阈值 Xsafe:低于该阈值的尾数低位翻转使 task metric 退化 <1%,而指数-尾数边界处单 bit 翻转导致灾难性崩溃。发生位置(权重 vs 激活 vs KV cache)方面:OrCA(ACM TODAES)证实 transformer 因参数冗余对低幅度权重故障"rarely cause noticeable effects";但 KV cache 中损坏条目会跨 decode step 持久化(见第 5 点)。

**2. 各方法族对轻度瞬态故障的灵敏度(核心)。**
- **语义级检测器(hallucination/UQ):原理上不敏感 + 文献空白。** semantic entropy(Farquhar et al., Nature 2024)、EigenScore/INSIDE(arXiv:2402.03744)、SAPLMA、P(True)(arXiv:2207.05221)、SelfCheckGPT、LLM-Check(NeurIPS 2024)的设计目标是"输出语义是否自洽/正确",信号来自多次采样的语义发散或对错误的隐状态可分性。一个不改变 argmax、被 mask 或仅改变个别 token 的数值扰动在语义层面不留签名。经跨 5 个检索角度的专门核查,**没有任何工作把 hallucination/uncertainty 检测指标放在硬件 fault injection 下测 dose-response(检测分数 vs 故障幅度)**:fault-injection 社区(arXiv:2601.19912、2604.00726、2604.10390)只用 loss/accuracy/gradient-norm/NaN-Inf 作检测信号;hallucination/UQ 社区从不注入硬件故障。最接近的相邻工作是把 UQ 放在量化精度(16/8/4-bit)下评估,但量化≠bit-flip/SDC。这是明确的文献空白。
- **geometric/statistical 检测器(Mahalanobis、range check、分位数监测):对单 token 尖峰有原理上的可检性,但被聚合稀释。** Dr. DNA(ASPLOS'24)用 neuron activation 分布签名"achieves 100% SDC detection rate for most cases, 95% detection rate on average and >90% across all cases",且"<1% memory overhead and <2.5% latency overhead";quantile-marker 监测(arXiv:2310.20349)对 activation peak shift(平台故障)和 bulk shift(输入故障)达 ~96% precision/~98% recall,开销低至 0.3%。但这些都是 per-inference 粒度;若把单 token 异常并入跨 token/跨请求聚合窗口,单点尖峰会被均值淹没——这正是瞬态信号的稀释问题。
- **trajectory/序列变化点检测:可从相邻文献外推,无 LLM-故障直接证据。** CUSUM 及其在线变体(Window-Limited CUSUM arXiv:2206.06777、Online Kernel CUSUM arXiv:2211.15070、NN-CUSUM arXiv:2210.17312)是常数内存、递归、面向流、具统计最优性的变点检测器,原理上可迁移到逐 token logprob/激活统计流(逐层余弦漂移、CUSUM 起点),但尚无人把它用于 LLM 推理瞬态故障检测。
- **cheap oracle 作为 ground truth:最可行。** recompute-oracle、batch-invariance 校验、speculative decoding draft/target 一致性可作为瞬态故障的近似真值。已知结构盲点:(i) draft 与 target 共享的计算路径故障对一致性信号不可见;(ii) post-forward 故障(采样层、解码循环、logits 变换)对所有隐藏态检测器原理上不可见。

**3. 可检测下限与噪声底(有直接实验证据)。** FP16 KV-cache 研究(arXiv:2604.15409,LLaMA-2-7B/Mistral-7B/Gemma-2-2B,GSM8K,31,500 run-pairs)量化了天然 run-to-run 变异,给出了对本问题最关键的定量结论:cache-ON vs cache-OFF 在**相同硬件、同一 batch** 下 mean KL >10、token 分歧率达 **100%(含 greedy 解码)**;切到 FP32 后 KV 分歧降至噪声底 **~2.5e-8**(下降 8 个数量级)、token flip 率精确归零。作者原文:"Mean KL >10 across all conditions indicates substantial redistribution of probability mass at every decode step; these are not small numerical perturbations." 这意味着 FP16 下天然数值噪声足以翻转 100% 的下游 token 采样,把轻度故障的签名彻底淹没——用户框架中 EPS_BATCH≈2.4e-4 噪声底与此定性一致(该数值系旧原型环境实测,不可直接继承,见主文档图例)。Thinking Machines batch-invariant kernels(He, Sep 2025)可把 batch 非确定性压到 bitwise 零,从而让原本低于噪声底的轻度故障变得可检——但代价是显著且**实现相关**的性能开销:TML 自测 batch-invariant GEMM 相对 cuBLAS "only lose about 20% performance",端到端约 61.5% 开销;SGLang+CUDA graphs 优化后"most slowdowns ranging from 25% to 45%, and average slowdown...being 34.35%"(相对 TML 基线 2.8× 加速),目标"reducing the performance gap to under 20%";LLM-42(arXiv:2601.17768)测得 Triton batch-invariant kernel "peaks at 194 TFLOPS, a slowdown of 63%" vs cuBLAS 527 TFLOPS。生产上只有对抽样流量启用才可摊销。

**4. 误报经济学与低 base rate(有直接实验证据 + 贝叶斯计算)。** Meta"Silent Data Corruptions at Scale"(arXiv:2102.11245)与其 2025 工程博客报告 fleet 中"about one fault per thousand devices"(约 1/1000 硅器件,且明确"much higher than cosmic-ray-induced soft errors",是基础硅缺陷而非粒子效应);Llama 3 训练中 SDC 造成"1.4% of unexpected GPU interruptions"(arXiv:2605.04213 引 Llama 3 Herd:54 天快照 466 次中断、419 次意外,其中约 78% 归因确认硬件问题、GPU 问题占意外中断 58.7%,16,384 H100 上约每 3 小时一次故障);Google 估计 Gemini 训练中"an SDC event occurs every week or two"(arXiv:2502.12340)。单请求粒度 base rate 更可能落在 1e-5~1e-8。在 base rate=1e-6、检测器工作点 TPR=0.9/FPR=0.1(对应 AUROC≈0.9)下,PPV≈(1e-6×0.9)/(1e-6×0.9+(1−1e-6)×0.1)≈9e-6——即约 11 万个告警才有 1 个真故障。要在 1e-6 base rate 下达到 PPV≈0.5,需 FPR≈1e-6 量级,远超语义检测器能力。这与稀有事件评估文献一致:AUROC 对 prevalence 不敏感,但在低 prevalence 下高 AUROC 可掩盖极低 PPV,应改用 AUPRC(Nature npj Digital Medicine 2018)。**结论:inline per-request 语义检测在经济上不可行。** 检测延迟 vs 故障持续时间的错配进一步恶化:瞬态故障(单事件、不留持久痕迹)常在聚合窗口关闭前已消失,事后无法复现验证——这也是 fleet-level in-production 测试(Ripple)相比 inline 事后检测的优势所在。

**5. 瞬态扰动在自回归解码中的传播动力学(有直接实验证据)。** 两种命运并存——衰减 vs 级联,取决于 logit margin 与传播路径。arXiv:2604.15409 显示扰动经 KV cache 持久化后跨 decode step 持续存在(divergence 持续所有步),且存在剧烈层间放大:**LLaMA-2 层 0→层 1 hidden state drift 从 3.45 跳到 361.96(单层 105× 放大);Mistral 从 25.76 跳到 902.36(35×)**,归因于 4:1 GQA 把一个共享 KV head 的舍入误差广播到 4 个 query head(相关误差而非独立噪声)。余弦相似度在第一个 transformer 层内即急剧去相关(Mistral 降到 0.015)。是否 lock-in 到错误分支取决于决策边界距离:靠近边界(小 logit margin)时任意小扰动即翻转 argmax,flip index 与 mean KL 负相关(Pearson r=−0.202~−0.120)——早翻转与高 KL 是模型不确定性的共同后果。activation patching 把分歧定位到**有状态的 KV cache 而非 residual stream**(patching 残差流仅恢复 <1% 分歧)。签名形态因此是双模态:既可能是"单 token 尖峰"(需 per-token 检测),也可能是"扰动后经 KV cache 持续漂移"(聚合窗口可见)。这直接决定了检测粒度选择。

## Details

**结论矩阵(方法族 × 故障幅度 × 持续时间 × 粒度 × 开销 × 低 base rate 实用性):**

| 方法族 | 重度故障 | 中度故障 | 轻度(近噪声底) | 瞬态单事件 | 持久 | 开销 | 低 base rate 实用性 | 证据档 |
|---|---|---|---|---|---|---|---|---|
| 语义检测器(SE/EigenScore/probes) | 部分(若改变语义) | 弱 | **原理不敏感** | 弱(需多采样,尖峰被吸收) | 部分 | 高(多次采样) | 差(PPV→0) | 空白(无 FI dose-response) |
| geometric/statistical(Mahalanobis/range/quantile) | 强(Dr.DNA 95–100%) | 中 | 弱(被聚合稀释) | per-inference 可,聚合稀释 | 强 | 低(<2.5%) | 中(需低 FPR) | 直接(非 LLM/CNN-ViT) |
| trajectory/变点(CUSUM) | 强 | 中 | 中(常开流) | **适合尖峰+漂移** | 强 | 低(常数内存) | 中 | 外推(无 LLM-故障) |
| cheap oracle(recompute/batch-invariance) | 强 | 强 | **强(bitwise)** | 强(抽样) | 强 | 高(20–63%,抽样摊销) | 好(oracle 非概率) | 直接 |
| speculative draft/target 一致性 | 中 | 中 | 中 | 中(部分路径) | 中 | 零(已开启时) | 中 | 外推 |
| ABFT(ALBERTA/Flash-ABFT) | 强(~100% 覆盖) | 强 | 中(受 eps 阈值限) | 强 | 强 | 中(GEMM 定制) | 好(校验数值恒等式) | 直接 |

**文献空白清单(完全无直接实验证据的组合):**
1. 任何 hallucination/UQ 检测器在硬件 fault injection 下的 dose-response(检测分数 vs 故障幅度)——**核心空白**。
2. CUSUM/在线变点检测应用于 LLM 逐 token logprob/激活流做瞬态故障检测——无 LLM 场景实验。
3. 轻度瞬态故障幅度分布 vs 天然 run-to-run 变异幅度分布的直接定量并置(arXiv:2604.15409 给了噪声底一侧,但未系统对比故障幅度分布)。
4. batch-invariant kernel 抽样启用对轻度故障可检性提升的生产级量化。

**故障编目框架对齐。** 用户的 `infra_off`(把隐状态推离健康流形:SDC、bit flip、attention mask bug、内核损坏)正对应 fault-injection 文献中的 critical/SDC-causing faults,geometric(Mahalanobis/范数)和 statistical(分位/峰度)度量能捕捉——Ranger(arXiv:2003.13874)正是利用"critical fault 表现为异常大的 activation 偏离"把 error resilience 提升 3–50x。`model_on`(停留在自洽流形:量化、能力上限、RoPE 缩放)对应被模型吸收的扰动,语义检测器和特征空间检测器都难分辨,因为隐状态仍在健康流形内。`ambiguous`(输入/dtype/长度/batch 门控)对应 arXiv:2604.15409 揭示的 FP16 数值非等价——只在特定 batch/cache 配置下出现签名,是 dtype/长度门控的典型例证。

**度量族到故障族的映射。** geometric 与 statistical 度量最适合 `infra_off` 重度故障;spectral(有效秩/稳定秩/EigenScore)是语义信号,对 infra 故障不敏感;infotheoretic(预测熵/EigenScore)同样是语义信号;cheap(batch-invariance、recompute-oracle)是唯一对轻度瞬态故障有原理覆盖的度量族。EigenScore(arXiv:2402.03744)的机制——对多次生成的 sentence embedding 协方差矩阵特征值取对数均值(通过 SVD 求解)——本质是语义多样性度量,数值扰动若不改变多次采样的语义分布则完全不产生信号。

**部署约束下的可行性。** 大规模 MoE(数百 B)、W4A8/W8A8、vLLM、speculative decoding、continuous batching 下,<5% 延迟预算排除了 DMR(2x)/TMR(3x)全冗余(arXiv:2605.04213 明确 detection 需 2× overhead、correction 需 3×)。ALBERTA(arXiv:2310.03841)的 checksum-based GEMM 保护对最脆弱层可做到接近 100% 瞬态错误覆盖,并用逐层 numerical imprecision 分布区分"数值误差"与"真故障"(正对应用户框架的 eps 阈值问题);Flash-ABFT(arXiv:2507.16676)对 attention 层(含 softmax 归一化)做到 5.3% 面积、<1.9% 能耗开销。这些 ABFT 方法是比隐状态检测器更对症的轻度瞬态故障手段,因为它们校验数值恒等式而非语义。

**speculative decoding 作为部分免费 oracle。** draft/target 校验本质是一个持续运行的一致性检查:target argmax 与 draft 提议不符即 mismatch。生产中已开启 speculative decoding 时,这是零额外开销的部分 oracle——但仅覆盖 draft 与 target 不共享的计算路径,且 post-sampling 故障不可见。这与用户框架中标注的已知结构盲点完全一致。

## Recommendations

**分层方案(按 base rate 与开销分工):**

1. **Fleet-level 离线/在线筛查为主力**(证据档:有直接实验证据)。沿用 Meta Fleetscanner(out-of-production,分钟级)+ Ripple(in-production,毫秒级)模式:官方数据显示"in-production testing can, within 15 days, detect 70 percent of the fleet data corruptions that opportunistic testing may take about 6 months to detect",而"the last 30 percent...are only detectable with opportunistic testing"(Fleetscanner 93% 覆盖/23% 独有,Ripple 7% 独有,arXiv:2203.08989)。这是低 base rate 事件的正确统计尺度——把检测问题从 per-request 移到 per-device,并解决瞬态故障"事后无法复现"的时序错配。

2. **抽样 cheap oracle 做 inline 兜底**(证据档:可从相邻文献外推)。仅对抽样流量(如 0.1–1%)启用 batch-invariant kernels + recompute-oracle,在 <5% 全局延迟预算内换取对轻度瞬态故障的 bitwise 可检性。基准:当抽样流量的 recompute mismatch 率超过 FP16 噪声底基线(arXiv:2604.15409 的 ~2.5e-8 FP32 参照或 batch-invariant bitwise 零)时升级排查。若 batch-invariant 开销从当前 34–63% 降到 <20%(SGLang 路线目标),可上调抽样比例。

3. **per-token 统计流变点检测做廉价常开层**(证据档:纯外推,LLM 场景无直接证据)。对逐 token logprob/激活范数流跑 Window-Limited CUSUM(arXiv:2206.06777)或 Online Kernel CUSUM(arXiv:2211.15070),常数内存、可流式、per-request 粒度,同时捕捉"单 token 尖峰"与"KV cache 持续漂移"两种签名。前置工作:需先用 fault injection 建立 dose-response 曲线(当前空白第 1 项)以标定阈值与 ARL/EDD。

4. **聚合层特征空间监测做慢变兜底**(证据档:有直接实验证据,非 LLM)。Dr. DNA/quantile-marker 式监测(<2.5% 延迟)部署在聚合窗口,捕捉持续性 `infra_off` 漂移与 `model_on` 的慢变退化。

5. **对最脆弱 GEMM/attention 层加 ABFT**(证据档:有直接实验证据)。ALBERTA/Flash-ABFT 在 <2% 量级开销下对轻度瞬态故障提供数值恒等式校验,比任何隐状态检测器都更对症。

**不建议:** 把 hallucination/semantic-entropy/EigenScore 检测器搬到生产推理路径监控 infra 故障——原理不敏感(语义不变的数值扰动无签名)、低 base rate 下 PPV 趋零、开销高(需多次采样)。

**改变建议的阈值/基准:** (a) 若出现把 UQ 指标放在 fault injection 下的 dose-response 研究并证明某工作点在 1e-6 base rate 下 FPR<1e-5,则可重新评估 per-request 语义检测;(b) 若 batch-invariant kernel 开销降到 <20%,抽样比例可上调甚至全流量启用;(c) 若单请求 base rate 经实测高于 1e-4(如故障器件富集的特定 fleet 分区),PPV 计算重算后 inline 统计检测的性价比上升。

## Caveats
- **核心文献空白:** 无任何工作直接测量 hallucination/UQ 检测器对硬件故障幅度的 dose-response,本报告对语义检测器"原理上不敏感"的判断基于机制推理与两社区文献的不相交(跨 5 检索角度的负结果),非直接反证;不能穷尽性证明其不存在。
- 多数瞬态故障 masking 数据来自 CNN/ViT;LLM instruction-level 数据(arXiv:2601.19912、2604.00726、2604.10390、2605.04213 等)为 2025–2026 近期成果,样本、模型规模与 GPU 代际有限,且多为训练场景(推理专项更少)。
- PPV 计算依赖假设的工作点(TPR/FPR)与假设的 per-request base rate(1e-5~1e-8);真实单请求粒度 base rate 缺乏公开测量,现有 1/1000 device、1.4% GPU 中断等均为 device/job 粒度。
- batch-invariant kernel 开销数字随实现快速变化(TML 61.5% → SGLang 34.35% → 目标 <20%);抽样启用的可摊销性未经生产验证。EigenAI(arXiv:2602.00182)宣称 1–5% 确定性开销为厂商自报,与独立复现数字冲突,需谨慎。
- arXiv:2604.15409 的 105×/35× 层放大与 100% token flip 结论基于 7B 级 dense/GQA 模型;数百 B MoE 的传播动力学(专家路由、更深层数)可能不同,外推需谨慎。
# 基于模型内部信号的 Hallucination / Quality-Degradation / OOD Detection 方法综述

## TL;DR
- 面向 LLM 推理可靠性,基于模型内部信号(hidden states、attention、logits/logprobs、派生谱统计量、SAE 特征、梯度)的故障检测已形成三条相互交叉的技术线:hallucination detection、OOD detection、quality-degradation/SDC detection;其中最工程可用的是**单次前向、gray-box 的概率/谱信号**(SEP、EigenScore、LLM-Check、Lookback Lens),而 supervised probe(SAPLMA、Geometry of Truth)虽在同分布内 AUROC 高,但**跨任务/跨域泛化已被多篇工作证伪**。
- 三条线共享同一套数学工具箱(Mahalanobis 距离、energy score、协方差 eigen-spectrum、密度估计、likelihood ratio),但**参考分布的构造方式**(全局边缘阈值 vs 分层条件化基线 vs 学习密度 vs 请求内自参考)才是决定成败的关键轴;经典分类器时代的 OOD 方法凡依赖分类头/类别标签结构者(MSP、ODIN、ViM、class-conditional Mahalanobis)难以直接迁移到自回归生成,仅依赖特征空间者(RMD、kNN、energy、SAE 重构误差)可迁移。
- 结构性盲区在于**输出层简并**(不同病因——知识缺失、上下文不忠、量化退化、硬件 SDC——在 hidden/logit 上产生高度相似签名),以及**marginal 统计被 domain 方差主导**;因此可靠的生产监测应组合"gray-box 概率信号 + 特征空间条件化基线 + 数值一致性校验(batch-invariant kernel)",而非依赖单一探针。
- （2026-08-03 补注,配合项目目的转向监控）**输出层简并对「诊断」是坏消息,对「监控」是好消息**:不同病因在 hidden/logit 上签名高度相似,意味着难以从内部信号**区分**病因;但同一个检测器因此能以一套信号**覆盖**多种病因的「有没有问题」判定——生产监控恰好只需要后者,病因归因交给运维侧证据(版本、配置、硬件事件)完成。

## Key Findings

1. **概率/不确定性 gray-box 信号是最实用的一层**:semantic entropy(Farquhar, Kossen, Kuhn & Gal, *Nature* 630:625–630, 2024,doi:10.1038/s41586-024-07421-0)"entropy-based uncertainty estimators for LLMs to detect a subset of hallucinations—confabulations—which are arbitrary and incorrect generations",即通过对语义等价簇取熵检测 confabulation,但需 5–10× 采样开销;SEP(Kossen et al., arXiv:2406.15927)用线性探针从单次前向的 hidden state 逼近 SE,将开销降到近零,证明 hidden states 隐式编码语义熵。P(True)(Kadavath et al., arXiv:2207.05221)显示大模型对自身答案有一定校准的自评估能力,但 RLHF 后校准显著变差(需温度 T≈2.5 修正)。

2. **无监督内部信号方法**用响应自一致性或激活谱作为打分:INSIDE/EigenScore(Chen et al., ICLR 2024, arXiv:2402.03744)取多次采样 hidden state 协方差矩阵特征值的 log-determinant; [AI Models](https://www.aimodels.fyi/papers/arxiv/inside-llms-internal-states-retain-power-hallucination)  [alphaXiv](https://www.alphaxiv.org/hi/overview/2402.03744v2) LLM-Check(Sriramanan et al., NeurIPS 2024)进一步做到**单次响应、单次前向**,用 attention kernel map 与 hidden state 的 eigenvalue 分析 + output token 不确定性; [GitHub](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection) HaloScope(Du et al., NeurIPS 2024 Spotlight, arXiv:2409.17504)用未标注生成的激活子空间 SVD 做 membership estimation,再训练二分类器,论文报告其在 TruthfulQA 上无监督 AUROC 达 78.64%,"favorably matches the performance of the supervised oracle (AUROC: 81.04%)",并"exhibiting 16.47% and 26.71% improvement over Semantic Entropy and EigenScore on the challenging TruthfulQA task"。

3. **Supervised probe 泛化失败是这条线的核心裂缝**:SAPLMA(Azaria & Mitchell, EMNLP Findings 2023)在跨 topic 时保持 60–80% 准确率, [arXiv](https://arxiv.org/html/2304.13734) 但后续复现显示其**无法泛化到未见的句式结构**;Levinstein & Herrmann(*Philosophical Studies* 182(7):1539–1565, 2025, arXiv:2307.00175)实证 SAPLMA 与 CCS "以非常基本的方式无法泛化";"LLMs Know More Than They Show"(Orgad et al., ICLR 2025, arXiv:2410.02707)证明 truthfulness 信息集中在 **exact answer tokens**、且 **encoding 不是普适而是 multifaceted/skill-specific**,探针跨数据集失效。 [arXiv](https://arxiv.org/abs/2410.02707v1) Geometry of Truth(Marks & Tegmark, COLM 2024, arXiv:2310.06824)则给出反向证据:足够规模下真/假有线性结构、difference-in-mean 探针可跨简单数据集迁移且有因果作用—— [arxiv](https://arxiv.org/pdf/2310.06824) 两方结论的差异取决于数据集难度与是否长文本生成。

4. **经典 OOD 谱系构成迁移的地基**:MSP(Hendrycks & Gimpel 2016)、ODIN、class-conditional Mahalanobis(Lee et al. 2018)、energy score(Liu et al. 2020)、ViM(Wang et al. CVPR 2022)、kNN-OOD(Sun et al. 2022)、activation shaping(ReAct/ASH/DICE)、GradNorm。RMD(Ren et al., arXiv:2106.09022)用全局高斯归一化 class-conditional 距离,专治 near-OOD,在 CLINC intent detection、Genomics OOD 等任务上验证(genomics 上 +15% AUROC),是**最容易迁移到 LLM 特征空间的方法**。

5. **生成式 likelihood 的 OOD 悖论**:Nalisnick et al.(ICLR 2019, arXiv:1810.09136)发现深度生成模型对 OOD 反而赋予更高 likelihood(CIFAR-10 训练的 Glow 给 SVHN 更高 likelihood), [arxiv](https://arxiv.org/pdf/1810.09136) 因此**直接用 LLM 序列 likelihood/perplexity 做 OOD 是不可靠的**;修正手段是 likelihood ratio(Ren et al. 2019,扣除 background 统计)、input complexity 校正(Serrà et al. 2020)与 typicality test(Nalisnick et al. 2019)。

6. **Quality-degradation / silent degradation 线文献分散但可拼接**:量化退化在内部信号上表现为 activation outliers 撑大动态范围、低比特量化引起 activation 分布向均值塌缩(可用量化前后 histogram 的 KL divergence 量化);repetition/degeneration 有注意力与 logit 分布签名;硬件 SDC(Meta arXiv:2102.11245:"we have run a vast library of silent error test scenarios across hundreds of thousands of machines in our fleet. This has resulted in hundreds of CPUs detected for these errors… We have monitored SDCs for a period longer than 18 months";Google "Cores that don't count", HotOS 2021)可用 range-based/activation-distribution 分位数漂移检测(Dr. DNA, ASPLOS 2024;arXiv:2310.20349),以及 ABFT 类 checksum(ALBERTA, arXiv:2310.03841:"our approach achieves over 99% coverage for errors that result in a mismatch with less than 0.2% and 0.01% computation and memory overheads, respectively")。推理数值不一致的根因被 Thinking Machines Lab(He et al., 2025)归结为 kernel 缺乏 batch-invariance,而非单纯浮点非结合性。 [NextBigFuture](https://www.nextbigfuture.com/2025/11/defeating-nondeterminism-in-llm-inference-by-thinking-machines.html)

## Details

### 一、Taxonomy(分类框架)

建议沿四个正交轴组织所有方法:

**轴 A — 信号来源**
- **Hidden states / residual stream**:SAPLMA、Geometry of Truth、SEP、MIND、HaloScope、INSIDE(embedding 协方差)、Mahalanobis/RMD/kNN(特征空间)、SAE 重构误差。
- **Attention weights/patterns**:Lookback Lens(lookback ratio 线性探针)、LLM-Check(attention kernel map 谱)、LapEigvals(attention Laplacian 谱特征)、ITI(定位 truthful attention heads)。
- **Logits / token-level logprobs / entropy**:predictive entropy、length-normalized logprob、P(True)、semantic entropy、SelfCheckGPT-Logprob、SAR、energy score、MSP/ODIN/ViM。
- **派生统计量**:eigen-spectrum(EigenScore、LLM-Check)、**intrinsic dimension / LID**(TwoNN、MLE/GeoMLE 等估计器;可用于逐样本打分或逐层 ID 曲线做自动选层,见 §二 (2b))、Mahalanobis/RMD、energy、density/likelihood-ratio、GradNorm、topological features(HalluZig 的 zigzag persistence)。

**轴 B — 监督程度**
- **Supervised probe**(需标注 true/false 或 correct/incorrect):SAPLMA、Geometry of Truth、ITI、Lookback Lens、SEP(用 SE 作弱监督标签)。
- **Unsupervised / zero-resource**:INSIDE/EigenScore、MIND(用 Wikipedia 自动构造伪标签)、HaloScope(未标注混合数据 membership estimation)、SelfCheckGPT(采样一致性)、semantic entropy、LID 打分(Yin et al.;需邻域表示集而非 K 次回答采样)、FEPoID 选层(TwoNN-ID 曲线,training-free)。
- **Calibration-only**(仅需 ID 校准集估计均值/协方差/阈值):Mahalanobis、RMD、energy、kNN、density。(注:此处 "ID" 指 in-distribution 校准集,与 intrinsic dimension 缩写无关。)

**轴 C — 参考分布构造方式**(本综述认为这是最有解释力的轴)
- **全局边缘分布阈值**:predictive entropy、MSP、perplexity 阈值、EigenScore 阈值——**易被 domain 方差主导**。
- **分层/条件化基线**:RMD(全局高斯归一化)、class-conditional Mahalanobis、Dr. DNA(逐类激活分布)、activation 分位数漂移监测。
- **学习到的密度模型**:HaloScope 子空间、SAE 重构、likelihood-ratio、ViM 残差子空间。
- **请求内自参考/一致性**:SelfCheckGPT、INSIDE、semantic entropy、SAR、Lookback Lens(context vs generation 注意力比)。

**轴 D — 访问要求 × 采样次数**
- **white-box 单次前向**:SAPLMA、SEP、MIND、Lookback Lens、LLM-Check、Geometry of Truth、ITI、Mahalanobis/RMD/energy/kNN(需特征)、LID/FEPoID(需 hidden;FEPoID 另需跨层一批表示估 TwoNN-ID)。
- **gray-box 单次前向**:P(True)、predictive entropy、length-normalized logprob、SelfCheckGPT-Logprob。
- **需多次采样**:semantic entropy、INSIDE、SelfCheckGPT(N=20)、SAR、任何 consistency-based 方法。

### 二、Hallucination detection via model internals

**(1) Supervised probing 线**

- **SAPLMA**(Azaria & Mitchell, EMNLP Findings 2023, arXiv:2304.13734):信号为特定 hidden layer 激活(OPT-6.7B 用第 20 层, [arXiv](https://arxiv.org/pdf/2304.13734) 共 32 层, [ResearchGate](https://www.researchgate.net/publication/370338601_The_Internal_State_of_an_LLM_Knows_When_its_Lying) 中后层最佳);在 hidden state 上训练前馈分类器输出"陈述为真"的概率。 [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.68/) 需标注的 true/false 数据集(6 个 topic),在 held-out topic 上评测。 [ResearchGate](https://www.researchgate.net/publication/370338601_The_Internal_State_of_an_LLM_Knows_When_its_Lying) 准确率 60–80%,优于 few-shot prompting(≤56%)。 [arXiv](https://arxiv.org/html/2304.13734) 失效模式:后续复现(Stanford CS224N;sisinflab 复现)显示对**未见句式结构不泛化**、对训练数据过拟合(MIND 论文指出 SAPLMA 在实时生成场景下 suboptimal)。white-box、单次前向、低开销。

- **Geometry of Truth**(Marks & Tegmark, COLM 2024, arXiv:2310.06824):在简单 true/false 陈述上可视化 hidden states 呈现清晰线性结构;提出 **mass-mean probing**(difference-in-mean), [arxiv](https://arxiv.org/pdf/2310.06824) 在 LLaMA-13B 第 13 层的 end-of-clause token 位置提取激活; [arXiv](https://arxiv.org/html/2310.06824v2) transfer 实验显示跨数据集泛化、且因果干预可翻转模型对真假的处理。 [arxiv](https://arxiv.org/pdf/2310.06824) 这是对 probe 怀疑论的部分反驳,但仅在**简单、无歧义陈述**上成立。

- **Representation Engineering / RepE**(Zou et al., arXiv:2310.01405)与 **ITI**(Li et al., NeurIPS 2023, arXiv:2306.03341):ITI 用线性探针定位一组 truthfulness 相关 attention heads,推理时沿 truth-correlated 方向按标准差平移这些 head 的激活; [arXiv](https://arxiv.org/html/2306.03341v6) 原文报告 "On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from 32.5% to 65.1%… ITI locates truthful directions using only few hundred examples"。ITI 主要是干预/缓解方法,但其"定位 truthful heads"的探针即是检测信号。数据高效(几百样本)。 [arXiv](https://arxiv.org/abs/2306.03341)

- **Lookback Lens**(Chuang et al., arXiv:2407.07071):信号为 **lookback ratio**——每个 attention head 上下文 token 注意力权重与新生成 token 注意力权重之比。 [arXiv](https://arxiv.org/abs/2407.07071) 对每层每 head 计算该比值,拼接为特征训练线性分类器 [arXiv](https://arxiv.org/pdf/2407.07071) 检测**上下文幻觉**(summary/QA 与输入不符)。关键结论:仅用 attention ratio 的线性分类器**与用完整 hidden state 或 NLI 模型的检测器同样有效**, [arXiv](https://arxiv.org/abs/2407.07071) 且可跨任务、映射 head 后可跨模型迁移; [Medium](https://medium.com/@techsachin/lookback-lens-detect-and-mitigate-hallucinations-in-llms-with-attention-maps-8241a06d3f2c) 可引导解码缓解(XSum 上降 9.6%)。 [arxiv](https://arxiv.org/pdf/2511.14776) white-box、需 attention map、单次前向。

- **Semantic Entropy Probes (SEP)**(Kossen et al., arXiv:2406.15927):用线性探针从**单次生成的 hidden state** 直接预测(二值化的)semantic entropy, [alphaXiv](https://www.alphaxiv.org/abs/2406.15927v1) 作为弱监督标签而非 accuracy 标签(作者论证 SE 标签比 accuracy 标签更 robust)。 [arXiv](https://arxiv.org/pdf/2406.15927) 中后层、second-last token 效果最佳,AUROC 0.7–0.95;甚至能在生成前(last input token)预测 SE。 [arxiv](https://arxiv.org/pdf/2406.15927) 比直接预测 accuracy 的 probe 泛化更好。 [arXiv](https://arxiv.org/abs/2406.15927) white-box、单次前向、近零额外开销。

**(2) Unsupervised / zero-resource 内部信号线**

- **INSIDE / EigenScore**(Chen et al., ICLR 2024, arXiv:2402.03744):对同一问题采样 K 个响应,取各响应 hidden state(penultimate layer)的句向量, [GitHub](https://github.com/0xkerem/halluscope) 构造协方差矩阵,**EigenScore = 特征值的 log-determinant**, [GitHub](https://github.com/0xkerem/halluscope) 度量 dense embedding 空间中的语义一致性/多样性( [alphaXiv](https://www.alphaxiv.org/hi/overview/2402.03744v2) 高多样性→幻觉)。另加 **test-time feature clipping** 截断极端激活 [alphaXiv](https://www.alphaxiv.org/hi/overview/2402.03744v2) 以抑制过自信幻觉。white-box、需多次采样。局限:无法用于 black-box 模型; [arXiv](https://arxiv.org/pdf/2402.03744) 评测框架与其他方法略不同(多响应而非单响应)。 [OpenReview](https://openreview.net/pdf?id=LYx4w3CAgy)

- **MIND**(Su et al., ACL Findings 2024, arXiv:2403.06448):无监督框架,从 Wikipedia 截断文章 + 续写正确性自动构造伪训练数据(无需人工标注);在 LLM 推理过程中用 MLP 对每个 token 的 contextual embedding(最后一层最后 token)做**实时**幻觉检测。 [Liner](https://liner.com/review/unsupervised-realtime-hallucination-detection-based-on-internal-states-large-language) 附带 HELM benchmark(6 个 LLM 的输出 + 内部状态)。 [arXiv](https://arxiv.org/html/2403.06448v2) 相比 SelfCheckGPT 延迟低 10×。 [arxiv](https://arxiv.org/pdf/2403.06448) white-box、单次前向、可流式。

- **HaloScope**(Du et al., NeurIPS 2024 Spotlight, arXiv:2409.17504):利用部署中自然产生的**未标注混合生成**(真+假), [alphaXiv](https://www.alphaxiv.org/abs/2409.17504) 对 LLM latent space 做 SVD 得到 membership estimation score(投影到"truthfulness 子空间"),据此自动分离真/假,再训练二分类器。 [arXiv](https://arxiv.org/abs/2409.17504)  [Emergent Mind](https://www.emergentmind.com/papers/2409.17504) 无需人工标注。TruthfulQA 上无监督 AUROC 78.64%,逼近 supervised oracle(81.04%),较 Semantic Entropy / EigenScore 分别提升 16.47% / 26.71%。white-box、需 ID 未标注语料。

- **LLM-Check**(Sriramanan et al., NeurIPS 2024):**单次响应、单次前向**,两条打分线——(a) 内部表示的 Eigenvalue Analysis(hidden state 协方差 + self-attention kernel similarity map 谱,发现 kernel map 对真/假 token 敏感);( [GitHub](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection) b) output token uncertainty(仅需输出概率分布的 gray-box 变体)。 [OpenReview](https://openreview.net/pdf?id=LYx4w3CAgy) 覆盖 zero-resource(FAVA)、多响应(SelfCheck)、有参考(RAGTruth)三种设置。 [OpenReview](https://openreview.net/pdf?id=LYx4w3CAgy) 强调可实时,内存/计算开销远低于 consistency-based 方法。

- **attention 谱方法**:LapEigvals 用 attention map 的 Laplacian 图谱特征训练分类器,TriviaQA 上 AUC-ROC 88.9%;HalluZig(arXiv:2601.01552)用 attention dynamics 的 zigzag persistence 拓扑特征。 [arxiv](https://arxiv.org/pdf/2510.06265) 这些是单篇初步结论,尚未被广泛复现。

**(2b) Intrinsic dimension / LID 几何量与自动选层**(工具箱定位,非与 SE/Lookback 同级的主检测器)

本征维(intrinsic dimension, 几何 ID)估计表示流形的有效自由度;与分类头无关,属 hidden 派生统计量。两条互补用法:

- **(a) LID 作为逐样本 truthfulness 分数**——**Characterizing Truthfulness … with Local Intrinsic Dimension**(Yin, Srinivasa & Chang, arXiv:2402.18048;代码 [LID-HallucinationDetection](https://github.com/fanyin3639/LID-HallucinationDetection)):假设正确续写更接近自然语言结构、**LID 偏小**,幻觉续写混杂 prompt 与模型捏造分布、**LID 偏大**。主估计器为适合逐点估计的 **LID-MLE / LID-GeoMLE**(Levina–Bickel 系);文中比较 TwoNN 等全局/集合估计器,真实数据上数量级可接近,但作者强调逐样本场景优先 MLE 系。四个生成式 QA 上相对 entropy / verbalized uncertainty 有提升;并报告层间 ID 呈 hunchback、检测 AUROC 曲线相对 ID 峰**滞后约 1–2 层**。white-box;需要邻域表示集(同分布或文中显示一定跨集邻居仍可用),**不是** K 次回答采样。

- **(b) TwoNN-ID 曲线用于自动选层**——**Automatic Layer Selection for Hallucination Detection / FEPoID**(Wang, Cao, Wilson & Zeng, arXiv:2605.26366, ICML 2026;代码 [Automatic-Layer-Selection-for-Hallucination-Detection](https://github.com/DesoloYw/Automatic-Layer-Selection-for-Hallucination-Detection)):动机是幻觉相关信号多在**中间层**,手选层不稳。对每层 hidden 用 **TwoNN**(Facco et al. 2017;`scikit-dimension` + Faiss)估 ID,取曲线上**第一个有效峰(First Effective Peak of Intrinsic Dimension, FEPoID)** 作为探测层,再做 hidden-state probing;选层 **training-free**、开销低。另提 **First-Sentence Truncation (FST)**:在首句末 token 取表示以减弱 EOS/句末噪声。主贡献是**选层准则**,检测器本体仍是探针(或其它表示分数);文中对比 Val Loss、SNR 等准则,称 FEPoID 更接近 oracle 层。QA 与 summarization、多尺度 LLaMA/Mistral 上验证。

- **定位与边界**:二者补的是「几何可观测量 + 如何选层」,**不应**替代阶段一 gray-box 概率信号,也**不能**消解 supervised probe 跨任务不泛化问题——FEPoID 选对层 ≠ 探针跨域一定成立。实现时注意:**不要**默认「上 TwoNN = 复现 Yin」——Yin 主用 MLE-LID 打分,FEPoID 主用 TwoNN 做层选择。成熟度:Yin 2024 有可运行代码与多数据集表;FEPoID 为 2026 新工作(已标 ICML 2026),跨栈复现仍有限,见 Caveats。

**(3) 概率/不确定性线(gray-box)**

- **Semantic Entropy**(Kuhn et al., ICLR 2023;Farquhar et al., *Nature* 630:625–630, 2024):对多个采样生成按语义等价(NLI 聚类)分簇,再对簇分布取熵。 [arxiv](https://arxiv.org/pdf/2606.08098) 检测 confabulation(任意且错误的生成)无需 ground truth。 [arxiv](https://arxiv.org/pdf/2601.22290) 5–10× 采样开销是其主要缺点。 [arXiv](https://arxiv.org/abs/2406.15927v1) gray-box + NLI 模型、需多次采样。

- **P(True) / P(IK)**(Kadavath et al., arXiv:2207.05221):模型先提出答案再自评"该答案为真"的概率 P(True); [arXiv](https://arxiv.org/pdf/2207.05221) P(IK) 预测"我知道答案"。大模型在正确格式下对多选/判断题校准良好, [arXiv](https://arxiv.org/abs/2207.05221) 自评能区分对错; [arXiv](https://arxiv.org/pdf/2207.05221) 但 RLHF policy 校准差(T≈2.5 可修正), [SciSpace](https://scispace.com/papers/language-models-mostly-know-what-they-know-zkg0pou4) P(IK) 跨新任务校准仍难。 [BibBase](https://bibbase.org/network/publication/kadavath-conerly-askell-henighan-drain-perez-schiefer-hatfielddodds-etal-languagemodelsmostlyknowwhattheyknow-2022) gray-box、单次(或少次)。注意:P(True) 是用模型概率读数,不属于被排除的 LLM-as-judge 自然语言评判。

- **SelfCheckGPT 非 prompt 变体**(Manakul et al., EMNLP 2023, arXiv:2303.08896):核心假设——若模型掌握知识,采样响应相互一致;幻觉则发散矛盾。 [ACL Anthology](https://aclanthology.org/2023.emnlp-main.557/) 需 N=20 随机采样。非 LLM-judge 变体:**SelfCheck-Logprob**(gray-box,句子 token 的 Avg/Max 负 log-probability;Max(−logp) NonFact AUC-PR 87.51 优于 Avg 83.21)、** [GitHub](https://github.com/potsawee/selfcheckgpt/blob/main/README.md) NLI 变体**(DeBERTa-v3 MNLI 判 contradiction,NonFact AUC-PR 92.50,是非 prompt 变体中最强)、** [GitHub](https://github.com/potsawee/selfcheckgpt/blob/main/README.md) n-gram/unigram 变体**(85.63)。WikiBio GPT-3 数据集、AUC-PR 指标。(其 LLM-prompt 变体 ~93.4 属被排除范围。)

- **SAR (Shifting Attention to Relevance)**(Duan et al., ACL 2024, arXiv:2307.01379):针对"generative inequality"——语义稀薄的 token/句子被等权甚至重权计入不确定性。 [arXiv](https://arxiv.org/html/2307.01379v1) SAR 用"删除某 token 对句义的改变量"衡量 token relevance,在 token 与 sentence 两级用 relevance 对负 log-prob 重加权( [OpenReview](https://openreview.net/forum?id=yZJapMWdHZ) 建立在 semantic entropy 上)。在 CoQA/TriviaQA/SciQ 及 Vicuna/LLaMA-2 等上以较大 margin 提升 AUROC;在 LM-Polygraph 的 28 种 UQ 方法中排第一。 [GitHub](https://github.com/jinhaoduan/SAR) gray-box、需多次采样。

- **"LLMs Know More Than They Show"**(Orgad et al., ICLR 2025, arXiv:2410.02707):在中间层 hidden state 上做探针,发现 truthfulness 信息集中在 **exact answer tokens**(定义:修改它会改变答案对错的 token), [arXiv](https://arxiv.org/html/2410.02707) 在该位置探针 AUC 显著提升;但**探针跨数据集不泛化**,说明 truthfulness encoding 是 multifaceted/skill-specific 而非普适;并揭示"内部编码正确答案但外部生成错误答案"的行为—内部脱节现象。white-box、单次前向。

**(4) truthfulness 方向普适性争论**:Levinstein & Herrmann(*Philosophical Studies* 2025, arXiv:2307.00175)实证 SAPLMA 与 Burns et al. CCS "以非常基本的方式无法泛化",并论证即便 LLM 有 belief,这些方法出于概念原因也难成功。 [arXiv](https://arxiv.org/abs/2307.00175) Orgad et al. 的 skill-specific 结论与之一致;Geometry of Truth 则在简单陈述上给出普适线性结构的反证。当前共识:**存在可线性读出的 truth-correlated 方向,但它是任务/领域条件化的,不存在单一普适 truthfulness 方向**。

### 三、OOD detection(经典谱系 → 自回归 LLM 迁移)

**经典分类器时代**
- **MSP**(Hendrycks & Gimpel, ICLR 2017):max softmax probability 作为 ID 置信,baseline。依赖分类头。
- **ODIN**:temperature scaling + input perturbation 放大 ID/OOD softmax 差距。依赖分类头。
- **class-conditional Mahalanobis**(Lee et al., NeurIPS 2018):对每类拟合共享协方差的高斯, [arxiv](https://arxiv.org/pdf/2505.18032) 打分 S(x)=max_c −(ψ(x)−μ_c)ᵀΣ⁻¹(ψ(x)−μ_c)。 [arxiv](https://arxiv.org/pdf/2210.07485) 需类别标签估计 μ_c;可用 penultimate feature。
- **RMD**(Ren et al., arXiv:2106.09022):在 class-conditional 距离基础上减去一个全局(class-agnostic)高斯的 Mahalanobis 距离, [ResearchGate](https://www.researchgate.net/publication/392086015_Mahalanobis_Improving_OOD_Detection_via_Feature_Normalization) 归一化背景效应,专治 near-OOD;CIFAR-100 vs CIFAR-10、CLINC intent、Genomics OOD 上比 MD 提升(genomics 上 +15% AUROC)。** [arXiv](https://arxiv.org/abs/2106.09022) 只需特征空间,最易迁移**。Mahalanobis++(Mueller & Hein 2025)进一步用 L2 归一化使特征更符合高斯假设。 [arXiv](https://arxiv.org/pdf/2510.15202)
- **energy score**(Liu et al., NeurIPS 2020):E(x)=−T·logΣexp(logit_k/T),理论上比 softmax 更贴合密度;依赖 logits。
- **ViM**(Wang et al., CVPR 2022):从特征对主子空间的**残差**生成 virtual logit,与真实 logits 匹配缩放后取 softmax; [arXiv](https://www.arxiv.org/pdf/2511.11934) 融合 feature-space class-agnostic 分数与 logit-space class-dependent 分数。 [Academia.edu](https://www.academia.edu/101081107/ViM_Out_Of_Distribution_with_Virtual_logit_Matching) 依赖 logits + 特征。
- **kNN-OOD**(Sun et al., ICML 2022):第 k 近邻距离作打分,无分布假设,** [arxiv](https://arxiv.org/pdf/2507.00570) 只需特征空间,可迁移**。
- **activation shaping**:ReAct(截断极端激活)、ASH、DICE(权重剪枝);**GradNorm**(用输出对 softmax 的梯度范数);gram matrix 方法。多数需 logits/分类结构。

**文本/Transformer 域**
- Podolskiy et al.(AAAI 2021):在 Transformer(BERT)embedding 上用 Mahalanobis 距离,优于 softmax/likelihood-ratio,奠定"PLM 特征空间 + 距离度量"范式。
- Arora et al.(EMNLP 2021):区分 **background shift** vs **semantic shift**,并对比 density-based(PPL/likelihood-ratio)与 calibration-based(MSP)方法,指出没有单一方法对两类 shift 都最优——直接影响 LLM OOD 的诊断。
- 基于 PLM embedding 密度/likelihood ratio、contrastive 表征 + OOD(如 holistic sentence embedding, arXiv:2210.07485)。

**LLM 时代**
- 基于 LLM embedding 的距离/密度 OOD(RMD、kNN 直接搬到 hidden state);"From Out-of-Distribution Detection to Hallucination Detection: A Geometric View"(arXiv:2602.07253)明确把 OOD 几何工具迁移到幻觉检测。
- **SAE 重构误差用于 OOD/anomaly**:SAE 在 ID 激活上训练,OOD 输入重构误差升高(RE=Σ(x̂−x)²); [arxiv](https://arxiv.org/pdf/1912.11831) vision 领域 Top-k SAE 对 ViT [CLS] token 做 OOD(arXiv:2604.26409,提出 Class Activation Profiles + Energy Profile Divergence)。** [arXiv](https://arxiv.org/html/2604.26409) 但 SAE 已知 off-distribution 不稳健**(Sam Marks 等观察),这对其 OOD 用途是根本性限制。
- **生成式 likelihood 悖论**:Nalisnick et al.(arXiv:1810.09136)——likelihood 对 OOD 反而更高; [arxiv](https://arxiv.org/pdf/1906.02994) 修正:likelihood ratio(Ren et al. 2019)、input complexity(Serrà et al. 2020)、 [arxiv](https://arxiv.org/pdf/2108.09976) typicality test(Nalisnick et al. 2019)。**直接告诫:LLM perplexity 阈值做 OOD 不可靠**。

**迁移性标注**:依赖分类头/类别标签结构者(MSP、ODIN、class-conditional Mahalanobis、ViM、energy on class logits、GradNorm、DICE)**难以直接迁移**到自回归生成(无固定类别集);仅需特征空间者(RMD 的 global 部分、kNN、SAE 重构、density)**可迁移**;LLM 的 vocab logits 虽是"类别"但为 token 级、开放序列,energy/entropy 可在 token 级用但语义不等价于样本级 OOD。

### 四、Quality degradation / silent degradation detection

**(1) perplexity/logprob 分布漂移监控**:基于生成的 perplexity 或 token logprob 分布随时间漂移做质量监控;但受 Nalisnick 悖论与 domain 方差影响,marginal PPL 阈值易误报,需条件化基线。

**(2) 量化/压缩退化的内部签名**:activation outliers(systematic channel-wise outliers 出现在少数通道/层,massive outliers 是 token-specific [arxiv](https://arxiv.org/pdf/2506.01967) 且幅度可超中位数上万倍,主要在 down-projection [arxiv](https://arxiv.org/pdf/2506.01967) 与 inter-block hidden states)撑大动态范围,低比特均匀量化时少数大值决定量化范围,迫使多数正常值向均值塌缩。 [arXiv](https://arxiv.org/html/2605.26175) 可用**量化前后 activation histogram 的 KL divergence**量化敏感度(InfoQuant, arXiv:2605.26175;Turning LLM Activations Quantization-Friendly, arXiv:2506.01967),或用输出分布的 KL(p_θ‖p_θ̂)作为量化质量的直接度量。这提供了"用内部信号预测量化敏感层"的路径。

**(3) repetition / degeneration 的内部签名**:神经文本退化(重复、空洞)伴随注意力塌缩与 logit 分布特征;nucleus sampling(Holtzman et al. 2020)即针对退化的输出分布问题。此子线在内部信号层面文献较散,多为初步观察。

**(4) 生产 serving 的非 judge 漂移监控**:embedding drift(对输出/隐层 embedding 统计做 population stability、KS、MMD 等距离)与输出分布统计监控属工程实践,缺乏单一权威 primary 论文;其模型内部信号的先例是激活分布分位数漂移监测与 Dr. DNA 式逐类激活签名。

**(5) 硬件故障 / SDC 对内部状态的影响与检测**:
- **现象**:Meta "Silent Data Corruptions at Scale"(Dixit et al., arXiv:2102.11245)——SDC 不被 CPU 错误上报机制捕获,跨栈传播为应用级问题;原文"we have run a vast library of silent error test scenarios across hundreds of thousands of machines in our fleet. This has resulted in hundreds of CPUs detected for these errors… We have monitored SDCs for a period longer than 18 months",约 1/1000 机器受影响、data-dependent、afflict 特定 core。Google "Cores that don't count"(Hochschild et al., HotOS 2021)提出 CEE(Corrupt Execution Errors),检出率远超工程师预期、可在安装后很久才显现、通常只影响特定 core。 [Sigops](https://sigops.org/s/conferences/hotos/2021/papers/hotos21-s01-hochschild.pdf)
- **DNN/LLM 影响**:LLM 对随机 bit flip 的韧性低于假设,且**规模增大不提升韧性**;模型大小与微调策略影响 SDC 脆弱性。 [arXiv](https://arxiv.org/pdf/2604.00726)
- **检测**:range-based(activation 值域边界检查,10% 检测 margin)、activation-distribution 分位数漂移(arXiv:2310.20349,监测少至两层各一个分位标记即达 ~96% precision / ~98% recall,开销 0.3–1.6%)、 [arxiv](https://arxiv.org/pdf/2310.20349) Dr. DNA(ASPLOS 2024,用 Distribution of Neuron Activations 签名,平均 95%、多数 100% SDC 检出,<1% 内存 / <2.5% 延迟开销); [FB](https://engineering.fb.com/2024/06/19/data-infrastructure/parameter-vulnerability-factor-pvf-ai-silent-data-corruption/) ABFT/checksum(ALBERTA, arXiv:2310.03841,原文"our approach achieves over 99% coverage for errors that result in a mismatch with less than 0.2% and 0.01% computation and memory overheads, respectively",含 <2% 自纠正开销;FT-Transformer, arXiv:2504.02211,端到端容错 attention,检测开销 ~4.7%)。 [arXiv](https://arxiv.org/pdf/2504.02211)

**(6) 推理数值不一致 / non-determinism**:Thinking Machines Lab "Defeating Nondeterminism in LLM Inference"(He et al., 2025, DOI 10.64434/tml.20250910)——LLM 推理非确定性的**主因是 kernel 缺乏 batch-invariance**(输出依赖 forward pass 的 batch size,而 batch size 随负载非确定变化), [Thinkingmachines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 而非单纯浮点非结合性(前向 kernel 无 atomic add、run-to-run 确定)。 [thinkingmachines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 需让 RMSNorm、matmul、attention 三类含 reduction 的算子对 batch size 固定 reduction 顺序。实测 Qwen3-235B(prompt "Tell me about Richard Feynman")在 temp 0 下采样 1000 次生成产生 80 种不同结果、最常见的出现 78 次,1000 次共享前 102 token、在第 103 token 分叉("Queens, New York" vs "New York City");启用 batch-invariant kernel 后 1000 次完全一致。 [thinkingmachines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) 这对"质量退化是模型问题还是数值/系统问题"的归因至关重要——是所有 marginal 输出统计监控的隐藏混淆源。

## 跨方法对比表

| 方法 (出处) | 信号来源 | 监督 | 访问要求 | 多次采样? | 开销/可流式 | 代表性能 | 主要局限 |
|---|---|---|---|---|---|---|---|
| SAPLMA (Azaria&Mitchell, EMNLP-F 2023) | hidden state(中后层) | supervised | white-box | 否 | 极低/可 | topic 内 60–80% acc | 跨句式/跨域不泛化、过拟合 |
| Geometry of Truth (Marks&Tegmark, COLM 2024) | hidden state(end-of-clause) | supervised (diff-in-mean) | white-box | 否 | 极低/可 | 简单陈述跨集可迁移+因果 | 仅简单无歧义陈述 |
| ITI (Li et al., NeurIPS 2023) | attention head 激活 | supervised probe | white-box | 否 | 低/可(兼干预) | Alpaca TruthfulQA 32.5→65.1% [arXiv](https://arxiv.org/abs/2306.03341) | 主为干预;方向任务相关 |
| Lookback Lens (Chuang et al., 2024) | attention ratio | supervised(线性) | white-box(attn) | 否 | 低/可 | 与 hidden/NLI 检测器相当;可跨模型 | 仅上下文幻觉 |
| SEP (Kossen et al., 2024) | hidden state(中后层) | 弱监督(SE 标签) | white-box | 否 | 近零/可 | AUROC 0.7–0.95 | 依赖 SE 标签质量 |
| INSIDE/EigenScore (Chen et al., ICLR 2024) | hidden state 协方差谱 | unsupervised | white-box | 是(K 采样) | 中/难 | QA 上超 logit/consistency 基线 | 不适 black-box;多响应 |
| MIND (Su et al., ACL-F 2024) | hidden state(末层末token) | unsup(伪标签) | white-box | 否 | 低/可(实时) | 超 SOTA reference-free [ACL Anthology](https://aclanthology.org/2024.findings-acl.854/) | 需自建伪数据 |
| HaloScope (Du et al., NeurIPS 2024) | 激活子空间 SVD | unsup(未标注) | white-box | 否 | 中/可 | TruthfulQA AUROC 78.64%(≈oracle 81.04) | 需 ID 未标注语料 |
| LLM-Check (Sriramanan et al., NeurIPS 2024) | hidden+attn 谱, logits | unsup/calib | white/gray | 否(单响应) | 低/可 | 多设置超基线 | 谱信号对模型/层敏感 |
| Semantic Entropy (Farquhar et al., Nature 2024) | 采样+NLI 聚类熵 | unsup | gray-box+NLI | 是(5–10×) | 高/难 | confabulation 检测 SOTA | 采样开销大 |
| SelfCheck-NLI (Manakul et al., 2023) | 采样一致性(NLI) | unsup | black-box | 是(N=20) | 高/难 | NonFact AUC-PR 92.50 | 延迟 10×生成 |
| SelfCheck-Logprob (同上) | token logprob | unsup | gray-box | 是 | 中/难 | Max(−logp) AUC-PR 87.51 | 逊于 NLI/prompt |
| P(True) (Kadavath et al., 2022) | 自评概率读数 | calib | gray-box | 否/少 | 低/可 | 大模型校准良好 | RLHF 后校准差 |
| SAR (Duan et al., ACL 2024) | relevance 重加权 logprob | unsup | gray-box | 是 | 中/难 | LM-Polygraph 28法第1 | 需采样+相似度模型 |
| LLMs Know More (Orgad et al., ICLR 2025) | hidden(exact answer token) | supervised | white-box | 否 | 低/可 | exact-token 处 AUC 峰值 | 跨数据集不泛化 |
| LID-Truthfulness (Yin et al., arXiv:2402.18048) | hidden 的局部本征维(LID-MLE/GeoMLE) | unsup(需邻域集) | white-box | 否* | 中/可 | 四 QA 上优于 entropy/verbalized UQ | 邻域与 k 敏感;非 TwoNN 主路径 |
| FEPoID (Wang et al., ICML 2026 / arXiv:2605.26366) | TwoNN 层 ID 曲线→选层(+可选 FST) | 选层无监督;探针可要标 | white-box | 否* | 选层近零/可 | 自动选近 oracle 中间层;增益来自选层+探针 | 2026 新;主贡献是选层非新分数 |
| Mahalanobis (Lee et al., 2018) | penultimate feature | calib(需类) | feature | 否 | 低/可 | 经典 baseline | 依赖类结构;near-OOD 弱 |
| RMD (Ren et al., 2021) | feature(全局归一) | calib | feature | 否 | 低/可 | genomics +15% AUROC | near-OOD 专用 |
| Energy (Liu et al., 2020) | logits | calib | logits | 否 | 低/可 | 优于 MSP | 依赖类 logits |
| ViM (Wang et al., 2022) | feature 残差+logits | calib | logits+feat | 否 | 低/可 | 融合优势 | 需分类结构 |
| kNN-OOD (Sun et al., 2022) | feature | calib | feature | 否 | 中/可 | 无分布假设强 | 需存 ID 特征库 |
| SAE 重构误差 | SAE 特征/重构 | unsup | white-box | 否 | 中/可 | vision OOD 有效 | off-distribution 不稳健 |
| Likelihood ratio (Ren et al., 2019) | 序列 likelihood | calib | gray-box | 否 | 低/可 | 修正 likelihood 悖论 | 需 background 模型 |
| Dr. DNA (ASPLOS 2024) | 神经元激活分布 | calib | white-box | 否 | <2.5%延迟/可 | 平均 95% SDC 检出 | 需逐类 profile |
| ALBERTA (arXiv:2310.03841) | GEMM checksum | 无(ABFT) | 计算层 | 否 | <0.2%/可 | >99% mismatch 覆盖 | 只查数值错非语义 |
| batch-invariant kernels (He et al., 2025) | 数值一致性 | 无 | 系统层 | 否 | ~1.6×延迟/可 | 1000 次生成完全一致 | 需改 kernel;不查语义 |

## Recommendations

**阶段一(即刻可部署,gray-box 单次前向)**:在生产推理服务上首先接入**不需要 white-box、不需要采样**的信号——length-normalized logprob、predictive entropy、P(True)、SelfCheck-Logprob(若可承受少量采样)。这些开销近零、无需改动 serving 栈。基准:先用 TruthfulQA / TriviaQA / SQuAD 幻觉设置离线标定阈值,AUROC 若 <0.75 说明 marginal 信号被 domain 方差主导,应进入阶段二。

**阶段二(white-box 特征信号 + 条件化基线)**:若可拿到 hidden states,部署 **SEP 或 MIND**(单次前向、可流式)做语义熵/幻觉探针,并**同时**用 **RMD 或 kNN** 在 hidden state 特征空间做 OOD——关键是用**分层/条件化基线**而非全局阈值,以抵消 domain 方差。对 RAG 场景加 **Lookback Lens**(attention ratio)检测上下文不忠。若采用 hidden-state probing 且不愿手选/穷举层,可用 **FEPoID**(逐层 TwoNN-ID,取第一有效峰)作 **training-free 选层**,必要时配合 **FST**(首句末 token);这是选层工具,不替代条件化基线,也不保证探针跨域泛化。可选对照:Yin et al. 的 **LID 分数**(MLE 系)作为与 entropy 正交的几何基线。基准:RAGTruth(约 17,790 条词/span 级标注、2,965 prompts、6 个 LLM)、HaluEval(35,000 样本:5,000 人工标注用户查询 + 30,000 任务生成样本)。阈值切换条件:跨域 AUROC 掉幅 >0.1 即说明探针过拟合训练分布,应改用 unsupervised 的 EigenScore/HaloScope 或引入目标域校准集。

**阶段三(数值/系统层完整性)**:对高可靠场景,叠加**非语义**的数值校验——ALBERTA 式 GEMM checksum(<0.2% 开销)或 activation range/分位数监测(Dr. DNA / arXiv:2310.20349,~96% precision、<1.6% 开销)捕获 SDC;并采用 **batch-invariant kernels**(He et al. 2025)消除 batch-size 引起的非确定性,以便把"质量退化"与"数值抖动"解耦。这一步是归因前提:没有数值一致性,任何 marginal 输出统计漂移都无法区分模型退化与系统噪声。

**方法选择决策规则**:(a) 只有 API/logprob → 阶段一 + SelfCheck 系列;(b) 有权重但要求实时单次 → SEP/MIND/LLM-Check/Lookback Lens;(b′) 做 hidden probe 且需自动选层 → FEPoID(TwoNN-ID)+可选 FST,或对照 Yin-LID 分数;(c) 可承受采样、要最高检测率 → semantic entropy / SAR / INSIDE;(d) 关注输入 novelty/domain shift → RMD/kNN 特征空间(勿用 perplexity 阈值);(e) 关注量化/硬件退化 → KL(p_θ‖p_θ̂) + activation range/checksum。**不要**依赖单一 supervised probe 作为唯一防线。

## Caveats

- **输出层简并 / 同签名多病因**:知识缺失型幻觉、上下文不忠、量化塌缩、SDC 位翻转可能在 hidden/logit 上产生高度相似签名;单一信号无法区分病因,必须组合"语义信号 + 特征 OOD + 数值校验"三层。
- **marginal 统计被 domain 方差主导**:全局 entropy/PPL/EigenScore 阈值在多域流量下,域间方差常盖过幻觉信号;这是 RMD 相对 class-conditional MD、条件化基线相对全局阈值胜出的根因,也是很多"高 AUROC"离线结果无法复现到生产的原因。
- **supervised probe 泛化失败**:SAPLMA、部分 Geometry of Truth 结论、Orgad et al. 的 skill-specific 发现、Levinstein & Herrmann 的批评共同表明——**不存在单一普适 truthfulness 方向**;probe 的高 AUROC 高度依赖训练分布,跨任务/跨句式/跨模型规模均可能崩。
- **对 decoding 策略与采样温度敏感**:consistency/entropy 类方法的分数随温度、nucleus 参数、采样数变化;跨配置比较 AUROC 需固定 decoding。
- **generative likelihood 悖论**:LLM 序列 likelihood/perplexity 对 OOD 可能反向,直接阈值不可靠,须用 likelihood-ratio/typicality/complexity 校正。
- **SAE 用于 OOD 的根本局限**:SAE 在略微 off-distribution 时重构就退化,使其 OOD/anomaly 用途存疑;其重构误差既可能来自真 OOD,也可能来自 SAE 自身泛化不足。
- **结论成熟度差异**:经典 OOD 谱系(MSP/Mahalanobis/RMD/energy/ViM/kNN)、SAPLMA 的域内有效性、semantic entropy、SelfCheckGPT-NLI、ITI 的 TruthfulQA 提升已被广泛复现;而 LapEigvals/HalluZig 的拓扑谱方法、部分 attention 谱结论、SAE-OOD 仍是单篇初步结果,应谨慎。部分 arXiv 编号(如 2602.x、2604.x、2605.x、2606.x)为 2026 年预印本,属最新且未经充分同行复现。
- **ID/LID 与 TwoNN 的定位边界**:intrinsic dimension 是**几何工具**,不是与 Semantic Entropy / Lookback Lens 同级的端到端主检测算法。(i) Yin et al. (2024) 用 **LID 分数**检测,主路径是 MLE/GeoMLE,不宜默认换成 TwoNN 并声称复现;(ii) FEPoID (Wang et al., 2026, arXiv:2605.26366) 用 **TwoNN** 做**自动选层**,检测器仍是 hidden probe 等——选对层不等于解决 probe 跨任务失败;(iii) 估计依赖邻域构造与样本量,marginal ID 阈值同样可能被 domain 方差主导;(iv) FEPoID 属 2026 新工作,跨服务栈复现仍有限,不宜列入阶段一「即刻可部署」清单。
- **Thinking Machines 的 batch-invariance 分析**为实验室博客(非同行评审),但有 DOI 且提供可运行代码,结论(batch-variance 而非浮点非结合性为主因)技术上自洽。
- **对比表注***:LID/FEPoID 行的「多次采样?=否*」指**不需要** K 次独立生成回答;但仍需要一批 hidden 向量构造邻域或逐层 ID 曲线(校准/缓存集),与 semantic entropy 的采样开销不是同一概念。
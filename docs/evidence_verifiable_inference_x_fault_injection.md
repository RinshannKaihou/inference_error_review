# 补齐证据基础:verifiable inference 协议线 × fault-injection 联合测量证据

## TL;DR
- **verifiable inference 协议线已足够成熟到可直接落地**:Token-DiFR / Activation-DiFR(arXiv:2511.20621)提供 anchored、provider 零开销、有 vLLM 开源集成的 per-token 参照检验,4-bit 量化 AUC>0.999(Token 300 tokens、Activation 2 tokens),是当前压缩「有害且不可见」象限的最强单一工具;但它与所有 replay 系协议一样,只覆盖 provider 侧偏离 spec 的持久/间歇故障,对 deployment-wide model_on(参照实现本身就错)结构性不可见。
- **fault-injection 文献里「任务影响 × 内部信号响应」的联合测量几乎不存在**:2025–2026 的 LLM 专项工作(LLMFI arXiv:2606.02430、arXiv:2601.19912、Sun et al. SC'25、LLM-PRISM arXiv:2604.10390)都测了任务影响,但检测侧要么是 recompute/checksum 冗余(非「内部信号」),要么根本没有检测器;对六个度量族(geometric/spectral/trajectory/statistical/infotheoretic/cheap)做硬件故障 dose-response 的工作仍是**全面文献空白**,最接近的仍是 Dr. DNA / quantile-marker 的 vision-DNN 结果。
- **可直接读出 P(detect | task-impact) 的行只有 vision-DNN 侧的 Dr. DNA、quantile-marker、ALBERTA**;LLM 侧唯一能给条件结构的是 arXiv:2601.19912(200/3000≈6.67% SDC,但无检测器)与 LLMFI 的 selective checksum(80.04–90.31% 的「准确率损失恢复覆盖率」,非硬件信号检出率,且无 FPR)。结论未变:per-request 有害故障检测仍依赖 paired reference,聚合路径覆盖持久 model_on,单发瞬态 on-manifold 残余仍不可归约。

---

## Key Findings

1. **DiFR 系是本轮最重要的深化对象**。Token-DiFR 的核心统计量是 seed 同步下的 Gumbel-Max logit 差:`logit_diff = logits[verifier_token] − logits[claimed_token]`,clip 到 99.9 分位。null 校准依赖「seed 固定时 >98% token 精确匹配、分歧时通常仅 2–3 个候选」这一近确定性事实。检测粒度分层明确:错 seed ~100 tokens、4-bit 量化 ~1,000 tokens、KV cache 量化 ~10,000 tokens;Activation-DiFR 用随机正交投影压缩隐状态,4-bit 量化 AUC>0.999 仅需 2 tokens,通信开销较 TOPLOC 降 25–75%(1–4 bytes/token)。

2. **跨硬件/跨栈 null 展宽是真实存在但可控的局限**。作者实测 A100 vs H200、单卡 vs 4-GPU TP、HuggingFace vs vLLM、prefill vs decode 的良性噪声「一致小于」真实故障信号——但有一个明确例外:**Qwen3-30B-A3B 上 HuggingFace 与 vLLM 的差距与 FP8 KV-cache 量化同量级**,这直接迫使部署方必须选定一个参照 spec 并将超出者判为偏离。这正是 replay 系协议对 deployment-wide model_on 故障不可见的量化证据。

3. **同族协议各有明确覆盖边界**。Rinberg et al.(arXiv:2511.02620,含 Anthropic、Harvard 作者)在 MoE-Qwen-30B 上把可外泄信息压到 <0.5%、FPR<10⁻²(对抗方 >200× 减速);TOPLOC(Ong, Di Ferrante et al. / Prime Intellect,arXiv:2501.16007,ICML 2025 poster #46281)LSH 指纹「detect unauthorized modifications to models, prompts, or precision with 100% accuracy, achieving no false positives or negatives」、验证快 100×、258 bytes/32 tokens(较直存 262 KB 降 1024×),但阈值法**对量化这类微小偏离检测失败**(DiFR 明确指出);SVIP(arXiv:2410.22307)proxy-task 法 FNR<5%/FPR<3%、<0.01s/query,但需训练检测模型;zkML(zkLLM 每 forward 986s+803s,外推 2000-token 生成约 23 天承诺+18 天证明)对高吞吐 LLM 完全不可行;RUT(Zhu et al.,arXiv:2506.06975)在 Gemma-2-9b 4-bit 替换检测上统计功效显著优于 MMD 和 KS,且抗对抗性 reroute。

4. **确定性基础设施线的开销数字已收敛**。Thinking Machines Lab(He, 2025,"Defeating Nondeterminism in LLM Inference",Sept 2025)的 batch-invariant kernels 基线约 60% 吞吐代价,并证明「1,000 identical runs now produce 1,000 identical outputs」(此前 Qwen3-235B temp-0 生成 80 个 unique 输出);SGLang 用优化的 FlashInfer/FlashAttention 3 backend 集成后降至 ~34%(34.35%);LLM-42(arXiv:2601.17768)的 decode-verify-rollback 使开销正比于需确定性的流量比例,其论文指出对混合 batch 强制 batch-invariant kernel 会使「throughput to collapse by 56% to about 415 tokens/s—penalizing every in-flight request for a single deterministic one」;DeepSeek-V4(arXiv:2606.19348)声称用 DeepGEMM 替换 cuBLAS 实现端到端 bitwise batch-invariant「开销可忽略」。MarginGate(arXiv:2605.30218)把 LLM-42 式验证稀疏化:Llama-3.1-8B 仅 18.56% 触发率、Qwen2.5-14B 15.05%、DSR1-Distill-Qwen-7B 49.50%,即恢复 100% 序列级确定性,较 always-on 降延迟 2.23×/1.99×。

5. **LLM 专项 fault-injection 在 2025–2026 密集涌现,但检测侧仍是短板**。LLMFI(arXiv:2606.02430,ICS 2026)是最完整的:单 bit-flip、float16、13 任务 3 模型,任务影响用准确率变化定义(reasoning 首 token flip、generative 用 exact-match/unit-test);头条数字是 multi-choice reasoning 准确率降幅最高 4.25%、generative 5.01%,bit 14(float16 exponent MSB)造成灾难性传播,MoE 对 memory fault 更鲁棒(仅 20–30% 专家激活遮蔽故障)。其 selective checksum 覆盖 80.04–90.31%(仅保护 30–32% GEMM),Dual First-token Generation 用冗余重算+token 比较——**但两者都不是「内部信号」检测器,且都不报 FPR**。

6. **arXiv:2601.19912 提供了 LLM 侧唯一干净的 Masked/SDC/DUE 三分条件结构**:3000 次注入 → 精确 200 次 SDC(≈6.67%),其中 178 数值错误、22 地址错误;bit 30(FP32 exponent MSB)SDC 率达 23.4–24.2%,是低位的 26.52–95.8×;但**该文不评测任何检测器**,无检出率。

---

## Details

### 交付物 1:verifiable inference 协议对比总表

| 协议 (arXiv) | 检验统计量 | 检测粒度 | 覆盖用户框架格 | provider 开销 | verifier 开销 | 部署前提 | 明确不覆盖 |
|---|---|---|---|---|---|---|---|
| **Token-DiFR** (2511.20621) | Gumbel-Max logit_diff vs seed 同步参照,clip 99.9% | per-token,可聚合 | infra_off + model_on(持久/间歇);catastrophic→moderate;subtle 量化需更多 token | 零(用未改 vLLM) | 需跑参照实现(大模型不可行,除非 provider 暴露 logprob) | seed 日志同步 or temp-0 spot-check;需参照实现或官方 API logprob | 单发瞬态 on-manifold 单请求;deployment-wide model_on(参照本身错) |
| **Activation-DiFR** (2511.20621) | 随机正交投影隐状态指纹距离 | per-token,极省样本(2 tokens) | forward-pass 类 infra_off/model_on | 1–4 bytes/token 通信 | 需存指纹+比对 | 需 provider 传指纹;同 spec | **无法验证 sampling**(可事后伪造 token 再算指纹) |
| **Weight-Exfil Verifier** (2511.02620) | FSSL:GM-LS(Gumbel)/IPT-LS(CGS) | per-token 似然 | infra_off(篡改/隐写);持久 | 低(解耦验证服务器) | 独立隔离栈 | seed 同步;decoupled verifier | on-manifold 天然错误;非篡改性 subtle |
| **TOPLOC** (2501.16007) | top-128 activation LSH + 多项式编码 | per-32-token commit | 模型/prompt/precision 篡改;catastrophic | 258 bytes/32 tokens | 验证快 100× | 同模型;单一全局阈值 | **量化等微小偏离检测失败**;不验 sampling |
| **SVIP** (2410.22307) | hidden-state proxy-task 分类器 | per-request | 模型替换(小模型冒充);catastrophic/moderate | 返回压缩 hidden vec | <0.01s/query | **需训练 proxy 模型**;secret 周期更新 | subtle 量化;需重训 |
| **RUT** (2506.06975) | reference logits 下 target token 的 log-rank 均匀性检验 | 聚合(100 prompt×100 ref) | model_on 替换/量化/微调;持久 | 需 top-k logprob | 本地跑 reference | logit 访问;抗对抗 reroute | 单请求;需 logit |
| **Model Equality Testing** (2410.20247) | MMD + Hamming 字符/token 核 | 聚合(~250 completions,median 77.4% power) | model_on 分布偏移;持久 | 黑盒仅需样本 | 本地参照分布 | 仅需 P、Q 样本 | 单请求;subtle 单事实翻转 |
| **zkLLM/zkML** | 交互式 ZKP + sumcheck | per-forward 密码学证明 | 全部(密码学强) | **986s+803s/forward,外推~41 天/2000 tokens** | 极高 | — | **高吞吐 LLM 完全不可行** |
| **bit-exact verify** (2606.00279) | 记录并发 batch composition 后 bitwise 重算 | per-token bitwise pass/fail | infra_off/model_on(provider 侧,同栈);全幅度 | 记录 batch 元数据 | 软件仿真跨 GPU 重算 | 记录 batch size/组成;无 atomic backend | deployment-wide model_on;需元数据完整 |
| **MarginGate** (2605.30218) | top-1/top-2 logit margin 阈值触发 LLM-42 验证器 | per-token 稀疏触发 | batch 诱导 flip(ambiguous 门控);瞬态/间歇 | 15–49.5% 触发的验证成本 | 确定性重算 + K/V 列修复 | 需确定性 verifier contract;MATH500 校准 | on-manifold 非 flip 损害;margin 高但错 |
| **DeepSeek-V4 kernels** (2606.19348) | DeepGEMM bitwise batch-invariant | 训练/推理 bitwise 对齐 | infra_off(非确定性根因消除) | 声称「可忽略」 | 一次重算即可 exact 比对 | 同硬件同实现 | 跨硬件;model_on |

**关键判读**:所有 replay/参照系协议(DiFR、TOPLOC、bit-exact)共享同一盲区——它们锚定于「参照实现是对的」这一前提,故对 deployment-wide model_on(整个部署过度量化/RoPE 配错/模板错,且参照实现也这样配)结构性不可见。DiFR 论文自曝的 Qwen3-30B-A3B「HF vs vLLM 差距 ≈ FP8 KV 量化」正是这一盲区的量化下限:当两个「正确」实现的良性差异已达 subtle 故障量级,任何同栈 replay 都无法区分「另一个正确实现」和「一个 subtle 故障」。唯一部分绕开此盲区的是分布系(MET、RUT)——它们对照的是独立参照分布而非同栈重算,但代价是丧失 per-request 粒度、退回聚合。

### 交付物 2:联合测量证据表(任务影响 × 检测信号)

| 工作 (出处) | 故障模型 | 任务影响定义 | 检测信号 | 条件检出率 P(detect\|task-impact) | FPR | 开销 | 证据档位 |
|---|---|---|---|---|---|---|---|
| **Dr. DNA** (ASPLOS'24) | 4 error model,3 vision task,10 模型 | SDC = 输出与 golden 不符(top-1) | 神经元激活分布签名 (DNA) | **平均 95%,多数 case 100%,全 case >90%**;较 baseline 提升 20–70% | 论文报为「高精度」但未给统一数值 | <1% mem, <2.5% latency | **直接**(但 vision-DNN,非 LLM) |
| **quantile-marker** (2310.20349) | memory fault + input corruption,vision DNN | critical SDC(激活分布 quantile 偏移) | 2 层各 1 quantile marker | **~98% recall, ~96% precision**(platform fault 层);input fault 分层测量 | 由 precision 推 | 0.3–1.6% | **直接**(vision) |
| **ALBERTA** (2310.03841) | transient HW error,ViT,FP/INT | 「导致 mismatch 的错误」 | GEMM checksum(算法级) | **>99% 覆盖导致 mismatch 的错误** | checksum 内建低误报 | <0.2% compute, <0.01% mem;纠错 <2% | **直接**(ViT) |
| **LLMFI** (2606.02430) | 单 bit-flip,float16,13 任务 3 LLM | 准确率变化(reasoning 首 token flip;generative exact-match/unit-test) | selective checksum;Dual First-token(冗余重算+比较) | checksum **80.04–90.31%「准确率损失恢复覆盖率」**(非 HW 信号检出);Dual First-token **无检出率** | **均未报 FPR** | checksum 仅保护 30–32% GEMM | **部分直接**(检测=冗余,非内部信号) |
| **2601.19912** | instruction-level bit-flip,NVBitFI | Masked/SDC/DUE 三分;SDC=正常运行但输出错 | **无检测器** | 200/3000≈**6.67% SDC**(178 数值+22 地址);bit-30 SDC 率 23.4–24.2% | N/A | 300+1750 GPU-h | **条件结构直接,检测空白** |
| **Sun et al.** (SC'25) | 单 fault,memory/compute | Masked vs SDC(final answer ≠ ref);generative 用质量指标 | 未验证内部检测器 | 聚合准确率掩盖文本质量退化(Falcon3-7B GSM8K 降 9.36%,Qwen2.5-7B 13.09%) | N/A | — | **任务影响直接,检测空白** |
| **LLM-PRISM** (2604.10390) | permanent GPU fault,RTL 仿真,7664 run,Megatron-LM | 训练 loss/divergence | **NaN/Inf 检测器**(量化其盲点) | 量化 NaN/Inf 检测器有效性与盲点(训练侧) | 报 blind spot 非 FPR | RTL 仿真 | **训练侧直接,推理空白** |
| **LLM-SDC-training** (2604.00726) | intermittent MatMul fault,NVBit | loss spike/grad norm/attention logit spike/param divergence | 轻量 param-update 异常检测 + recompute | 定性识别「有害 update」,recompute 恢复 | 未量化 | 单 GPU | **训练侧直接** |
| **Ranger** (2003.13874) | transient,DNN | SDC→benign 转化 | 激活值域限幅 | 大幅降 SDC 率(条件:限幅生效于 out-of-range) | 低(值域内不触发) | 极低 | **直接**(CNN) |

**核心判定(未变、现已加固)**:没有任何工作把**六个度量族中任一内部信号检测器**对**硬件故障做 dose-response 联合测量**。LLMFI 是最接近的 LLM 联合测量,但其「检测」是 ABFT checksum 与冗余重算(recompute/replay 的另一种形态),不是 geometric/spectral/trajectory/statistical/infotheoretic/cheap 中任一族的内部信号。Dr. DNA / quantile-marker 的激活分布签名在概念上属 statistical 族,是**唯一有直接 dose-response 的族,但仅在 vision-DNN**。

### 交付物 3:判定矩阵逐格更新(族 × 持续性 × 幅度)

- **infra_off × 持久 × catastrophic**:证据档位从「外推」**升到「直接」**。DiFR Token/Activation(2511.20621)+ TOPLOC(2501.16007)+ LLMFI bit-14/bit-30 灾难性传播证据(2606.02430, 2601.19912)共同锚定。P(detect)≈1.0,只要有参照且故障持久。
- **infra_off × 持久 × moderate(5–20% 掉点)**:**升到「直接」**。DiFR 4-bit AUC>0.999@1k tokens + RUT 对 4-bit 替换功效 + MET(median 77.4% power)。聚合路径可靠。
- **infra_off × 持久 × subtle(单事实翻转)**:仍「部分外推」。Activation-DiFR 2-token 检测量化是最强证据,但对「单发单请求 subtle」无覆盖;DiFR 自曝 HF-vs-vLLM 良性噪声可达 subtle 量级 → 存在不可消除的 subtle 假阴带。
- **infra_off × 瞬态 × catastrophic**:「直接」。乱码/NaN 传播,任何输出侧监控即捕获(2601.19912 DUE、LLM-PRISM NaN/Inf)。但 2605.04213 提示 **NaN/±INF 仅占 SDC 结果的 1.01%**,即多数 SDC 不表现为 catastrophic → catastrophic 瞬态是「易检但稀有」。
- **infra_off × 瞬态 × subtle/moderate**:**仍是核心空白**(单发 on-manifold 单请求)。无 per-request 参照则不可归约;压缩杠杆仍是 MarginGate 式风险加权选择性验证(margin 触发 15–49.5%)+ seed 日志事后审计。
- **model_on × 持久 × 任意幅度(deployment-wide)**:**仍是最深的结构空白**。所有同栈 replay 结构性不可见;唯一手段是独立金集精度回归 + RUT/MET 对独立参照分布——但金集面临 MoE 专家覆盖问题(见下)。
- **model_on × 持久 × 局部(部分请求量化)**:「直接」升级,RUT 专测「一部分 API 响应来自量化模型」的比例 q 下的统计功效。
- **ambiguous × 间歇 × 任意**:MarginGate(2605.30218)+ FP16 KV-cache divergence(2604.15409,100% token divergence, FP32 降 8 数量级)+ Stage-Replay(2607.28495,BF16 replica exact 但 live-vs-prefill 166/200 suffix 分歧)把「batch/dtype/KV 门控」这一格从「外推」**升到「直接」**——现有确凿证据表明 dtype 与 KV cache 构造是可复现的 flip 源。

### 交付物 4:六个度量族的故障注入响应测量现状

1. **geometric(隐状态范数/距离流形):文献空白**。最接近:Ranger(2003.13874)激活值域限幅、ALBERTA(2310.03841)GEMM checksum——都是几何邻域的粗代理,但非流形几何检测器 dose-response。
2. **spectral(激活谱/奇异值):文献空白**。无任何故障注入下谱响应测量。相邻:Code-MUE(2607.12273)spectral graph entropy 做 UQ,但对天然错误非硬件故障。
3. **trajectory(逐层/逐 token 轨迹):部分**。LLMFI(2606.02430)layer-wise/stage-wise 传播(最后 3–4 层退化)+ Stage-Replay(2607.28495)是轨迹级测量,但测的是「故障如何传播」,非「轨迹检测器如何响应」。
4. **statistical(激活分布/quantile):唯一有直接 dose-response 的族,但仅 vision**。Dr. DNA + quantile-marker(2310.20349)。LLM 侧空白。
5. **infotheoretic(熵/KL/cross-entropy):部分**。DiFR mean cross-entropy 可检量化但「易被 temperature 篡改」;RUT log-rank 均匀性是信息论量。均为参照系,非内部信号自足检测。硬件故障 dose-response 空白。
6. **cheap(logit margin/top-k gap/max-prob):最接近落地**。MarginGate(2605.30218)top-1/top-2 margin 是 flip 风险的直接 reference-free 信号,且已做「flip vs margin」联合测量(K/V 扰动 flip 前平坦,低 margin 暴露风险)。但这是「batch flip」非「硬件故障」dose-response。

### 交付物 5:harness 实验清单修订版

**实验 A(harm-gated 三元 dose-response)**:注入 infra_off/model_on/ambiguous × 瞬态/间歇/持久 × 三幅度,对每个注入同时记录(i)下游任务指标变化 与(ii)六族内部信号。**这是填补最大空白的实验**——现有全部 LLM fault-injection 都缺 (ii) 的内部信号轴。可直接复用:LLMFI 开源框架(NVBit 基础)做注入层,DiFR 的 vLLM 集成(github.com/adamkarvonen/difr)做参照锚,quantile-marker 方法迁移做 statistical 族探针。基准阈值:若某族在 subtle 幅度下 AUC<0.7 则判该族对该格无用。

**实验 B(margin 介导翻转定向注入)**:用 MarginGate 的 margin 信号定位低 margin 步,在这些步定向注入,测「margin 阈值 → flip → 任务翻转」的传递函数。复用:MarginGate 校准协议(MATH500)+ FP16 KV divergence(2604.15409)作为已知 100% divergence 的阳性对照。基准:若定向注入的 P(task-flip|low-margin) 显著 > 随机注入,则 margin 是有效风险特征,可驱动实验 C 的预算分配。

**实验 C(选择性重算预算富集率对比)**:对比均匀抽样 replay vs margin/uncertainty 加权 replay 的「单位验证预算命中有害事件率」。复用:DiFR Token-DiFR 做 replay 引擎,MarginGate 的 15–49.5% 触发率作为预算参考点。基准:富集率 <1.5× 则选择性验证不值得工程复杂度;Heteroskedastic Signals(2606.15841)警告 uncertainty 质量的结构异质性会限制此增益,应作为对照假设。

**实验 D(金集门控轴 + 专家覆盖审计)**:针对 deployment-wide model_on 空白设计。金集须(1)独立于生产栈重算(绕开 replay 盲区),(2)审计 MoE 专家覆盖——现有金集实践(Langfuse/Galileo 建议 100–1000 条 diverse 项)**从不检查金集是否激活全部专家**。修订:构造覆盖全 expert 路由的探针集(可借 MoE serving trace 如 BurstGPT 反推激活分布),确保金集精度回归能触发每个 expert。基准:若金集专家覆盖 <80% 则 deployment-wide model_on 检测有系统盲区。serving 侧 provenance(prefix cache 命中、抢占、迁移、batch 组成)应作为故障风险特征记录——bit-exact verify(2606.00279)已证明记录 batch composition 使验证坍缩为 bitwise pass/fail,这是 provenance-as-risk-feature 的直接先例。

---

## Recommendations

1. **立即部署 Token-DiFR + Activation-DiFR 作为 anchored baseline**(复用 github.com/adamkarvonen/difr 的 vLLM 集成)。对自有推理栈:一个参照实例随机抽查生成输出,seed 同步。阈值设定:选定单一参照 spec,超出即 flag。这一步直接覆盖 infra_off/model_on 的持久与间歇故障,provider 零开销。**升级触发条件**:若发现 HF-vs-vLLM 类良性噪声达 subtle 故障量级(如 Qwen3-30B-A3B 案例),必须补充独立金集回归(见 3)。

2. **对 ambiguous/batch 门控格上 MarginGate 式稀疏验证**。margin 阈值触发 15–49.5% 步的确定性重算,恢复序列级确定性,较 always-on 省 ~2×。**阈值调整**:触发率 >50%(如 DSR1 案例)时改用全 batch-invariant kernel(SGLang ~34% 开销)更划算;LLM-42 证据表明混合 batch 上强制 batch-invariant 会使吞吐塌陷 56%,故稀疏触发在低确定性流量占比时严格占优。

3. **补金集回归覆盖 deployment-wide model_on**——这是 replay 系结构性盲区,只能靠独立参照。金集设计**必须**加专家覆盖审计轴(现有实践全缺);辅以 RUT/MET 对独立参照分布做聚合检验(RUT 每 prompt 1 target + 100 reference completions,MET 约 250 completions、median 77.4% power)。

4. **优先做实验 A**(harm-gated dose-response with 内部信号轴),因为它填补的是全文献最大空白:六度量族对硬件故障的响应从未被联合测量。这是本研究方向能产出「直接证据」升级最多格子的单一实验。

5. **不要投入 zkML 路线**:zkLLM 每 2000-token 生成外推 ~41 天,对生产 LLM 完全不可行,DiFR/TOPLOC 的统计参照法以极小开销达成同等实用检测。

---

## Caveats

- **证据档位诚实标注**:LLM 侧「任务影响 × 内部信号」联合测量的**直接证据为零**;所有升到「直接」的格子要么依赖 vision-DNN 外推(Dr. DNA、quantile-marker、ALBERTA),要么依赖参照/冗余系而非内部信号(DiFR、LLMFI checksum)。
- **Sun et al.(SC'25)的「聚合准确率掩盖文本质量退化」精确数值分割未能从 ACM 全文取得**(ACM 反爬阻断),仅有 Falcon3-7B GSM8K 降 9.36%、Qwen2.5-7B 13.09% 等片段;引用前应核对原 PDF。
- **NaN/±INF 仅占 SDC 结果 1.01%**(arXiv:2605.04213,production GPU gate-level 注入),意味着依赖 NaN/Inf 监控只能捕获极小部分 SDC,catastrophic 瞬态「易检但稀有」的判断由此加固;但该数据来自训练/通用 GPU 场景,推理侧比例未知。
- **单发瞬态 on-manifold 单请求损害仍是不可归约的概率残余**,本轮所有新证据未改变这一结论,只是把压缩杠杆(风险加权选择性验证 + seed 日志事后审计)的工具从概念变为有开源实现(MarginGate、DiFR)。
- 部分 2026 年 arXiv 预印本(2606.19348 DeepSeek-V4、2607.28495、2601.17768)为近期未经同行评审的预印本,开销/功效数字以作者自报为准,应视为初步。
- Thinking Machines Lab 的 batch-invariant 开销(~60% 基线、SGLang ~34%)来自其 2025-09 博客而非同行评审论文;第三方复现(如 SGLang 34.35%)基本吻合,但硬件/backend 依赖大,迁移到自有栈前需实测。
# Annotated research register

Research date: 5 October 2026. Primary papers, author project pages and official documentation. These are original reading notes, not reproductions of the papers. `methods` means selected full-text methods were examined; `abstract` does not mean the full paper or code was audited. No third-party implementation or full paper is redistributed. Bibliographic publication years may differ from first arXiv posting.

## Reading sequence

Start with S02, S04, S05, S07 and S09. Then read S10, S12, S14–S16 for representations; S22–S26 for model risk; S29–S34 for evaluation. S36–S39 are optional extension/current-literature reading, not dependencies for the first pilot.

## [S01] Mastering diverse control tasks through world models

Hafner, Danijar; Pasukonis, Jurgis; Ba, Jimmy; Lillicrap, Timothy. 2025. Nature. Review depth: **methods**.

Source: <https://www.nature.com/articles/s41586-025-08744-2>

**Finding:** World-model prediction and imagined policy improvement can be combined in a working control system.

**Use in Rineya:** Borrow the separation of dynamics, utility and control; qualify model fidelity before imagined learning.

**Boundary:** Control results do not establish language-model transfer or feasibility on a laptop.

## [S02] Successor Features for Transfer in Reinforcement Learning

Barreto, Andre; others. 2017. NeurIPS. Review depth: **methods**.

Source: <https://arxiv.org/abs/1606.05312>

**Finding:** A linear reward feature decomposition permits reuse of policy-conditioned expected future features across reward functions.

**Use in Rineya:** Define consequence features, task weights, continuation policy and generalized policy improvement explicitly.

**Boundary:** Transfer assumptions include compatible dynamics, reward features and approximation quality.

## [S03] Universal Successor Features Approximators

Borsa, Diana; others. 2019. ICLR. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1812.07626>

**Finding:** Successor-feature approximation can condition on both task and policy representations.

**Use in Rineya:** Log policy identities; compare a policy bank with a plan-conditioned predictor.

**Boundary:** A goal embedding alone does not specify how later actions will be chosen.

## [S04] Distributional Successor Features Enable Zero-Shot Policy Optimization

Zhu, Chuning; Wang, Xinqi; Han, Tyler; Du, Simon S.; Gupta, Abhishek. 2024. NeurIPS; arXiv v4 2025. Review depth: **methods**.

Source: <https://arxiv.org/abs/2403.06328>

**Finding:** DiSPO models a distribution of achievable successor features and a policy for realizing outcomes supported by its dataset.

**Use in Rineya:** Compare direct long-horizon consequence prediction with recursive latent rollouts.

**Boundary:** Dataset support and reward representation still constrain what can be achieved.

## [S05] Successor Features for Efficient Multisubject Controlled Text Generation

Cao, Meng; Fatemi, Mehdi; Cheung, Jackie Chi Kit; Shabanian, Samira. 2023. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2311.04921>

**Finding:** SF-GEN applies successor features to multiple text-generation controls while leaving the base language-model parameters unchanged.

**Use in Rineya:** Include a frozen-language-model successor-feature baseline.

**Boundary:** This is direct prior art; successor features in language generation are not a new Rineya invention.

## [S06] Decision Transformer: Reinforcement Learning via Sequence Modeling

Chen, Lili; others. 2021. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2106.01345>

**Finding:** A Transformer can model actions conditioned on trajectory history and desired return.

**Use in Rineya:** Use return-conditioned behavior modeling as a policy baseline.

**Boundary:** Conditioning on a desired return is not an action-conditioned world model.

## [S07] Offline Reinforcement Learning as One Big Sequence Modeling Problem

Janner, Michael; Li, Qiyang; Levine, Sergey. 2021. NeurIPS; author project. Review depth: **project overview**.

Source: <https://trajectory-transformer.github.io/>

**Finding:** Trajectory Transformer models discretized trajectories and uses search to plan.

**Use in Rineya:** Use typed trajectory slots and compare explicit search with direct scoring.

**Boundary:** Tokenizing states, actions and rewards is established prior art, not evidence of causal understanding.

## [S08] UniMASK: Unified Inference in Sequential Decision Problems

Carroll, Micah; others. 2022. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2211.10869>

**Finding:** Different masks over a trajectory can express multiple sequential inference tasks.

**Use in Rineya:** Specify mask families separately for prediction, inverse modeling and retrospective diagnostics.

**Boundary:** A convenient mask does not establish that its visible information exists at deployment.

## [S09] Masked Trajectory Models for Prediction, Representation, and Control

Wu, Philipp; Majumdar, Arjun; Stone, Kevin; Lin, Yixin; Mordatch, Igor; Abbeel, Pieter; Rajeswaran, Aravind. 2023. ICML / PMLR. Review depth: **abstract**.

Source: <https://proceedings.mlr.press/v202/wu23o.html>

**Finding:** Masked trajectory reconstruction supports forward, inverse and control uses within one network.

**Use in Rineya:** Compare causal and plan-conditioned masks without changing information budgets.

**Boundary:** Continuous-control results do not establish faithful multi-step language reasoning.

## [S10] Learning Action Representations for Reinforcement Learning

Chandak, Yash; Theocharous, Georgios; Kostas, James; Jordan, Scott; Thomas, Philip. 2019. ICML / PMLR. Review depth: **abstract**.

Source: <https://proceedings.mlr.press/v97/chandak19a.html>

**Finding:** Learned action representations can exploit structure in large discrete action spaces.

**Use in Rineya:** Represent operator, arguments and executor semantics rather than a free-standing action name.

**Boundary:** Similar action embeddings need not imply equivalent effects in every state.

## [S11] Mastering Atari, Go, chess and shogi by planning with a learned model

Schrittwieser, Julian; others. 2020. Nature; primary arXiv version. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1911.08265>

**Finding:** MuZero learns quantities needed for planning rather than requiring full observation reconstruction.

**Use in Rineya:** Test a task-sufficient consequence model against a richer state predictor.

**Boundary:** Predicting reward and value is not itself a novel architecture.

## [S12] DeepMDP: Learning Continuous Latent Space Models for Representation Learning

Gelada, Carles; others. 2019. ICML / PMLR. Review depth: **abstract**.

Source: <https://proceedings.mlr.press/v97/gelada19a.html>

**Finding:** Reward and latent-transition prediction give a model-based representation-learning objective.

**Use in Rineya:** Retain transition structure in addition to utility supervision.

**Boundary:** Theoretical guarantees depend on assumptions not automatically satisfied by hidden LLM states.

## [S13] Learning Invariant Representations for Reinforcement Learning without Reconstruction

Zhang, Amy; McAllister, Rowan; Calandra, Roberto; Gal, Yarin; Levine, Sergey. 2021. ICLR. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2006.10742>

**Finding:** Behaviorally relevant state representations can suppress irrelevant observational detail without reconstructing all pixels.

**Use in Rineya:** Test nuisance invariance while protecting distinctions needed by different goals.

**Boundary:** Task-specific invariance can erase features needed by a later task.

## [S14] Representation Learning with Contrastive Predictive Coding

van den Oord, Aaron; Li, Yazhe; Vinyals, Oriol. 2018. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1807.03748>

**Finding:** Predictive contrastive objectives learn representations from future observations.

**Use in Rineya:** Try contrastive consequence targets with carefully constructed same-state alternatives.

**Boundary:** Naive negatives can penalize two genuinely equivalent successful outcomes.

## [S15] Neural Discrete Representation Learning

van den Oord, Aaron; Vinyals, Oriol; Kavukcuoglu, Koray. 2017. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1711.00937>

**Finding:** A learned discrete codebook can represent latent data structure.

**Use in Rineya:** Compare continuous consequences, train-only clustering and vector quantization.

**Boundary:** A codebook does not automatically discover useful or human-interpretable reward categories.

## [S16] VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning

Bardes, Adrien; Ponce, Jean; LeCun, Yann. 2022. ICLR. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2105.04906>

**Finding:** Variance and covariance regularization can accompany an invariance objective to discourage collapsed embeddings.

**Use in Rineya:** Use explicit anti-collapse diagnostics before adding a codebook.

**Boundary:** A noncollapsed representation can still be irrelevant to decisions.

## [S17] Diversity is All You Need: Learning Skills without a Reward Function

Eysenbach, Benjamin; others. 2019. ICLR. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1802.06070>

**Finding:** Diverse behaviors can be learned through an intrinsic information objective rather than task reward labels.

**Use in Rineya:** Use behavior diversity for transition coverage, then separately ground utility.

**Boundary:** Intrinsic objectives are still design choices; unsupervised structure does not determine human preferences.

## [S18] Generalized Decision Transformer for Offline Hindsight Information Matching

Furuta, Hiroki; Matsuo, Yutaka; Gu, Shixiang Shane. 2021. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2111.10364>

**Finding:** Future trajectory statistics can provide alternative conditioning signals for sequential behavior modeling.

**Use in Rineya:** Compare consequence-summary conditioning with scalar return conditioning.

**Boundary:** Hindsight information must not be presented as prospectively available evidence.

## [S19] RUDDER: Return Decomposition for Delayed Rewards

Arjona-Medina, Jose A.; others. 2019. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1806.07857>

**Finding:** Return decomposition addresses difficult delayed-credit settings.

**Use in Rineya:** Study long-distance attribution without double-counting terminal reward.

**Boundary:** Predictive attribution is not by itself a causal effect estimate.

## [S20] Deep Reinforcement Learning from Human Preferences

Christiano, Paul F.; others. 2017. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1706.03741>

**Finding:** Comparisons between behaviors can train a reward model for policy learning.

**Use in Rineya:** Offer preference pairs as one sparse grounding route.

**Boundary:** Preference noise and the intended objective must remain separately audited.

## [S21] Let's Verify Step by Step

Lightman, Hunter; others. 2023. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2305.20050>

**Finding:** Process supervision distinguishes intermediate reasoning feedback from final-answer supervision.

**Use in Rineya:** Evaluate verifier-grounded intermediate consequences alongside terminal correctness.

**Boundary:** A learned process score must not become its own independent evaluation oracle.

## [S22] When to Trust Your Model: Model-Based Policy Optimization

Janner, Michael; Fu, Justin; Zhang, Marvin; Levine, Sergey. 2019. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1906.08253>

**Finding:** Short model rollouts provide a practical way to limit the effects of model error.

**Use in Rineya:** Gate imagined updates behind short-horizon fidelity and real-state re-anchoring.

**Boundary:** A successful control recipe is not a guarantee against language-world model exploitation.

## [S23] Conservative Q-Learning for Offline Reinforcement Learning

Kumar, Aviral; others. 2020. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2006.04779>

**Finding:** Offline value learning must address distribution shift and overestimation outside data support.

**Use in Rineya:** Track candidate support and include conservative value baselines.

**Boundary:** An uncertainty penalty without validation is not a certified conservative bound.

## [S24] A Distributional Perspective on Reinforcement Learning

Bellemare, Marc G.; Dabney, Will; Munos, Remi. 2017. ICML. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1707.06887>

**Finding:** Predicting a return distribution exposes information hidden by its mean.

**Use in Rineya:** Distinguish aleatoric outcome variation from uncertainty about model parameters.

**Boundary:** Marginal return distributions do not establish joint multi-objective or multi-step dependence.

## [S25] Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles

Lakshminarayanan, Balaji; Pritzel, Alexander; Blundell, Charles. 2017. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1612.01474>

**Finding:** Independent predictors provide a practical uncertainty-estimation baseline.

**Use in Rineya:** Use small independent heads or models, then test calibration and coverage.

**Boundary:** Agreement can coexist with a shared blind spot.

## [S26] Scaling Laws for Reward Model Overoptimization

Gao, Leo; Schulman, John; Hilton, Jacob. 2022. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2210.10760>

**Finding:** Optimizing a proxy reward more strongly can separate proxy improvement from actual quality.

**Use in Rineya:** Measure evaluator exploitability as candidate search width grows.

**Boundary:** A higher predicted utility is not evidence of better real outcomes.

## [S27] LoRA: Low-Rank Adaptation of Large Language Models

Hu, Edward J.; others. 2022. ICLR. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2106.09685>

**Finding:** Low-rank trainable updates provide a parameter-efficient adaptation mechanism.

**Use in Rineya:** Compare bounded adapters with a frozen-backbone sidecar.

**Boundary:** Few trainable parameters do not guarantee preservation of existing behaviors.

## [S28] QLoRA: Efficient Finetuning of Quantized LLMs

Dettmers, Tim; others. 2023. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2305.14314>

**Finding:** Quantized frozen weights can reduce the memory cost of adapter tuning.

**Use in Rineya:** Treat quantized loading as a separately qualified memory option.

**Boundary:** Exact deployment arithmetic and small-delta fidelity must be checked on Rineya.

## [S29] RULER: What's the Real Context Size of Your Long-Context Language Models?

Hsieh, Cheng-Ping; others. 2024. COLM. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2404.06654>

**Finding:** Long-context evaluation should test more than a single simple retrieval behavior.

**Use in Rineya:** Separate dependency distance, distractors, aggregation and horizon length.

**Boundary:** Good text-context performance does not establish action planning competence.

## [S30] LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding

Bai, Yushi; others. 2024. ACL. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2308.14508>

**Finding:** Multiple long-context task types provide broader coverage than one metric.

**Use in Rineya:** Use a budgeted external context suite after synthetic transition gates.

**Boundary:** Benchmark access, licenses and contamination still require dataset-specific review.

## [S31] TextWorld: A Learning Environment for Text-based Games

Cote, Marc-Alexandre; others. 2018. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1806.11532>

**Finding:** Text-based interactive worlds provide controllable sequential-action tasks.

**Use in Rineya:** Introduce external environments after exact synthetic snapshots work.

**Boundary:** Natural-language observations introduce aliasing and partially observed state.

## [S32] ALFWorld: Aligning Text and Embodied Environments for Interactive Learning

Shridhar, Mohit; others. 2021. ICLR. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2010.03768>

**Finding:** Text and embodied tasks can share goals and interaction structure.

**Use in Rineya:** Evaluate transfer from symbolic tasks to a richer text-action environment.

**Boundary:** Success on the text setting does not establish visual or physical competence.

## [S33] WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents

Yao, Shunyu; others. 2022. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2207.01206>

**Finding:** A grounded web-shopping environment exposes multi-step decisions and task rewards.

**Use in Rineya:** Use a sandbox benchmark, never real purchases, for later agent evaluation.

**Boundary:** A benchmark simulator is not an authorization to act on real accounts.

## [S34] Deep Reinforcement Learning at the Edge of the Statistical Precipice

Agarwal, Rishabh; others. 2021. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2108.13264>

**Finding:** Aggregate RL claims need uncertainty-aware evaluation across tasks and runs.

**Use in Rineya:** Preserve paired parent outcomes and independent training seeds.

**Boundary:** A toy run or a few seeds cannot certify a small practical gain.

## [S35] Hindsight Experience Replay

Andrychowicz, Marcin; others. 2017. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1707.01495>

**Finding:** Achieved outcomes can support goal relabeling under suitable task semantics.

**Use in Rineya:** Reuse transitions under valid alternative preference weights.

**Boundary:** Relabeling cannot change what physically happened or cross sealed splits.

## [S36] V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Assran, Mido; others. 2025. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2506.09985>

**Finding:** Self-supervised video representations can be connected to action-conditioned prediction and planning.

**Use in Rineya:** Study visual latent targets only after a language/symbolic control benefit is established.

**Boundary:** The reported scale and training resources are not the proposed laptop recipe.

## [S37] DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Zhou, Gaoyue; Pan, Hengkai; LeCun, Yann; Pinto, Lerrel. 2024. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2411.04983>

**Finding:** Pretrained visual features can support learned latent dynamics and planning.

**Use in Rineya:** Prefer frozen visual encoders over pixel generation for a later compact variant.

**Boundary:** A chosen visual representation may omit a task-critical state variable.

## [S38] From Word to World: Can Large Language Models be Implicit Text-based World Models?

Li, Yixia; others. 2025. arXiv preprint; revision 2026. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2512.18832>

**Finding:** Language-world modeling can be evaluated by prediction fidelity, scaling behavior and downstream agent utility.

**Use in Rineya:** Use independent fidelity and actual-control endpoints.

**Boundary:** Do not repeat unverified percentage gains or venue claims from earlier brainstorming.

## [S39] V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

Mur-Labadia, Lorenzo; others. 2026. arXiv preprint. Review depth: **abstract**.

Source: <https://arxiv.org/abs/2603.14482>

**Finding:** Dense hierarchical self-supervision offers additional ideas for predictive representations.

**Use in Rineya:** Track multi-level targets and visual tokenizers as later reading.

**Boundary:** Newer representation results do not prove the proposed Rineya consequence geometry.

## [S40] Attention Is All You Need

Vaswani, Ashish; others. 2017. NeurIPS. Review depth: **abstract**.

Source: <https://arxiv.org/abs/1706.03762>

**Finding:** Attention provides a sequence-modeling mechanism with parallel training computations.

**Use in Rineya:** Reuse attention while specifying exactly which information each query can access.

**Boundary:** Parallel tensor operations do not remove environment causality or combinatorial search.

## [S41] torch.nn.Transformer documentation

{PyTorch contributors}. 2026. official documentation; mutable. Review depth: **API reference**.

Source: <https://docs.pytorch.org/docs/2.10/generated/torch.nn.Transformer.html>

**Finding:** Attention masks are part of the API contract and conventions differ across attention interfaces.

**Use in Rineya:** Test blocked-versus-allowed Boolean semantics against the installed version.

**Boundary:** Documentation versions and native Rineya attention kernels may differ.

## [S42] Qwen3 model documentation

{Hugging Face contributors}. 2026. official documentation; mutable. Review depth: **API reference**.

Source: <https://huggingface.co/docs/transformers/model_doc/qwen3>

**Finding:** The documented model API provides the stock-model reference for embeddings and model outputs.

**Use in Rineya:** Use it as an interface reference, then qualify the actual Rineya implementation.

**Boundary:** Stock Qwen3 compatibility is not proof of custom checkpoint compatibility.

## [S43] Gymnasium Env API

{Farama Foundation contributors}. 2026. official documentation; mutable. Review depth: **API reference**.

Source: <https://gymnasium.farama.org/api/env/>

**Finding:** The environment interface distinguishes true termination from time-limit truncation.

**Use in Rineya:** Record both flags and create explicit reproducible snapshot contracts.

**Boundary:** A seed is not a complete snapshot of an arbitrary external environment.

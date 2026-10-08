# Related work for RCWM-T1: primary sources checked as of 2026-10-06

## Is there precedent for "a tiny fine-tuning delta acts mostly as a constant activation offset"?

**Yes, and most of it is recent (2023–2026).** The closest precedents show that a LoRA or narrow fine-tune is, in effect, adding a near-constant steering vector. As far as I found, none quantify the variance share at the answer slot the way RCWM's 97–99% figure does. None test it on a skill that looks conditional, like state-transition generation. Wang et al. come closest: they say a backdoor task "seems like it must involve conditional behavior," yet an unconditional vector was enough. Your pending steering test is the causal test this literature treats as decisive: train or transplant the vector and see whether it reproduces the behaviour. One caution: a constant offset at late layers could be the downstream echo of earlier changes that do depend on the input. The 1–3% input-specific residual may be where the actual skill lives, so test it separately.

- **Simple Mechanistic Explanations for Out-Of-Context Reasoning.** Wang, Engels, … Nanda, 2025. ICML 2025 R2-FM workshop, arXiv:2507.08218. https://arxiv.org/abs/2507.08218 — Finds that "the LoRA fine-tuning essentially adds a constant steering vector." The LoRA-added vectors have absolute cosine near 1 across tokens and prompts, and steering vectors trained from scratch reproduce the effect. This is the strongest direct precedent for finding (5).
- **Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences.** Minder et al., 2025. ICLR 2026, arXiv:2510.13900. https://arxiv.org/abs/2510.13900 — Narrow fine-tuning creates strong activation biases that show up even on the first few tokens of random text. These biases are readable and usable for steering, and the work covers Qwen and others at 1B–32B. The paper links the biases to overfitting and catastrophic forgetting: ablating the bias raises fine-tuning loss and lowers pretraining loss. I took this detail from a summary of the HTML, so check it in the PDF.
- **Convergent Linear Representations of Emergent Misalignment.** Soligo et al., 2025. arXiv:2506.11618. https://arxiv.org/abs/2506.11618 — Uses rank-1 LoRAs on Qwen-14B. A single direction mediates the fine-tune-induced behaviour, and it transfers across separate fine-tunes.
- **Steering LLM Reasoning Through Bias-Only Adaptation.** Sinii et al., 2025. EMNLP 2025, arXiv:2505.18706. https://arxiv.org/abs/2505.18706 — Additive per-layer steering vectors (about 0.0016% extra parameters on an 8B model) recover, and sometimes beat, fully RL-tuned accuracy on GSM8K and MATH.
- **Persona Vectors.** Chen et al., 2025. arXiv:2507.21509. https://arxiv.org/abs/2507.21509 — Behaviour shifts after fine-tuning correlate strongly with activation shifts along single trait directions, and steering during training can prevent them.
- **A Mechanistic Understanding of Alignment Algorithms (DPO/toxicity).** Lee et al., 2024. ICML 2024, arXiv:2401.01967. https://arxiv.org/abs/2401.01967 — DPO bypasses toxic capability with an offset rather than removing it, and the change is easy to reverse.

## Fine-tuning as reweighting existing mechanisms ("mode" framing)

- **Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking.** Prakash et al., 2024. ICLR 2024, arXiv:2402.14811. https://arxiv.org/abs/2402.14811 — Base and fine-tuned models use the same entity-tracking circuit, and the gains come from better handling of positional information.
- **Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks.** Jain et al., 2024. ICLR 2024, arXiv:2311.12786. https://arxiv.org/abs/2311.12786 — Fine-tuning rarely changes capabilities. It learns a minimal "wrapper" on top of them.
- **Understanding Catastrophic Forgetting in LMs via Implicit Inference.** Kotha et al., 2024. ICLR 2024, arXiv:2309.10105. https://arxiv.org/abs/2309.10105 — Fine-tuning skews which task the model implicitly infers. A per-task "mode vector" is a natural mechanistic correlate of that.

## Offset-style adaptation methods (constant shifts are often enough)

- **BitFit.** Ben Zaken et al., 2022. ACL 2022 (short), arXiv:2106.10199. https://arxiv.org/abs/2106.10199 — Tuning only bias terms (under 0.1% of parameters, i.e. literally constant offsets) is competitive with full fine-tuning on small to medium data.
- **Towards a Unified View of Parameter-Efficient Transfer Learning.** He et al., 2022. ICLR 2022, arXiv:2110.04366. https://arxiv.org/abs/2110.04366 — Recasts adapters, prefix tuning and LoRA as modifications Δh to hidden states.
- **ReFT.** Wu et al., 2024. NeurIPS 2024, arXiv:2404.03592. https://arxiv.org/abs/2404.03592 — Interventions on hidden representations match or beat LoRA with 15–65x fewer parameters.
- **LoFiT.** Yin, Ye & Durrett, 2024. NeurIPS 2024, arXiv:2406.01563. https://arxiv.org/abs/2406.01563 — Learned offset vectors on 3–10% of attention heads match LoRA.
- **Activation steering foundations:**
  - Turner et al., 2023, ActAdd, arXiv:2308.10248. https://arxiv.org/abs/2308.10248
  - Zou et al., 2023, Representation Engineering, arXiv:2310.01405. https://arxiv.org/abs/2310.01405
- **In-context analogues** (a single vector carries the task):
  - Hendel et al., 2023, In-Context Learning Creates Task Vectors, Findings of EMNLP 2023, arXiv:2310.15916. https://arxiv.org/abs/2310.15916
  - Todd et al., 2024, Function Vectors, ICLR 2024, arXiv:2310.15213. https://arxiv.org/abs/2310.15213

## Sparse deltas, task vectors and merging

- **Task-Specific Skill Localization.** Panigrahi et al., 2023. ICML 2023, arXiv:2302.06600. https://arxiv.org/abs/2302.06600 — Grafting about 0.01% of fine-tuned parameters keeps over 95% of performance. This is the closest precedent for Titan's 0.05%-norm sparse delta.
- **Editing Models with Task Arithmetic.** Ilharco et al., 2023. ICLR 2023, arXiv:2212.04089. https://arxiv.org/abs/2212.04089 — Weight-space task vectors can be added and negated.
- **TIES-Merging.** Yadav et al., 2023. NeurIPS 2023, arXiv:2306.01708. https://arxiv.org/abs/2306.01708 — Resets small-magnitude delta entries and resolves sign conflicts.
- **DARE ("Language Models are Super Mario").** Yu et al., 2024. ICML 2024, arXiv:2311.03099. https://arxiv.org/abs/2311.03099 — Most delta parameters can be dropped at random and the rest rescaled without hurting SFT abilities.
- **How this fits (5):** these papers show weight deltas are redundant. Your finding adds that the delta's *activation-space* effect is about rank-1 per task. That is the bridge Wang et al. and Minder et al. start to build.

## Probing for world and entity state (findings 3 and 1)

- **Implicit Representations of Meaning in Neural Language Models.** Li, Nye & Andreas, 2021. ACL 2021, arXiv:2106.00737. https://arxiv.org/abs/2106.00737 — Entity state in BART and T5 can be read out linearly, and editing it has predictable effects.
- **Othello-GPT.**
  - Li et al., 2023, Emergent World Representations, ICLR 2023, arXiv:2210.13382. https://arxiv.org/abs/2210.13382 — The board state representation is nonlinear.
  - Nanda, Lee & Wattenberg, 2023, Emergent Linear Representations in World Models, BlackboxNLP 2023, arXiv:2309.00941. https://arxiv.org/abs/2309.00941 — It becomes linear in a "mine vs. theirs" basis. Lesson: probe failure may reflect the wrong basis. This matters for your change-mask probes.
- **Entity Tracking in Language Models.** Kim & Schuster, 2023. ACL 2023, arXiv:2305.02363. https://arxiv.org/abs/2305.02363 — Text-only pretraining does not surface entity tracking. Fine-tuned models generalize non-trivially to longer operation sequences but degrade.
- **Designing and Interpreting Probes with Control Tasks.** Hewitt & Liang, 2019. EMNLP 2019, arXiv:1909.03368. https://arxiv.org/abs/1909.03368 — Introduces selectivity. Your selectivity of about 0 means the probe can compute consequences from surface features, so it is not evidence the representation encodes them.
- **LLMs Know More Than They Show.** Orgad et al., 2025. ICLR 2025, arXiv:2410.02707. https://arxiv.org/abs/2410.02707 — Internal states encode more than outputs show. Your result runs the other way: the skill is visible through the LM head but not through linear probes.
- **Evaluating the World Model Implicit in a Generative Model.** Vafa et al., 2024. NeurIPS 2024, arXiv:2406.03689. https://arxiv.org/abs/2406.03689 — Models pass next-step diagnostics but have incoherent world models. This matches strong one-step accuracy with multi-step collapse.
- **Can Language Models Serve as Text-Based World Simulators?** Wang et al., 2024. ACL 2024, arXiv:2406.06485. https://arxiv.org/abs/2406.06485 — Uses one-step state-transition prediction as a benchmark and finds GPT-4 unreliable.
- **Faith and Fate.** Dziri et al., 2023. NeurIPS 2023, arXiv:2305.18654. https://arxiv.org/abs/2305.18654 — Errors compound on compositional multi-step tasks. Relevant to your k=2/4 collapse being faster than independent composition predicts.
- **Reasoning via Planning (RAP).** Hao et al., 2023. EMNLP 2023, arXiv:2305.14992. https://arxiv.org/abs/2305.14992 — Uses the LM as a world model rolled out step by step for planning. Precedent for self-rollout (finding 2).

## Goal transfer (finding 4)

- **Successor Features for Transfer in RL.** Barreto et al., 2017. NeurIPS 2017, arXiv:1606.05312. https://arxiv.org/abs/1606.05312 — Separating dynamics features from linear reward weights lets a model transfer to new reward weightings. This is the formal analogue of your out-of-span-goal advantage.

## Forgetting measured through representational drift (d)

- **What Happens To BERT Embeddings During Fine-tuning?** Merchant et al., 2020. BlackboxNLP 2020, arXiv:2004.14448. https://arxiv.org/abs/2004.14448 — Fine-tuning is conservative and mainly changes the top layers, consistent with your late-layer offset.
- **Anatomy of Catastrophic Forgetting.** Ramasesh, Dyer & Raghu, 2021. ICLR 2021, arXiv:2007.07400. https://arxiv.org/abs/2007.07400 — Deeper layers drive forgetting.
- **Spurious Forgetting in Continual Learning of LMs.** Zheng et al., 2025. ICLR 2025, arXiv:2501.13453. https://arxiv.org/abs/2501.13453 — Many apparent performance drops are lost task alignment, not lost knowledge. This fits the "mode vector" reading.

## Not verified or not covered

- **Not searched, so UNVERIFIED as citations:** Feng & Steinhardt (binding IDs), Biderman et al. "LoRA Learns Less and Forgets Less", Kornblith et al. (CKA).
- **Finding (6):** I did not search for literature on TDR/thermal throttling.
- **Specific numbers:** figures quoted above (cosine near ±1, ablation loss directions, 0.0016% parameters) come from abstracts or summaries of the HTML. Check them against the PDFs before citing numbers.
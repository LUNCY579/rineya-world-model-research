## Successor features vs. scalar goal-conditioned evaluators: prior work for positioning RCWM-T1

I checked every source below on arXiv, PMLR, NeurIPS/ICLR/ACL pages or Europe PMC. I read the full text for Barreto 2017, Lehnert 2017 and UVFA. For the other papers I relied on their abstract or HTML pages.

### A. The SF/GPI core: exact transfer to any linear goal, and what "span" means

1. **Successor Features for Transfer in RL.** Barreto et al., NeurIPS 2017, arXiv:1606.05312. https://arxiv.org/abs/1606.05312
   - Value splits into dynamics times goal: Q = ψ^π·w. For a fixed policy this gives exact evaluation of any w whose reward is linear in φ.
   - Theorem 2 bounds the GPI loss by the distance min_j‖w_i − w_j‖ to tasks already seen, plus an approximation error ε. Finding w is described as "a supervised learning problem."
   - **Relevance to us:** we select among 8 precommitted plans, which is a fixed policy library. So consequence features plus a linear readout is exactly SF policy evaluation followed by a max over policies. The distance-to-seen-w term only measures how good the plan library is, not how accurate the evaluation is. That predicts our out-of-span advantage (finding 4).

2. **Universal Successor Features Approximators (USFA).** Borsa et al., ICLR 2019, arXiv:1812.07626. https://arxiv.org/abs/1812.07626
   - Trained only on the canonical basis vectors of R⁴, then tested on mixed and negative-weight tasks such as [-1,-1,0,0].
   - "USFAs perform considerably better than the unstructured UVFAs... the former exploit the structure while the latter cannot." Negative rewards were "never experienced by the agents during training."
   - **Relevance:** the closest precedent for finding 4: a factorized model beats a scalar goal-conditioned model on goals outside the training span.

3. **Universal Value Function Approximators (UVFA).** Schaul et al., ICML 2015, PMLR 37. https://proceedings.mlr.press/v37/schaul15.html
   - The canonical scalar goal-conditioned evaluator.
   - Interpolation to held-out goals works "because the training set is broad enough to cover the different parts of goal space."
   - Extrapolation is weak: trained on goals in three rooms and tested on the fourth, it reaches only **24% ± 8% policy quality**.
   - **Relevance:** directly supports the pattern "about a tie in-span, scalar evaluator loses out-of-span."

4. **Transfer in Deep RL Using SFs and GPI.** Barreto et al., ICML 2018, arXiv:1901.10964. https://arxiv.org/abs/1901.10964
   - Relaxes the assumption that rewards are linear in a fixed feature set, and extends the guarantees to any tasks that differ only in reward.

5. **Fast RL with Generalized Policy Updates.** Barreto et al., PNAS 2020, doi:10.1073/pnas.1907370117 (checked via Europe PMC; the PNAS page returned 403).
   - "If the reward function of a task can be well approximated as a linear combination of the reward functions of tasks previously solved, we can reduce a reinforcement-learning problem to a simpler linear regression."
   - **Relevance:** this is the in-feature-span label-efficiency argument.

6. **Optimistic Linear Support and SFs (SFOLS).** Alegre et al., ICML 2022, PMLR 162:394-413, arXiv:2206.11326. https://proceedings.mlr.press/v162/alegre22a.html
   - Builds a policy set whose SFs form a convex coverage set, which gives optimal transfer to any linearly expressible task.
   - **Relevance:** a formal answer to "which plan library suffices for all w."

7. **Successor Feature Representations.** Reinke and Alameda-Pineda, TMLR 2023, arXiv:2110.15701. https://arxiv.org/abs/2110.15701
   - Removes the linear-reward restriction by modelling the discounted distribution of features.

**Terminology warning for the write-up.** In the literature "span" usually means *rewards in the feature span*. Our "out-of-span" means *goal weights outside the span of the scalar evaluator's training goals*, with every goal still linear in features. That second meaning matches the USFA and UVFA setups, not SFR or Barreto 2018. Define it explicitly.

### B. Limits of SFs

8. **Advantages and Limitations of SFs for Transfer.** Lehnert, Tellex, Littman, ICML 2017 Lifelong RL workshop, arXiv:1708.00102. https://arxiv.org/abs/1708.00102
   - "The learned representation is not transferrable between optimal policies"; SFs must be re-learned when the optimal policy changes.
   - **Relevance:** this does not apply to our fixed-plan evaluation. Say so, because reviewers may raise it.

9. **SFs Combine Elements of Model-Free and Model-Based RL.** Lehnert and Littman, JMLR 21(196), 2020. https://www.jmlr.org/beta/papers/v21/19-060.html
   - Learning to predict future reward outcomes "is equivalent to learning successor features."
   - **Relevance:** supports reading our consequence head as a linear-model / SF hybrid.

10. **Does Zero-Shot RL Exist?** Touati, Rapin, Ollivier, ICLR 2023, arXiv:2209.14935. https://arxiv.org/abs/2209.14935
    - SF performance depends heavily on how features are learned: Laplacian features work well, while autoencoder, transition-model and contrastive features are inconsistent.
    - Forward-backward representations reach 85% of supervised RL zero-shot.
    - **Relevance:** our features are simulator-defined (the oracle-feature case), which sidesteps this question. Flag it as a scope limit.

### C. Rollouts vs. single-pass "consequence" models (findings 1 and 2)

11. **Distributional SFs Enable Zero-Shot Policy Optimization (DiSPO).** Zhu, Wang, Han, Du, Gupta, NeurIPS 2024, arXiv:2403.06328. https://arxiv.org/abs/2403.06328
    - Motivated by "autoregressive model rollouts suffer from compounding error." Evaluation under a new reward becomes linear regression; at test time they relabel **10,000 transitions**.
    - Beats USFA, FB, MOPO, COMBO and GC-IQL on D4RL tasks (e.g. antmaze-medium-diverse: 631 vs USFA 394, COMBO 418).

12. **RaMP: Self-Supervised RL that Transfers using Random Features.** Chen et al., NeurIPS 2023, arXiv:2305.17250. https://arxiv.org/abs/2305.17250
    - Random-feature Q-basis plus regressed weights. Lower policy-evaluation error than model rollouts (Hopper 25.7 vs 50.2); "PETS ... suffers from compounding error." Online it collects 2,500 steps for the regression.
    - Notes that standard SFs "barely transfer" in several environments.

13. **Multi-Step GPI (h-GPI).** Alegre, Bazzan, Nowé, da Silva, NeurIPS 2023. https://proceedings.neurips.cc/paper_files/paper/2023/hash/77c7faab15002432ba1151e8d5cc389a-Abstract-Conference.html
    - Interpolates between model-free GPI and model-based planning through a horizon parameter h. Its lower bound is strictly better than GPI's, with bounds stated in terms of model and policy-library error.

**Relevance of C.** Finding 1 (end-to-end multi-step generation collapses) fits the compounding-error story. Finding 2 goes the other way: our model's serial self-rollout *beats* single-pass heads trained on hidden states. DiSPO and RaMP report the opposite ordering in continuous control. h-GPI is the best theoretical frame for why short-horizon rollouts (k=4) plus a linear readout can win when one-step model error is low relative to the error of the single-pass heads. Report this as a contrast with prior results, not as a replication.

### D. Language-model precedents

14. **SF-GEN: Successor Features for Efficient Multi-Subject Controlled Text Generation.** Cao, Fatemi, Cheung, Shabanian, ICML 2024, PMLR 235:5567-5583, arXiv:2311.04921. https://proceedings.mlr.press/v235/cao24a.html
    - The first use of SFs for text generation. The LLM's dynamics are decoupled from rewards, and each new subject "only requires solving a simple linear regression problem."
    - It uses one SF network for all subjects, where the Rect baseline needs one model per subject. It is "slightly behind" Rect on a single subject "due to the linearity constraint."
    - **Relevance:** the closest LM analogue of finding 4. Note that SF-GEN uses large label sets (e.g. ~2.1M sentiment-labelled sentences), not about 1k.

15. **FUDGE: Controlled Text Generation with Future Discriminators.** Yang and Klein, NAACL 2021, arXiv:2104.05218. https://aclanthology.org/2021.naacl-main.276
    - A scalar attribute predictor applied to partial sequences, which is the LM counterpart of our scalar evaluator.

16. **Decoding-Time LM Alignment with Multiple Objectives (MOD).** Shi et al., NeurIPS 2024, arXiv:2406.18853. https://arxiv.org/abs/2406.18853
    - Next-token prediction from a linear combination of objective-specific models, for any weighting. Linear goal weights at decoding time, but over policies rather than consequence features.

### What is already known, and where RCWM-T1 adds something

- **Known:**
  - Factorized (SF) evaluation transfers exactly to new linear goals.
  - Scalar goal-conditioned evaluators interpolate within the span of training goals but extrapolate poorly (UVFA 24%; USFA vs UVFA on negative and mixed w).
  - In-feature-span weights are cheap to fit by regression (PNAS 2020, SF-GEN, DiSPO, RaMP).
  - Our "about a tie in-span, win out-of-span" result is therefore the expected pattern, and should be framed as confirming it in LM hidden-state / text-transition models.
- **Not found in my search** (absence not proven): a comparison of factorized and scalar evaluators trained on an LM's own hidden states at matched label budgets (about 1.1k). The prior work above uses 2.5k–10k reward samples or millions of labels, and none matches the label budget across the two arms.
- **Contrasts with prior work:** self-rollout beating single-pass heads (finding 2) runs against DiSPO and RaMP. A symbolic learner with 4,000 labels winning on one family echoes Touati's point that the choice of features and representation dominates.
- **Not covered by these sources:** findings 3 (probe selectivity), 5 (mode vector) and 6 (TDR resets). Search the interpretability and steering literature separately for those.
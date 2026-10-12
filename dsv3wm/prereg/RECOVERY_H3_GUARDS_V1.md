# H3 development dynamics and behavior guards

Prospective source addition before any H3 guard model output. Existing teacher,
student-training and proposer contracts remain unchanged. The new evaluator
accepts the original full StudentConfig and actual completed artifact lineage;
it does not relabel C3/C4 as A3/A4 or infer a student's warm parent from its name.

Use the same fixed24-parent recovery DEV bank and the same six original-format
behavior fixtures. Dynamics uses greedy32 tokens, prompt512, h1/h2/h4 direct
responses and a four-step own-prediction rollout. Missing/malformed predictions
are not repaired with truth. Preserve parent/source-index/phase/step tags and
raw replies, then replay the causal schedule to check that the next state came
from a prediction. Report state, legality, termination and cumulative attempted
failure with their existing contracts. All dynamics and six behavior generations
must finish before scoring; source labels never become generator inputs.

Read out unchanged final A0/A2 warm parents first. These guard readouts are a new
instrument with explicit trace tags and measured decoder counters; previous
warm dynamics/behavior results remain unchanged. After qualified teacher corpus
and completed student/control training, freeze each actual output alongside its
own warm-parent guard result. Report paired losses by parent and behavior ID.
An already failed warm-parent fixture cannot demonstrate further loss of a
previously possessed capability. No six-case broad retention pass is assumed.

Use the frozen FP32/TF32-off unmerged parent/adapters, shared four-thread CPU
lane and RAM thresholds. Record physical decoder-hook positions/forward counts
separately from the legacy conservative position proxy. These are inference
counters, not measured FLOPs. Preserve partial raw traces and failed status on
interruption. New exclusive output paths and an actual-artifact/source/data/core
freeze are required; append the shared execution dependency closure before use.

The original H3 continuation thresholds remain those in its draft: valid greedy
proposal count, paired planning gain, no lost matched learned-rerank success,
no unexplained behavior regression and ID h4 net loss at most2/12. These guards
are necessary measurements, not new evidence that a student passes. Calibration,
multi-seed efficacy, broad retention and independent confirmation remain separate.

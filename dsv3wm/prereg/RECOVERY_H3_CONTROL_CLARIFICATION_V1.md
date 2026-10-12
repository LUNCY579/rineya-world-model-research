# H3 continuation-control clarification v1

Prospective clarification before any native H3 student/control outcome. Preserve
the teacher-pinned `RECOVERY_H3_DRAFT_V1.md` and its freeze byte-identically.
This new record specifies the previously missing executable C3/C4 forward slot;
it does not revise teacher labels, candidate ranking or any earlier outcome.

Every study/control starts from its prescribed final32 warm artifact: A3/C3 from
A2, A4/C4 from A0. They use the same qualified positive corpus for admission,
the same source TRAIN trajectory order, the same32 optimizer updates, batch1,
FP32/unmerged rank16/all original projections/dropout.05, and declared AdamW
recipe. Controls do not consume positive plan targets in their training losses.

- A3: h1 + .5h2 + .5h4 + verified-plan CE, four forwards per update.
- C3: h1 + .5h2 + .5h4 + an additional h1 CE, four forwards per update.
- A4: h1 + verified-plan CE, two forwards per update.
- C4: h1 + an additional h1 CE, two forwards per update.

The control's additional h1 uses the **same current trajectory row, exact prompt
and exact target** as that step's ordinary h1. It runs a separate real forward
with the same independently fixed `... +9` dropout seed used for the plan slot.
It contributes unit-weight loss/gradients; it is neither a discarded forward
nor zero-weight compute. Both slot types use per-response-token mean CE.

The explicit configuration contract is `plan_weight=1` with
`control_extra_one_step=false` for A3/A4, or `plan_weight=0` with
`control_extra_one_step=true` for C3/C4. Zero plan weight without an explicit
extra-h1 control is refused. C3/C4 may use their A3/A4 architecture arm names in
typed configs, but records and comparisons must identify them as continuation
controls using these fields. Pure warm-baseline readout can retain study-config
fields while loading the unchanged warm artifact; it performs no optimization.

Thus A3/C3 each have128 nominal post-warm decoder forwards and A4/C4 each64,
with32 optimizer updates in all cases. Plan-prompt length, response length,
vocabulary work, labels and effective supervision differ from a repeated
one-step row. These are nominal forward/update controls, **not** equal measured
FLOPs or equal-label experiments. Report actual input/supervised tokens,
per-slot losses, slot source, forwards, run time, memory and parameter-token
proxies with their limitations. Keep teacher search/verification costs visible.

No native execution is authorized by this clarification alone: source/config/
corpus/warm/qualification inputs still require the prospective student freeze.
No H1/H2 gate is bypassed and no efficacy/retention claim follows from CPU tests
of this implementation.

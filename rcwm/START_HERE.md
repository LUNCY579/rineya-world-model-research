# Rineya Consequence World Model / RCWM-001

**Research and implementation bundle — 5 October 2026**

The proposal: learn what executable actions cause as reusable consequence representations, then
interpret those consequences through goal-conditioned utility. Test the benefit against a matched
scalar trajectory evaluator before adding clustering, native adapters or imagined RL.

## Start with these resources

| Resource | Purpose |
|---|---|
| [Main report](report/Rineya_RCWM_Research_Plan.pdf) | 31-page research, architecture, experiment and roadmap report |
| [Editable master](report/Rineya_RCWM_Research_Plan.md) | Full Markdown source; individual chapters are also included |
| [Offline HTML](report/Rineya_RCWM_Research_Plan.html) | Searchable browser edition with embedded CSS and MathML |
| [Annotated reading](research/ANNOTATED_READING.md) | 43 primary sources, findings, implementation uses and limitations |
| [Architecture contract](specs/ARCHITECTURE_CONTRACT.md) | Modules, inputs, outputs and native-versus-sidecar boundaries |
| [Implementation handoff](roadmap/IMPLEMENTATION_HANDOFF_PROMPT.md) | Copy-ready, gated engineering instructions |
| [Reference package](reference/README.md) | Executable CPU mechanics; no Rineya weights required |
| [Implementation status](IMPLEMENTATION_STATUS.md) | Exactly what is supplied, tested and still planned |

## Suggested reading order

Read report sections 1–5, then 7–10 and 13. Read the masking and data contracts before writing a
native integration. Use the annotated sources S02, S04, S05, S07 and S09 to understand the closest
prior art. Check the run/evaluation templates before opening any sealed evaluation data.

## What was actually validated

60 CPU tests passed. Three JSON schemas, four illustrative trajectory records, seven configurations
and the 43-entry source register passed the bundled resource checks. A tiny supervised synthetic
Transformer training run was executed. **Its low average error does not qualify its per-plan fidelity**:
a known prerequisite-chain prediction failure is preserved in the report and validation notes.
No native Rineya model, GPU, long-context training, retention evaluation or remote repository action
was executed. No first-in-LLM or general intelligence breakthrough is claimed.

## Run the reference checks

From a short local extraction path (for example `C:\Rineya\RCWM` on Windows), create a Python virtual
environment and install appropriate dependencies. From `reference`:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m rcwm.demo --out ../validation/reproduced_demo --steps 250 --seed 17
```

The test extra includes JSON Schema and YAML tooling used by resource tests. Dependency installation
may contact package repositories; the reference package itself makes no network calls or downloads.
Observed CPU versions are in `validation/environment_observed.json`; they are not a GPU installer.

From the bundle root:

```bash
python tools/validate_resources.py
python tools/verify_manifest.py
```

The YAML files are declarative planning/run contracts, not an implemented native training CLI.
The CPU YAML records the actual demo command. Null native hashes, budgets and promotion thresholds
mean **blocked**, not unlimited permission. Report rebuilding requires separately installed Pandoc,
XeLaTeX and the named fonts; font files are not distributed.

## Directory map

`report/` contains the PDF, editable Markdown/LaTeX and offline HTML.
`research/` contains annotations, CSV/JSON source metadata, BibTeX and claim-evidence mapping.
`specs/` contains implementation, data, masks, training, security, evaluation and native runbooks.
`configs/`, `schemas/` and `examples/` contain machine-readable contracts and synthetic examples.
`reference/` contains the small original Python package and its tests.
`roadmap/` contains 21 gated experiment entries and an implementation prompt.
`templates/` contains experiment, run, gate, decision, model and dataset documentation templates.
`validation/` contains actual scoped test/demo/resource evidence and known limitations.
`tools/` contains build, validation and integrity utilities.

Cited third-party papers, datasets and code are linked, not copied into the archive. Review licenses and
pin revisions before reuse. The supplied `.pt` file is a tiny original synthetic demonstration checkpoint,
not Rineya. The final manifest excludes itself from hashing.

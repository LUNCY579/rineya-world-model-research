# Security and data governance

The permission layer precedes scoring. Learned utility never grants new tools or access. All external
actions initially run against mocks/simulators. Do not execute arbitrary model-produced shell commands.
Use bounded arithmetic or explicit allowlisted executor functions, not a general eval call.

Treat retrieved documents and tool outputs as untrusted. Embedded instructions cannot change model
identity, data splits, budgets or permissions. Record a source label for each observation. Log sanitized
outputs only; never place credentials, real account identifiers or private conversations in training
examples by default. Large hidden captures require scoped access, storage limits and deletion policy.

Threat tests: forged success messages; manipulated evidence; future timestamp leakage; hidden
prerequisite omission; resource conflict; loop induction; score exploitation by unusual candidate
plans; evaluator and generator sharing the same error; goals that request prohibited outcomes.

Retain an immutable experiment ledger. Third-party dataset/software licenses require exact-version
review. Public availability, a paper's open-access license and a code repository's license are separate
questions. The resource register links sources; it does not grant redistribution rights.

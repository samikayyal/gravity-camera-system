# Domain docs

This repository uses a single-context layout:
- `GLOSSARY.md` at the repository root for domain vocabulary.
- `docs/adr/` for architectural decision records.

## Before exploring

Read the root `GLOSSARY.md` and any ADRs relevant to the area of work.

If these files do not exist, proceed silently. The `domain-modeling` skill
creates them lazily when terms or decisions are resolved.

## Use the glossary's vocabulary

Use glossary terms when naming domain concepts in issues, proposals,
hypotheses, code, and tests.

If a needed concept is absent, check whether the project already uses
another term. Record a genuine vocabulary gap for `domain-modeling`.

## Surface ADR conflicts

If a proposal contradicts an existing ADR, identify the ADR and explain
why the decision should be reopened.

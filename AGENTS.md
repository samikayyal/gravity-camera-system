# Repository instructions

## Python

Use `uv` for all Python work:
- Manage dependencies with `uv add <package>`.
- Run Python files with `uv run <file.py>`.
- Run Python tools with `uv run <tool>`.

## Scratch files

Create one-off helper scripts, bundle builders, temporary notes, drafts, and
review images under `.scratch/`. This directory is local and ignored by Git.
Keep maintained program code, tests, configuration, and durable documentation
in their normal project locations; keep training outputs and checkpoints in `runs/`.

## Agent skills

### Issue tracker

Track issues and specs in GitHub Issues. Before reading, creating, or updating tickets, read `docs/agents/issue-tracker.md`.

### Triage labels

Use the five default triage labels. Before triaging or applying a triage label, read `docs/agents/triage-labels.md`.

### Domain docs

Use a single-context layout: root `GLOSSARY.md` and `docs/adr/`. Before exploring the codebase, read `docs/agents/domain.md`.

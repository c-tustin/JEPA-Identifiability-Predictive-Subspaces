# Standalone layout check

2026-10-28: this branch was rebuilt directly on main at `8a885a6`, extracting only the original-paper reproduction from source commit `07c7e42`.

- Code and documentation now live in the top-level `predictive-closure-reproduction/` folder.
- No earlier experiment branch is an ancestor of the new study commit. Main's existing files are preserved; the root README describes this branch.
- All eight source/planner/runtime/checkpoint tests pass from the relocated folder, including the separate-process resume test. The pinned upstream checkout and verified CIFAR data were reused for testing.
- No scientific training budget, result, checkpoint or protocol was changed by this move. Recorded execution evidence remains historical; no full paper result is complete.
- The original local checkout and its paused-run files were left untouched.

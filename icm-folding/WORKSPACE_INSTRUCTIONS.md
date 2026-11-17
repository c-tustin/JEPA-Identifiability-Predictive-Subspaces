# ICM folding work

The user wants `icm-folding/` to stay current as new code and findings arrive.

- Put new code and experiment directories about ICM policy occupancy, representation folding, fold mechanisms, prevention, or repair directly under `icm-folding/`.
- When related code is created or discovered elsewhere during a task, integrate it into this folder in the same task. Consolidate compatible code into the relevant experiment directory, preserving provenance and resolving imports and output paths. For existing studies with path dependencies, retain their original location and expose them through a relative symbolic link and an index entry rather than breaking working experiments or duplicating checkpoints.
- Existing source-folder links already expose changes to their target code immediately; do not copy those targets into competing versions.
- Update `icm-folding/README.md` for new code, protocols, figures, and results. Update `icm-folding/PROGRESS.md` with substantive changes, dates, evidence, and remaining work. Label unrun code and pending results accurately; preserve the distinction between empirical fold checks and analytic guarantees.
- Verify affected imports or appropriate checks and validate new document and source-folder links. Preserve unrelated work and nested repositories.
- Publish updates directly to `main` inside their research folders. Do not create Codex-named folders or feature branches for this workspace.

Apply this during future work in this workspace. This instruction does not itself create a background monitor or authorize merging unrelated Git branches.

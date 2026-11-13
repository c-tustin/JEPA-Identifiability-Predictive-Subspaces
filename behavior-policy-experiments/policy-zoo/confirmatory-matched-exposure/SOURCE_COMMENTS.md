# Explanatory source comments and the frozen experiment

Comments added on November 13, 2026 explain policy actions at common reset
states, alignment versus SIGReg inputs, shared versus policy-specific predictor
data, and the paired TOST hypotheses. This is a comment-only Python change:
the parsed syntax trees, including existing docstrings, are unchanged.

The experiment hashes raw source bytes, so comments intentionally change the
computed signature. Existing locks, results, checkpoints and verification
records have not been rewritten. The signature check remains strict and will
reject historical artifacts from this annotated checkout. Do not run `--freeze`
again or replace the saved lock to accept the comments.

To reproduce the saved experiment, use the unchanged historical commit in a
separate checkout, with the dependencies and commands in the study README:

```sh
git worktree add --detach ../kempner-frozen-replay 1b6fa039094324960f15b802b764cd263a893ffb
```

That commit predates these annotations and preserves the source bytes used by
the saved signature. The annotated files are for reading the same implementation;
they do not define a new trained cohort or new scientific results.

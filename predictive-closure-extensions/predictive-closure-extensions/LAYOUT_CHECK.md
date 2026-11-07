# Standalone layout check

2026-10-28: this branch was rebuilt directly on main at `8a885a6`, extracting only the Predictive Closure study from source commit `101956d`.

- All study files now live in top-level `predictive-closure-extensions/`. No other experimental branch is needed.
- The six-input scalar encoder is included in `models.py`; its definition is structurally identical to the earlier encoder. All 20 existing checkpoints load and give exactly the same outputs as the earlier class on the parity-check inputs.
- `sigreg.py` is byte-for-byte identical to the implementation used by the original study; its license notice is retained.
- All **29 saved output files**, including 20 model checkpoints, are byte-for-byte unchanged. No model was retrained and no saved numerical result was regenerated.
- All **10 standalone scientific tests pass** after the move. Earlier 68-test counts included unrelated experiments and are not the standalone test count.
- The root README describes this branch. The practice folder is intentionally excluded; main's license and original research proposal are retained. Earlier experimental branches are not ancestors of the new study commit.

# Third-party notices

`SIGReg1D` in `behavior-policy-experiments/identifiability/sigreg.py` (also exported by `reproduce.py`) adapts the scalar form of SIGReg in the following MIT-licensed project:

- Project: [LeJEPA Identifiability](https://github.com/klindtlab/lejepa-identifiability)
- Source: [`experiments/lejepa_id/losses.py`](https://github.com/klindtlab/lejepa-identifiability/blob/main/experiments/lejepa_id/losses.py)
- Copyright holder: David Klindt

The reference expression used by the numerical checks is also based on that implementation. The upstream license follows.

```text
MIT License

Copyright (c) 2026 David Klindt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## Curiosity-driven exploration (Pathak et al., 2017)

The ICM and actor–critic implementation in `behavior-policy-experiments/curiosity/experiment.py` is a source-derived PyTorch port of [pathak22/noreward-rl](https://github.com/pathak22/noreward-rl), commit `3e220c2177fc253916f12d980957fc40579d577a`. The relevant unmodified sources and complete BSD-style and OpenAI MIT notices are retained under [the upstream snapshot](behavior-policy-experiments/curiosity/upstream/noreward-rl/LICENSE). Those notices apply to the snapshot and adapted portions; the repository's MIT license does not replace them. The curiosity README documents exactly what is ported and changed.

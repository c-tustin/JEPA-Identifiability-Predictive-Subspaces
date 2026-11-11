# Data and source attribution

Data: Chen, S. (2015). *Beijing PM2.5* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5JS49. Licensed CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/. Original hourly observations are
downloaded unmodified. Our derived inputs apply log1p to pollution, cumulative
wind, snow and rain, one-hot encode wind direction, and standardize using training
years. Dataset files and model outputs are ignored by Git.

Theory: Levy, M., Lidayan, A., and Goldberg, Y. (2026).
*Predictive Closure: A Set-Level Characterization of JEPA's Feature Selection*.
https://arxiv.org/abs/2610.27364v1.

The SIGReg expression in models.py adapts jepa/losses.py from the official
https://github.com/Mosh0110/predictive-closure repository, pinned commit
2964fa8f737fa43871d47928edc9aac16fa9298f. Other pipeline code is new.
The upstream notice follows:

MIT License

Copyright (c) 2026 Predictive Closure contributors

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

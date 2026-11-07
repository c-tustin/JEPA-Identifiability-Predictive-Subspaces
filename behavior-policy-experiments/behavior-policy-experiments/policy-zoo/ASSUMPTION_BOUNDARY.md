# Are these experiments within Klindt's theorem?

**The fixed-OU control matches the theoretical data setup. The physical-policy zoo tests departures from it.**

The paper's forward setting uses stationary Gaussian latent pairs with positive-correlation OU transitions. Its approximate result still assumes that world; finite optimization is a separate issue. [Paper, §§3, 5 and 7](https://arxiv.org/html/2605.26379v1).

| Ingredient | Fixed-OU reference here | New physical controllers |
| --- | --- | --- |
| One latent and one encoder output | Yes: log-distance and one scalar | Preserved |
| Informative observation | Fixed triangle, x(d)=V/d | Preserved; no extra world coordinates |
| Gaussian log-distance | Unbounded Gaussian samples | Bounded physical states; at best truncated/approximate Gaussian |
| OU pair law | Explicit Gaussian conditional transition | Discrete physical steps, feedback, rejections or policy memory |
| Fixed pair-generating distribution | Fixed source | Handover changes the sampled mixture over encoder updates |
| Ideal optimization | Approximated by finite SIGReg/Adam training | Also approximated; no new optimization guarantee |

Making a controller's visitation histogram Gaussian-looking does **not** make its conditional transition Gaussian OU. Our Metropolis controller can target a distribution on a physical-step lattice; its next state still comes from discrete actions. That is why it is an extension rather than an exact realization of the OU control.

The old positive-correlation Gaussian filtered sources match the one-pair Gaussian law in their ideal stationary setting. The negative-correlation sources and the physical policies depart from that control. Their empirical behavior must be evaluated separately.

The fixed-OU run is itself a numerical experiment: finite samples, a restricted neural encoder, finite-frequency SIGReg and 4,000 updates. We have not certified a population error bound for each run. Nor does a high R² establish correct action-conditioned dynamics or planning ability.

**A nuance for the curricula:** their final OU-only phase returns to the OU pair law. If the final encoder satisfies the theorem's optimality/representation conditions under that law, its earlier pretraining does not prevent the theorem from applying. We have not established those conditions from R² alone. The paper does not say that OU-only initialization and training must be the fastest route to a good representation in a fixed number of updates.

Consequently, a new policy/OU curriculum beating the fixed-OU run means **better recovery under this finite training protocol**. It is not a counterexample to the paper, a proof for arbitrary behavior policies, or a new identifiability guarantee. The new controllers remain useful tests of when the recipe works outside its ideal setting.

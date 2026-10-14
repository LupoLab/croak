# API source review

All 50 original modules included by the five API domain pages were reviewed in source.
This inventory counts the 527 original module/class/function docstrings read;
it includes some nested helpers beyond the rendered public surface. Rendered
attribute comments were checked separately. The previously unrendered public
`croak.delay_origin` module was also reviewed (five docstrings) and added to the
retrieval API, bringing the rendered module total to 51. Existing adequate descriptions were
retained; changed prose remains in the source docstrings, not duplicated in the
manual. Public call signatures, defaults and runtime statements are unchanged.

The main corrections concern objective normalisation, supported model options,
field planes, chromatic focal filtering, coherent collection, density Jacobians,
noise and covariance assumptions, temporal/spectral phase alignment and session
reproducibility. Historical performance claims and implementation chronology
were removed from user-facing API descriptions. Numerical assumptions, equations
and user contracts remain; contributor interfaces stay discoverable in the
API and developer guide.

| Module | Original docstrings reviewed |
|---|---:|
| `croak.retrieve` | 3 |
| `croak.solver` | 13 |
| `croak.copra` | 7 |
| `croak.copra_jax` | 10 |
| `croak.lbfgs` | 3 |
| `croak.lbfgs_hand` | 4 |
| `croak.lbfgs_ad` | 5 |
| `croak.optimistix_lbfgs` | 4 |
| `croak.lm` | 5 |
| `croak.optimistix_lm` | 6 |
| `croak.cmaes` | 5 |
| `croak.warm_lbfgs` | 2 |
| `croak.result` | 10 |
| `croak.progress` | 7 |
| `croak.metrics` | 6 |
| `croak.io` | 30 |
| `croak.preprocess` | 17 |
| `croak.pipeline` | 3 |
| `croak.processing` | 20 |
| `croak.truth_metrics` | 7 |
| `croak.marginal_checks` | 9 |
| `croak.dispersion` | 9 |
| `croak.mirrors` | 7 |
| `croak.refractive_db` | 6 |
| `croak.save` | 7 |
| `croak.uncertainty` | 22 |
| `croak.covariance` | 10 |
| `croak.session.params` | 10 |
| `croak.session.options` | 7 |
| `croak.session.pipeline` | 20 |
| `croak.session.dispersion` | 4 |
| `croak.session.engine` | 3 |
| `croak.session.retarget` | 2 |
| `croak.cli` | 2 |
| `croak.scripting` | 2 |
| `croak.grid` | 10 |
| `croak.pulses` | 35 |
| `croak.interactions` | 20 |
| `croak.forward` | 8 |
| `croak.smearing` | 12 |
| `croak.focal` | 13 |
| `croak.collection` | 11 |
| `croak.materials` | 8 |
| `croak.gases` | 2 |
| `croak.maths` | 9 |
| `croak.constants` | 1 |
| `croak.forward_jax` | 12 |
| `croak.metrics_jax` | 8 |
| `croak._jax_pulse` | 14 |
| `croak.plotting` | 77 |
| `croak.delay_origin` (new API entry) | 5 |

Package ASTs were compared with the approved layout commit after removing only
docstring expressions. They match, establishing that runtime statements,
signatures, decorators and defaults did not change. Formatting, lint, type and
repository tests provide separate checks; final results are in `verification.md`.

# Retrieval and numerical methods

Retrieval adjusts a trial pulse until its predicted trace matches the data.
croak poses this as (regularised) nonlinear least squares and offers ten solvers
behind one {func}`~croak.retrieve.retrieve` call. This chapter covers which
solver to use, what the objective is, how gradients are computed, how
regularisation and fitted instrument parameters enter, and how to judge
convergence.

## Which page do I need?

| I want to… | Read |
|---|---|
| pick a solver, set restarts and stopping criteria | [Choosing a solver](../howto/solver_selection.md) |
| know exactly what is minimised and how the trace scale is fitted | [Retrieval as nonlinear least squares](../explanation/retrieval_theory.md) |
| understand how COPRA, L-BFGS, LM and CMA-ES work | [Retrieval algorithms](../explanation/algorithms.md) |
| see the analytic adjoint and the automatic-differentiation route | [Gradients](../explanation/gradients.md) |
| stabilise a noisy or ill-posed retrieval, or use a measured spectrum | [Regularisation](../howto/regularisation.md) |
| fit the medium thickness, delay zero or smearing width | [Fitting the medium thickness and delay-zero](../howto/fitting_thickness_tau0.md) |

```{toctree}
:maxdepth: 1

../howto/solver_selection
../explanation/retrieval_theory
../explanation/algorithms
../explanation/gradients
../howto/regularisation
../howto/fitting_thickness_tau0
```

## Key concepts

The trace error $R$ is the weighted RMS residual after an optimal intensity
scale $\mu$ (one per frequency with `R_omega=True`). The objective is
non-convex, so the result depends on the start: the default `warm-lbfgs` starts
the autodiff L-BFGS solver from a COPRA sweep, and several random restarts are
the standard check. The penalty-capable solvers add smoothness and
spectral-match penalties by default (`reg_spectrum=0.01`, `reg_amp=0.03`);
`reg_spectrum` binds only when a measured spectrum is supplied.

## Main entry points

- {func}`~croak.retrieve.retrieve`: retrieve from a trace on a grid;
  {data}`~croak.retrieve.ALGORITHMS` lists the solvers.
- {func}`~croak.pipeline.retrieve_from_tracedata`: retrieve from a preprocessed
  {class}`~croak.preprocess.TraceData`.
- {class}`~croak.result.RetrievalResult`: the uniform result object.
- {data}`~croak.solver.REG_DEFAULTS`: the default penalty weights.

The [algorithm comparison](../tutorials/02_choosing_an_algorithm.md) runs the
solvers side by side. After a run, continue with
[interpreting results](results.md).

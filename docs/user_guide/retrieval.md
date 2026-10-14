# Retrieval and numerical methods

Retrieval adjusts a trial pulse to reduce the discrepancy between the data and
the predicted trace. Choose a solver that supports your model and constraints,
then examine convergence and sensitivity to the starting field. Different
initial guesses can reach different solutions in this nonlinear problem.

The practical solver guide is followed by the objective, algorithms, gradient
derivations, regularisation, and the fitting of instrument parameters.

```{toctree}
:maxdepth: 1

../howto/solver_selection
../explanation/retrieval_theory
../explanation/algorithms
../explanation/gradients
../howto/regularisation
../howto/fitting_thickness_tau0
```

Return to [your first retrieval](../tutorials/01_first_retrieval.md) for the
basic call, or explore [the algorithm comparison](../tutorials/02_choosing_an_algorithm.md).
Look up parameters in the [retrieval API](../reference/api/retrieval.md).
After a run, continue with [interpreting results](results.md).

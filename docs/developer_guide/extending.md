# Extending interactions and solvers

An extension must specify both the physical model it implements and the
interfaces through which it is used. Start with the existing implementation
closest to the intended behaviour, and state unsupported combinations explicitly.

## Interactions

The interaction defines the nonlinear signal and its derivative convention.
The existing [interaction extension example](../explanation/interactions.md#extending-to-a-new-geometry)
shows the interface. Check the carrier-frequency relation, conjugations, and
arm timing against the physical process. Support in the thin-medium model does
not establish support for dispersive propagation or spatial mixtures.

## Solvers

The retrieval wrapper selects a solver through
{data}`croak.retrieve.ALGORITHMS`. See the
[retrieval API](../reference/api/retrieval.md) for the base interface, result
object, and progress reporting. An added solver must expose its supported
parameters consistently and identify which forward-model and regularisation
options it accepts.

Check derivatives and numerical behaviour using the approach in
[numerical validation](validation.md). Document scientific behaviour in the
[user guide](../user_guide/retrieval.md) as well as the API.

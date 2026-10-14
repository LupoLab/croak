# Extending interactions and solvers

An extension must specify both the physical model it implements and the
interfaces through which it is used. Start with the existing implementation
closest to the intended behaviour, and state unsupported combinations explicitly.

## Interactions

Define the signal, carrier relation and derivative convention together. Test the
local derivative with complex perturbations before checking a full trace. The
[user guide](../explanation/interactions.md) gives the implemented physical kernels.

### Define a local kernel

A new geometry is a small subclass:

```python
import numpy as np
from croak.interactions import Interaction

class THG(Interaction):           # third-harmonic generation, s = E^2 G
    name = "thg"
    omega0_scale = 3.0            # signal near 3*omega0

    def signal(self, test, gate):
        return test**2 * gate

    def signal_adjoint(self, test, gate, s_bar):
        test_bar = 2.0 * np.conj(test) * np.conj(gate) * s_bar
        gate_bar = np.conj(test**2) * s_bar
        return test_bar, gate_bar
```

This defines a thin-medium local kernel and its analytic derivative. Register an
instance in {data}`~croak.interactions.INTERACTIONS`, or pass an instance to a
NumPy interface that accepts it. Registration alone does **not** add a new JAX
kernel: the JAX forward implementation dispatches supported names explicitly.
Implement and test that path before advertising support in AD solvers. Automatic
differentiation can differentiate a JAX kernel once it is expressed there; it
cannot infer the implementation from an arbitrary NumPy subclass.

To support [geometric smearing](../explanation/forward_model.md#geometric-smearing) as well, add
{meth}`~croak.interactions.Interaction.smeared_shifts` and
{meth}`~croak.interactions.Interaction.smeared_signal`: the same signal written as a
product of *three* separately shifted replicas, because the factors that coincide
in the 1-D model acquire different arrival times across the focal spot. PG and SD
implement them; a geometry without a three-arm form (SHG) simply does not, and the
forward model raises if a kernel is supplied.


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

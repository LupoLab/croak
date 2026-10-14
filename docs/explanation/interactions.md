# Interactions

Choose an **interaction** to describe how the overlapping pulse replicas generate
the nonlinear signal. This choice determines the trace carrier, sensitivity to
phase and available ambiguities. The [first tutorial](../tutorials/01_first_retrieval.md)
uses SHG; the [dispersive tutorial](../tutorials/03_dispersive_retrieval.md) uses PG.
The kernels below describe the instantaneous local response; propagation and
collection are developed in the [forward model](forward_model.md).

Each interaction takes a *test* field $E$ (undelayed) and a *gate* field $G$
(delayed by $\tau$), both in the time domain, and returns the signal $s(t)$.

## The three geometries

| Class | `name` | Signal $s$ | $\omega_0$ scale |
|-------|--------|-----------|------------------|
| {class}`~croak.interactions.SHG` | `"shg"` | $E\,G$ | 2.0 |
| {class}`~croak.interactions.SD` | `"sd"` | $E^2\,G^*$ | 1.0 |
| {class}`~croak.interactions.PG` | `"pg"` | $E\,\lvert G\rvert^2$ | 1.0 |

`omega0_scale` records where the signal sits relative to the pulse carrier: the
SHG signal is generated near $2\omega_0$, while SD and PG stay at the fundamental
$\omega_0$. This single number has two consequences:

- it sets the wavelength axis of an SHG-FROG trace (handled in
  [Grids and filtering](../howto/grids_and_filtering.md));
- it determines which geometries support **dispersive propagation**. The slab
  propagation constant $\beta(\omega)$ is built at the fundamental carrier, so it
  is only physically consistent for signals that stay there. **Dispersive
  propagation is therefore supported for PG and SD, but not SHG** — croak raises a
  `ValueError` if you request a dispersive SHG model.

```{admonition} TG-FROG uses the PG kernel
:class: tip
Transient-grating FROG — the geometry demonstrated in the
[companion paper](validation.md) — has the third-order signal
$E_1 E_2^* E_3$ with three replicas of the pulse. When the gate replicas coincide in time and share the same local field, this
reduces to $E\,\lvert G\rvert^2$, i.e. the **PG** kernel. Use
`interaction="pg"` for TG-FROG data.
```

## Resolving an interaction

Anywhere an interaction is expected you may pass its name (case-insensitive) or an
instance; {func}`~croak.interactions.get_interaction` resolves both, and
{data}`~croak.interactions.INTERACTIONS` is the name→instance registry.

```python
from croak.interactions import get_interaction, PG
get_interaction("PG")    # -> PG()
get_interaction(PG())    # -> the same instance, unchanged
```

## The adjoint convention

croak supports automatic differentiation and analytic adjoints
(see [Gradients](gradients.md)). The analytic interaction interface provides
`signal(test, gate)` but also `signal_adjoint(test, gate, s_bar)`, which
back-propagates a cotangent through the nonlinearity.

The convention is **Wirtinger**: the cotangent of a complex variable $v$ is

```{math}
\bar v \;\equiv\; \frac{\partial L}{\partial v^*},
```

which is the quantity gradient descent on a real loss $L$ actually needs. Given
the output cotangent $\bar s$, `signal_adjoint` returns $(\bar E, \bar G)$.

For example, polarisation gating $s = E\lvert G\rvert^2$ has

```{math}
\bar E = \lvert G\rvert^2\,\bar s,
\qquad
\bar G = 2\,G\,\operatorname{Re}\!\bigl(E^*\,\bar s\bigr),
```

which is exactly what {meth}`PG.signal_adjoint <croak.interactions.PG.signal_adjoint>`
implements. The holomorphic SHG kernel and the mixed-holomorphy SD kernel have
similarly compact adjoints; see the [Gradients](gradients.md) page for the full
set and the derivation.

## Extending to a new geometry

Contributor interfaces and backend support are described in
[extending interactions and solvers](../developer_guide/extending.md#interactions).
The signal and adjoint equations above remain the physical reference for users.

## API

See {mod}`croak.interactions` in the [API reference](../reference/api/index.md).

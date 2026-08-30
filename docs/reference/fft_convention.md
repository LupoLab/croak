# FFT convention

Subtle factor-of-$2\pi$ and ordering bugs are the bane of pulse-retrieval code.
croak fixes one convention and applies it everywhere; this page records it.

## The continuous transform

croak uses the **physics convention**: the $1/2\pi$ lives on the inverse transform
and the forward kernel is $e^{+i\omega t}$.

```{math}
\tilde E(\omega) = \int E(t)\, e^{+i\omega t}\,\mathrm{d}t,
\qquad
E(t) = \frac{1}{2\pi}\int \tilde E(\omega)\, e^{-i\omega t}\,\mathrm{d}\omega.
```

This differs from the engineering convention (opposite sign) and the symmetric
convention ($1/\sqrt{2\pi}$ on both); mixing them is a classic source of a flipped
or mis-scaled spectrum.

## The discrete transform

On a uniform grid the integrals become Riemann sums, so
{meth}`Grid.fft <croak.grid.Grid.fft>` and {meth}`Grid.ifft <croak.grid.Grid.ifft>`
carry the grid spacings:

```{math}
\tilde E \approx \Delta t \sum E\, e^{+i\omega t},
\qquad
E \approx \frac{\Delta\omega}{2\pi} \sum \tilde E\, e^{-i\omega t}.
```

These operate on **centred** arrays (zero frequency in the middle): each is an
`ifftshift` → transform → `fftshift` sandwich with the appropriate scale. Because
the scale factors are physical, `g.fft` followed by `g.ifft` is the identity and
the transform approximates the continuous integral (Parseval's energy relation
holds with the $1/2\pi$).

## Raw transforms

The [forward model](../explanation/forward_model.md) runs in raw DFT-bin order for
speed, using **unnormalised, unshifted** transforms. These are exposed as
{func}`croak.grid.raw_fft` (`scipy.fft.fft`) and {func}`croak.grid.raw_ifft`
(`scipy.fft.ifft`, with the $1/N$). Their adjoints — needed for the
[Wirtinger gradient](../explanation/gradients.md) — are:

- adjoint of `raw_fft` is $N \times$ `raw_ifft`;
- adjoint of `raw_ifft` is $\tfrac1N \times$ `raw_fft`.

These appear directly in {meth}`ForwardModel.adjoint_single
<croak.forward.ForwardModel.adjoint_single>`.

## Practical implications

- Pass **centred** arrays to `Grid.fft`/`Grid.ifft` and to every public function;
  croak shifts to DFT-bin order internally where needed.
- A delay $\tau$ is a spectral phase ramp $e^{+i\omega\tau}$ in the raw domain
  (see {meth}`Grid.tshift <croak.grid.Grid.tshift>`), matching the forward kernel
  sign above.
- All transforms use {mod}`scipy.fft`.

## See also

- [Conventions](conventions.md) — units, axes and the baseband grid.
- {class}`croak.grid.Grid` in the [API reference](api/index.md).

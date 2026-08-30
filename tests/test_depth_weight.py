"""Generation-envelope weights (``depth_weight``) on the depth quadrature."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from croak import Grid, retrieve
from croak.forward_jax import make_param_trace_fn
from croak.pulses import gaussian_pulse

C_LIGHT = 299_792_458.0
OMEGA0 = 2 * np.pi * C_LIGHT / 260e-9


@pytest.fixture
def setup():
    """A 1 fs DUV pulse and a 40 um dispersive-slab configuration."""
    g = Grid(256, dt=0.5e-15)
    ew = gaussian_pulse(g, 1.0e-15)
    delays = np.linspace(-20e-15, 20e-15, 41)
    kwargs = dict(material="SiO2", thickness=40e-6, npoints=40, omega0=OMEGA0)
    return g, ew, delays, kwargs


def envelope(npoints: int, z_r_um: float = 20.0) -> np.ndarray:
    """A Gaussian-crossing-like complex envelope over the node ladder."""
    u = (np.arange(npoints) + 0.5) / npoints * 2.0 - 1.0  # ~centered slab
    u = u * (20.0 / z_r_um)
    return (1.0 + u**2) ** -0.5 * np.exp(1j * np.arctan(u))


def test_uniform_weight_is_a_noop(setup):
    g, ew, delays, kw = setup
    base = make_param_trace_fn(g.omega, delays, "pg", **kw)
    ones = make_param_trace_fn(
        g.omega, delays, "pg", depth_weight=np.ones(kw["npoints"]), **kw
    )
    a = np.asarray(base(jnp.asarray(ew), kw["thickness"], 0.0))
    b = np.asarray(ones(jnp.asarray(ew), kw["thickness"], 0.0))
    np.testing.assert_allclose(b, a, rtol=0.0, atol=1e-14 * a.max())


def test_envelope_changes_the_trace(setup):
    g, ew, delays, kw = setup
    base = make_param_trace_fn(g.omega, delays, "pg", **kw)
    env = make_param_trace_fn(
        g.omega, delays, "pg", depth_weight=envelope(kw["npoints"]), **kw
    )
    a = np.asarray(base(jnp.asarray(ew), kw["thickness"], 0.0))
    b = np.asarray(env(jnp.asarray(ew), kw["thickness"], 0.0))
    assert np.max(np.abs(a / a.max() - b / b.max())) > 1e-3


def test_depth_weight_validation(setup):
    g, ew, delays, kw = setup
    with pytest.raises(ValueError, match="one entry per quadrature node"):
        make_param_trace_fn(
            g.omega, delays, "pg", depth_weight=np.ones(kw["npoints"] + 1), **kw
        )
    with pytest.raises(ValueError, match="dispersive slab"):
        make_param_trace_fn(g.omega, delays, "pg", depth_weight=np.ones(1))


def test_envelope_aware_retrieval_beats_uniform_on_enveloped_trace(setup):
    """Synthesize with a strong envelope; the matching model must fit better."""
    g, ew, delays, kw = setup
    dw = envelope(kw["npoints"], z_r_um=8.0)  # strong: slab spans 5 z_R
    synth = make_param_trace_fn(g.omega, delays, "pg", depth_weight=dw, **kw)
    trace = np.asarray(synth(jnp.asarray(ew), kw["thickness"], 0.0))

    common = dict(
        algorithm="lbfgs-ad", guess=ew, maxiters=60, R_omega=False, progress=False
    )
    uniform = retrieve(trace, g.omega, delays, "pg", **common, **kw)
    aware = retrieve(trace, g.omega, delays, "pg", depth_weight=dw, **common, **kw)
    assert aware.error < 0.2 * uniform.error


def test_generation_envelope_pipeline_helper():
    from croak.retrieve import algorithm_params
    from croak.session.params import RetrieveParams
    from croak.session.pipeline import generation_envelope

    off = RetrieveParams(dispersive=True, thickness_um=40.0, npoints=40)
    assert generation_envelope(off) is None

    p = RetrieveParams(
        dispersive=True,
        thickness_um=40.0,
        npoints=40,
        envelope=True,
        envelope_zr_um=131.0,
        envelope_lw_um=355.0,
        envelope_zf_um=0.0,
    )
    g = generation_envelope(p)
    assert g is not None and g.shape == (40,) and np.iscomplexobj(g)
    # front focus: monotone amplitude decay across the slab
    assert np.all(np.diff(np.abs(g)) < 0)
    with pytest.raises(ValueError, match="envelope_zr_um"):
        generation_envelope(
            RetrieveParams(dispersive=True, thickness_um=40.0, envelope=True)
        )
    # the pipeline guard has somewhere to route to (and something to refuse)
    assert "depth_weight" in algorithm_params("lbfgs-ad")
    assert "depth_weight" not in algorithm_params("copra")

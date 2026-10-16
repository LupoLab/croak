"""The :class:`~croak.session.options.SessionOptions` bundle: the serialisable session.

``SessionOptions`` groups the per-stage parameter dataclasses, and the name of
the stage-1 loader that produced the trace, into one object that round-trips to
the ``options.toml`` layout (one TOML table per stage). It is the single source
of truth for session serialisation — the GUI's ``WizardState`` and the headless
engine both go through it — so a session saved in one place reloads identically
in the other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from .. import gases, materials
from ..save import load_options, save_options
from .params import (
    BeamPathMirror,
    DispersionParams,
    LoadParams,
    PreprocParams,
    RetrieveParams,
    SimulatedLoadParams,
    UncertaintyParams,
)

__all__ = ["ENTRIES", "SessionOptions"]


def _from_mapping(cls, data: Any):
    """Build a dataclass from a mapping, ignoring unknown keys and keeping defaults.

    Permissive on purpose: an ``options.toml`` written by an older (or newer)
    croak may omit or add keys, so missing keys fall back to the dataclass default
    and unknown keys are dropped rather than raising.
    """
    obj = cls()
    if not isinstance(data, dict):
        return obj
    valid = {f.name for f in fields(cls)}
    for key, value in data.items():
        if key in valid:
            setattr(obj, key, value)
    return obj


def _mirror_from_mapping(data: Any) -> BeamPathMirror:
    """Build a (frozen) :class:`BeamPathMirror` from a mapping, dropping unknowns."""
    if not isinstance(data, dict):
        return BeamPathMirror()
    valid = {f.name for f in fields(BeamPathMirror)}
    return BeamPathMirror(**{k: v for k, v in data.items() if k in valid})


def _dispersion_from_mapping(data: Any) -> DispersionParams:
    """Build :class:`~croak.session.params.DispersionParams` and its nested mirrors.

    Raises
    ------
    ValueError
        If the session applies a refractiveindex.info material, which croak no
        longer supports: dropping it as an unknown key would silently change the
        compensated pulse.
    """
    obj = DispersionParams()
    if not isinstance(data, dict):
        return obj
    if data.get("ri_page") and data.get("ri_thickness_mm"):
        raise ValueError(
            "this session applies the refractiveindex.info material "
            f"{data.get('ri_book')}/{data['ri_page']}, which croak no longer "
            "supports; register its n(λ) with croak.materials.register_material "
            "and re-create the dispersion stage"
        )
    valid = {f.name for f in fields(DispersionParams)}
    for key, value in data.items():
        if key == "mirrors":
            obj.mirrors = [_mirror_from_mapping(m) for m in value]
        elif key == "material_thickness_mm":
            obj.material_thickness_mm = _canonical_thicknesses(value)
        elif key in valid:
            setattr(obj, key, value)
    if obj.gas_name:
        obj.gas_name = gases.canonical_name(obj.gas_name)
    return obj


def _canonical_thicknesses(value: dict[str, Any]) -> dict[str, float]:
    """Rebuild ``material_thickness_mm`` under the current material names.

    A session written by an earlier croak may name a material by a name it has
    since retired (``SiO2-Franta`` is now ``SiO2``); thicknesses of what is now one
    material add, as two slabs of it would.
    """
    out: dict[str, float] = {}
    for name, mm in value.items():
        key = materials.canonical_name(str(name))
        out[key] = out.get(key, 0.0) + float(mm)
    return out


def _retrieve_from_mapping(data: Any) -> RetrieveParams:
    """Build :class:`~croak.session.params.RetrieveParams` under current names."""
    obj = _from_mapping(RetrieveParams, data)
    obj.material = materials.canonical_name(obj.material)
    return obj


#: Stage section names written to / read from the TOML, in order.
_SECTION_NAMES: tuple[str, ...] = (
    "load",
    "simulated",
    "preproc",
    "retrieve",
    "dispersion",
    "uncertainty",
)

#: Which loader produced the trace — i.e. which of the mutually exclusive stage-1
#: entries the session started from. ``"experimental"`` reads ``[load]``,
#: ``"simulated"`` reads ``[simulated]``; ``"synthetic"`` has no serialised
#: parameters yet (the generator's settings live only in its GUI controls) and is
#: recorded so a reload can say so rather than silently loading nothing.
ENTRIES: tuple[str, ...] = ("experimental", "simulated", "synthetic")


@dataclass
class SessionOptions:
    """A complete retrieval session: which loader it used and the stage parameters.

    Round-trips to the ``options.toml`` layout (one TOML table per stage, plus a
    scalar ``entry`` key naming the loader) via :meth:`to_dict`/:meth:`from_dict`.
    The dispersion table carries a nested ``material_thickness_mm`` sub-table and
    a ``mirrors`` array-of-tables, which :meth:`from_dict` rebuilds into
    :class:`~croak.session.params.BeamPathMirror`.

    ``load`` and ``simulated`` are alternatives, selected by ``entry`` (see
    :data:`ENTRIES`): a measured session fills ``[load]`` and leaves
    ``[simulated]`` at its defaults, and vice versa. Both are always written, so
    the file records which trace the retrieval actually came from — without
    ``entry`` a simulated session would look like an empty experimental one.

    Notes
    -----
    The recipe refers to the external input and calibration files rather than
    embedding them, and records no retrieval RNG seed; synthetic-generator
    settings are not serialised for headless replay. See
    :doc:`/howto/saving_and_loading`.
    """

    entry: str = "experimental"
    load: LoadParams = field(default_factory=LoadParams)
    simulated: SimulatedLoadParams = field(default_factory=SimulatedLoadParams)
    preproc: PreprocParams = field(default_factory=PreprocParams)
    retrieve: RetrieveParams = field(default_factory=RetrieveParams)
    dispersion: DispersionParams = field(default_factory=DispersionParams)
    uncertainty: UncertaintyParams = field(default_factory=UncertaintyParams)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the ``{entry, section: {key: value}}`` options layout.

        ``entry`` comes first because TOML requires scalar keys to precede the
        tables they share a level with.
        """
        out: dict[str, Any] = {"entry": self.entry}
        out.update({name: asdict(getattr(self, name)) for name in _SECTION_NAMES})
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionOptions:
        """Build from a nested options mapping (version header keys are ignored).

        A file written before ``entry`` existed has no simulated workflow to
        restore, so it reads back as an experimental session.
        """
        return cls(
            entry=str(data.get("entry", "experimental")),
            load=_from_mapping(LoadParams, data.get("load", {})),
            simulated=_from_mapping(SimulatedLoadParams, data.get("simulated", {})),
            preproc=_from_mapping(PreprocParams, data.get("preproc", {})),
            retrieve=_retrieve_from_mapping(data.get("retrieve", {})),
            dispersion=_dispersion_from_mapping(data.get("dispersion", {})),
            uncertainty=_from_mapping(UncertaintyParams, data.get("uncertainty", {})),
        )

    def to_toml(self, path: str) -> str:
        """Write the session to ``path`` as an options TOML (with version header)."""
        return save_options(self.to_dict(), path)

    @classmethod
    def from_toml(cls, path: str) -> SessionOptions:
        """Read a session from an ``options.toml`` file."""
        return cls.from_dict(load_options(path))

    def with_(self, **changes: Any) -> SessionOptions:
        """Return a copy with the given per-stage params replaced (e.g. ``load=``)."""
        return replace(self, **changes)

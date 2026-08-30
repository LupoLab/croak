# Workflow

Loading files, cleaning and regridding measured traces, the high-level pipeline,
post-processing, dispersion tuning (including chirped mirrors, back-propagated
beam-path coating mirrors and the refractiveindex.info database) and saving. The
dispersion sources here are described in
[Materials and mirrors](../../howto/materials_and_mirrors.md).

## I/O

```{eval-rst}
.. automodule:: croak.io
   :members:
```

## Preprocessing

```{eval-rst}
.. automodule:: croak.preprocess
   :members:
```

## Pipeline

```{eval-rst}
.. automodule:: croak.pipeline
   :members:
```

## Processing

```{eval-rst}
.. automodule:: croak.processing
   :members:
```

## Known-truth errors

Errors of a retrieval against a *known* pulse — available for the synthetic and
simulated workflows, where the answer is known in advance. Distinct from
{mod}`croak.metrics`, which scores the retrieval against the measured **trace**:
a small trace error does not imply a correct pulse. See
[Loading simulated traces](../../howto/loading_simulated.md).

```{eval-rst}
.. automodule:: croak.truth_metrics
   :members:
```

## Marginal consistency checks

Pre-retrieval trace QC from the frequency marginal: the marginal predicted from
the independent spectrum (exact for SHG, transform-limited for PG/TG/SD), its
centroid/width anchors, and the centroid-matching efficiency-exponent picker. The
background is in [FROG trace marginals as consistency checks](../../explanation/marginals.md).

```{eval-rst}
.. automodule:: croak.marginal_checks
   :members:
```

## Dispersion tuning

```{eval-rst}
.. automodule:: croak.dispersion
   :members:
```

## Mirrors

Built-in and custom mirror reflectivity / dispersion — chirped compressors
({data}`~croak.mirrors.BUILTIN_MIRRORS`) and beam-path coatings
({data}`~croak.mirrors.BUILTIN_COATINGS`, which carry reflectivity and are usually
back-propagated) — registered into the {data}`croak.dispersion.MIRRORS` table via
{func}`~croak.dispersion.register_mirror`.

```{eval-rst}
.. automodule:: croak.mirrors
   :members:
```

## Refractive-index database

Optional bridge to the [refractiveindex.info](https://refractiveindex.info)
database for browsing arbitrary materials (requires the `croak[ridb]` extra).

```{eval-rst}
.. automodule:: croak.refractive_db
   :members:
```

## Saving and loading

```{eval-rst}
.. automodule:: croak.save
   :members:
```

## Uncertainty analysis

Statistical and systematic error bars on the retrieved FWHM (the resampling
bootstraps and the substrate-thickness systematic), and the fast analytic
linearised Gauss–Newton covariance. Each estimator takes an optional
`propagation` transfer function (from
{func}`~croak.session.dispersion.transfer_function`) to report the bar at a
different beamline point. The background and caveats are in
[Uncertainty of the retrieved pulse duration](../../explanation/uncertainty_estimation.md).

```{eval-rst}
.. automodule:: croak.uncertainty
   :members:

.. automodule:: croak.covariance
   :members:
```

## Sessions (headless replay)

A whole retrieval — load, preprocess, retrieve, dispersion compensation and
uncertainty — captured as serialisable per-stage parameters and replayed without
the GUI. {func}`~croak.session.run_session` drives the same stage mappings the
wizard uses; {func}`~croak.session.retarget` re-points an *experimental* session at
a new dataset folder. {class}`~croak.session.options.SessionOptions` also records
which stage-1 loader the trace came from (`entry`, see
{data}`~croak.session.options.ENTRIES`), so a simulated session replays through
{func}`~croak.session.pipeline.assemble_simulated_load_data` rather than the
measured loader — see [Saving and loading](../../howto/saving_and_loading.md).

```{eval-rst}
.. automodule:: croak.session.params
   :members:

.. automodule:: croak.session.options
   :members:

.. automodule:: croak.session.pipeline
   :members:

.. automodule:: croak.session.dispersion
   :members:

.. automodule:: croak.session.engine
   :members:

.. automodule:: croak.session.retarget
   :members:
```

## Command line and script generation

The `croak` console script ({mod}`croak.cli`) replays sessions and retargets them
at new datasets; {func}`croak.scripting.generate_script` writes a standalone,
editable Python script that reproduces a retrieval.

```{eval-rst}
.. automodule:: croak.cli
   :members:

.. automodule:: croak.scripting
   :members:
```

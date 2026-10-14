# Core physics

Grids and transforms, pulse parameterisations, nonlinear interactions, the
forward model, material dispersion and numerical helpers. Begin with
[measurement fundamentals](../../user_guide/fundamentals.md) for units and
conventions, or [modelling the measurement](../../user_guide/models.md) for the
physical assumptions behind these interfaces. The [first retrieval](../../tutorials/01_first_retrieval.md)
shows how the grid, pulse and trace functions fit together.

## Grid and transforms

```{eval-rst}
.. automodule:: croak.grid
   :members:
```

## Pulses

```{eval-rst}
.. automodule:: croak.pulses
   :members:
```

## Interactions

```{eval-rst}
.. automodule:: croak.interactions
   :members:
```

## Forward model

```{eval-rst}
.. automodule:: croak.forward
   :members:
```

## Geometric smearing

Mask geometry to smearing-kernel statistics for a non-collinear BOXCARS setup;
consumed by the forward model.

```{eval-rst}
.. automodule:: croak.smearing
   :members:
```

## Chromatic focal mixture

An explicit quadrature over the focal plane, with wavelength-dependent amplitude
filters and arrival offsets at each node. It resolves effects approximated by
the reduced Gaussian kernel; matching second moments does not make the models
identical for every field.

```{eval-rst}
.. automodule:: croak.focal
   :members:
```

(the-phase-the-incoherent-sum-throws-away)=
(where-the-aperture-sits)=
(validity)=
(regime)=
## Collection aperture

The focal mixture without its full-beam-collection assumption. See
[Modelling the collection aperture](../../howto/collection_aperture.md).

```{eval-rst}
.. automodule:: croak.collection
   :members:

.. autodata:: croak.collection.WEIGHTINGS

.. autodata:: croak.collection.EDGE_SECOND_MOMENT

.. autofunction:: croak.collection.aperture_offset

.. autofunction:: croak.collection.quadrature_hole_diameter
```

## Materials

```{eval-rst}
.. automodule:: croak.materials
   :members:
```

## Gases

```{eval-rst}
.. automodule:: croak.gases
   :members:
```

## Maths utilities

```{eval-rst}
.. automodule:: croak.maths
   :members:
```

## Constants

```{eval-rst}
.. automodule:: croak.constants
   :members:
```

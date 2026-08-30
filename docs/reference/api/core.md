# Core physics

Grids and transforms, pulse parameterisations, nonlinear interactions, the
forward model, material dispersion and numerical helpers.

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

The smearing kernel without its achromatic reduction: an explicit quadrature over the
focal plane, each node carrying its own chromatic amplitude filter and its own definite
arrival offsets.

```{eval-rst}
.. automodule:: croak.focal
   :members:
```

## Collection aperture

The focal mixture without its full-beam-collection assumption. See
[Modelling the collection aperture](../../howto/collection_aperture.md).

```{eval-rst}
.. automodule:: croak.collection
   :members:
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

"""Opt-in neural-network column physics, inference only (plan phase P2).

docs/dev/DIFFERENTIABLE-HYBRID-PLAN.md section 4 and docs/nn-physics.md.
A column model maps a column's state, ``[feature, column]``, to physical
tendencies, ``[output, column]``.  This package holds the parts every
seat shares:

* :mod:`woof.nn.manifest` -- the artefact: weights, their SHA-256, the
  input and output vocabulary with units, normalisation statistics and
  the trained input range;
* :mod:`woof.nn.models` -- runners: PyTorch on CUDA (CuPy arrays in and
  out through DLPack on woof's stream), MLX on Metal (the facade's
  ``mx.array`` in and out), and a NumPy reference;
* :mod:`woof.nn.bridge` -- the zero-copy array exchange those runners use;
* :mod:`woof.nn.safety` -- out-of-range refusal or fallback, input
  clipping, output limits, positivity and a column-integral fixer.

Nothing on the default path imports this package, torch or ``mlx.nn``
(``tests/test_nn_default_path_imports.py``).  An NN option is not a WRF
scheme: a run that selects one is not WRF-identical, and the identity
doors (``GPUWM_WRF_EXACT=1``, ``physics_mode = "exact"``) refuse it.
"""

from __future__ import annotations

#: The label every NN-bearing run, registry row and receipt carries.
NOT_WRF_IDENTICAL = "not-wrf-identical: neural-network column physics (woof.nn)"

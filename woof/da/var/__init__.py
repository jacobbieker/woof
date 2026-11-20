"""EXPERIMENTAL: WRFDA-style incremental 3D-Var (docs/dev/WRFDA-3DVAR-PLAN.md).

This package is a separate door.  No forecast path imports it
(``tests/test_var_import_graph.py`` holds that), it launches no kernel unless
a caller asks for an analysis, and it changes no default forecast word.  An
analysis it makes reaches a forecast only through the existing
``woof run --analysis BUNDLE`` door.

The arithmetic follows WRFDA 4.7.1 (public domain; see
``licenses/LICENSE-WRF-public-domain.txt`` and ``PROVENANCE.md``) in
operation order, so that the oracle tier can agree with a serial WRFDA
build word for word.  WRF's bundled CRTM 2.3.0 is never used.

Modules (landed phase by phase):

- :mod:`woof.da.var.be_dat` -- WRFDA ``be.dat`` reader and
  ``da_setup_be_regional`` (binary64, host side).
- :mod:`woof.da.var.reference` -- NumPy binary64 mirrors of the WRFDA
  routines, in WRFDA's operation order; the tests' and the kernels' reference.
- :mod:`woof.da.var.cost` -- ``VarCost``: J and grad J in WRFDA's order
  (``da_calculate_j``, ``da_calculate_gradj``), against an ``ObsTerm``.
- :mod:`woof.da.var.minimise` -- ``da_minimise_cg`` and the outer loop.
- :mod:`woof.da.var.oracle` -- reader for the oracle build's array dumps
  (``tools/wrfda_oracle``).
"""

STATUS = "experimental"

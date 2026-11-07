"""The MPAS-ICE (land-ice) port (core_landice), a coupled component of the hex model.

A GPU port of the MPAS-ICE v8.4.2 core, to be bit-exact to the pinned native
oracle and to run on CUDA and MLX (Metal) through the shared facade. The land-ice
core is built on the Albany and Trilinos finite-element frameworks (its
``external_albany_version`` and ``external_trilinos_version`` pins), which makes
it the most architecturally different of the three lanes: the model-specific layer
is small (21 Fortran files) but it rides on a large FEM dependency. This landing
is the lane's scaffolding: the door, the pinned options inventory and the coupling
interface. The port lands on feat/mpas-ice.
"""

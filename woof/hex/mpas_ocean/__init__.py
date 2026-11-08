"""The MPAS-Ocean port (core_ocean), a coupled component of the hex model.

A GPU port of the MPAS-Ocean v8.4.2 core (the MOM-lineage ocean on the unstructured
hexagonal mesh), to be bit-exact to the pinned native oracle and to run on CUDA
and MLX (Metal) through the shared facade. This landing is the lane's scaffolding:
the door, the pinned options inventory and the coupling interface. The dynamical
core and its schemes land piece by piece; docs/mpas-coupled-roadmap.md records
what is ported and what remains.
"""

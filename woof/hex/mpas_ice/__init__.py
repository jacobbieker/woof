"""The MPAS land-ice (MALI) port (core_landice), a coupled component of the hex model.

A GPU port of the MPAS land-ice v8.4.2 core, to be bit-exact to the pinned native
oracle and to run on CUDA and MLX (Metal) through the shared facade. The core's
own Fortran (about 22,000 lines in 21 files: the shallow-ice (SIA) velocity
solver that is its default, the thermal solver, advection, calving and
subglacial hydrology) is ported against the gfortran oracle. The higher-order
velocity solvers (``config_velocity_solver`` = 'L1L2', 'FO', 'Stokes') are
external dycores -- Albany on Trilinos, C++ finite elements reached through
``Interface_velocity_solver.cpp`` -- and are refused by name until each can be
held to a reference. This landing is the lane's scaffolding: the door, the
pinned options inventory and the coupling interface;
docs/mpas-coupled-roadmap.md records what is ported and what remains.
"""

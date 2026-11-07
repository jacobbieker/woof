"""The MPAS-Seaice port (core_seaice), a coupled component of the hex model.

A GPU port of the MPAS-Seaice v8.4.2 core, to be bit-exact to the pinned native
oracle and to run on CUDA and MLX (Metal) through the shared facade. This
landing is the lane's scaffolding: the door, the pinned options inventory and
the coupling interface. The dynamical core and its schemes land on
feat/mpas-seaice, gated the same way the hex (MPAS-Atmosphere) port is.
"""

"""Smoke source physics: fire emission, injection height and removal.

A stand-alone library of array-in/array-out functions, not wired into the
dycore, each one a port of WRF v4.7.1's own Fortran held to it word for
word (``tools/smoke_wrf471_oracle``):

* :mod:`woof.smoke.emission` -- WRF-SFIRE's fire smoke tracer emission
  (``phys/module_fr_fire_atm.F``, ``tracer_opt = 3``) and PM2.5 emission
  factors by fuel;
* :mod:`woof.smoke.injection` -- the lowest-layer and boundary-layer
  injection profiles;
* :mod:`woof.smoke.plume_rise` -- the Freitas et al. (2007) 1-D plume-rise
  model of WRF-Chem (``chem/module_chem_plumerise_scalar.F``);
* :mod:`woof.smoke.deposition` -- the Zhang et al. (2001) particle dry
  deposition velocity (``chem/module_aer_drydep.F``) and GOCART
  gravitational settling (``chem/module_gocart_settling.F``).

Every physics function has a CuPy form (the production path: whole-domain
arrays in and out, one NVRTC kernel launch, :mod:`woof.smoke._kernels`) and
a NumPy reference (suffix ``_np``) that repeats WRF's arithmetic operation
for operation in WRF's precision.  The two agree with each other, and with
the compiled WRF, at zero ULP.

Import the submodules directly; this package imports nothing on its own,
so importing it never touches CuPy.
"""

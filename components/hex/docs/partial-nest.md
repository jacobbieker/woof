# Partial-depth band (height-dependent horizontal resolution)

A partial-depth band is a finer regional MPAS mesh that carries the
atmosphere from the ground up to a flat height `z_i`. Above that height the
coarser parent alone carries it. This is the hex counterpart of the WRF
partial-depth nest (plans/sorted-mapping-hummingbird.md, Phase D5). The code
is in:

- `woof/hex/partial_nest.py`: the band, its namelist group and the refusals;
- `woof/hex/partial_nest_run.py`: the coupled run;
- `woof/hex/partial_nest_coupling.py`: the host tables;
- `woof/hex/cuda_partial_nest_v841.py`: the kernels.

## How the band couples to its parent

- **Interface.** The interface is a parent `zgrid` face at which every column
  of both meshes stands at one height. The child is the parent's own discrete
  column cut at that face.
  - The acoustic solve takes the parent's `rw` there as a specified top.
  - The interface face fluxes of theta, u, w and the scalars use the parent's
    own second-order face value. They do not upwind: an upwinded face carries
    heat into the top layer in stratification (D4, reproduced here).
- **Top zone.** The `relax_levels` layers below the interface relax toward
  the parent. They use MPAS's relaxation-zone arithmetic
  (`atm_bdy_adjust_dynamics_relaxzone_tend`), with ring strength `5 - d` on
  the layer `d` below the interface.
- **Lateral zones.** These are filled online through the regional LBC
  machinery: each parent step's end becomes the far end of the child's LBC
  interval.
- **Coarse to fine.** MPAS's barycentric interpolation
  (`mpas_calculate_barycentric_weights_for_points`). Winds go through
  `mpas_reconstruct_2d` and init_atmosphere's edge-normal projection.
- **Fine to coarse.** A conservative overlap remap feeds back theta and the
  scalars, below the interface, on the parent cells the child's interior
  covers wholly. It never feeds back `rho_zz`, w or the vertical coordinate.
  After feedback the band re-reads its planes and lateral frame from the
  parent.
- **Mass budget.** The uniform interface residual is added to the
  interface `rw` as a face-flux correction.

## Namelist

The band is configured by the WOOF-only group `&woof_partial_nest`, read by
`woof.hex.partial_nest.configs_from_namelist`:

```
&woof_partial_nest
    config_partial_nest_interface_level = 18      ! 0-based zgrid face
    config_partial_nest_time_step_ratio = 2       ! child steps per parent step
    config_partial_nest_relax_levels = 4          ! 1..4
    config_partial_nest_feedback = 1
    config_partial_nest_feedback_fields = 'theta'
    config_partial_nest_mass_correction = .true.
    config_partial_nest_face_scheme = 'second_order'
/
```

Every other route that reads a namelist (the dycore loader and the forecast
drivers) refuses the group by name. A namelist without the group maps
exactly as before.

## What is refused, by name

- feedback of w, `rho_zz` or the coordinate (a measured delayed-feedback
  instability in WRF), and of u;
- column physics (`config_physics_suite`): the band's radiation would need
  the extended column, and no hex seam carries a column taller than the
  dycore's;
- `config_moist_physics` (moist acoustic coefficients at the interface);
- upper u Rayleigh damping, CAM damping, and a w-damping layer below the
  interface;
- the monotonic or positive-definite scalar transport;
- 3-D mixing or LES through the interface;
- the fused acoustic unit;
- any schedule other than v8.4.1 split-three.

## Metal

The hex lane runs on Metal by default (`woof/hex/metal_lane.py`;
`WOOF_HEX_METAL=0` opts out). The band's oracle and envelope have been
measured there; the rest of the lane against CUDA is in
`docs/dev/mlx-backend.md`.

## Evidence

- `components/hex/tests/test_partial_nest_oracle_v841.py` holds the
  coupling operators to compiled MPAS v8.4.1
  (`tools/partial_nest_mpas_oracle`). It also holds the remap to its exact
  binary64 reference.
- `components/hex/tests/test_partial_nest_envelope_v841.py` and the receipts
  in `tests/data/receipts/partial-nest-d5` hold the envelope. Those receipts
  carry the measured numbers, including where the band does not beat the
  parent alone.

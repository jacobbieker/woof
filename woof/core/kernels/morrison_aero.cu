// woof/core/kernels/morrison_aero.cu
//
// WRF v4.7.1 aerosol-aware Morrison two-moment microphysics (mp_physics =
// 40, package morr_tm_aero, phys/module_mp_morr_two_moment_aero.F).
//
// The loader assembles this unit AFTER morrison_aero_constants.cuh,
// morrison_aero_support.cuh and morrison.cu (kernels/__init__.py
// _EXTRA_HEADERS["morrison_aero"]).  The aero module is the mp_physics =
// 10 module with prescribed-aerosol droplet activation and ice nucleation
// added and four init words changed, so its entry points are morrison.cu's
// own (mo_process, mo_sediment_64/_256, mo_finalize, mo_exner,
// mo_refl10cm) compiled with MORR_AERO defined by the constants header:
// every line the aero module adds or changes is a MORR_AERO block there,
// and the mp_physics = 10 module ("morrison") compiles none of them.
// Graded against WRF's own Fortran by tools/morrison_aero_wrf471_oracle.

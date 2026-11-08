# NEMO kernels (opt-in)

CeCILL v2.0; read `NOTICE.md` and `LICENSE` before using or redistributing.
A build that loads these kernels can be redistributed only under CeCILL or
the GNU GPL as a whole.

```
WOOF_ENABLE_NEMO=1 pixi run -e default python -m pytest tests/test_nemo_traqsr_parity.py
```

Ported: 2-band penetrative solar radiation (`traqsr.F90`, `ln_qsr_2bd`,
linear free surface, leapfrog); the equation of state (`eosbn2.F90`: in situ
and potential density, expansion ratios, N^2) for TEOS-10, EOS-80 and S-EOS;
TKE vertical mixing (`zdfsh2.F90`, `zdftke.F90`) without sea ice; implicit
vertical diffusion of T and S (`trazdf.F90`); and `NemoColumnOcean`, those
pieces stepped in `stpmlf.F90`'s order with `sbc_cpl_rcv`, the bottom drag,
`tra_sbc` and the Asselin filter, as a `woof.ocean.interface.ExternalOcean`
behind `woof.ocean.wrf_cpl` that couples SST only (an ocean at rest).  Also
flux-corrected transport advection (`traadv_fct.F90`, `TracerAdvectionFct`,
2nd/4th order, Zalesak limiter) given the transports, and iso-level lateral diffusion (`traldf_lev.F90`,
`LateralDiffusionLevel`, laplacian or bilaplacian) given the diffusivities.
Not yet: the iso-neutral operators and NEMO's dynamics
(docs/coupling/WRF-HYDRO-NEMO.md).

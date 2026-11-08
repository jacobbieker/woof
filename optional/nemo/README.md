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
TKE vertical mixing (`zdfsh2.F90`, `zdftke.F90`) without sea ice.  It consumes WRF's SURF_NET_SOLAR from
`woof.ocean.wrf_cpl`.  Not yet: the rest of NEMO's tracer and dynamics core
(docs/coupling/WRF-HYDRO-NEMO.md).

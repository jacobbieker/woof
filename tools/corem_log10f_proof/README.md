# corem_log10f.cuh against glibc 2.43's log10f, bit for bit

`woof/core/kernels/corem_log10f.cuh` (`cm_log10f`) is glibc 2.43's float32
`log10f`: CORE-MATH's correctly rounded binary32 log10 (revision bc385c2),
the function glibc ships from 2.42 on.  It is the libm of the repository's
WRF v4.7.1 column oracles (Ubuntu 26.04, `2.43-2ubuntu2.4`; see
`tools/glibc_trig_flt32_proof`), and WRF-Solar's `cal_cldfra3`
(`woof/core/kernels/cldfra3.cu`, icloud = 3) takes its `LOG10` from it.
glibc 2.39's `log10f` is FDLIBM's and is a different function: it differs
from this one on 29,787,060 of the 2^32 inputs (`gfk_log10`,
`glibc_flt32_extra.cuh`, keeps that one for the WRF v4.6.1 oracles).

* `exhaustive-final.log`: the header compiled for the host
  (`../glibc_trig_flt32_proof/host_shim.hpp`) against glibc 2.43's
  `libm.so.6` in an `ubuntu:26.04` container on pb, all 2^32 inputs:
  0 mismatches.
* `gpu-final.log`: the header compiled by NVRTC 12.9 on the RTX A6000 as
  the cldfra3 unit compiles it (`--fmad=false --ftz=false`) and with
  contraction on, all 2^32 inputs against the pb host's glibc 2.44
  `log10f` (the same correctly rounded function): 0 mismatches.
* Metal: `tests/test_cldfra3_wrf471_oracle.py` sweeps the arguments
  `cal_cldfra3` can pass (1e-10 < qc+qi < 1e-6) through the translated unit.

```sh
G=$(pixi exec -s gxx sh -c 'echo $CONDA_PREFIX')
$G/bin/g++ -O2 -std=c++17 -ffp-contract=off -fno-builtin -fopenmp exhaustive.cpp -ldl -lm -o exhaustive
docker run --rm -v $G:$G:ro -v $PWD:/w -w /w ubuntu:26.04 ./exhaustive > exhaustive-final.log
$G/bin/g++ -O2 -std=c++17 -fno-builtin -fopenmp -shared -fPIC gpu_oracle.cpp -lm -o gpu_oracle.so
LD_LIBRARY_PATH=$G/lib python gpu_check.py > gpu-final.log
```

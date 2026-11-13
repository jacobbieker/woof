# WRF-Chem PNNL files (opt-in)

These are Battelle/PNNL-terms code. Read `NOTICE.md` and `LICENSE` before
using or redistributing anything here.

```
WOOF_ENABLE_WRFCHEM_PNNL=1 WOOF_WRF_SOURCE=/path/to/WRF-4.7.1 \
    pixi run -e default python -m pytest tests/test_wrfchem_pnnl_activate_parity.py
```

```python
from woof import optin
pnnl = optin.load("wrfchem_pnnl")   # refuses by name unless enabled
```

Plan (roadmap P7): `phys/module_mixactivate.F` (`activate`, then the column
driver `mixactivate` with `explmix` and `loadaer`, and the SORGAM wrapper
`module_mixactivate_wrappers.F`), then the modal optics (`optical_driver.F`,
`module_optical_averaging.F`, `module_fastj_mie.F`). Each is ported exactly
and graded bitwise against a gfortran -O0 oracle of the pinned file. The
clean-room defaults (P8 activation, P9 optics) are graded against these
ports within a stated tolerance.

Done so far: the opt-in door, and `activate` (with `maxsat`), bitwise against its oracle on 112 cases
(`oracle/activate`, `woof_wrfchem_pnnl/activate.py`).

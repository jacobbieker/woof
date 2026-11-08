"""Four dycore steps of the small mixed land/water physics case with no
[external_ocean] table; prints the digest of every state array and driver
field, whether woof.ocean.external was imported, and whether the driver
carries an external_ocean.  Run as a subprocess by
tests/test_external_ocean_absent_noop.py."""

import datetime
import hashlib
import sys

import numpy as np


def main():
    sys.path.insert(0, "tests")
    import conftest  # noqa: F401  (installs the backend as the suite does)
    import cupy as cp
    from test_physics_driver import _full_state

    from woof.core.dycore import step

    state, cfg, driver = _full_state(
        ra_physics=0, ra_lw_physics=0, ra_sw_physics=1,
        radiation_start_time=datetime.datetime(2011, 6, 26, 12),
        radiation_latitude=np.full((2, 6), 45.0), radiation_longitude=np.full((2, 6), -5.0))
    for _ in range(4):
        step(state, cfg)
    h = hashlib.sha256()

    def add(name, a):
        if isinstance(a, cp.ndarray):
            h.update(name.encode())
            h.update(cp.asnumpy(a).tobytes())

    for k in sorted(vars(state)):
        add("s." + k, getattr(state, k))
    for k in sorted(driver.fields):
        add("f." + k, driver.fields[k])
    print(h.hexdigest(), "external" if "woof.ocean.external" in sys.modules else "-",
          hasattr(driver, "external_ocean"))


if __name__ == "__main__":
    main()

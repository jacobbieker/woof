"""Read the km_opt = 5 oracle's cases.bin and oracle.bin into woof layout.

Every 3-D array comes back in woof's C order (k, j, i): mass (nz, ny, nx),
w levels (nz + 1, ny, nx), u (nz, ny, nx + 1), v (nz, ny + 1, nx); planes
(ny, nx).  Fortran level k is index k - 1.
"""
import numpy as np

PROFILES = ("fnm", "fnp", "dn", "dnw", "c1h", "c2h", "c1f", "c2f", "znw",
            "t_base", "qv_base", "u_base", "v_base")
PLANES = ("msf", "mut", "ust", "ustm", "hfx", "qfx", "br", "rmol", "wspd",
          "u10", "v10", "gamu", "gamv", "pblh")
MASS_IN = ("t", "p", "pb", "alt", "qv", "qc", "qi", "tke")
REALS = ("dx", "dy", "dt", "c_s", "c_k", "tke_heat_flux",
         "tke_drag_coefficient", "cf1", "cf2", "cf3", "p_top")
INTS = ("nx", "ny", "nz", "bnd", "isfflx", "mix_full_fields", "use_theta_m")


def oracle_fields(nx, ny, nz):
    """(name, (k, j, i) extents, dtype) in run_sms3dtke.F90's write order."""
    m = (nz, ny, nx)
    p = (ny, nx)
    return (
        ("rdzw", m, "f4"), ("rdz", m, "f4"),
        ("zx", (nz + 1, ny, nx + 1), "f4"), ("zy", (nz + 1, ny + 1, nx), "f4"),
        ("th", m, "f4"), ("rho", m, "f4"), ("u_phy", m, "f4"), ("v_phy", m, "f4"),
        ("p8w1", p, "f4"), ("t8w1", p, "f4"), ("p8wt", p, "f4"), ("t8wt", p, "f4"),
        ("d11", m, "f4"), ("d22", m, "f4"), ("d12", m, "f4"),
        ("nlflux", (nz + 1, ny, nx), "f4"),
        ("gamu", p, "f4"), ("gamv", p, "f4"), ("pblh", p, "f4"), ("kpbl", p, "i4"),
        ("hfx_nonlocal", p, "f4"),
        ("elmin", m, "f4"), ("dlk", m, "f4"), ("hfx_meso", p, "f4"),
        ("xkmh", m, "f4"), ("xkmv", m, "f4"), ("xkhh", m, "f4"), ("xkhv", m, "f4"),
        ("xkmv_meso", m, "f4"), ("bn2", m, "f4"),
        ("shear", m, "f4"), ("tke_rhs", m, "f4"), ("l_diss", m, "f4"),
        ("ru", (nz, ny, nx + 1), "f4"), ("rv", (nz, ny + 1, nx), "f4"),
        ("rw", (nz + 1, ny, nx), "f4"), ("rt", m, "f4"), ("tke_tend", m, "f4"),
        ("rqv", m, "f4"), ("rqc", m, "f4"), ("rqi", m, "f4"),
        ("hfx_final", p, "f4"),
    )


class _Reader:
    def __init__(self, raw):
        self.raw = raw
        self.pos = 0

    def take(self, count, dtype):
        size = np.dtype(dtype).itemsize * count
        out = np.frombuffer(self.raw, dtype=dtype, count=count, offset=self.pos)
        self.pos += size
        return out.copy()

    def field(self, shape, dtype="<f4"):
        """A (k, j, i) array stored with i fastest, then k, then j."""
        if len(shape) == 2:
            return self.take(shape[0] * shape[1], dtype).reshape(shape)
        nk, nj, ni = shape
        a = self.take(nk * nj * ni, dtype).reshape(nj, nk, ni)
        return np.ascontiguousarray(a.transpose(1, 0, 2))


def read_cases(cases_path, oracle_path=None):
    rin = _Reader(open(cases_path, "rb").read())
    rout = _Reader(open(oracle_path, "rb").read()) if oracle_path else None
    (ncase,) = rin.take(1, "<i4")
    if rout is not None:
        (nout,) = rout.take(1, "<i4")
        assert nout == ncase
    cases = []
    for _ in range(int(ncase)):
        c = dict(zip(INTS, (int(x) for x in rin.take(7, "<i4"))))
        c.update(zip(REALS, rin.take(11, "<f4")))
        nx, ny, nz = c["nx"], c["ny"], c["nz"]
        for name in PROFILES:
            c[name] = rin.take(nz + 1, "<f4")
        for name in PLANES:
            c[name] = rin.field((ny, nx))
        c["u"] = rin.field((nz, ny, nx + 1))
        c["v"] = rin.field((nz, ny + 1, nx))
        for name in ("w", "ph", "phb"):
            c[name] = rin.field((nz + 1, ny, nx))
        for name in MASS_IN:
            c[name] = rin.field((nz, ny, nx))
        if rout is not None:
            c["wrf"] = {name: rout.field(shape, "<" + dt)
                        for name, shape, dt in oracle_fields(nx, ny, nz)}
        cases.append(c)
    assert rin.pos == len(rin.raw)
    if rout is not None:
        assert rout.pos == len(rout.raw)
    return cases

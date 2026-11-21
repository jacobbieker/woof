"""Hybrid 3DEnVar: WRFDA's alpha control variable (docs/dev/WRFDA-3DVAR-PLAN.md,
section 1.10; plan item V3.1).

With ``ensdim_alpha = N > 0`` WRFDA extends the control vector with ``N``
alpha fields and adds an ensemble increment to the static ``U v``::

    x' = U_s v_s + sum_n alpha_n o x'_n,     alpha = U_a v_a

``x'_n`` are the ensemble perturbations scaled by ``1/sqrt(N-1)``; ``U_a``
is the alpha fields' correlation: the recursive filter of
``alpha_corr_scale`` (the static ``B``'s ``da_transform_through_rf``) and a
vertical localisation (``alpha_vertloc_opt``: 0 none, 1 ``be.vertloc.dat``,
2 computed in log p).  The cost is ``J = Jb + Jo + Je`` with ``Jb =
jb_factor/2 v_s^T v_s`` and ``Je = je_factor/2 v_a^T v_a``, the weights from
``da_setup_background_errors``: ``jb_factor = je_factor / (je_factor - 1)``
for ``je_factor > 1`` (so ``1/jb_factor + 1/je_factor = 1``); ``-999`` for
``je_factor <= 1`` (full ensemble: the static control variables are
dropped); 1 with no ensemble.

The arithmetic follows WRFDA 4.7.1 in operation order (``da_setup_be_regional``'s
alpha block, ``da_get_alpha_vertloc``, ``da_setup_flow_predictors``,
``da_transform_vtovv``/``da_vertical_transform`` for alpha,
``da_calc_flow_dependence_xa(_adj)``, ``da_add_xa``), for the default
``alphacv_method = 2`` (xa space), regional, single resolution, 3D-Var (no
4DEnVar time levels), no hydrometeor perturbations.

Three tiers, as for the static ``U`` (:mod:`woof.da.var.cost`):
:class:`HybridReferenceU` (NumPy binary64) and :class:`HybridDeviceU`
(``oracle``: binary64, the reference's words on CUDA and Metal;
``production``: binary32).  The perturbations can stay where the LETKF keeps
its members (:meth:`Ensemble.from_members` takes NumPy or CuPy stacks);
:func:`read_ep` and :func:`write_ep` are WRFDA's ``ep`` files
(``ep_format`` 1 and 11), for the oracle lane.

Derived from WRFDA 4.7.1 (``var/da/da_setup_structures``,
``var/da/da_vtox_transforms``, ``var/da/da_minimisation``), public domain;
see ``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import dataclasses
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from woof.da.var import reference as ref

__all__ = [
    "AlphaB",
    "Ensemble",
    "HybridConfig",
    "HybridDeviceU",
    "HybridReferenceU",
    "alpha_vertloc",
    "drop_static",
    "ep_from_wrf",
    "ep_paths",
    "flow_xa",
    "flow_xa_adj",
    "hybrid_cost",
    "jb_factor_for",
    "member_fields",
    "perturbations_sha256",
    "read_ep",
    "read_vertloc",
    "setup_alpha",
    "wrf_times",
    "write_ep",
    "write_vertloc",
]

BASE_PRES = 100000.0
#: ``da_setup_flow_predictors``: file names and real kind per ``ep_format``
EP_FORMATS = {1: (("u", "v", "t", "q", "ps"), "f8"), 11: (("U", "V", "T", "QVAPOR", "PSFC"), "f4")}
FIELDS = ("u", "v", "t", "q", "ps")


@dataclass(frozen=True)
class HybridConfig:
    """The ``&wrfvar7``/``&wrfvar13``/``&wrfvar16`` keys of the hybrid (WRFDA names, defaults).

    ``je_factor`` has no default: WRFDA's (``Registry/registry.var``: 1.0)
    selects full-ensemble mode, since ``da_setup_background_errors`` takes
    every ``je_factor <= 1`` (including 0 < je_factor < 1, which it neither
    clamps nor resets) to ``jb_factor = -999`` and drops the static ``B``.
    Pass it explicitly: e.g. 2 for an even hybrid, 1 for full ensemble.
    """

    je_factor: float | None = None
    alpha_corr_scale: float = 200.0  # km
    alpha_std_dev: float = 1.0
    alpha_vertloc_opt: int = 2
    max_vert_var_alpha: float = 99.0
    alphacv_method: int = 2
    alpha_hydrometeors: bool = False
    anal_type_hybrid_dual_res: bool = False
    use_4denvar: bool = False

    def check(self) -> None:
        """:meth:`check_options` and a given, positive ``je_factor`` (WRFDA has no check for <= 0)."""
        self.check_options()
        if self.je_factor is None:
            raise ValueError("HybridConfig.je_factor must be given: WRFDA's default 1.0 is full-ensemble mode "
                             "(jb_factor = -999, no static B)")
        if not self.je_factor > 0.0:
            raise ValueError(f"je_factor must be > 0 (got {self.je_factor})")

    def check_options(self) -> None:
        """The options this port implements (everything but the weights)."""
        for key, bad in (
            ("alphacv_method = 1 (vp space)", self.alphacv_method != 2),
            ("alpha_hydrometeors", self.alpha_hydrometeors),
            ("anal_type_hybrid_dual_res", self.anal_type_hybrid_dual_res),
            ("use_4denvar", self.use_4denvar),
        ):
            if bad:
                raise NotImplementedError(f"woof var hybrid: {key} is not implemented")
        if self.alpha_vertloc_opt not in (0, 1, 2):
            raise ValueError(f"alpha_vertloc_opt must be 0, 1 or 2 (got {self.alpha_vertloc_opt})")


def jb_factor_for(ne: int, je_factor: float) -> float:
    """``da_setup_background_errors``: the static weight WRFDA derives."""
    if ne > 0:
        if je_factor > 1.0:
            return je_factor / (je_factor - 1.0)
        return -999.0
    return 1.0


def drop_static(b):
    """``da_setup_be_regional`` with ``jb_factor <= 0``: every static ``mz`` is 0."""
    from woof.da.var.be_dat import BeSub

    jy, kz = b.jy, b.kz
    zero = tuple(BeSub(s.name, 0, np.zeros((jy, 0)), np.zeros((jy, kz, 0)), np.zeros(0), np.zeros(0),
                       np.zeros((kz, 0)), np.zeros(0)) for s in b.v)
    return dataclasses.replace(b, v=zero)


# ------------------------------------------------------------ vertical localisation


def _seq_sum(a) -> float:
    acc = 0.0
    for x in np.ravel(a):
        acc = acc + float(x)
    return acc


def _eof_decomposition(cov: np.ndarray):
    """``da_eof_decomposition``: LAPACK ``dsyev('V', 'U')``, modes in descending order."""
    from scipy.linalg import lapack

    w, v, info = lapack.dsyev(np.asfortranarray(cov), compute_v=1, lower=0)
    if info != 0:
        raise ValueError(f"dsyev failed, info = {info}")
    return w[::-1].copy(), v[:, ::-1].copy()


def alpha_vertloc(znw, ptop: float):
    """``da_get_alpha_vertloc`` with ``alpha_vertloc_opt = 2`` (log p): ``(alpha_val, alpha_evec)``.

    ``alpha_val`` is the square root of the localisation matrix's
    eigenvalues, ``alpha_evec`` (kz, kz) its eigenvectors (columns).  The
    eigenvectors' signs are LAPACK's; WRFDA's own words come from its
    ``be.vertloc.dat`` (:func:`read_vertloc`).
    """
    znw = np.asarray(znw, dtype=np.float64)
    kz = znw.size - 1
    cutoff = 1.0 / 2.71828
    p_w = znw * (BASE_PRES - ptop) + ptop
    ln_p_w = np.log(np.maximum(p_w, 0.0001))
    kcutoff = np.ones(kz)
    for k in range(kz):
        for k1 in range(k + 1, kz):
            if abs(ln_p_w[k] - ln_p_w[k1]) <= cutoff:
                kcutoff[k] = float(k1 - k + 1)
    smth = kcutoff.copy()
    for _ in range(2):
        for k in range(1, kz - 1):
            smth[k] = kcutoff[k] + 0.25 * ((kcutoff[k - 1] + kcutoff[k + 1]) - 2.0 * kcutoff[k])
        kcutoff = smth.copy()
    rho = np.empty((kz, kz))
    for k in range(kz):
        kscale = kcutoff[k]
        inv = 1.0 / (kscale * kscale)
        for k1 in range(k, kz):
            kdist = k1 - k
            rho[k, k1] = np.exp((-1.0 * float(kdist * kdist)) * inv)
            rho[k1, k] = rho[k, k1]
    ev, evec = _eof_decomposition(rho)
    totvar = _seq_sum(ev)
    ev = np.where(ev < 0.0, 0.0, ev)
    totvar_inv = 1.0 / _seq_sum(ev)
    ev = (ev * totvar) * totvar_inv
    return np.sqrt(ev), evec


def _records(data: bytes):
    """Fortran sequential records, either endianness."""
    end = None
    for e in (">", "<"):
        if len(data) >= 8:
            (n,) = struct.unpack(e + "i", data[:4])
            if 0 <= n <= len(data) - 8 and struct.unpack(e + "i", data[4 + n : 8 + n])[0] == n:
                end = e
                break
    if end is None:
        raise ValueError("not a Fortran sequential file")
    out, pos = [], 0
    while pos < len(data):
        (n,) = struct.unpack(end + "i", data[pos : pos + 4])
        out.append(data[pos + 4 : pos + 4 + n])
        (m,) = struct.unpack(end + "i", data[pos + 4 + n : pos + 8 + n])
        if m != n:
            raise ValueError("corrupt Fortran record markers")
        pos += 8 + n
    return end, out


def _record(payload: bytes, end: str = ">") -> bytes:
    n = struct.pack(end + "i", len(payload))
    return n + payload + n


def read_vertloc(path):
    """``be.vertloc.dat`` (``alpha_vertloc_opt = 1``; WRFDA writes it with option 2)."""
    end, recs = _records(Path(path).read_bytes())
    (kz,) = struct.unpack(end + "i", recs[0][:4])
    val = np.frombuffer(recs[1], dtype=end + "f8", count=kz).astype(np.float64)
    evec = np.frombuffer(recs[2], dtype=end + "f8", count=kz * kz).reshape((kz, kz), order="F").astype(np.float64)
    return val, evec


def write_vertloc(path, val, evec, endian: str = ">") -> None:
    kz = int(np.size(val))
    data = _record(struct.pack(endian + "i", kz), endian)
    data += _record(np.asarray(val, dtype=endian + "f8").tobytes(), endian)
    data += _record(np.asarray(evec, dtype=endian + "f8").tobytes(order="F"), endian)
    Path(path).write_bytes(data)


# ------------------------------------------------------------ alpha B


@dataclass(frozen=True)
class AlphaB:
    """WRFDA's ``be%alpha`` for ``ne`` members: modes, ``val`` (ny, mz), ``evec`` (ny, kz, mz), ``rf_alpha``."""

    ne: int
    mz: int
    val: np.ndarray
    evec: np.ndarray
    rf_alpha: np.ndarray
    rf_lengthscale: np.ndarray
    rf_passes: int
    alpha_val: np.ndarray  # the vertical localisation's sqrt-eigenvalues (kz,)
    alpha_evec: np.ndarray  # (kz, kz)


def setup_alpha(ne: int, *, jy: int, kz: int, ds: float, config: HybridConfig = HybridConfig(), rf_passes: int = 6,
                znw=None, ptop: float | None = None, vertloc=None) -> AlphaB:
    """The alpha block of ``da_setup_be_regional`` (``sigma_alpha = alpha_std_dev``).

    ``vertloc`` is ``(alpha_val, alpha_evec)`` (option 1's ``be.vertloc.dat``,
    or WRFDA's own words for option 2); otherwise option 2 computes it from
    ``znw`` and ``ptop``.
    """
    from woof.da.var.be_dat import _vertical_truncation

    config.check_options()
    if ne < 2:
        raise ValueError(f"the alpha control variable needs ensdim_alpha >= 2 (got {ne}; ep is scaled by "
                         "1/sqrt(ne-1))")
    opt = config.alpha_vertloc_opt
    if opt > 0:
        if vertloc is None:
            if opt == 1:
                raise ValueError("alpha_vertloc_opt = 1 needs be.vertloc.dat (vertloc=read_vertloc(path))")
            if znw is None or ptop is None:
                raise ValueError("alpha_vertloc_opt = 2 needs znw and ptop")
            vertloc = alpha_vertloc(znw, float(ptop))
        a_val = np.asarray(vertloc[0], dtype=np.float64)
        a_evec = np.asarray(vertloc[1], dtype=np.float64)
        if a_val.size != kz or a_evec.shape != (kz, kz):
            raise ValueError(f"vertical localisation is for {a_val.size} levels, the domain has {kz}")
        # WRFDA truncates on the sqrt-eigenvalues: da_get_alpha_vertloc stores
        # eval = sqrt(eval) as alpha_val (and writes those to be.vertloc.dat),
        # and da_setup_be_regional passes alpha_val to da_get_vertical_truncation
        mz = _vertical_truncation(config.max_vert_var_alpha, a_val, kz)
    else:
        mz = 1
        a_val = np.zeros(kz)
        a_val[0] = 1.0
        a_evec = np.ones((kz, kz))
    sigma_alpha = config.alpha_std_dev
    val = np.empty((jy, mz))
    evec = np.empty((jy, kz, mz))
    for m in range(mz):
        val[:, m] = sigma_alpha * a_val[m]
        evec[:, :, m] = a_evec[None, :, m]
    ls = np.full(mz, (1000.0 * config.alpha_corr_scale) / ds)
    rf_alpha, sf = ref.calculate_rf_factors(ls, rf_passes)
    for m in range(mz):
        val[:, m] = sf[m] * val[:, m]
    return AlphaB(ne=int(ne), mz=int(mz), val=val, evec=evec, rf_alpha=np.asarray(rf_alpha), rf_lengthscale=ls,
                  rf_passes=int(rf_passes), alpha_val=a_val, alpha_evec=a_evec)


# ------------------------------------------------------------ perturbations


@dataclass
class Ensemble:
    """``ep``: the scaled perturbations, u, v, t, q (nx, ny, nz, ne) and ps (nx, ny, ne).

    u and v on mass points (A grid), t temperature (K), q specific humidity,
    ps surface pressure (Pa): ``da_setup_flow_predictors``'s variables for
    ``alphacv_method = 2``.  Arrays are NumPy or CuPy, binary64.
    """

    u: object
    v: object
    t: object
    q: object
    ps: object

    @property
    def ne(self) -> int:
        return int(self.ps.shape[-1])

    @property
    def shape(self) -> tuple:
        return tuple(int(n) for n in self.u.shape[:3])

    def arrays(self):
        return (self.u, self.v, self.t, self.q, self.ps)

    def __post_init__(self):
        # the device kernels index ep by these shapes: a mismatch would read out of bounds
        shape = tuple(int(n) for n in np.shape(self.u))
        if len(shape) != 4:
            raise ValueError(f"ensemble u must be (nx, ny, nz, ne), got {shape}")
        for k in ("v", "t", "q"):
            if tuple(int(n) for n in np.shape(getattr(self, k))) != shape:
                raise ValueError(f"ensemble {k} is {np.shape(getattr(self, k))}, u is {shape}")
        want = shape[:2] + shape[3:]
        if tuple(int(n) for n in np.shape(self.ps)) != want:
            raise ValueError(f"ensemble ps is {np.shape(self.ps)}, want {want} (nx, ny, ne)")
        if shape[3] < 2:
            raise ValueError(f"the ensemble needs at least 2 members (got {shape[3]})")

    @classmethod
    def from_perturbations(cls, perts: dict) -> "Ensemble":
        """``ep = (1/sqrt(ne-1)) * x'`` from unscaled perturbation stacks.

        ``perts`` maps u, v, t, q to (ne, nx, ny, nz) and ps to (ne, nx, ny)
        arrays (NumPy or CuPy): an ``ep`` file's contents, member by member.
        """
        xp = _xp(perts["u"])
        ne = int(perts["u"].shape[0])
        if ne < 2:
            raise ValueError(f"the ensemble needs at least 2 members (got {ne}; ep is scaled by 1/sqrt(ne-1))")
        scale = 1.0 / np.sqrt(float(ne - 1))
        out = {}
        for k in FIELDS:
            a = xp.asarray(perts[k], dtype=xp.float64)
            out[k] = xp.ascontiguousarray(xp.moveaxis(scale * a, 0, -1))
        return cls(**out)

    @classmethod
    def from_members(cls, members: dict) -> "Ensemble":
        """Perturbations about the members' mean (members summed in order, then ``/ ne``).

        ``members`` as :meth:`from_perturbations`'s ``perts`` but full
        states, e.g. the LETKF's member stacks on the device.
        """
        xp = _xp(members["u"])
        perts = {}
        for k in FIELDS:
            a = xp.asarray(members[k], dtype=xp.float64)
            acc = a[0].copy()
            for n in range(1, a.shape[0]):
                acc = acc + a[n]
            mean = acc / float(a.shape[0])
            perts[k] = a - mean[None]
        return cls.from_perturbations(perts)


def _xp(a):
    if type(a).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


def ep_paths(directory, ne: int, ep_format: int = 1) -> list:
    """The ``ep`` files :func:`read_ep` reads, in its order.

    Raises ValueError for a missing directory or file, naming the first.
    """
    if ep_format not in EP_FORMATS:
        raise NotImplementedError(f"woof var hybrid: ep_format = {ep_format} is not implemented (1 and 11 are)")
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    out = [directory / f"{name}.e{n:03d}" for n in range(1, ne + 1) for name in EP_FORMATS[ep_format][0]]
    missing = [f for f in out if not f.is_file()]
    if missing:
        raise ValueError(f"{directory} has {len(out) - len(missing)} of the {len(out)} ep files for {ne} members "
                         f"(ep_format {ep_format}); the first missing is {missing[0].name}")
    return out


def perturbations_sha256(perts: dict) -> str:
    """SHA-256 of unscaled perturbation stacks: each field's name, shape and little-endian binary64 words."""
    import hashlib

    h = hashlib.sha256()
    for k in FIELDS:
        a = np.ascontiguousarray(perts[k], dtype="<f8")
        h.update(f"{k}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def wrf_times(path) -> list:
    """A WRF file's ``Times`` records as strings (its ``START_DATE`` if it has none)."""
    import netCDF4

    with netCDF4.Dataset(Path(path)) as nc:
        nc.set_auto_mask(False)
        if "Times" in nc.variables:
            v = nc.variables["Times"]
            v.set_auto_chartostring(False)
            return [b"".join(np.asarray(r).ravel()).decode() for r in v[:]]
        return [str(nc.getncattr("START_DATE"))]


def read_ep(directory, ne: int, ep_format: int = 1, shape=None) -> dict:
    """WRFDA's ``ep`` files (``<dir>/<var>.e001`` ...) as unscaled stacks for :meth:`Ensemble.from_perturbations`."""
    if ep_format not in EP_FORMATS:
        raise NotImplementedError(f"woof var hybrid: ep_format = {ep_format} is not implemented (1 and 11 are)")
    names, kind = EP_FORMATS[ep_format]
    out = {k: [] for k in FIELDS}
    for n in range(1, ne + 1):
        for key, name in zip(FIELDS, names):
            end, recs = _records((Path(directory) / f"{name}.e{n:03d}").read_bytes())
            ni, nj, nk = struct.unpack(end + "3i", recs[0][:12])
            if key == "ps":
                nk = 1
            if shape is not None and (ni, nj) != tuple(shape[:2]):
                raise ValueError(f"{name}.e{n:03d} is {ni}x{nj}, the domain {shape[0]}x{shape[1]}")
            a = np.frombuffer(recs[1], dtype=end + kind, count=ni * nj * nk).astype(np.float64)
            a = a.reshape((ni, nj) if key == "ps" else (ni, nj, nk), order="F")
            out[key].append(a)
    return {k: np.stack(v) for k, v in out.items()}


def member_fields(path, time_index=0) -> dict:
    """One member's u, v, t, q, ps as ``gen_be_ep2`` builds them (the full state, not a perturbation).

    ``u`` and ``v`` averaged to mass points, ``t`` from ``theta = 300 + T`` and
    ``p = PB + P`` (``da_get_trh``), ``q = QVAPOR / (1 + QVAPOR)``, ``PSFC``;
    (nx, ny, nz) and (nx, ny), the layout of :class:`Ensemble`.
    """
    from woof.da.var import gen_be as gb

    f = gb.read_wrf_history(Path(path), int(time_index))
    big_u, big_v = f["U"], f["V"]
    dim1, dim2 = big_u.shape[0] - 1, big_v.shape[1] - 1
    theta = gb.T0 + f["T"]
    pres = f["PB"] + f["P"]
    x = f["QVAPOR"]
    return {"u": 0.5 * (big_u[0:dim1] + big_u[1:dim1 + 1]), "v": 0.5 * (big_v[:, 0:dim2] + big_v[:, 1:dim2 + 1]),
            "t": theta * np.power(pres / gb.BASE_PRES, gb.KAPPA), "q": x / (1.0 + x), "ps": f["PSFC"].copy()}


def ep_from_wrf(paths, *, time_index=0) -> dict:
    """``gen_be_ep2``: unscaled perturbations of WRF history files about their mean.

    ``paths`` are the members' history files (``wrfout``, or a woof history
    file with the same variables), valid at the analysis time, in member
    order.  Per member, as ``gen_be_ep2.f90``: ``u`` and ``v`` averaged to
    mass points, ``t`` from ``theta = 300 + T`` and ``p = PB + P``
    (``da_get_trh``), ``q = QVAPOR / (1 + QVAPOR)`` and ``PSFC``; the mean is
    its running form ``((n - 1) mean + x) / n`` (``1/n`` first), and each
    member's perturbation ``x - mean``.  ``time_index`` is the record read,
    one for all members or one per member (``gen_be_ep2`` reads the first,
    ``da_get_field.inc`` 47-50).  Returns the stacks
    :meth:`Ensemble.from_perturbations` takes, as :func:`read_ep` does.
    Hydrometeor perturbations (``mp_physics > 0``) are not read
    (``alpha_hydrometeors`` is not implemented).
    """
    paths = [Path(x) for x in paths]
    if len(paths) < 2:
        raise ValueError(f"the ensemble needs at least 2 members (got {len(paths)})")
    indices = [int(time_index)] * len(paths) if np.ndim(time_index) == 0 else [int(i) for i in time_index]
    if len(indices) != len(paths):
        raise ValueError(f"{len(indices)} time indices for {len(paths)} members")
    members = []
    shape = None
    for path, index in zip(paths, indices):
        m = member_fields(path, index)
        if shape is None:
            shape = m["t"].shape
        elif m["t"].shape != shape:
            raise ValueError(f"{path.name} is {m['t'].shape}, the first member {shape}")
        members.append(m)
    mean = {k: np.zeros_like(members[0][k]) for k in FIELDS}
    for n, m in enumerate(members, start=1):
        inv = 1.0 / float(n)
        for k in FIELDS:
            mean[k] = (float(n - 1) * mean[k] + m[k]) * inv
    return {k: np.stack([m[k] - mean[k] for m in members]) for k in FIELDS}


def write_ep(directory, perts: dict, ep_format: int = 1, endian: str = ">") -> None:
    """Unscaled perturbation stacks as WRFDA ``ep`` files (gen_be_ep2's layout)."""
    if ep_format not in EP_FORMATS:
        raise NotImplementedError(f"woof var hybrid: ep_format = {ep_format} is not implemented (1 and 11 are)")
    names, kind = EP_FORMATS[ep_format]
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ne = int(np.shape(perts["u"])[0])
    for n in range(ne):
        for key, name in zip(FIELDS, names):
            a = np.asarray(perts[key][n], dtype=np.float64)
            nk = 1 if key == "ps" else a.shape[2]
            head = _record(struct.pack(endian + "3i", a.shape[0], a.shape[1], nk), endian)
            body = _record(np.asarray(a, dtype=endian + kind).tobytes(order="F"), endian)
            (d / f"{name}.e{n + 1:03d}").write_bytes(head + body)


# ------------------------------------------------------------ the reference


def flow_xa(alpha_vp, ep: Ensemble) -> dict:
    """``da_calc_flow_dependence_xa``: ``xa_ens = sum_n alpha_n * ep_n`` from zero, n ascending."""
    a = np.asarray(alpha_vp, dtype=np.float64)
    nx, ny, nz, ne = a.shape
    out = {k: np.zeros((nx, ny, nz)) for k in ("u", "v", "t", "q")}
    out["psfc"] = np.zeros((nx, ny))
    e = {k: np.asarray(x) for k, x in zip(FIELDS, ep.arrays())}
    for n in range(ne):
        an = a[..., n]
        out["u"] = out["u"] + an * e["u"][..., n]
        out["v"] = out["v"] + an * e["v"][..., n]
        out["t"] = out["t"] + an * e["t"][..., n]
        out["q"] = out["q"] + an * e["q"][..., n]
        out["psfc"] = out["psfc"] + a[:, :, 0, n] * e["ps"][..., n]
    return out


def flow_xa_adj(xa: dict, ep: Ensemble) -> np.ndarray:
    """``da_calc_flow_dependence_xa_adj`` into a zeroed ``vp%alpha`` (nx, ny, nz, ne)."""
    e = {k: np.asarray(x) for k, x in zip(FIELDS, ep.arrays())}
    nx, ny, nz, ne = e["u"].shape
    out = np.zeros((nx, ny, nz, ne))
    for n in range(ne):
        out[:, :, 0, n] = out[:, :, 0, n] + e["ps"][..., n] * xa["psfc"]
        acc = np.zeros((nx, ny, nz))
        acc = acc + e["q"][..., n] * xa["q"]
        acc = acc + e["t"][..., n] * xa["t"]
        acc = acc + e["v"][..., n] * xa["v"]
        acc = acc + e["u"][..., n] * xa["u"]
        out[..., n] = out[..., n] + acc
    return out


class _HybridLayout:
    """The control vector: the static blocks v1..v5, then alpha (nx, ny, mz, ne) in Fortran order."""

    def _layout(self, b, alpha: AlphaB, ep: "Ensemble", nx: int, ny: int, nz: int, jb_factor: float):
        if ep.shape != (nx, ny, nz) or ep.ne != alpha.ne:
            raise ValueError(f"ensemble {ep.shape} x {ep.ne} does not match the domain ({nx}, {ny}, {nz}) "
                             f"x {alpha.ne}")
        mz = int(alpha.mz)
        for name, got, want in (("val", np.shape(alpha.val), (ny, mz)), ("evec", np.shape(alpha.evec), (ny, nz, mz))):
            if tuple(got) != want:
                raise ValueError(f"alpha {name} is {tuple(got)}, the domain needs {want}")
        if not 1 <= mz <= nz or np.size(alpha.rf_alpha) < mz:
            raise ValueError(f"alpha mz {mz} needs 1 <= mz <= {nz} and {mz} rf_alpha values")
        self.nx, self.ny, self.nz = nx, ny, nz
        self.alpha = alpha
        self.jb_factor = float(jb_factor)
        self.b = b if jb_factor > 0.0 else drop_static(b)
        self.static_mz = tuple(int(s.mz) for s in self.b.v[:5])
        self.cv_mz = self.static_mz + (alpha.mz, alpha.ne)  # WRFDA's be%cv_mz
        self.cv_size_jb = nx * ny * sum(self.static_mz)
        self.cv_size_je = nx * ny * alpha.mz * alpha.ne
        self.cv_size = self.cv_size_jb + self.cv_size_je
        # da_minimise_cg's precondition_cg blocks (WRFDA 4.7.1
        # var/da/da_minimisation/da_minimise_cg.inc, lines 113-126): per static
        # mode one nx*ny chunk, then sz(nv_precon) = size_alphac split into
        # cv_mz = alpha%mz contiguous chunks of nn = nx*ny*ne.  WRFDA's alpha cv is
        # (ix, jy, mz, ne) in Fortran order (da_cv_to_vv.inc, line 163), so
        # with mz > 1 those chunks cross level and member boundaries; this is
        # WRFDA's grouping, kept as it is (the cv here has the same order).
        self.precon_blocks = [(m, nx * ny) for m in self.static_mz] + [(alpha.mz, nx * ny * alpha.ne)]


class HybridReferenceU(_HybridLayout):
    """``U`` and ``U^T`` of the hybrid in NumPy binary64 (:class:`~woof.da.var.cost.ReferenceU`'s interface)."""

    precision = "reference"
    dtype = np.float64

    def __init__(self, b, alpha: AlphaB, ep: Ensemble, xb: dict, jb_factor: float):
        nx, ny = (int(n) for n in np.shape(xb["grid_box_area"]))
        self._layout(b, alpha, ep, nx, ny, int(b.kz), jb_factor)
        self.ep = Ensemble(*(np.asarray(_host(a), dtype=np.float64) for a in ep.arrays()))
        self.xb = xb

    def _alpha_vv(self, cva):
        a = self.alpha
        vv = np.asarray(cva, dtype=np.float64).reshape((self.nx, self.ny, a.mz, a.ne), order="F").copy()
        for n in range(a.ne):
            vv[..., n] = ref.transform_through_rf(vv[..., n], a.rf_alpha, self.xb["grid_box_area"], a.rf_passes)
        return vv

    def apply(self, cv, stages: bool = False) -> dict:
        cv = np.asarray(_host(cv), dtype=np.float64)
        if cv.size != self.cv_size:
            raise ValueError(f"cv has {cv.size} values; the hybrid's is {self.cv_size}")
        nx, ny, nz, jb = self.nx, self.ny, self.nz, self.cv_size_jb
        if jb > 0:
            xa = ref.transform_u(cv[:jb], self.b, self.xb, with_xtoxa=False)["xa"]
        else:
            xa = {k: np.zeros((nx, ny, nz)) for k in ("u", "v", "t", "q")}
            xa["psfc"] = np.zeros((nx, ny))
        a = self.alpha
        vv = self._alpha_vv(cv[jb:])
        vp = np.stack([ref.vvtovp(a.evec, a.val, vv[..., n], nz) for n in range(a.ne)], axis=-1)
        ens = flow_xa(vp, self.ep)
        for k in ("u", "v", "t", "q", "psfc"):
            xa[k] = xa[k] + ens[k]
        xa = ref.xtoxa(xa, self.xb, self.xb["c1h"], self.xb["c2h"])
        out = {"xa": xa}
        if stages:
            out.update(alpha_vv=vv, alpha_vp=vp, xa_ens=ens)
        return out

    def adjoint(self, xa: dict, stages: bool = False) -> dict:
        xa = {k: np.asarray(_host(v), dtype=np.float64) for k, v in xa.items()}
        xa1 = ref.xtoxa_adj(xa, self.xb, self.xb["c1h"], self.xb["c2h"])
        avp = flow_xa_adj(xa1, self.ep)
        parts = []
        if self.cv_size_jb > 0:
            parts.append(ref.transform_u_adj(xa1, self.b, self.xb, with_xtoxa=False)["cv"])
        a = self.alpha
        vv = np.stack([ref.vvtovp_adj(a.evec, a.val, avp[..., n], a.mz) for n in range(a.ne)], axis=-1)
        vv_eof = vv.copy()
        for n in range(a.ne):
            vv[..., n] = ref.transform_through_rf_adj(vv[..., n], a.rf_alpha, self.xb["grid_box_area"], a.rf_passes)
        parts.append(vv.reshape(-1, order="F"))
        out = {"cv": np.concatenate(parts)}
        if stages:
            out.update(alpha_vp=avp, alpha_vv_eof=vv_eof, alpha_vv=vv)
        return out


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


# ------------------------------------------------------------ the device


class HybridDeviceU(_HybridLayout):
    """The hybrid ``U`` and ``U^T`` as kernel launches (``kernels/var_u.cu``), in the reference's order.

    The static part is :class:`~woof.da.var.transforms.DeviceU`'s; the alpha
    fields run the same recursive-filter line kernel (``ne * mz`` lines per
    row), ``var_alpha_vvtovp`` and ``var_flow_xa`` (and their adjoints).
    """

    def __init__(self, b, alpha: AlphaB, ep: Ensemble, xb: dict, jb_factor: float, precision: str = "oracle"):
        import cupy as cp

        from woof.da.var.transforms import DeviceU

        nx, ny = (int(n) for n in np.shape(xb["grid_box_area"]))
        self._layout(b, alpha, ep, nx, ny, int(b.kz), jb_factor)
        self.static = DeviceU(self.b, xb, precision)
        self.precision = precision
        self.dtype = self.static.dtype
        dev = lambda a: cp.ascontiguousarray(cp.asarray(a, dtype=self.dtype))  # noqa: E731
        self.ep = tuple(dev(a) for a in ep.arrays())
        self.aval = dev(alpha.val)
        self.aevec = dev(alpha.evec)
        self.arf = dev(np.tile(np.asarray(alpha.rf_alpha)[: alpha.mz], alpha.ne))

    def _launch(self, name, total, *args):
        self.static._launch(name, total, *args)

    def _i(self, *v):
        return tuple(np.int32(x) for x in v)

    def _rf(self, f, adjoint: bool):
        """``da_transform_through_rf(_adj)`` of every member's alpha (``var_rf_lines5`` with one field)."""
        nx, ny = self.nx, self.ny
        mz = self.alpha.mz * self.alpha.ne
        npass = self.alpha.rf_passes // 2
        first, last = (1, 0) if adjoint else (0, 1)
        for n, axis in enumerate((1, 0) if adjoint else (0, 1)):
            lines = (ny if axis == 0 else nx) * mz
            pre, post = (first, -1) if n == 0 else (-1, last)
            self._launch("var_rf_lines5", lines, f, f, f, f, f, self.arf, self.arf, self.arf, self.arf, self.arf,
                         *self._i(mz, 0, 0, 0, 0, nx, ny, axis, npass, 1 if adjoint else 0), self.static.px,
                         *self._i(pre, post))

    def _evec_args(self):
        e, v = self.aevec, self.aval
        return self._i(self.nx, self.ny, self.nz, self.alpha.mz, self.alpha.ne, e.shape[1], e.shape[2], v.shape[1])

    def apply(self, cv, stages: bool = False) -> dict:
        import cupy as cp

        nx, ny, nz, jb = self.nx, self.ny, self.nz, self.cv_size_jb
        cv = cp.ascontiguousarray(cp.asarray(cv, dtype=self.dtype)).ravel()
        if cv.size != self.cv_size:
            raise ValueError(f"cv has {cv.size} values; the hybrid's is {self.cv_size}")
        if jb > 0:
            xa = self.static.apply(cv[:jb], with_xtoxa=False)["xa"]
        else:
            xa = {k: self.static._zeros(nx, ny, nz) for k in ("u", "v", "t", "q")}
            xa["psfc"] = self.static._zeros(nx, ny)
        a = self.alpha
        mzn = a.mz * a.ne
        vv = self.static._zeros(nx, ny, mzn)
        self._launch("var_cv_block", nx * ny * mzn, cp.ascontiguousarray(cv[jb:]), vv, *self._i(nx, ny, mzn, 0))
        self._rf(vv, adjoint=False)
        vp = self.static._zeros(nx, ny, nz, a.ne)
        self._launch("var_alpha_vvtovp", vp.size, self.aevec, self.aval, vv, vp, *self._evec_args())
        out = {"alpha_vv": vv.copy(), "alpha_vp": vp.copy()} if stages else {}
        self._launch("var_flow_xa", nx * ny * nz, vp, *self.ep, xa["u"], xa["v"], xa["t"], xa["q"], xa["psfc"],
                     *self._i(nx, ny, nz, a.ne))
        self.static._xtoxa(xa)
        out["xa"] = xa
        return out

    def adjoint(self, xa: dict, stages: bool = False) -> dict:
        import cupy as cp

        from woof.da.var.transforms import XA_KEYS

        nx, ny, nz = self.nx, self.ny, self.nz
        a = {k: cp.array(xa[k], dtype=self.dtype, copy=True, order="C") for k in XA_KEYS}
        a["psfc"] = cp.ascontiguousarray(a["psfc"].reshape(nx, ny))
        self.static._xtoxa_adj(a)
        al = self.alpha
        avp = self.static._zeros(nx, ny, nz, al.ne)
        self._launch("var_flow_xa_adj", avp.size, avp, *self.ep, a["u"], a["v"], a["t"], a["q"], a["psfc"],
                     *self._i(nx, ny, nz, al.ne))
        parts = []
        if self.cv_size_jb > 0:
            parts.append(self.static.adjoint(a, with_xtoxa=False)["cv"])
        mzn = al.mz * al.ne
        vv = self.static._zeros(nx, ny, mzn)
        self._launch("var_alpha_vvtovp_adj", vv.size, self.aevec, self.aval, avp, vv, *self._evec_args())
        out = {"alpha_vp": avp, "alpha_vv_eof": vv.copy()} if stages else {}
        self._rf(vv, adjoint=True)
        block = self.static._zeros(nx * ny * mzn)
        self._launch("var_cv_block", block.size, block, vv, *self._i(nx, ny, mzn, 1))
        parts.append(block)
        if stages:
            out["alpha_vv"] = vv
        out["cv"] = cp.concatenate(parts)
        return out


# ------------------------------------------------------------ the cost


def hybrid_cost(b, xb: dict, obs, ep: Ensemble, config: HybridConfig, *, tier: str = "reference",
                vertloc=None, **kw):
    """A :class:`~woof.da.var.cost.VarCost` with the alpha control variable.

    ``xb`` needs :class:`~woof.da.var.cost.ReferenceU`'s keys, ``ds`` and,
    for ``alpha_vertloc_opt = 2`` without ``vertloc``, ``znw`` and ``ptop``.
    """
    from woof.da.var.cost import VarCost

    config.check()
    nx, ny = (int(n) for n in np.shape(xb["grid_box_area"]))
    jb_factor = jb_factor_for(ep.ne, config.je_factor)
    alpha = setup_alpha(ep.ne, jy=ny, kz=int(b.kz), ds=float(xb["ds"]), config=config, rf_passes=b.rf_passes,
                        znw=xb.get("znw"), ptop=xb.get("ptop"), vertloc=vertloc)
    keys = ("grid_box_area", "coefx", "coefy", "qs", "q", "psac", "dnw", "rho", "p", "t", "c1h", "c2h")
    xbu = {k: xb[k] for k in keys}
    if tier == "reference":
        u = HybridReferenceU(b, alpha, ep, xbu, jb_factor)
    else:
        u = HybridDeviceU(b, alpha, ep, xbu, jb_factor, tier)
    return VarCost(u, obs, tier=tier, jb_factor=jb_factor, je_factor=config.je_factor, **kw)

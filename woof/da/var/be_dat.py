"""WRFDA ``be.dat`` reader and ``da_setup_be_regional`` in binary64.

``be.dat`` is a Fortran sequential unformatted file (4-byte record markers)
written by gen_be.  :func:`read_be_dat` reads it byte for byte: it detects
the byte order from the first record marker (12: three 4-byte integers) and
the real size from the ``lat_min`` record (24 bytes means real*8), reads the
optional ``eta_be`` record, and refuses a variable-name mismatch, trailing
bytes, or a record whose two markers disagree.  Values are kept in binary64
(real*4 files are widened, which is exact); nothing is rounded to a narrower
type before the device upload.

:func:`setup_be_regional` is ``da_setup_be_regional`` for the regional
path with ``interpolate_stats = .false.`` (the be.dat levels are the
model's), ``vert_corr = 2`` and ``use_rf = .true.``: regression
coefficients by latitude bin (averaged over ``j`` unless ``bin_type = 5`` or
``lat_stats_option``), vertical truncation by ``max_vert_var``, eigenvalues
and eigenvectors (global or per-latitude by ``vert_evalue``), and the
recursive-filter alphas and variance rescaling of
``da_rescale_background_errors``.  Every array equals WRFDA's word for word
(gate O1, ``tests/test_var_be_dat.py``).  ``interpolate_stats``,
``cloud_cv_options >= 2``, ``use_cv_w`` and WRF-Chem control variables are
refused by name.

Derived from WRFDA 4.7.1 (``var/da/da_setup_structures``), public domain;
see ``licenses/LICENSE-WRF-public-domain.txt``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from woof.da.var.reference import calculate_rf_factors, seq_sum

__all__ = [
    "BeDat",
    "BeEigen",
    "BeSub",
    "BeConfig",
    "RegionalB",
    "read_be_dat",
    "setup_be_regional",
    "write_be_dat",
]

# Accepted names per control variable, as da_setup_be_regional's select cases.
_CV3D_NAMES = (("psi", "u"), ("chi_u", "v"), ("t_u", "t"), ("rh_u", "rh"))
_CV5_NAMES = ("ps_u", "ps")
# cv_options = 6's nine regression records, by name, in WRFDA's select order.
_CV6_REGRESSIONS = {
    "regcoeff_psi_chi": "psi_chi",
    "regcoeff_psi_t": "psi_t",
    "regcoeff_psi_ps": "psi_ps",
    "regcoeff_psi_rh": "psi_rh",
    "regcoeff_chi_u_t": "chi_u_t",
    "regcoeff_chi_u_ps": "chi_u_ps",
    "regcoeff_chi_u_rh": "chi_u_rh",
    "regcoeff_t_u_rh": "t_u_rh",
    "regcoeff_ps_u_rh": "ps_u_rh",
}
# Shapes of the regression records: "1" (num_bins), "2" (nk, nb2d), "3" (nk, nk, nb2d).
_REG_RANK = {
    "psi_chi": 1,
    "psi_ps": 2,
    "psi_t": 3,
    "psi_rh": 3,
    "chi_u_t": 3,
    "chi_u_ps": 2,
    "chi_u_rh": 3,
    "t_u_rh": 3,
    "ps_u_rh": 2,
}


class BeDatError(ValueError):
    """``be.dat`` does not have the layout WRFDA reads."""


class _Records:
    """Fortran sequential unformatted records with 4-byte markers."""

    def __init__(self, data: bytes):
        self.data = data
        if len(data) < 8:
            raise BeDatError("be.dat: shorter than one record")
        head_le = int.from_bytes(data[:4], "little")
        head_be = int.from_bytes(data[:4], "big")
        if head_be == 12:
            self.endian = ">"
        elif head_le == 12:
            self.endian = "<"
        else:
            raise BeDatError(
                f"be.dat: first record marker is {head_be} (big) / {head_le} (little); "
                "expected 12 (ni, nj, nk)"
            )
        # (offset, length) views into the one buffer, not copies of it
        view = memoryview(data)
        self.records: list[memoryview] = []
        off = 0
        order = "big" if self.endian == ">" else "little"
        while off < len(data):
            if off + 4 > len(data):
                raise BeDatError(f"be.dat: {len(data) - off} trailing bytes at offset {off}")
            n = int.from_bytes(data[off : off + 4], order, signed=True)
            end = off + 4 + n
            if n < 0 or end + 4 > len(data):
                raise BeDatError(f"be.dat: record at offset {off} (length {n}) runs past the end")
            tail = int.from_bytes(data[end : end + 4], order, signed=True)
            if tail != n:
                raise BeDatError(f"be.dat: record at offset {off}: markers {n} and {tail} disagree")
            self.records.append(view[off + 4 : end])
            off = end + 4
        self.pos = 0

    def peek_len(self, ahead: int = 0) -> int | None:
        i = self.pos + ahead
        return len(self.records[i]) if i < len(self.records) else None

    def next(self) -> memoryview:
        if self.pos >= len(self.records):
            raise BeDatError("be.dat: ended before the records WRFDA reads")
        r = self.records[self.pos]
        self.pos += 1
        return r

    def ints(self, count: int | None = None) -> np.ndarray:
        r = self.next()
        a = np.frombuffer(r, dtype=self.endian + "i4")
        if count is not None and a.size < count:
            raise BeDatError(f"be.dat: integer record has {a.size} values, {count} read")
        return a if count is None else a[:count]

    def reals(self, shape, rsize: int) -> np.ndarray:
        """A record read into an array of ``shape`` (column-major).

        As Fortran's ``read``, a record longer than the list is allowed (the
        rest is skipped); a shorter one is an error.
        """
        r = self.next()
        count = int(np.prod(shape))
        if len(r) < count * rsize:
            raise BeDatError(f"be.dat: real record has {len(r)} bytes, {count} values of {rsize} read")
        a = np.frombuffer(r[: count * rsize], dtype=f"{self.endian}f{rsize}").astype(np.float64)
        return a.reshape(shape, order="F")

    def name(self, width: int = 10) -> str:
        """A ``character*width`` read: the record's first ``width`` bytes, trimmed."""
        return bytes(self.next()[:width]).decode("ascii", errors="replace").strip()


@dataclass(frozen=True)
class BeEigen:
    """One variable's vertical EOF records (gen_be order)."""

    name: str
    evec_glo: np.ndarray  # (nk, nk)
    eval_glo: np.ndarray  # (nk,)
    evec_loc: np.ndarray  # (nk, nk, num_bins2d)
    eval_loc: np.ndarray  # (nk, num_bins2d)


@dataclass(frozen=True)
class BeDat:
    """A parsed ``be.dat`` (no model-grid processing)."""

    ni: int
    nj: int
    nk: int
    bin_type: int
    eta_be: np.ndarray | None
    lat_min: float
    lat_max: float
    binwidth_lat: float
    hgt_min: float
    hgt_max: float
    binwidth_hgt: float
    num_bins: int
    num_bins2d: int
    bin: np.ndarray  # (ni, nj, nk), 1-based bin numbers
    bin2d: np.ndarray  # (ni, nj)
    cv_options: int
    regcoeff: dict  # name -> array; see _REG_RANK
    eigen: tuple  # BeEigen x 4 (psi, chi_u, t_u, rh_u), in v1..v4 order
    eigen_ps: BeEigen  # nk = 1
    rf_lengthscale: tuple  # 4 arrays (nk,) in v1..v4 order
    rf_lengthscale_ps: np.ndarray  # (1,)
    endian: str
    real_size: int
    sha256: str


def read_be_dat(path, cv_options: int = 5) -> BeDat:
    """Read a WRFDA ``be.dat`` as ``da_setup_be_regional`` does.

    ``cv_options`` selects the regression records (5: three, 6: nine named,
    7: none), as in WRFDA, where it comes from the namelist, not the file.
    """
    if cv_options not in (5, 6, 7):
        raise BeDatError(f"cv_options should be 5, 6, or 7 (got {cv_options})")
    data = Path(path).read_bytes()
    rec = _Records(data)
    ni, nj, nk = (int(v) for v in rec.ints(3))
    bin_type = int(rec.ints(1)[0])
    # Optional eta_be (written by gen_be for interpolate_stats).  The lat and
    # hgt records are two 3-real records followed by the 8-byte bin counts.
    eta_be = None
    l0, l1, l2 = rec.peek_len(0), rec.peek_len(1), rec.peek_len(2)
    if not (l0 in (12, 24) and l1 == l0 and l2 == 8):
        r = rec.next()
        if len(r) == 4 * nk:
            eta_be = np.frombuffer(r, dtype=rec.endian + "f4").astype(np.float64)
        elif len(r) == 8 * nk:
            eta_be = np.frombuffer(r, dtype=rec.endian + "f8").copy()
        else:
            raise BeDatError(f"be.dat: record after bin_type has {len(r)} bytes; expected eta_be or lat_min")
    lat_len = rec.peek_len(0)
    if lat_len not in (12, 24):
        raise BeDatError(f"be.dat: lat_min record has {lat_len} bytes; expected 12 or 24")
    rsize = lat_len // 3
    lat_min, lat_max, binwidth_lat = (float(v) for v in rec.reals((3,), rsize))
    hgt_min, hgt_max, binwidth_hgt = (float(v) for v in rec.reals((3,), rsize))
    num_bins, num_bins2d = (int(v) for v in rec.ints(2))
    bins = rec.ints(ni * nj * nk).reshape((ni, nj, nk), order="F").astype(np.int64)
    bins2d = rec.ints(ni * nj).reshape((ni, nj), order="F").astype(np.int64)

    reg: dict = {}

    def reg_shape(key):
        rank = _REG_RANK[key]
        return {1: (num_bins,), 2: (nk, num_bins2d), 3: (nk, nk, num_bins2d)}[rank]

    if cv_options == 5:
        for key in ("psi_chi", "psi_ps", "psi_t"):
            reg[key] = rec.reals(reg_shape(key), rsize)
    elif cv_options == 6:
        for _ in range(len(_CV6_REGRESSIONS)):
            nm = rec.name(80)
            key = _CV6_REGRESSIONS.get(nm)
            if key is None:
                raise BeDatError(f"be.dat: unknown regression coefficient name {nm!r}")
            reg[key] = rec.reals(reg_shape(key), rsize)

    eig: list = [None] * 4
    for _ in range(4):
        nm = rec.name()
        nk_v, nb2d = (int(v) for v in rec.ints(2))
        idx = next((i for i, names in enumerate(_CV3D_NAMES) if nm in names), None)
        if idx is None:
            raise BeDatError(
                f"be.dat: eigenvectors for variable {nm!r}; expected one of "
                f"{[n for names in _CV3D_NAMES for n in names]} for cv_options={cv_options}"
            )
        # WRFDA allocates evec_loc/eval_loc once from the first variable's sizes.
        if eig.count(None) == 4:
            nk_alloc, nb_alloc = nk_v, nb2d
        evec_glo = rec.reals((nk, nk), rsize)
        eval_glo = rec.reals((nk,), rsize)
        evec_loc = rec.reals((nk_alloc, nk_alloc, nb_alloc), rsize)
        eval_loc = rec.reals((nk_alloc, nb_alloc), rsize)
        eig[idx] = BeEigen(nm, evec_glo, eval_glo, evec_loc, eval_loc)
    if any(e is None for e in eig):
        raise BeDatError("be.dat: a control variable's eigenvectors appear twice")
    nm = rec.name()
    nk_2d, nb2d = (int(v) for v in rec.ints(2))
    evec5 = rec.reals((1, 1), rsize)
    eval5 = rec.reals((1,), rsize)
    evec5_loc = rec.reals((nk_2d, nk_2d, nb2d), rsize)
    eval5_loc = rec.reals((nk_2d, nb2d), rsize)
    if nm not in _CV5_NAMES:
        exp = "ps" if cv_options == 7 else "ps_u"
        raise BeDatError(f"be.dat: variable mismatch: expected {exp} but got {nm!r}")
    eig_ps = BeEigen(nm, evec5, eval5, evec5_loc, eval5_loc)

    rfls: list = [None] * 4
    for _ in range(4):
        nm = rec.name()
        idx = next((i for i, names in enumerate(_CV3D_NAMES) if nm in names), None)
        if idx is None:
            raise BeDatError(f"be.dat: lengthscales for variable {nm!r}")
        rfls[idx] = rec.reals((nk,), rsize)
    if any(r is None for r in rfls):
        raise BeDatError("be.dat: a control variable's lengthscales appear twice")
    nm = rec.name()
    if nm not in _CV5_NAMES:
        raise BeDatError(f"be.dat: lengthscales for variable {nm!r}; expected ps_u or ps")
    rfls_ps = rec.reals((1,), rsize)
    if rec.pos != len(rec.records):
        raise BeDatError(f"be.dat: {len(rec.records) - rec.pos} trailing records")
    return BeDat(
        ni=ni,
        nj=nj,
        nk=nk,
        bin_type=bin_type,
        eta_be=eta_be,
        lat_min=lat_min,
        lat_max=lat_max,
        binwidth_lat=binwidth_lat,
        hgt_min=hgt_min,
        hgt_max=hgt_max,
        binwidth_hgt=binwidth_hgt,
        num_bins=num_bins,
        num_bins2d=num_bins2d,
        bin=bins,
        bin2d=bins2d,
        cv_options=cv_options,
        regcoeff=reg,
        eigen=tuple(eig),
        eigen_ps=eig_ps,
        rf_lengthscale=tuple(rfls),
        rf_lengthscale_ps=rfls_ps,
        endian=rec.endian,
        real_size=rsize,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def write_be_dat(path, be: BeDat, *, endian: str = ">", real_size: int = 8) -> str:
    """Write ``be`` in WRFDA's layout (the record order :func:`read_be_dat` reads).

    Returns the file's SHA-256.  Names are written as WRFDA's
    ``character*10`` records.  ``real_size = 4`` narrows the reals (lossy
    unless they are binary32 values).
    """
    if endian not in (">", "<"):
        raise ValueError("endian must be '>' or '<'")
    fdt = f"{endian}f{real_size}"
    idt = f"{endian}i4"
    order = "big" if endian == ">" else "little"
    out = bytearray()

    def rec(payload: bytes):
        n = len(payload).to_bytes(4, order, signed=True)
        out.extend(n + payload + n)

    def reals(a):
        rec(np.asarray(a, dtype=np.float64).astype(fdt).tobytes(order="F"))

    def ints(a):
        rec(np.asarray(a).astype(idt).tobytes(order="F"))

    def name(s, width=10):
        rec(f"{s:<{width}}"[:width].encode("ascii"))

    ints([be.ni, be.nj, be.nk])
    ints([be.bin_type])
    if be.eta_be is not None:
        reals(be.eta_be)
    reals([be.lat_min, be.lat_max, be.binwidth_lat])
    reals([be.hgt_min, be.hgt_max, be.binwidth_hgt])
    ints([be.num_bins, be.num_bins2d])
    ints(be.bin)
    ints(be.bin2d)
    if be.cv_options == 5:
        for key in ("psi_chi", "psi_ps", "psi_t"):
            reals(be.regcoeff[key])
    elif be.cv_options == 6:
        for nm, key in _CV6_REGRESSIONS.items():
            name(nm, 80)
            reals(be.regcoeff[key])
    for e in list(be.eigen) + [be.eigen_ps]:
        name(e.name)
        ints([e.eval_glo.size, e.eval_loc.shape[-1]])
        reals(e.evec_glo)
        reals(e.eval_glo)
        reals(e.evec_loc)
        reals(e.eval_loc)
    for e, r in zip(be.eigen, be.rf_lengthscale):
        name(e.name)
        reals(r)
    name(be.eigen_ps.name)
    reals(be.rf_lengthscale_ps)
    data = bytes(out)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class BeConfig:
    """The namelist keys ``da_setup_be_regional`` reads (WRFDA names, defaults)."""

    cv_options: int = 5
    rf_passes: int = 6
    vert_corr: int = 2
    vert_evalue: int = 1
    max_vert_var1: float = 99.0
    max_vert_var2: float = 99.0
    max_vert_var3: float = 99.0
    max_vert_var4: float = 99.0
    max_vert_var5: float = 99.0
    var_scaling1: float = 1.0
    var_scaling2: float = 1.0
    var_scaling3: float = 1.0
    var_scaling4: float = 1.0
    var_scaling5: float = 1.0
    len_scaling1: float = 1.0
    len_scaling2: float = 1.0
    len_scaling3: float = 1.0
    len_scaling4: float = 1.0
    len_scaling5: float = 1.0
    psi_chi_factor: float = 1.0
    psi_t_factor: float = 1.0
    psi_ps_factor: float = 1.0
    psi_rh_factor: float = 1.0
    chi_u_t_factor: float = 1.0
    chi_u_ps_factor: float = 1.0
    chi_u_rh_factor: float = 1.0
    t_u_rh_factor: float = 1.0
    ps_u_rh_factor: float = 1.0
    lat_stats_option: bool = False
    interpolate_stats: bool = False
    use_rf: bool = True
    cloud_cv_options: int = 0
    use_cv_w: bool = False
    jb_factor: float = 1.0


@dataclass(frozen=True)
class BeSub:
    """One control variable's ``be_subtype`` after setup (WRFDA's ``be%vN``)."""

    name: str
    mz: int
    val: np.ndarray  # (jy, mz): sqrt(eigenvalue) * var_scaling * rf_scale_factor
    evec: np.ndarray  # (jy, kz, mz)
    rf_alpha: np.ndarray  # (mz,)
    val_g: np.ndarray  # (mz,) global eigenvalues
    evec_g: np.ndarray  # (kz, mz)
    rf_lengthscale: np.ndarray  # (mz,) non-dimensional, after len_scaling


@dataclass(frozen=True)
class RegionalB:
    """``be_type`` for the regional cv5/6/7 path, binary64."""

    v: tuple  # BeSub x 5 (v1..v5)
    reg: dict = field(default_factory=dict)  # "psi_chi" (jy,kz), "psi_t" (jy,kz,kz), ...
    ix: int = 0
    jy: int = 0
    kz: int = 0
    cv_options: int = 5
    rf_passes: int = 6  # the passes rf_alpha and the variance scale were normalised for
    be_sha256: str = ""

    @property
    def cv_mz(self) -> tuple:
        return tuple(s.mz for s in self.v)


def _vertical_truncation(max_vert_var: float, eigenval: np.ndarray, mz0: int) -> int:
    """``da_get_vertical_truncation``; ``mz0`` is the value WRFDA had before."""
    kz = eigenval.size
    mz = mz0
    if max_vert_var >= 100.0:
        mz = kz
        for k in range(kz):
            if eigenval[k] <= 0.0:
                mz = k
                break
        return mz
    tot = 0.0
    for k in range(kz):
        tot = tot + float(eigenval[k])
    cum = 0.0
    for k in range(kz):
        cum = cum + float(eigenval[k])
        if eigenval[k] <= 1e-12:
            mz = k
            break
        if cum / tot >= 0.01 * max_vert_var:
            mz = k + 1
            break
    if max_vert_var == 0.0:
        mz = 0
    if mz == 0:
        mz = 1
    return mz


def _allocate_be_sub(jy, kz, eval_glo, evec_glo, eval_loc, evec_loc, name, mz, vert_evalue):
    """``da_allocate_background_errors`` for ``vert_corr = 2``."""
    if mz <= 0:
        return BeSub(name, 0, np.zeros((jy, 0)), np.zeros((jy, kz, 0)), np.zeros(0), np.zeros(0),
                     np.zeros((kz, 0)), np.zeros(0))
    if vert_evalue == 1:
        val = np.empty((jy, mz))
        for m in range(mz):
            val[:, m] = np.sqrt(eval_glo[m])
        evec = np.empty((jy, kz, mz))
        evec[:, :, :] = evec_glo[None, :kz, :mz]
    else:
        if np.any(eval_loc[:, :mz] <= 0):
            raise ValueError(f"{name}: non-positive local eigenvalue with vert_evalue=2")
        val = np.sqrt(eval_loc[:, :mz])
        evec = np.array(evec_loc[:, :kz, :mz])
    return BeSub(name, mz, val, evec, np.ones(mz), np.array(eval_glo[:mz]),
                 np.array(evec_glo[:kz, :mz]), np.zeros(mz))


def _rescale(be_sub: BeSub, var_scaling: float, len_scaling: float, ds: float,
             s: np.ndarray, rf_passes: int) -> BeSub:
    """``da_rescale_background_errors`` (``s`` in metres on entry, per mode)."""
    mz = be_sub.mz
    if mz <= 0:
        return be_sub
    s = np.array(s[:mz], dtype=np.float64)
    s = len_scaling * s
    s = s / ds
    val = var_scaling * be_sub.val
    rf_alpha, sf = calculate_rf_factors(s, rf_passes)
    for m in range(mz):
        val[:, m] = sf[m] * val[:, m]
    return BeSub(be_sub.name, mz, val, be_sub.evec, rf_alpha, be_sub.val_g, be_sub.evec_g, s)


def setup_be_regional(be: BeDat, *, ix: int, jy: int, kz: int, ds: float,
                      config: BeConfig = BeConfig()) -> RegionalB:
    """``da_setup_be_regional`` (regional, ``interpolate_stats = .false.``).

    ``ix, jy, kz`` are the mass-grid sizes (WRFDA's ``ide-1``, ``jde-1``,
    ``kde-1``); ``ds`` is the grid spacing in metres (``xb%ds``).
    """
    c = config
    for key, bad in (
        ("interpolate_stats", c.interpolate_stats),
        ("cloud_cv_options >= 2", c.cloud_cv_options >= 2),
        ("use_cv_w", c.use_cv_w),
        ("use_rf = .false. (wavelet B)", not c.use_rf),
        ("vert_corr = 1", c.vert_corr != 2),
        ("jb_factor <= 0", c.jb_factor <= 0.0),
    ):
        if bad:
            raise NotImplementedError(f"woof var: {key} is not implemented")
    if c.cv_options != be.cv_options:
        raise ValueError(f"be.dat was read for cv_options={be.cv_options}, config has {c.cv_options}")
    if c.vert_evalue not in (1, 2):
        raise ValueError(f"vert_evalue must be 1 or 2 (got {c.vert_evalue})")
    if (be.ni, be.nj, be.nk) != (ix, jy, kz):
        raise ValueError(
            f"be.dat dimensions ({be.ni}, {be.nj}, {be.nk}) do not match the model's "
            f"({ix}, {jy}, {kz}); interpolate_stats is not implemented"
        )
    if be.eta_be is not None:
        raise NotImplementedError("woof var: be.dat carries eta_be, which only interpolate_stats reads")
    b2 = be.bin2d[0, :] - 1  # bin2d(1,j), 0-based

    reg: dict = {}
    if c.cv_options != 7:
        rc = be.regcoeff
        f = {"psi_chi": c.psi_chi_factor, "psi_ps": c.psi_ps_factor, "psi_t": c.psi_t_factor,
             "psi_rh": c.psi_rh_factor, "chi_u_t": c.chi_u_t_factor, "chi_u_ps": c.chi_u_ps_factor,
             "chi_u_rh": c.chi_u_rh_factor, "t_u_rh": c.t_u_rh_factor, "ps_u_rh": c.ps_u_rh_factor}
        b3 = be.bin[0, :, :] - 1  # bin(1,j,k), (nj, nk)
        reg["psi_chi"] = f["psi_chi"] * rc["psi_chi"][b3]
        keys2 = ["psi_ps"] + (["ps_u_rh", "chi_u_ps"] if c.cv_options == 6 else [])
        keys3 = ["psi_t"] + (["psi_rh", "chi_u_t", "chi_u_rh", "t_u_rh"] if c.cv_options == 6 else [])
        for key in keys2:
            # regcoeff(k, b) -> reg(j, k)
            reg[key] = f[key] * rc[key][:, b2].T
        for key in keys3:
            # regcoeff(i, k, b) -> reg(j, i, k)
            reg[key] = f[key] * np.moveaxis(rc[key][:, :, b2], 2, 0)
        if not c.lat_stats_option and be.bin_type != 5:
            for key in keys2 + ["psi_chi"] + keys3:
                a = reg[key]
                mean = seq_sum(a, axis=0) / float(jy)
                reg[key] = np.broadcast_to(mean, a.shape).copy()

    subs = []
    mvv = (c.max_vert_var1, c.max_vert_var2, c.max_vert_var3, c.max_vert_var4)
    for i in range(4):
        e = be.eigen[i]
        mz = _vertical_truncation(mvv[i], e.eval_glo, 1)
        eval_loc = e.eval_loc[:, b2].T  # (nj, nk)
        evec_loc = np.moveaxis(e.evec_loc[:, :, b2], 2, 0)  # (nj, nk, nk)
        subs.append(_allocate_be_sub(jy, kz, e.eval_glo, e.evec_glo, eval_loc, evec_loc,
                                     e.name, mz, c.vert_evalue))
    e = be.eigen_ps
    mz5 = 0 if c.max_vert_var5 == 0.0 else 1
    subs.append(_allocate_be_sub(jy, 1, e.eval_glo, e.evec_glo, e.eval_loc[:, b2].T,
                                 np.moveaxis(e.evec_loc[:, :, b2], 2, 0), e.name, mz5, c.vert_evalue))

    vs = (c.var_scaling1, c.var_scaling2, c.var_scaling3, c.var_scaling4, c.var_scaling5)
    ls = (c.len_scaling1, c.len_scaling2, c.len_scaling3, c.len_scaling4, c.len_scaling5)
    rfls = list(be.rf_lengthscale) + [be.rf_lengthscale_ps]
    out = []
    for i in range(5):
        s = rfls[i] * ds  # be*_rf_lengthscale(1:kz) * xb%ds
        out.append(_rescale(subs[i], vs[i], ls[i], ds, s, c.rf_passes))
    return RegionalB(v=tuple(out), reg=reg, ix=ix, jy=jy, kz=kz, cv_options=c.cv_options,
                     rf_passes=c.rf_passes,
                     be_sha256=be.sha256)

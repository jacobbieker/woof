"""Translate Fast-TUV (phot_opt = 3) into chem_ftuv.cu with tools/chem_f2cu.

    pixi run -e default python tools/chem_ftuv/transcribe_ftuv.py WRF_SRC [--check]

WRF_SRC is a WRF v4.7.1 tree; every file read must match
tools/chem_wrf471_oracle/ftuv/SOURCES.sha256.  The block between
``// BEGIN GENERATED`` and ``// END GENERATED`` in
woof/core/kernels/chem_ftuv.cu is rewritten (``--check`` only compares).

Translated, whole and unchanged: photo_inti and photo (module_ftuv_driver.F),
the routines they call in module_ftuv_subs.F and module_wave_data.F, and the
one-time initialisation those need (schu_inti, wave_data_inti, r44_inti,
r08_inti, inter_inti).  ftuv_driver's per-column glue is hand-written in the
.cu.  Not translated: the MOZART-only branches (pchem's xs_mvk, xs_macr,
xs_hyac, xs_glyald calls record a fatal if reached), ftuv_timestep_init and
p_interp (MOZART only, so esfact keeps its initial 1.0), and setaer/aer_init
(photoin never calls setaer).

Module state.  DATA-initialised module variables are constants: arrays are
file-scope __constant__ tables, scalars their literal.  Everything else the
routines touch is a field of FtuvModule: arrays as pointers into one shared
buffer the single-thread initialisation kernel fills (tv_bind_shared lays it
out), scalars by value.  sq is the one array the column routines write; each
column works on its own copy (tv_bind_column).  Automatic local arrays, and
any local array over WS_THRESHOLD elements, live in a per-column workspace
stack (ws_d / ws_i) instead of the thread's frame.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "chem_f2cu"))
import f2cu  # noqa: E402

KERNEL = ROOT / "woof/core/kernels/chem_ftuv.cu"
PINS = ROOT / "tools/chem_wrf471_oracle/ftuv/SOURCES.sha256"
WAVE = ["wave_data_inti", "r44_inti", "r08_inti", "r01", "fo3qy", "r04", "r44", "r08",
        "r06", "r10", "r11", "r14", "r15", "qyacet", "r17", "r18"]
SUBS = ["schu_inti", "inter_inti", "calc_zenith", "photoin", "setair", "setozo", "setcld",
        "aer_wrf2ftuv", "pchem", "spheres", "airmas", "setz", "calcoe", "set_o2_xsect",
        "schu", "lymana", "inter3", "rtlink", "ps2str", "tridec", "trislv"]
DRIVER = ["photo_inti", "photo"]
#: Registry/registry.chem chem_opt values photo_inti and pchem compare with.
STATE = """      integer, parameter :: mozart_kpp = 111, mozcart_kpp = 112, t1_mozcart_kpp = 114
      integer, parameter :: mozart_mosaic_4bin_kpp = 201, mozart_mosaic_4bin_aq_kpp = 202
"""
WS_THRESHOLD = 64
#: Fixed bounds of the ALLOCATABLE arrays.  Module ones are their allocate
#: statements' (wave_data_inti, r44_inti, r08_inti, inter_inti), except
#: xfrac, allocated (maxval(xcnt), nintervals) at run time and laid out here
#: with the upper bound (nwint, nw-1): every access indexes within the
#: allocated part, and the one whole-array statement (xfrac(:,:) = 1) fills
#: a superset.  photo's are its allocate statements' (module nz).
FIXED = {"sq": "nw-1, nz, nj", "a": "nw-1", "b": "nw-1", "suma": "nw-1", "sumb": "nw-1",
         "xi": "nw", "xcnt": "nw-1", "xfrac": "nwint, nw-1", "luse2usgs": "1"}
FIXED.update({f"photo.{n}_ph": "nz" for n in
              "z tlev airlev rh xlwc o3 acb1 acb2 aoc1 aoc2 aant aso4 asal".split()})
FIXED.update({f"photo.{n}_ph": "nz-1" for n in
              ("tlay tauaer300 tauaer400 tauaer600 tauaer999 waer300 waer400 waer600 "
               "waer999 gaer300 gaer400 gaer600 gaer999").split()})
FIXED["photo.ftuv"] = "nz, njout"
#: ftuv_driver's PARAMETERs the hand-written glue uses, folded as gfortran
#: folds them.
GLUE_PARAMS = {"xair_kg": "1.0e3_dp/28.97_dp*6.023e23_dp*1.0e-6_dp",
               "ph_no3o_factor": "1.1236_dp", "m2s": "60._dp"}


def check_pins(src):
    for line in PINS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split()
        got = hashlib.sha256((src / rel).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{rel}: sha256 {got}, pinned {digest}")


def _spec(text):
    """A module's specification part (everything before CONTAINS)."""
    return text[:re.search(r"^\s*contains\s*$", text, re.I | re.M).start()]


def translator(src: Path):
    texts = {n: (src / "chem" / n).read_text()
             for n in ("module_wave_data.F", "module_ftuv_subs.F", "module_ftuv_driver.F")}
    tr = f2cu.Translator([STATE] + [_spec(t) for t in texts.values()],
                         struct_name="FtuvModule", implicit=False, prefix="tv_",
                         fixed_dims=FIXED, kind_names={"dp": 8})
    tr.workspace = {"threshold": WS_THRESHOLD}
    tr.module_data_constants = True
    tr.unreachable_calls = {"xs_mvk", "xs_macr", "xs_hyac", "xs_glyald"}
    tr.add_source(texts["module_wave_data.F"], WAVE)
    tr.add_source(texts["module_ftuv_subs.F"], SUBS)
    tr.add_source(texts["module_ftuv_driver.F"], DRIVER)
    return tr


def _module_size(tr, v):
    """C expression of a module array's element count (nz is the bind
    function's argument)."""
    em = f2cu.Emitter(tr, tr.module, f2cu.Routine("bind", [], scope=tr.module))
    terms = []
    for lo, hi in v.dims:
        code = em.emit(("bin", "+", ("bin", "-", hi, lo), ("num", "1")))[0]
        terms.append(f"({code})")
    return "*".join(terms).replace("mod.nz", "nz")


def generate(src: Path) -> str:
    tr = translator(src)
    names = WAVE + SUBS + DRIVER
    bodies = [tr.emit_routine(tr.routines[n]) for n in names]
    out = ["// BEGIN GENERATED by tools/chem_ftuv/transcribe_ftuv.py from",
           "// chem/module_ftuv_driver.F, module_ftuv_subs.F and module_wave_data.F",
           "// (WRF v4.7.1).  Do not edit by hand."]
    for name, expr in GLUE_PARAMS.items():
        val, kind = tr.const(f2cu.parse_expr(tr.normalise(expr)), tr.module)
        out.append(f"#define TV_{name.upper()} {f2cu.literal(val, kind)}")
    for name in ("nw", "nj", "tuv_jmax", "nwint", "nref"):
        out.append(f"#define TV_{name.upper()} {tr.module.vars[name].param[0]}")
    for name in ("pid_no2", "pid_o31d", "pid_o33p", "pid_hono", "pid_hno3", "pid_hno4",
                 "pid_no3o2", "pid_no3o", "pid_h2o2", "pid_ch2om", "pid_ch2or", "pid_ald",
                 "pid_op1", "pid_op2", "pid_paa", "pid_ket", "pid_glya", "pid_glyb",
                 "pid_mgly", "pid_dcb", "pid_onit", "pid_macr", "pid_cl2", "pid_hocl",
                 "pid_fmcl", "pid_pan", "pid_n2o5"):
        out.append(f"#define TV_{name.upper()} {tr.module.vars[name].param[0]}")
    # FtuvModule: scalars by value, arrays as pointers, the workspace stack.
    used = sorted(tr.module_vars_used)
    arrays = [n for n in used if tr.module.vars[n].dims is not None]
    out.append("struct FtuvModule {")
    for n in used:
        v = tr.module.vars[n]
        star = "*" if v.dims is not None else ""
        out.append(f"    {f2cu._CTYPE[v.kind]} {star}{n};")
    out += ["    double *ws_d, *ws_d_end;  // workspace stack (binary64 arrays)",
            "    int *ws_i, *ws_i_end;     // workspace stack (integer arrays)",
            "    int f2cu_fatal;  // first source line of a wrf_error_fatal/stop, or 0",
            "};"]
    # The shared buffer: every module array, sq included (its initial state).
    for kind, buf in (("double", "d"), ("int", "i")):
        terms = [_module_size(tr, tr.module.vars[n]) for n in arrays
                 if tr.module.vars[n].kind == kind]
        out.append(f"#define TV_SHARED_{buf.upper()}(nz) ({' + '.join(terms) or '0'})")
    out.append("__device__ void tv_bind_shared(FtuvModule &mod, double *d, int *i, int nz) {")
    for n in arrays:
        v = tr.module.vars[n]
        buf = "d" if v.kind == "double" else "i"
        out.append(f"    mod.{n} = {buf}; {buf} += {_module_size(tr, v)};")
    out.append("}")
    out.append(f"#define TV_SQ_SIZE(nz) ({_module_size(tr, tr.module.vars['sq'])})")
    out += f2cu.HELPERS.strip().splitlines()
    out += [tr.module_tables[k] for k in sorted(tr.module_tables)]
    out += tr.constant_tables
    for name in names:
        out.append(tr.prototype(tr.routines[name]) + ";")
    for b in bodies:
        out += b
    # Workspace each column needs at most: every routine's frame at once
    # (no recursion, so no routine is live twice).  Sizes are polynomials in
    # the module nz; the launcher evaluates these.
    for k, kind in enumerate(("D", "I")):
        pts = [workspace_need(tr, nz)[k] for nz in (20, 40, 60, 80)]
        c = pts[0]
        # exact quadratic through nz = 20, 40, 60, checked at 80
        d1, d2 = pts[1] - pts[0], pts[2] - 2 * pts[1] + pts[0]
        a2 = d2 // (2 * 400)
        a1 = (d1 - a2 * (40 * 40 - 20 * 20)) // 20
        a0 = pts[0] - a1 * 20 - a2 * 400
        assert all(a2 * nz * nz + a1 * nz + a0 == v for nz, v in zip((20, 40, 60, 80), pts)), pts
        del c
        out.append(f"#define TV_WS_{kind}(nz) ({a2}*(nz)*(nz) + {a1}*(nz) + {a0})")
    out.append("// END GENERATED")
    return "\n".join(out) + "\n", tr


def workspace_need(tr, nz: int) -> tuple[int, int]:
    """(doubles, ints) of workspace one column needs for module nz."""
    env = {"nz": nz, "njout": tr.module.vars["tuv_jmax"].param[0],
           "nw": tr.module.vars["nw"].param[0]}  # rtlink(nz, nw, ...) <- photoin passes nw
    need = {"double": 0, "int": 0}
    for routine, name, kind, dims in tr.workspace_arrays:
        scope = tr.routines[routine].scope
        size = 1
        for lo, hi in dims:
            size *= _eval(tr, hi, scope, env) - _eval(tr, lo, scope, env) + 1
        need[kind] += size
    return need["double"], need["int"]


def _eval(tr, node, scope, env):
    """Integer value of a workspace bound.  The only run-time names the bounds
    use are nz -- photo's module nz, and every callee's nz dummy, which photo
    passes that same nz down -- and photo's njout (ftuv_driver passes
    tuv_jmax)."""
    try:
        return tr.const(node, scope)[0]
    except f2cu.NotConstant:
        pass
    tag = node[0]
    if tag == "ref" and node[2] is None:
        return env[node[1]]
    if tag == "bin":
        a, b = _eval(tr, node[2], scope, env), _eval(tr, node[3], scope, env)
        return {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[node[1]]
    if tag == "neg":
        return -_eval(tr, node[1], scope, env)
    raise ValueError(node)


def main(argv):
    src = Path(argv[1])
    check_pins(src)
    block, _ = generate(src)
    text = KERNEL.read_text()
    start = text.index("// BEGIN GENERATED")
    end = text.index("// END GENERATED") + len("// END GENERATED\n")
    new = text[:start] + block + text[end:]
    if "--check" in argv:
        if new != text:
            raise SystemExit("chem_ftuv.cu's generated block is stale")
        return
    KERNEL.write_text(new)


if __name__ == "__main__":
    main(sys.argv)

"""Transcribe WRF-Chem's idealized gas profile tables into a Python data module.

    pixi run -e mlx python tools/chem_gas_profile/transcribe.py WRF_SRC [--check]

WRF_SRC is a WRF v4.7.1 tree; chem/module_input_chem_data.F and
chem/module_data_sorgam.F must match their pins in
tools/chem_wrf471_oracle/gasprofile/SOURCES.sha256.  The script reads, as
written in the Fortran:

* module_input_chem_data.F's ``#define CASENAME`` (line 45) and its module
  DATA: ``dens`` (:145-148), ``ZFA_BDY`` (:154-156), ``ZFA`` (:159-161), the 34
  ``xl(n, :)`` profiles (:168-301) and ``so4vaptoaer`` (:83), with Fortran's
  ``r*value`` repeat counts (``kxm1``/``kx``) expanded;
* setup_gasprofile_map_radm_racm's ``iref`` / ``fracref`` defaults and their
  first 41 entries (:411-420) and ``ggnam`` labels (:422-429);
* module_data_sorgam.F's ``grav`` PARAMETER (:575), which
  module_input_chem_data USEs (:52) and input_chem_profile divides by (:714);

and writes woof/core/chem_gas_profile_data.py holding each number as the
decimal string the Fortran spells (``--check`` only compares).  The float32
words are taken from those strings at import and graded bit for bit against
the oracle's dump of the same arrays (tests/data/oracles/chem/gasprofile/tables).
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "woof" / "core" / "chem_gas_profile_data.py"
PINS = ROOT / "tools" / "chem_wrf471_oracle" / "gasprofile" / "SOURCES.sha256"
MODULE = "chem/module_input_chem_data.F"
SORGAM = "chem/module_data_sorgam.F"
KX, LO = 16, 34


def pins() -> dict[str, str]:
    out = {}
    for line in PINS.read_text().splitlines():
        if line and not line.startswith("#"):
            digest, rel = line.split()
            out[rel] = digest
    return out


def read_pinned(src: Path, rel: str) -> str:
    data = (src / rel).read_bytes()
    want = pins()[rel]
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise SystemExit(f"{rel}: sha256 {got}, pinned {want}")
    return data.decode("latin-1")


def _code(text: str) -> str:
    """Strip ``!`` comments and join ``&`` continuations (no string in the
    parsed statements holds a ``!`` or ``&``)."""
    lines = [ln.split("!", 1)[0].rstrip() for ln in text.splitlines()]
    return re.sub(r"&\s*\n\s*", " ", "\n".join(lines))


def _items(body: str, consts: dict[str, int]) -> list[str]:
    out: list[str] = []
    for item in (s.strip() for s in body.split(",")):
        if "*" in item:
            count, value = (s.strip() for s in item.split("*"))
            out += [value] * consts.get(count.lower(), int(count) if count.isdigit() else -1)
        else:
            out.append(item)
    return out


def tables(src: Path) -> dict:
    text = read_pinned(src, MODULE)
    case = re.search(r"^#define CASENAME (\d+)\s*$", text, re.M)
    spec = _code(text[:text.index("\nCONTAINS\n")])
    if not re.search(r"PARAMETER\(\s*kx=16,\s*kxm1=kx-1,\s*logg=350,\s*lo=34\)", spec):
        raise SystemExit("kx / lo PARAMETER changed")
    consts = {"kx": KX, "kxm1": KX - 1}

    def data(pattern: str) -> list[str]:
        found = re.findall(pattern + r"\s*/([^/]*)/", spec, re.I)
        if len(found) != 1:
            raise SystemExit(f"{pattern}: {len(found)} DATA statements")
        return _items(found[0], consts)

    dens = data(r"DATA\s+dens")
    zfa_bdy = data(r"DATA\s+ZFA_BDY")
    zfa = data(r"DATA\s+ZFA(?!_)")
    so4 = data(r"DATA\s+so4vaptoaer")
    xl = []
    for n in range(1, LO + 1):
        xl.append(data(rf"DATA\s*\(xl\({n},k_loop\),k_loop=1,kx\)"))
    body = text[text.index("SUBROUTINE setup_gasprofile_map_radm_racm"):
                text.index("END SUBROUTINE setup_gasprofile_map_radm_racm")]
    body = _code(body)
    iref_default = re.search(r"iref\(:\)\s*=\s*(\d+)", body).group(1)
    frac_default = re.search(r"fracref\(:\)\s*=\s*([0-9.Ee+-]+)", body).group(1)
    iref = _items(re.search(r"iref\(1:41\)\s*=\s*\(/([^/]*)/\)", body).group(1), {})
    frac = _items(re.search(r"fracref\(1:41\)\s*=\s*\(/([^/]*)/\)", body).group(1), {})
    names = [s.strip().strip("'").strip().lower() for s in
             re.search(r"ggnam\(1:41\)\s*=\s*\(/([^/]*)/\)", body).group(1).split(",")]
    sorgam = _code(read_pinned(src, SORGAM))
    grav = re.findall(r"PARAMETER\s*\(\s*grav\s*=\s*([0-9.Ee+-]+)\s*\)", sorgam, re.I)
    sizes = (len(dens), len(zfa), len(zfa_bdy), len(so4), [len(r) for r in xl],
             len(iref), len(frac), len(names), len(grav))
    if sizes != (KX, KX + 1, KX + 1, 1, [KX] * LO, 41, 41, 41, 1):
        raise SystemExit(f"unexpected table sizes {sizes}")
    return dict(casename=int(case.group(1)), dens=dens, zfa=zfa, zfa_bdy=zfa_bdy,
                so4vaptoaer=so4[0], xl=xl, iref_default=iref_default,
                fracref_default=frac_default, iref=iref, fracref=frac, ggnam=names,
                grav=grav[0])


def _tuple(values: list[str], indent: str, per: int = 6) -> str:
    rows = [", ".join(f'"{v}"' for v in values[i:i + per]) for i in range(0, len(values), per)]
    return "(\n" + "".join(f"{indent}    {r},\n" for r in rows) + f"{indent})"


def render(t: dict) -> str:
    p = pins()
    out = [
        '"""GENERATED by tools/chem_gas_profile/transcribe.py; do not edit.',
        "",
        f"WRF v4.7.1 {MODULE} (sha256 {p[MODULE]})",
        f"and {SORGAM} (sha256 {p[SORGAM]}), numbers",
        "spelled as the Fortran spells them.  woof/core/chem_gas_profile.py turns",
        "them into float32 words.",
        '"""',
        "",
        f"SOURCE_SHA256 = {{{MODULE!r}: {p[MODULE]!r},",
        f"                 {SORGAM!r}: {p[SORGAM]!r}}}",
        "",
        "#: module_input_chem_data.F:45.",
        f"CASENAME = {t['casename']}",
        "#: :73, kx (profile levels) and lo (NALROM species).",
        f"KX = {KX}",
        f"LO = {LO}",
        "#: module_data_sorgam.F:575 PARAMETER (grav=...).",
        f'GRAV = "{t["grav"]}"',
        "#: :83 DATA so4vaptoaer.",
        f'SO4VAPTOAER = "{t["so4vaptoaer"]}"',
        "#: :145-148 DATA dens (molecules cm-3, top of the profile first).",
        f"DENS = {_tuple(t['dens'], '', 4)}",
        "#: :154-156 DATA ZFA_BDY (m).",
        f"ZFA_BDY = {_tuple(t['zfa_bdy'], '')}",
        "#: :159-161 DATA ZFA (m).",
        f"ZFA = {_tuple(t['zfa'], '')}",
        "#: :168-301 DATA xl(n, 1:kx), n = 1 .. lo; index 1 is the top of the profile.",
        "XL = (",
    ]
    for n, row in enumerate(t["xl"], 1):
        out.append(f"    # xl({n}, :)")
        out.append(f"    {_tuple(row, '    ')},")
    out += [
        ")",
        "#: setup_gasprofile_map_radm_racm (:409-431): iref(:) / fracref(:) defaults,",
        "#: then entries 1:41 and the ggnam labels WRF gives them.",
        f"IREF_DEFAULT = {t['iref_default']}",
        f'FRACREF_DEFAULT = "{t["fracref_default"]}"',
        f"IREF = ({', '.join(t['iref'])},)",
        f"FRACREF = {_tuple(t['fracref'], '', 8)}",
        f"GGNAM = {_tuple(t['ggnam'], '', 8)}",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str]) -> None:
    src = Path(argv[1])
    new = render(tables(src))
    if "--check" in argv:
        if not OUT.is_file() or OUT.read_text() != new:
            raise SystemExit(f"{OUT.relative_to(ROOT)} is stale")
        return
    OUT.write_text(new)


if __name__ == "__main__":
    main(sys.argv)

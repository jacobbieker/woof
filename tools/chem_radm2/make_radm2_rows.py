"""Write woof/data/chem/species/radm2.json and the radm2 / radm2sorg set files,
and join the shared rows (o3, no2, co in cams_aq.json; so2, sulf in
gocart.json) to the two sets.

Usage: python tools/chem_radm2/make_radm2_rows.py [WORKTREE_ROOT] [--check]

Run after tools/chem_gocart/make_gocart_rows.py (which rewrites so2 and sulf
without the RADM2 membership).  The long names come from WRF v4.7.1's
Registry/registry.chem state lines (read from WOOF_WRF471_STOCK_TREE, default
~/Development/wrf-src/WRF-4.7.1); every other number is stated with its
source below.  ``--check`` rewrites nothing and fails when a file would
change.

One species is ONE row (woof.chem_table): the five species RADM2 shares with
the CAMS and GOCART sets gain the two sets and the RADM2 processes, gated to
the RADM2 sets by ``process_sets``, and every process they already ran is
gated to the sets that ran it, so no existing selection changes.
"""
import json
import os
import re
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if a != "--check"]
CHECK = "--check" in sys.argv[1:]
root = Path(args[0] if args else ".")
wrf = Path(os.environ.get("WOOF_WRF471_STOCK_TREE",
                          Path.home() / "Development/wrf-src/WRF-4.7.1"))

R, RS = "radm2", "radm2sorg"
BOTH = [R, RS]
EMIS, PHOT, WES, VERTMX = "emission.radm2", "photolysis.ftuv", "drydep.wesely", "mixing.vertmx"
GAS, AER, AERDEP = "chem.radm2", "aerosol.sorgam", "drydep.sorgam"
EPSILC = 1.0e-16

#: registry.chem:3929-3930 (package radm2 / radm2sorg), in package order.
GASES = ("so2 sulf no2 no o3 hno3 h2o2 ald hcho op1 op2 paa ora1 ora2 nh3 n2o5 "
         "no3 pan hc3 hc5 hc8 eth co ol2 olt oli tol xyl aco3 tpan hono hno4 "
         "ket gly mgly dcb onit csl iso hcl ho ho2").split()
AEROSOLS = ("so4aj so4ai nh4aj nh4ai no3aj no3ai naaj naai claj clai orgaro1j "
            "orgaro1i orgaro2j orgaro2i orgalk1j orgalk1i orgole1j orgole1i "
            "orgba1j orgba1i orgba2j orgba2i orgba3j orgba3i orgba4j orgba4i "
            "orgpaj orgpai ecj eci p25j p25i antha seas soila nu0 ac0 corn").split()
#: The rows WRF's sorgam_driver exchanges with the gas phase
#: (woof.core.chem_sorgam.GASES): they name aerosol.sorgam too, so the
#: ledger books SORGAM's change of them under chemistry.
SORGAM_GASES = ("sulf", "hno3", "nh3", "hcl")
#: Rows another set file already defines (woof/data/chem/species).
SHARED = {"o3": "cams_aq.json", "no2": "cams_aq.json", "co": "cams_aq.json",
          "so2": "gocart.json", "sulf": "gocart.json"}

# IUPAC 2013 conventional atomic weights (C 12.011, H 1.008, N 14.007,
# O 15.999, Cl 35.45); a lumped RADM2 species takes its Stockwell et al.
# (1990, JGR 95, 16343, Table 1) surrogate.  The mass ledger is the only
# reader (ppmv -> kg); no mechanism or emission arithmetic uses them.
FORMULA = {
    "no": ("NO", 30.006), "hno3": ("HNO3", 63.012), "h2o2": ("H2O2", 34.014),
    "ald": ("CH3CHO, the ALD surrogate", 44.053),
    "hcho": ("HCHO", 30.026), "op1": ("CH3OOH", 48.041),
    "op2": ("C2H5OOH, the OP2 surrogate", 62.068),
    "paa": ("CH3C(O)OOH", 76.051), "ora1": ("HCOOH", 46.025),
    "ora2": ("CH3COOH", 60.052), "nh3": ("NH3", 17.031),
    "n2o5": ("N2O5", 108.009), "no3": ("NO3", 62.004),
    "pan": ("CH3C(O)OONO2", 121.048), "hc3": ("propane, the HC3 surrogate", 44.097),
    "hc5": ("n-pentane, the HC5 surrogate", 72.151),
    "hc8": ("n-octane, the HC8 surrogate", 114.232), "eth": ("C2H6", 30.070),
    "ol2": ("C2H4", 28.054), "olt": ("propene, the OLT surrogate", 42.081),
    "oli": ("trans-2-butene, the OLI surrogate", 56.108),
    "tol": ("toluene", 92.141), "xyl": ("xylene", 106.168),
    "aco3": ("CH3C(O)OO", 75.043),
    "tpan": ("H(CO)CH=CHC(O)OONO2, the TPAN surrogate", 161.069),
    "hono": ("HONO", 47.013), "hno4": ("HO2NO2", 79.011),
    "ket": ("methyl ethyl ketone, the KET surrogate", 72.107),
    "gly": ("glyoxal", 58.036), "mgly": ("methylglyoxal", 72.063),
    "dcb": ("C5H6O2, the DCB surrogate", 98.101),
    "onit": ("C4H9ONO2, the ONIT surrogate", 119.120),
    "csl": ("cresol", 108.140), "iso": ("isoprene", 68.119),
    "hcl": ("HCl", 36.458), "ho": ("OH", 17.007), "ho2": ("HO2", 33.006),
}
GAS_FAMILY = {
    "no": "nox", "hno3": "noy", "n2o5": "noy", "no3": "noy", "pan": "noy",
    "tpan": "noy", "hono": "noy", "hno4": "noy", "onit": "noy",
    "h2o2": "oxidant", "op1": "oxidant", "op2": "oxidant", "paa": "oxidant",
    "nh3": "inorganic", "hcl": "inorganic", "ho": "radical", "ho2": "radical",
    "aco3": "radical",
}


def aer_family(name):
    if name in ("nu0", "ac0", "corn"):
        return "number"
    for prefix, family in (("so4", "sulfate"), ("nh4", "ammonium"),
                           ("no3", "nitrate"), ("na", "seasalt"),
                           ("cl", "seasalt"), ("orgpa", "oc"), ("org", "soa"),
                           ("ec", "bc"), ("seas", "seasalt"), ("soila", "dust")):
        if name.startswith(prefix):
            return family
    return "pm_other"


def registry_names():
    text = (wrf / "Registry/registry.chem").read_text(errors="replace")
    out = {}
    for name in (*GASES, *AEROSOLS):
        m = re.search(r'^state\s+real\s+%s\s+ikjftb\s+chem\s.*?"%s"\s+"([^"]*)"\s+"([^"]*)"'
                      % (name, name), text, re.M | re.I)
        if m is None:
            raise SystemExit(f"registry.chem has no chem state line for {name}")
        out[name] = (m.group(1).strip(), m.group(2).strip())
    return out


REG = registry_names()
SOURCE = ("registry.chem:3929-3930 packages radm2 (chem_opt 1) and radm2sorg "
          "(chem_opt 2); floor = WRF epsilc 1.E-16 (chem/module_data_radm2.F:5)")
INFLOW = ("default_inflow = epsilc: the row's value at an inflow face with no "
          "enabled boundary source (WOOF's constant stand-in for WRF's "
          "height-dependent bdy_chem_value profile; the receipt's inflow "
          "table names it)")


def gas_processes(name):
    out = [EMIS]
    if name == "o3":
        out.append(PHOT)
    out += [VERTMX, GAS]
    if name in SORGAM_GASES:
        out.append(AER)
    return out


def gas_row(name):
    desc, units = REG[name]
    assert units == "ppmv", (name, units)
    formula, mass = FORMULA[name]
    return {
        "name": name, "output_name": name, "long_name": desc, "units": "ppmv",
        "phase": "gas", "family": GAS_FAMILY.get(name, "voc"), "sets": BOTH,
        "transported": True, "floor": EPSILC, "default_inflow": EPSILC,
        "molar_mass_g_mol": mass, "processes": gas_processes(name),
        "emissions": [], "boundary": [], "drydep": None, "settling": None,
        "wetdep_ls_alpha": 0.0, "aging": None, "optics": None,
        "mp_coupling": None, "exclusive_group": None,
        "process_sets": {AER: [RS]} if name in SORGAM_GASES else None,
        "provenance": f"{SOURCE}; {INFLOW}; molar mass {mass} g/mol ({formula}), "
                      "read only by the mass ledger; wetdep_ls_alpha 0: WRF's "
                      "RADM2 packages run no wet removal with wetscav_onoff = 0",
    }


def aer_row(name):
    desc, units = REG[name]
    number = name in ("nu0", "ac0", "corn")
    assert units == ("#/kg-dryair" if number else "ug/kg-dryair"), (name, units)
    return {
        "name": name, "output_name": name, "long_name": desc,
        "units": "kg-1" if number else "ug kg-1", "phase": "aerosol",
        "family": aer_family(name), "sets": [RS], "transported": True,
        "floor": EPSILC, "default_inflow": EPSILC, "molar_mass_g_mol": None,
        "processes": [EMIS, AERDEP, VERTMX, AER], "emissions": [],
        "boundary": [], "drydep": None, "settling": None,
        "wetdep_ls_alpha": 0.0, "aging": None, "optics": None,
        "mp_coupling": None, "exclusive_group": None, "process_sets": None,
        "provenance": f"{SOURCE}; {INFLOW}; "
                      + ("a number row (particles per kg of dry air): the ledger "
                         "counts particles, not kilograms; " if number else "")
                      + "dry deposition velocity from sorgam_depdriver "
                        "(drydep.sorgam), mixed by vertmx",
    }


def join_shared(row):
    """The shared row ``row`` joins the RADM2 sets: its existing processes
    stay gated to the sets that ran them, the RADM2 ones gate to RADM2."""
    name = row["name"]
    old_sets = [s for s in row["sets"] if s not in BOTH]
    gates = dict(row.get("process_sets") or {})
    old = [p for p in row["processes"] if p not in (EMIS, PHOT, GAS, AER)]
    new = [p for p in gas_processes(name) if p not in old]
    for key in old:
        if key in (VERTMX, WES):
            continue  # every set mixes and Wesely-deposits these gases
        gates.setdefault(key, list(old_sets))
    for key in new:
        gates[key] = [RS] if key == AER else list(BOTH)
    row["sets"] = old_sets + BOTH
    row["processes"] = old + new
    row["process_sets"] = gates
    note = ("; RADM2/RADM2SORG (registry.chem:3929-3930) share this row: "
            "emission.radm2, chem.radm2"
            + (", photolysis.ftuv (the ozone column Fast-TUV reads)" if name == "o3" else "")
            + (", aerosol.sorgam (radm2sorg)" if name in SORGAM_GASES else "")
            + " act on it only in the RADM2 sets (process_sets)")
    if name == "sulf":
        note += ("; in RADM2 sulf is WRF's gas-phase H2SO4 (it keeps this "
                 "row's phase and the GOCART sulfate molar mass 96.0576, which "
                 "only the mass ledger and vertmx's deposited-mass diagnostic "
                 "read)")
    if row["phase"] != "gas":
        # sulf: GOCART's particulate sulfate takes CAMS's sulphate aerosol;
        # RADM2's gas-phase H2SO4 must not (it starts from WRF's profile).
        for ref in row["boundary"]:
            ref["sets"] = list(old_sets)
        note += "; its CAMS sulphate boundary entries fill it only in the GOCART sets"
    if note not in row["provenance"]:
        row["provenance"] += note
    return row


def dump(path, document):
    text = json.dumps(document, indent=2) + "\n"
    if CHECK:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            raise SystemExit(f"{path} is stale: rerun tools/chem_radm2/make_radm2_rows.py")
        return
    path.write_text(text, encoding="utf-8", newline="\n")


species = root / "woof/data/chem/species"
rows = [gas_row(n) for n in GASES if n not in SHARED]
rows += [aer_row(n) for n in AEROSOLS]
dump(species / "radm2.json", {
    "schema": "gpuwm.chem.species.v1", "owner": "aq-radm2",
    "description": "WRF-Chem's RADM2 gas phase (chem_opt 1) and its MADE/SORGAM "
                   "modal aerosol (chem_opt 2), WRF v4.7.1 Registry packages "
                   "radm2 and radm2sorg (registry.chem:3929-3930), one row per "
                   "species. o3, no2, co (cams_aq.json), so2 and sulf "
                   "(gocart.json) are shared rows that join both sets "
                   "(tools/chem_radm2/make_radm2_rows.py).",
    "rows": rows})
for filename in sorted(set(SHARED.values())):
    path = species / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    for row in document["rows"]:
        if SHARED.get(row["name"]) == filename:
            join_shared(row)
    dump(path, document)
SETS = {
    R: (1, "WRF-Chem's RADM2 gas-phase package, all 42 gases (registry.chem:3929): "
           "anthropogenic emissions (emiss_opt = 3), Fast-TUV photolysis "
           "(phot_opt = 3), the RADM2 QSSA mechanism, vertical mixing and dry "
           "deposition.",
        "registry.chem:3929 package radm2 chem_opt==1"),
    RS: (2, "WRF-Chem's RADM2SORG package, the RADM2 gases and the MADE/SORGAM modal "
            "aerosol, 80 species (registry.chem:3930): RADM2 plus aerosol "
            "emissions, SORGAM dynamics with RPMARES inorganic equilibrium "
            "(ISORROPIA is bring-your-own-source), clean-room SOA partitioning, "
            "SORGAM dry deposition and the PM2.5/PM10 sums.",
         "registry.chem:3930 package radm2sorg chem_opt==2"),
}
for name, (opt, description, provenance) in SETS.items():
    dump(root / f"woof/data/chem/sets/{name}.json", {
        "schema": "gpuwm.chem.set.v1", "name": name, "description": description,
        "wrf_chem_opt": opt, "wrf_tracer_opt": None, "provenance": provenance})

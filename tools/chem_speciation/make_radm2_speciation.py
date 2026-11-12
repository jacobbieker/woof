"""Write woof/data/chem/speciation/radm2.v1.json, WOOF's RADM2/SORGAM speciation.

The file maps the nine inventory fields WOOF's anthropogenic sources carry
(co, nox, nh3, nmvoc, so2, bc, oc, pm25, pm10; each a flux density in
kg m-2 s-1) onto every emis_ant row that WRF-Chem's emiss_opt = 3 reads for
RADM2 and RADM2SORG (woof.core.chem_radm2_emiss.EMISSIONS), as a linear
combination per row with the unit conversion folded into each weight:

* gases: kg m-2 s-1 -> mol km-2 hr-1, factor 1e6 * 3600 * 1000 / M[g mol-1];
* aerosols: kg m-2 s-1 -> ug m-2 s-1, factor 1e9.

Every choice below is WOOF's, not WRF's: WRF reads emis_ant ready-made
(chem/convert_emiss.F) and prescribes no speciation.  Run from the worktree
root; ``--check`` exits 1 if the committed file differs from what this
script writes.

    pixi run -e mlx python tools/chem_speciation/make_radm2_speciation.py [--check]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("woof/data/chem/speciation/radm2.v1.json")

#: IUPAC standard atomic weights (abridged, conventional values), g mol-1.
C, H, O, N, S, CL = 12.011, 1.008, 15.999, 14.007, 32.06, 35.45

GAS = 1.0e6 * 3600.0 * 1000.0      # kg m-2 s-1 -> g km-2 hr-1
AEROSOL = 1.0e9                    # kg m-2 s-1 -> ug m-2 s-1

M_CO = C + O
M_NO2 = N + 2 * O                  # EDGAR reports NOx as NO2 mass
M_NH3 = N + 3 * H
#: SO2 as WOOF's GOCART rows take it (woof/data/chem/species/gocart.json).
M_SO2 = 64.066

#: NOx molar split, NO : NO2 (N-conserving: both shares divide NOx-as-NO2
#: mass by M(NO2)).  EDGAR's own statement: primary NOx "can be assumed to
#: be composed of 90 % NO and 10 % NO2" (Crippa et al. 2018, Earth Syst. Sci.
#: Data 10, 1987-2013, p. 1989, the EDGAR v4.3.2 air-pollutant paper).
NOX_NO, NOX_NO2 = 0.9, 0.1
#: EDGAR's NOx is "expressed as NO2 mass unit" (Crippa et al. 2023, Earth
#: Syst. Sci. Data 15, 2667-2694, p. 2668, HTAP_v3 built on EDGAR).
CITE_NOX = ("NOx is NO2-equivalent mass (Crippa et al. 2023, ESSD 15, 2667-2694); "
            "molar split 90 % NO / 10 % NO2 of primary NOx, Crippa et al. 2018, "
            "ESSD 10, 1987-2013, p. 1989")

#: Aitken (i) : accumulation (j) mass split of primary fine aerosol for
#: MADE/SORGAM, Tuccella et al. (2012), JGR 117, D03303, as HERMESv3 applies
#: it (Guevara et al. 2019, Table 2: ECI bc*0.2, ECJ bc*0.8, ORGI/ORGJ).
AITKEN, ACCUMULATION = 0.2, 0.8

#: Organic matter per organic carbon for primary fine anthropogenic OC.
#: EDGAR's OC is carbon mass; SORGAM's orgpai/orgpaj rows are organic mass.
#: EMEP MSC-W's value (Simpson et al. 2012, ACP 12, 7825-7865, Sect. 6.3).
OM_OC = 1.3

#: IPCC TAR WG1 Ch. 4 Table 4.7b (Prather, Ehhalt et al. 2001): the OxComp
#: speciation of INDUSTRIAL NMVOC emissions (built on EDGAR), wt% and mean
#: carbon number per group.  The groups are the GEIA/EDGAR voc01-voc25
#: groups (isoprene and monoterpenes, voc10/voc11, carry no industrial
#: mass).  Molar mass of each group: its formula at the table's carbon
#: number (stated per group).  The RADM2 classes and the mass shares of the
#: split groups are HERMESv3's RADM2 profile (Guevara et al. 2019, Table 2),
#: whose classes follow Stockwell et al. (1990), JGR 95, 16343-16367, with
#: three corrections recorded in ``deviation``.  Each group's moles are its
#: mass over its molar mass; a split group's mass is shared before that
#: division (HERMESv3's expressions apply to group mass).
def _alkane(n):
    return n * C + (2 * n + 2) * H


def _alkene(n):
    return n * C + 2 * n * H


def _alkanol(n):
    return n * C + (2 * n + 2) * H + O


def _carbonyl(n):
    return n * C + 2 * n * H + O


def _acid(n):
    return n * C + 2 * n * H + 2 * O


def _alkylbenzene(n):
    return n * C + (2 * n - 6) * H


NMVOC_GROUPS = (
    # (GEIA group, TAR name, wt%, carbon number, formula, M, {class: mass share})
    ("voc01", "Alcohols", 3.2, 2.5, "CnH2n+2O", _alkanol(2.5), {"hc3": 0.95, "hc5": 0.05}),
    ("voc02", "Ethane", 4.7, 2.0, "C2H6", _alkane(2), {"eth": 1.0}),
    ("voc03", "Propane", 5.5, 3.0, "C3H8", _alkane(3), {"hc3": 1.0}),
    ("voc04", "Butanes", 10.9, 4.0, "C4H10", _alkane(4), {"hc3": 1.0}),
    ("voc05", "Pentanes", 9.4, 5.0, "C5H12", _alkane(5), {"hc5": 1.0}),
    ("voc06", "Higher alkanes", 18.2, 7.5, "CnH2n+2", _alkane(7.5), {"hc5": 0.43, "hc8": 0.57}),
    ("voc07", "Ethene", 5.2, 2.0, "C2H4", _alkene(2), {"ol2": 1.0}),
    ("voc08", "Propene", 2.4, 3.0, "C3H6", _alkene(3), {"olt": 1.0}),
    ("voc09", "Ethyne", 2.2, 2.0, "C2H2", 2 * C + 2 * H, {"hc3": 0.4}),
    ("voc12", "Other alkenes, alkynes, dienes", 3.8, 4.8, "CnH2n", _alkene(4.8), {"oli": 1.0}),
    ("voc13", "Benzene", 3.0, 6.0, "C6H6", 6 * C + 6 * H, {"tol": 0.293}),
    ("voc14", "Toluene", 4.9, 7.0, "C7H8", _alkylbenzene(7), {"tol": 1.0}),
    ("voc15", "Xylene", 3.6, 8.0, "C8H10", _alkylbenzene(8), {"xyl": 1.0}),
    ("voc16", "Trimethylbenzene", 0.7, 9.0, "C9H12", _alkylbenzene(9), {"xyl": 1.0}),
    ("voc17", "Other aromatics", 3.1, 9.6, "CnH2n-6", _alkylbenzene(9.6), {"xyl": 1.0}),
    ("voc18", "Esters", 1.4, 5.2, "CnH2nO2", _acid(5.2), {"hc3": 0.69, "hc5": 0.31}),
    ("voc19", "Ethers", 1.7, 4.7, "CnH2n+2O", _alkanol(4.7), {"hc8": 1.0}),
    ("voc20", "Chlorinated HC's", 0.5, 2.6, "C2HCl3 (trichloroethene, WOOF's representative)",
     2 * C + H + 3 * CL, {"hc3": 1.0}),
    ("voc21", "Formaldehyde", 0.5, 1.0, "CH2O", _carbonyl(1), {"hcho": 1.0}),
    ("voc22", "Other aldehydes", 1.6, 3.7, "CnH2nO", _carbonyl(3.7), {"ald": 1.0}),
    ("voc23", "Ketones", 1.9, 4.6, "CnH2nO", _carbonyl(4.6), {"ket": 1.0}),
    ("voc24", "Acids", 3.6, 1.9, "CnH2nO2", _acid(1.9), {"ora2": 0.56}),
    ("voc25", "Others", 8.1, 4.9, "unspecified", None, {}),
)

DEVIATIONS = (
    "HERMESv3's printed RADM2 profile omits voc15 (xylene) from every class and "
    "lists voc17 (other aromatics) in both HC8 and XYL; WOOF assigns xylene and "
    "other aromatics once each, to XYL (Stockwell et al. 1990: XYL is xylene and "
    "the more reactive aromatics).",
    "Acids: HERMESv3 gives 0.44 to ORA1 and 0.56 to ORA2; add_anthropogenics' "
    "RADM2 branch adds no e_ora1 (woof.core.chem_radm2_emiss.GAS_TERMS), so the "
    "formic share is not emitted and only ORA2's 0.56 is.",
    "Ethyne: 0.4 of its mass to HC3 (HERMESv3's reactivity-weighted share); the "
    "rest, like voc25 'Others' (8.1%, no stated composition) and the shares of "
    "benzene outside TOL, is not emitted.",
)


def _nmvoc_moles():
    """{class: mol per kg NMVOC} and the per-class derivation strings."""
    moles, why = {}, {}
    for group, name, wt, ncarbon, formula, molar, shares in NMVOC_GROUPS:
        for cls, share in shares.items():
            mol_per_kg = wt / 100.0 * share * 1000.0 / molar
            moles[cls] = moles.get(cls, 0.0) + mol_per_kg
            why.setdefault(cls, []).append(
                f"{group} {name} {wt}% x {share} / {molar:.3f} g mol-1 ({formula}, n={ncarbon})")
    return moles, why


GAS_E = ("e_so2", "e_co", "e_no", "e_no2", "e_nh3", "e_hcl", "e_csl", "e_iso",
         "e_ald", "e_hcho", "e_ora2", "e_hc3", "e_hc5", "e_hc8", "e_eth",
         "e_olt", "e_oli", "e_tol", "e_xyl", "e_ket", "e_ol2", "e_hono")
AEROSOL_E = ("e_pm25i", "e_pm25j", "e_pm_10", "e_eci", "e_ecj", "e_orgi",
             "e_orgj", "e_so4i", "e_so4j", "e_no3i", "e_no3j", "e_naaj",
             "e_naai")
INPUTS = ("co", "nox", "nh3", "nmvoc", "so2", "bc", "oc", "pm25", "pm10")

CITE_HERMES = ("Guevara et al. (2019), HERMESv3 Part 1, Geosci. Model Dev. 12, "
               "1885-1907, doi:10.5194/gmd-12-1885-2019, Table 2")
CITE_TAR = ("IPCC TAR WG1 Ch. 4 (Prather, Ehhalt et al. 2001), Table 4.7b, "
            "industrial NMVOC")


def _row(units, terms, provenance, clamp=False):
    return {"units": units,
            "terms": [{"field": f, "weight": float(f"{w:.9e}")} for f, w in terms],
            "clamp_at_zero": clamp, "provenance": provenance}


def build() -> dict:
    voc, why = _nmvoc_moles()
    out = {}
    gas = "mol km-2 hr-1"
    aer = "ug m-2 s-1"
    out["e_so2"] = _row(gas, [("so2", GAS / M_SO2)],
                        f"SO2 mass / {M_SO2} g mol-1; all of EDGAR's SO2 is gas "
                        "(no primary sulfate split: see e_so4i)")
    out["e_co"] = _row(gas, [("co", GAS / M_CO)], f"CO mass / {M_CO:.3f} g mol-1")
    out["e_no"] = _row(gas, [("nox", NOX_NO * GAS / M_NO2)],
                       f"{NOX_NO} of NOx moles (mass / {M_NO2:.4f} g mol-1) as NO. "
                       + CITE_NOX)
    out["e_no2"] = _row(gas, [("nox", NOX_NO2 * GAS / M_NO2)],
                        f"{NOX_NO2} of NOx moles (mass / {M_NO2:.4f} g mol-1) as "
                        "NO2, the complement of e_no. " + CITE_NOX)
    out["e_nh3"] = _row(gas, [("nh3", GAS / M_NH3)], f"NH3 mass / {M_NH3:.3f} g mol-1")
    for e in ("e_hcl", "e_hono"):
        out[e] = _row(gas, [], "not emitted: EDGAR v8.1 AP carries no HCl or HONO "
                      f"and the RADM2 profile of {CITE_HERMES} assigns none")
    out["e_csl"] = _row(gas, [], f"not emitted: {CITE_TAR} has no cresol group")
    out["e_iso"] = _row(gas, [], f"not emitted: {CITE_TAR} carries no isoprene "
                        "(voc10); isoprene is biogenic")
    for cls in ("ald", "hcho", "ora2", "hc3", "hc5", "hc8", "eth", "olt", "oli",
                "tol", "xyl", "ket", "ol2"):
        out[f"e_{cls}"] = _row(
            gas, [("nmvoc", voc[cls] / 1000.0 * GAS)],
            f"{voc[cls]:.4f} mol per kg NMVOC = sum of " + "; ".join(why[cls])
            + f". Profile {CITE_TAR}; classes and shares {CITE_HERMES}")
    fine = "max(pm25 - bc - {om} oc, 0)".format(om=OM_OC)
    for e, share in (("e_pm25i", AITKEN), ("e_pm25j", ACCUMULATION)):
        out[e] = _row(aer, [("pm25", share * AEROSOL), ("bc", -share * AEROSOL),
                            ("oc", -share * OM_OC * AEROSOL)],
                      f"{share} x {fine}: unspeciated primary PM2.5 (its BC and "
                      f"organic matter removed, OM:OC {OM_OC}, Simpson et al. 2012 "
                      "Sect. 6.3, which also caps EC+OM below PM); Aitken:"
                      f"accumulation {AITKEN}:{ACCUMULATION}, Tuccella et al. (2012) "
                      f"via {CITE_HERMES}", clamp=True)
    out["e_pm_10"] = _row(aer, [("pm10", AEROSOL), ("pm25", -AEROSOL)],
                          "max(pm10 - pm25, 0): the coarse primary PM", clamp=True)
    for e, share in (("e_eci", AITKEN), ("e_ecj", ACCUMULATION)):
        out[e] = _row(aer, [("bc", share * AEROSOL)],
                      f"{share} of BC mass, Tuccella et al. (2012) via {CITE_HERMES}")
    for e, share in (("e_orgi", AITKEN), ("e_orgj", ACCUMULATION)):
        out[e] = _row(aer, [("oc", share * OM_OC * AEROSOL)],
                      f"{share} x {OM_OC} x OC: SORGAM's orgpa rows are organic "
                      f"mass, EDGAR's OC is carbon (OM:OC {OM_OC}, Simpson et al. "
                      f"2012, ACP 12, 7825-7865, Sect. 6.3); split {CITE_HERMES}")
    for e in ("e_so4i", "e_so4j"):
        out[e] = _row(aer, [], "not emitted: EDGAR v8.1 AP has no primary sulfate "
                      "field; whatever primary sulfate it counts stays inside PM2.5 "
                      f"(e_pm25i/j). {CITE_HERMES} likewise sets SO4I/SO4J to 0")
    for e in ("e_no3i", "e_no3j"):
        out[e] = _row(aer, [], "not emitted: no primary nitrate field in EDGAR v8.1 "
                      f"AP; {CITE_HERMES} sets NO3I/NO3J to 0")
    for e in ("e_naaj", "e_naai"):
        out[e] = _row(aer, [], "not emitted: anthropogenic primary sodium has no "
                      "EDGAR v8.1 AP field (sea salt is a separate process)")
    ordered = {e: out[e] for e in GAS_E + AEROSOL_E}
    assert set(ordered) == set(out)
    return {
        "schema": "gpuwm.chem.speciation.v1",
        "name": "radm2",
        "description": (
            "WOOF's speciation of anthropogenic inventory fields (kg m-2 s-1) into "
            "the emis_ant rows WRF-Chem's emiss_opt = 3 reads for RADM2 / RADM2SORG "
            "(gases mol km-2 hr-1, aerosols ug m-2 s-1). Each row is the sum of "
            "its terms (field x weight, float64), clamped at zero where "
            "clamp_at_zero is set. WOOF's choices, not WRF's: WRF reads emis_ant "
            "ready-made."),
        "inputs": list(INPUTS),
        "input_units": "kg m-2 s-1",
        "constants": {
            "nox_molar_split_no_no2": [NOX_NO, NOX_NO2],
            "aitken_accumulation_mass_split": [AITKEN, ACCUMULATION],
            "om_per_oc": OM_OC,
            "molar_mass_g_mol": {"co": round(M_CO, 6), "no2": round(M_NO2, 6),
                                 "nh3": round(M_NH3, 6), "so2": M_SO2},
            "nmvoc_mol_per_kg": {k: float(f"{v:.9e}") for k, v in sorted(voc.items())},
            "nmvoc_mass_fraction_emitted": round(sum(
                wt / 100.0 * sum(shares.values())
                for _g, _n, wt, _c, _f, _m, shares in NMVOC_GROUPS), 6),
        },
        "deviations": list(DEVIATIONS),
        "outputs": ordered,
        "caps": [{
            "members": ["e_eci", "e_ecj", "e_orgi", "e_orgj"],
            "limit": {"field": "pm25", "weight": 1.0e9},
            "provenance": "EC + OM never exceeds the cell's PM2.5 (Simpson et al. "
                          "2012, ACP 12, 7825-7865, Sect. 6.3): where BC + 1.3 OC "
                          "is above EDGAR's PM2.5 (two independent inventories and "
                          "a fixed OM:OC), the four carbon rows are scaled by "
                          "PM2.5 / (their sum), keeping their i/j and EC/OM ratios; "
                          "the residual PM2.5 rows are then zero by their clamp."}],
        "provenance": (
            f"Generated by tools/chem_speciation/make_radm2_speciation.py. NMVOC: "
            f"{CITE_TAR} mapped to RADM2 (Stockwell et al. 1990) with {CITE_HERMES}; "
            "NOx split Crippa et al. 2018; i/j split Tuccella et al. 2012; "
            "OM:OC Simpson et al. 2012. Gas weights fold 1e6 m2 km-2 x 3600 s hr-1 "
            "x 1000 g kg-1 / M; aerosol weights fold 1e9 ug kg-1."),
    }


def text() -> str:
    return json.dumps(build(), indent=1) + "\n"


def main(argv) -> int:
    root = Path(".")
    path = root / OUT
    if "--check" in argv:
        if path.read_text(encoding="utf-8") != text():
            print(f"{OUT} differs from the generator; rerun without --check")
            return 1
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text(), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

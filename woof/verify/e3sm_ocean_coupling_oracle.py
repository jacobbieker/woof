"""E3SM v3.2.0's MPAS-Ocean coupler import and export: cases, stream, GPU run.

``tools/e3sm_coupler_oracle/run_ocean_coupling.F90`` drives E3SM v3.2.0's
``ocn_import_mct`` and ``ocn_export_mct`` (cut verbatim out of
``components/mpas-ocean/driver/ocn_comp_mct.F``, with ``mpaso_cpl_indices``,
``ocn_constants`` and ``ocn_freezing_temperature`` as E3SM builds them) over
MPAS pools read from a stream, built with gfortran -O0 -ffp-contract=off.
Each case is one process: the coupler's field lists and switches, ``x2o`` and
``o2x``, and the configs, packages and arrays the two routines read. The
oracle writes ``o2x`` after the export and every array after both routines.

The lists are the G case's (``e3sm_ocean_merge_oracle.GCASE``'s ``x2o``)
plus the waves coupled two ways and the river nutrients, and an ``o2x`` with
the land-ice, removed-runoff, frazil and 3-D thermal-forcing fields. The
cases reach: absent forcing fields (a package that is off) left alone, halo
cells (the arrays are longer than ``nCellsSolve``), the removed Antarctic
runoff on both sides of (and on) its latitudes, the shortwave's ``max`` and
the sea-ice pressure's ``min`` at signed zeros and the cap, the precipitation
factor, the 10 m wind's square root, the active-wave stress, the frazil's
three branches (positive mass, the freezing-point energy, land ice) with a
land-ice mask and without one, the land-ice modes ``standalone``, ``data``
and ``coupled``, and the thermal forcing's mask threshold; and the stops
E3SM makes on a list that lacks a required field or an unknown ``glc_nzoc``.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_ocean_merge_oracle import GCASE, NUTRIENTS, word_mismatches  # noqa: F401

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_ocean_coupling"

#: the float32 rounding of mpas_constants' default-real ``gravity = 9.80616``
GRAVITY = float(np.float32(9.80616))

WAVE_X2O = ([f"Sw_ustokes_wavenumber_{m}" for m in range(1, 7)] + [f"Sw_vstokes_wavenumber_{m}" for m in range(1, 7)]
            + "Sw_Hs Sw_Fp Sw_Dp Sw_Z0 Sw_Ustar Sw_Charn Faww_Tawx Faww_Tawy Fwow_Twox Fwow_Twoy Faow_Tocx "
              "Faow_Tocy".split())
NUTRIENT_X2O = [f"Foxx_rof{t}" for t in NUTRIENTS]
O2X_BASE = ("So_u So_t Faoo_h2otemp So_s So_v So_ssh So_dhdx So_dhdy Fioo_q Fioo_frazil So_blt So_bls So_htv "
            "So_stv So_rhoeff").split()
O2X_LAND_ICE = "Foxo_q_li Foxo_frazil_li Foxo_ismw Foxo_ismh".split()
O2X_REMOVED = "Foxo_rrofl Foxo_rrofi Foxo_rrofih".split()


def tf3d_names(glc_nzoc: int) -> list[str]:
    return [f"So_tf3d{k:02d}" for k in range(1, glc_nzoc + 1)] + [f"So_tf3d_mask{k:02d}"
                                                                  for k in range(1, glc_nzoc + 1)]


#: the import's 1-D forcing fields (x2o's field each copies, or a formula)
IMPORT_1D = ("windStressZonal windStressMeridional latentHeatFlux sensibleHeatFlux longWaveHeatFluxUp "
             "longWaveHeatFluxDown evaporationFlux seaIceHeatFlux icebergHeatFlux snowFlux seaIceFreshWaterFlux "
             "icebergFreshWaterFlux seaIceSalinityFlux riverRunoffFlux iceRunoffFlux shortWaveHeatFlux rainFlux "
             "atmosphericPressure seaIcePressure iceFraction removedRiverRunoffFlux removedIceRunoffFlux "
             "significantWaveHeight peakWaveFrequency peakWaveDirection z0wav ustarwav charnockSeaState "
             "waveSupportedZonalWindStress waveSupportedMeridionalWindStress waveToOceanZonalWindStress "
             "waveToOceanMeridionalWindStress waveNetOceanZonalWindStress waveNetOceanMeridionalWindStress "
             "landIceFreshwaterFlux landIceHeatFlux landIceFraction windSpeed10m").split()
IMPORT_2D = ("stokesDriftZonalWavenumber", "stokesDriftMeridionalWavenumber", "landIceInterfaceTracers")
#: configs the two routines look up (logical), and their G-case values
LOGICAL_CONFIGS = ("config_use_ecosysTracers config_use_DMSTracers config_use_MacroMoleculesTracers "
                   "config_use_ecosysTracers_sea_ice_coupling config_use_DMSTracers_sea_ice_coupling "
                   "config_use_MacroMoleculesTracers_sea_ice_coupling config_use_CFCTracers "
                   "config_remove_ais_river_runoff config_remove_ais_ice_runoff config_cvmix_kpp_use_theory_wave "
                   "config_eke_correction_enable config_use_precip_scaling").split()
#: the forcing pool's tracer-index dimensions (1-based, as MPAS's)
EXPORT_DIMS = ("index_avgTemperatureSurfaceValue", "index_avgSalinitySurfaceValue", "index_avgSurfaceVelocityZonal",
               "index_avgSurfaceVelocityMeridional", "index_avgSSHGradientZonal", "index_avgSSHGradientMeridional")

_POOLS = {"configs": 1, "packages": 2, "mesh": 3, "forcing": 4, "state": 5}


def _uniform(rng, low: float, high: float, n) -> np.ndarray:
    """``rng.random`` scaled with separate numpy operations (no contraction)."""
    return (rng.random(n) * (high - low)) + low


def _spec(name, *, lsize=23, halo=3, wav="none", nutrients=False, one_way=False, glc_nzoc=0, n_glc_z=0,
          active_wave=False, precip=None, remove=False, kpp=False, eke=False, frazil=True, mask=False,
          land_ice="off", tf="off", absent=(), freezing=None, dt=1800, aborts=False, x2o_drop=(), o2x_drop=(),
          o2x_extra=(), density0=1026.0, hof=3.34e5):
    return dict(name=name, lsize=lsize, halo=halo, wav=wav, nutrients=nutrients, one_way=one_way,
                glc_nzoc=glc_nzoc, n_glc_z=n_glc_z, active_wave=active_wave, precip=precip, remove=remove,
                kpp=kpp, eke=eke, frazil=frazil, mask=mask, land_ice=land_ice, tf=tf, absent=tuple(absent),
                freezing=freezing, dt=dt, aborts=aborts, x2o_drop=tuple(x2o_drop), o2x_drop=tuple(o2x_drop),
                o2x_extra=tuple(o2x_extra), density0=density0, hof=hof)


_WAVE_FIELDS = ("stokesDriftZonalWavenumber stokesDriftMeridionalWavenumber significantWaveHeight peakWaveFrequency "
                "peakWaveDirection z0wav ustarwav charnockSeaState waveSupportedZonalWindStress "
                "waveSupportedMeridionalWindStress waveToOceanZonalWindStress waveToOceanMeridionalWindStress "
                "waveNetOceanZonalWindStress waveNetOceanMeridionalWindStress").split()
_LAND_ICE_FIELDS = "landIceFreshwaterFlux landIceHeatFlux landIceFraction landIceInterfaceTracers".split()

SPECS = (
    # a G case: no waves, no land ice, frazil on, no land-ice mask
    _spec("gcase", absent=_WAVE_FIELDS + _LAND_ICE_FIELDS + ["removedRiverRunoffFlux", "removedIceRunoffFlux"]),
    # land ice standalone: the mask, the removed runoff, the precipitation factor, the theory-wave 10 m wind
    _spec("standalone", lsize=37, mask=True, land_ice="standalone", remove=True, precip=0.9873046875000123,
          kpp=True, o2x_extra=O2X_LAND_ICE + O2X_REMOVED, absent=_WAVE_FIELDS, density0=1027.25),
    # land ice data, other freezing coefficients and heat of fusion, an odd coupling interval
    _spec("data", lsize=31, mask=True, land_ice="data", o2x_extra=O2X_LAND_ICE, absent=_WAVE_FIELDS,
          freezing=(0.0901, -0.0575, -7.61e-4, 0.0, -18.48 * 1.0371), dt=1237, hof=3.337e5),
    # land ice coupled, 3-D thermal forcing (nGlcZLevels < glc_nzoc), no frazil, the EKE 10 m wind, one way
    _spec("coupled_tf3d", lsize=29, land_ice="coupled", tf="3d", glc_nzoc=4, n_glc_z=3, frazil=False, eke=True,
          one_way=True, o2x_extra=tf3d_names(4), absent=_WAVE_FIELDS),
    _spec("tf3d_30", lsize=11, halo=0, tf="3d", glc_nzoc=30, n_glc_z=30, o2x_extra=tf3d_names(30),
          absent=_WAVE_FIELDS),
    # the waves coupled two ways with the active-wave stress; fields whose packages are off
    _spec("waves_active", lsize=27, wav="twoway", active_wave=True, nutrients=True, remove=True,
          o2x_extra=O2X_REMOVED,
          absent=["icebergHeatFlux", "seaIcePressure", "peakWaveDirection", "landIceFraction"]),
    _spec("waves_passive", lsize=19, wav="twoway", absent=["stokesDriftMeridionalWavenumber", "ustarwav",
                                                           "windStressMeridional", "rainFlux"]),
    # the stops
    _spec("abort_missing_x2o", lsize=5, aborts=True, x2o_drop=["Sg_blit"]),
    _spec("abort_missing_x2o_icemask", lsize=5, aborts=True, x2o_drop=["Sg_icemask_floating"]),
    _spec("abort_missing_o2x", lsize=5, aborts=True, o2x_drop=["So_rhoeff"]),
    _spec("abort_missing_tf3d", lsize=5, aborts=True, glc_nzoc=4, n_glc_z=4, tf="3d",
          o2x_extra=[n for n in tf3d_names(4) if n != "So_tf3d_mask03"]),
    _spec("abort_missing_wave", lsize=5, aborts=True, wav="twoway", x2o_drop=["Sw_Charn"]),
    _spec("abort_missing_nutrient", lsize=5, aborts=True, nutrients=True, x2o_drop=["Foxx_rofDSi"]),
    _spec("abort_glc_nzoc", lsize=5, aborts=True, glc_nzoc=5),
)


def _x2o_list(spec) -> list[str]:
    names = list(GCASE["x2o"])
    if spec["wav"] == "twoway":
        names += WAVE_X2O
    if spec["nutrients"]:
        names += NUTRIENT_X2O
    return [n for n in names if n not in spec["x2o_drop"]]


def _o2x_list(spec) -> list[str]:
    names = O2X_BASE + list(spec["o2x_extra"])
    return [n for n in names if n not in spec["o2x_drop"]]


def _x2o_values(rng, names, lsize) -> np.ndarray:
    out = _uniform(rng, -300.0, 300.0, (lsize, len(names)))
    for k, name in enumerate(names):
        if name == "So_duu10n":
            out[:, k] = _uniform(rng, 0.0, 400.0, lsize)
            out[0, k] = 0.0
        elif name == "Foxx_swnet":
            out[:, k] = _uniform(rng, -50.0, 600.0, lsize)
            out[:4, k] = (-0.0, 0.0, -1.0e-300, 5.0e-324)
        elif name == "Si_bpress":
            out[:, k] = _uniform(rng, 0.0, 1.0e5, lsize)
            out[:3, k] = (-0.0, 0.0, -12.5)
        elif name.startswith(("Faxa_rain", "Faxa_snow", "Foxx_rof")):
            out[:, k] = _uniform(rng, 0.0, 1.0e-4, lsize)
    return out


def _lat(rng, n) -> np.ndarray:
    lat = _uniform(rng, -1.5707963267948966, 1.5707963267948966, n)
    lat[:4] = (-1.04719666667, -0.99483767345, np.nextafter(-1.04719666667, 0.0), -1.3)
    return lat


def _arrays(rng, spec, x2o_names) -> dict:
    """pool -> name -> (kind, value); arrays are in Fortran's memory order:
    a (k, nCells) field is a C array (nCells, k)."""
    ns, nc = spec["lsize"], spec["lsize"] + spec["halo"]
    absent = set(spec["absent"])
    configs = {name: False for name in LOGICAL_CONFIGS}
    configs.update(config_remove_ais_river_runoff=spec["remove"], config_remove_ais_ice_runoff=spec["remove"],
                   config_cvmix_kpp_use_theory_wave=spec["kpp"], config_eke_correction_enable=spec["eke"],
                   config_use_precip_scaling=spec["precip"] is not None)
    pools = {"configs": {k: (1, v) for k, v in configs.items()}}
    pools["configs"]["config_density0"] = (2, spec["density0"])
    pools["configs"]["config_land_ice_flux_mode"] = (3, spec["land_ice"])
    pools["configs"]["config_glc_thermal_forcing_coupling_mode"] = (3, spec["tf"])
    pools["packages"] = {"frazilIceActive": (1, spec["frazil"])}
    pools["mesh"] = {"nCellsSolve": (4, ns), "latCell": (5, _lat(rng, nc)),
                     "areaCell": (5, _uniform(rng, 1.0e8, 4.0e8, nc))}
    forcing = {}
    for name in IMPORT_1D:
        forcing[name] = (8, None) if name in absent else (5, _uniform(rng, -10.0, 10.0, nc))
    for name in IMPORT_2D:
        width = 2 if name == "landIceInterfaceTracers" else 6
        forcing[name] = (9, None) if name in absent else (6, _uniform(rng, -1.0, 1.0, (nc, width)))
    forcing["index_landIceInterfaceTemperature"] = (4, 1)
    forcing["index_landIceInterfaceSalinity"] = (4, 2)
    # export: surface tracers (T in K, S), velocities, the SSH gradient, each with spare rows
    for k, name in enumerate(EXPORT_DIMS):
        forcing[name] = (4, (3, 1, 2, 1, 1, 2)[k])
    tsv = np.empty((nc, 3))
    tsv[:, 0] = _uniform(rng, 0.0, 40.0, nc)  # salinity
    tsv[:, 1] = _uniform(rng, -5.0, 5.0, nc)
    tsv[:, 2] = _uniform(rng, 268.0, 300.0, nc)  # temperature, K
    tsv[:6, 2] = (271.0, 290.0, 268.5, 273.15, 271.25, 272.5)
    forcing["avgTracersSurfaceValue"] = (6, tsv)
    forcing["avgSurfaceVelocity"] = (6, _uniform(rng, -2.0, 2.0, (nc, 2)))
    forcing["avgSSHGradient"] = (6, _uniform(rng, -1.0e-4, 1.0e-4, (nc, 2)))
    forcing["avgTotalFreshWaterTemperatureFlux"] = (5, _uniform(rng, -1.0e-3, 1.0e-3, nc))
    if spec["mask"]:
        forcing["landIceMask"] = (7, (np.arange(nc) % 5 == 1).astype(np.int32) + (np.arange(nc) % 7 == 3))
    else:
        forcing["landIceMask"] = (10, None)
    if spec["frazil"]:
        forcing["seaIceEnergy"] = (5, _uniform(rng, -1.0e6, 1.0e6, nc))
        forcing["frazilSurfacePressure"] = (5, _uniform(rng, 0.0, 100.0, nc))
    if spec["land_ice"] in ("standalone", "data"):
        forcing["avgLandIceFreshwaterFlux"] = (5, _uniform(rng, -1.0e-4, 1.0e-4, nc))
        forcing["avgLandIceHeatFlux"] = (5, _uniform(rng, -100.0, 100.0, nc))
    if spec["remove"]:
        forcing["avgRemovedRiverRunoffFlux"] = (5, _uniform(rng, 0.0, 1.0e-4, nc))
        forcing["avgRemovedIceRunoffFlux"] = (5, _uniform(rng, 0.0, 1.0e-4, nc))
        forcing["avgRemovedIceRunoffHeatFlux"] = (5, _uniform(rng, -50.0, 0.0, nc))
    if spec["tf"] == "3d":
        nz = max(spec["n_glc_z"], 1)
        forcing["nGlcZLevels"] = (4, spec["n_glc_z"])
        forcing["avgThermalForcingAtZLevels"] = (6, _uniform(rng, -2.0, 5.0, (nc, nz)))
        maskz = _uniform(rng, 0.0, 1.0, (nc, nz))
        maskz.flat[:4] = (0.9, np.nextafter(0.9, 1.0), 0.0, 1.0)
        forcing["avgThermalForcingAtZLevelsMask"] = (6, maskz)
    pools["forcing"] = forcing
    state = {"ssh": (5, _uniform(rng, -2.0, 2.0, nc))}
    lt = _uniform(rng, 0.0, 20.0, (nc, 4))
    lt[1:3, 0] = 0.0  # a zero energy of either sign into min(.., 0)
    state["layerThickness"] = (6, lt)
    if spec["frazil"]:
        acc = _uniform(rng, -5.0, 5.0, nc)
        acc[:5] = (0.0, -0.0, 0.0, 2.5, -1.0)
        state["accumulatedFrazilIceMass"] = (5, acc)
    pools["state"] = state
    return pools


def cases() -> list[dict]:
    out = []
    for seed, spec in enumerate(SPECS):
        rng = np.random.default_rng(20261120 + 101 * seed)
        x2o_names, o2x_names = _x2o_list(spec), _o2x_list(spec)
        ns = spec["lsize"]
        case = dict(spec)
        case["lists"] = {"x2o": x2o_names, "o2x": o2x_names}
        case["x2o"] = _x2o_values(rng, x2o_names, ns)
        case["o2x"] = _uniform(rng, -7.0, 7.0, (ns, len(o2x_names)))
        nc = ns + spec["halo"]
        case["blt"] = _uniform(rng, -2.0, 35.0, (nc, 2))
        case["tv"] = _uniform(rng, 0.0, 1.0e-4, (nc, 2))
        case["pools"] = _arrays(rng, spec, x2o_names)
        out.append(case)
    return out


def _freezing(case) -> tuple:
    return case["freezing"] or (0.0, 0.0, 0.0, 0.0, -18.48)


def _entries(case):
    for pool in ("configs", "packages", "mesh", "forcing", "state"):
        for name, (kind, value) in case["pools"][pool].items():
            yield pool, name, kind, value


def input_bytes(case: dict) -> bytes:
    """run_ocean_coupling.F90's stream."""
    head = struct.pack("<9i", case["dt"], 0, int(case["nutrients"]), int(case["one_way"]), case["glc_nzoc"],
                       int(case["active_wave"]), 0, 0, 0)
    head += struct.pack("<11d", case["precip"] if case["precip"] is not None else 1.0, case["hof"],
                        0.0, 0.0, 0.0, 0.0, *_freezing(case))
    head += case["wav"].ljust(8).encode()
    nbl = case["lsize"] + case["halo"]
    parts = [head, struct.pack("<5i", 1, 2, 1, 2, nbl),
             np.ascontiguousarray(case["blt"], dtype="<f8").tobytes(),
             np.ascontiguousarray(case["tv"], dtype="<f8").tobytes()]
    for key in ("x2o", "o2x"):
        rlist = ":".join(case["lists"][key]).encode()
        parts.append(struct.pack("<i", len(rlist)) + rlist)
    parts.append(struct.pack("<3i", case["lsize"], len(case["lists"]["x2o"]), len(case["lists"]["o2x"])))
    parts.append(np.ascontiguousarray(case["x2o"], dtype="<f8").tobytes())
    parts.append(np.ascontiguousarray(case["o2x"], dtype="<f8").tobytes())
    entries = list(_entries(case))
    parts.append(struct.pack("<i", len(entries)))
    for pool, name, kind, value in entries:
        parts.append(struct.pack("<2i", _POOLS[pool], kind) + name.ljust(64).encode())
        if kind == 1:
            parts.append(struct.pack("<i", int(value)))
        elif kind == 2:
            parts.append(struct.pack("<d", value))
        elif kind == 3:
            parts.append(value.ljust(64).encode())
        elif kind == 4:
            parts.append(struct.pack("<i", value))
        elif kind == 5:
            parts.append(struct.pack("<i", value.size) + np.ascontiguousarray(value, dtype="<f8").tobytes())
        elif kind == 6:
            parts.append(struct.pack("<2i", value.shape[1], value.shape[0])
                         + np.ascontiguousarray(value, dtype="<f8").tobytes())
        elif kind == 7:
            parts.append(struct.pack("<i", value.size) + np.ascontiguousarray(value, dtype="<i4").tobytes())
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def array_entries(case: dict) -> list[tuple[str, str, int, object]]:
    """The arrays the oracle writes back, in its order."""
    return [(p, n, k, v) for p, n, k, v in _entries(case) if k in (5, 6, 7)]


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """``o2x`` and ``<pool>/<name>`` for every array, from the oracle's output."""
    out, at = {}, 0
    ns, no = case["lsize"], len(case["lists"]["o2x"])
    out["o2x"] = np.frombuffer(raw, dtype="<f8", count=ns * no, offset=at).reshape(ns, no).copy()
    at += 8 * ns * no
    for pool, name, kind, value in array_entries(case):
        dtype = "<i4" if kind == 7 else "<f8"
        count = value.size
        out[f"{pool}/{name}"] = np.frombuffer(raw, dtype=dtype, count=count, offset=at).reshape(value.shape).copy()
        at += 4 * count if kind == 7 else 8 * count
    assert at == len(raw), (case["name"], at, len(raw))
    return out


# --- the port ------------------------------------------------------------------------------------------------


def indices_for(case: dict):
    from woof.hex.coupled import ocn_coupling_e3sm as occ
    return occ.indices(case["lists"]["x2o"], case["lists"]["o2x"], rof2ocn_nutrients=case["nutrients"],
                       wav_ocn_coup=case["wav"], ocn_lnd_one_way=case["one_way"], glc_nzoc=case["glc_nzoc"])


def options_for(case: dict):
    from woof.hex.coupled import ocn_coupling_e3sm as occ
    c0, cs, cp_, cps, az1 = _freezing(case)
    return occ.Options(
        ocn_cpl_dt=case["dt"], config_density0=case["density0"],
        config_use_precip_scaling=case["precip"] is not None,
        precip_fact=case["precip"] if case["precip"] is not None else 1.0,
        config_remove_ais_river_runoff=case["remove"], config_remove_ais_ice_runoff=case["remove"],
        config_cvmix_kpp_use_theory_wave=case["kpp"], config_eke_correction_enable=case["eke"],
        config_momentum_use_active_wave=case["active_wave"], frazil_ice_active=case["frazil"],
        config_land_ice_flux_mode=case["land_ice"], config_glc_thermal_forcing_coupling_mode=case["tf"],
        config_frazil_heat_of_fusion=case["hof"], open_ocean_freezing=(c0, cs, cp_, cps, az1))


def run_gpu(case: dict) -> dict[str, np.ndarray]:
    """The port: the import then the export, outputs as the oracle writes them."""
    import cupy as cp
    from woof.hex.coupled import ocn_coupling_e3sm as occ
    idx, opts = indices_for(case), options_for(case)
    dev = {}
    for pool, name, kind, value in _entries(case):
        if kind in (5, 6):
            dev[pool, name] = cp.asarray(np.ascontiguousarray(value, dtype=np.float64))
        elif kind == 7:
            dev[pool, name] = cp.asarray(np.ascontiguousarray(value, dtype=np.int32))
    forcing = {n: a for (p, n), a in dev.items() if p == "forcing"}
    state = {n: a for (p, n), a in dev.items() if p == "state"}
    dims = {n: v for n, (k, v) in case["pools"]["forcing"].items() if k == 4}
    ns = case["lsize"]
    x2o = cp.asarray(case["x2o"].astype(np.float64))
    o2x = cp.asarray(case["o2x"].astype(np.float64))
    occ.import_x2o(idx, opts, x2o, n_cells_solve=ns, lat_cell=dev["mesh", "latCell"], forcing=forcing)
    diagnostics = {"landIceBoundaryLayerTracers": cp.asarray(case["blt"].astype(np.float64)),
                   "landIceTracerTransferVelocities": cp.asarray(case["tv"].astype(np.float64)),
                   "indexBLT": 1, "indexBLS": 2, "indexHeatTrans": 1, "indexSaltTrans": 2}
    occ.export_o2x(idx, opts, o2x, n_cells_solve=ns, forcing=forcing, state=state, dims=dims,
                   diagnostics=diagnostics)
    out = {"o2x": cp.asnumpy(o2x)}
    for (pool, name), array in dev.items():
        out[f"{pool}/{name}"] = cp.asnumpy(array)
    return out


def load_fixture() -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    data = np.load(FIXTURE_DIR / "oracle-r8.npz")
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        case, _, rest = key.partition("/")
        outputs.setdefault(case, {})[rest] = data[key]
    aborts = json.loads((FIXTURE_DIR / "aborts.json").read_text(encoding="utf-8"))
    return outputs, aborts


def measure() -> dict[str, dict[str, int]]:
    """case -> output (``o2x`` column or array) -> words that differ from the oracle."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        if case["aborts"]:
            continue
        got, want = run_gpu(case), outputs[case["name"]]
        fields = {}
        for k, name in enumerate(case["lists"]["o2x"]):
            fields[f"o2x/{name}"] = word_mismatches(got["o2x"][:, k], want["o2x"][:, k])
        for key in want:
            if key != "o2x":
                fields[key] = word_mismatches(got[key], want[key])
        report[case["name"]] = fields
    return report

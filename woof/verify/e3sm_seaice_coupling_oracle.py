"""E3SM v3.2.0's MPAS-Seaice coupler import and export: cases, stream, GPU run.

``tools/e3sm_coupler_oracle/run_seaice_coupling.F90`` drives E3SM v3.2.0's
``ice_import_mct`` and ``ice_export_mct`` (cut verbatim out of
``components/mpas-seaice/driver/ice_comp_mct.F`` with ``basal_pressure`` and
``frazil_mass``, over ``mpassi_cpl_indices``, ``seaice_constants`` and
Icepack's column physics as E3SM builds them) on MPAS pools read from a
stream, built with gfortran -O0 -ffp-contract=off. Each case is one process:
the coupler's field lists and switches, ``x2i`` and ``i2x``, and the configs,
dimensions and arrays the two routines read. The oracle writes ``i2x`` after
the export, every array after both routines, and the rotation's sine and
cosine words per cell (``trig``).

The cases reach: the ocean surface types ``free`` (the frazil mass, mushy
and not, both Si0new branches and both liquidus branches) and not; every
``tfrz_option`` (``mushy`` under mushy and BL99 thermodynamics,
``linear_salt``, ``constant``, ``minus1p8``); ice fractions of 0, -0, below,
at and above 1; the rotated and unrotated grids (with a pole and the
``atan2(0, -0)`` point); topographic ponds; the column shortwave; icebergs;
the bulk and modal aerosols and the z-aerosols (with their coupled dust
iron); the waves coupled two ways with and without the column waves and
the floe size distribution; halo cells; and the stops MCT makes on a list
that lacks a required field.
"""
from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import numpy as np

from woof.verify.e3sm_ocean_merge_oracle import word_mismatches  # noqa: F401

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "data" / "oracles" / "e3sm_seaice_coupling"

AEROSOL_X2I = ("Faxa_bcphidry Faxa_bcphodry Faxa_bcphiwet Faxa_ocphidry Faxa_ocphodry Faxa_ocphiwet "
               "Faxa_dstdry1 Faxa_dstdry2 Faxa_dstdry3 Faxa_dstdry4 Faxa_dstwet1 Faxa_dstwet2 Faxa_dstwet3 "
               "Faxa_dstwet4").split()
#: a shuffled x2i (the order is the coupler's; the indices follow it)
X2I_BASE = ("Sa_z So_s Faxa_lwdn So_t Sa_u So_u Sa_v So_v Fioo_q Sa_tbot Sa_ptem So_dhdx Sa_shum So_dhdy Sa_dens "
            "Faxa_rain Fioo_frazil Faxa_snow Faxa_swndr Faxa_swvdr Faxa_swndf Faxa_swvdf").split()
I2X_BASE = ("Si_t Faii_taux Si_bpress Si_tref Si_qref Si_ifrac Si_ithick Si_avsdr Si_anidr Si_avsdf Si_anidf "
            "Si_snowh Si_u10 Si_u10withgusts Faii_tauy Faii_lat Faii_sen Faii_lwup Faii_evap Faii_swnet Fioi_swpen "
            "Fioi_melth Fioi_meltw Fioi_salt Fioi_taux Fioi_tauy PFioi_bergh PFioi_bergw").split()

OCEAN_COUPLING = ("seaSurfaceTemperature seaSurfaceSalinity seaFreezingTemperature freezingMeltingPotential "
                  "frazilMassAdjust uOceanVelocity vOceanVelocity seaSurfaceTiltU seaSurfaceTiltV").split()
ATMOS_COUPLING = ("airLevelHeight airPotentialTemperature airTemperature airSpecificHumidity airDensity "
                  "shortwaveVisibleDirectDown shortwaveVisibleDiffuseDown shortwaveIRDirectDown "
                  "shortwaveIRDiffuseDown longwaveDown rainfallRate snowfallRate uAirVelocity vAirVelocity "
                  "atmosReferenceSpeed10m atmosReferenceTemperature2m atmosReferenceHumidity2m").split()
TRACERS_AGGREGATE = ("iceAreaCell iceVolumeCell snowVolumeCell pondDepthCell pondLidThicknessCell pondAreaCell "
                     "surfaceTemperatureCell").split()
VELOCITY_SOLVER = "airStressCellU airStressCellV oceanStressCellU oceanStressCellV".split()
SHORTWAVE = ("albedoVisibleDirectCell albedoIRDirectCell albedoVisibleDiffuseCell albedoIRDiffuseCell "
             "absorbedShortwaveFlux").split()
ATMOS_FLUXES = "latentHeatFlux sensibleHeatFlux longwaveUp evaporativeWaterFlux".split()
OCEAN_FLUXES = "oceanHeatFlux oceanShortwaveFlux oceanFreshWaterFlux oceanSaltFlux".split()
#: biogeochemistry pool arrays of the z-aerosols, (name, components)
ZAEROSOL = (("atmosBlackCarbonFlux", 2), ("atmosDustFlux", 4), ("atmosWetDustFlux", 4), ("atmosDryDustFlux", 4),
            ("oceanZAerosolConc", 6))

KTHERM = {"zero layer": 0, "BL99": 1, "mushy": 2}


def _spec(name, *, lsize=23, halo=3, thermo="mushy", tfrz="mushy", surface="free", rotate=True, ponds=False,
          column_sw=True, bergs=False, aerosols=False, modal=False, zaer=False, couple_bgc=False, column_bgc=False,
          wav="none", nfreq=0, column_waves=False, n_frequencies=0, fsd=False, aborts=False, x2i_drop=(),
          i2x_drop=(), i2x_extra=(), radius=6371229.0, seed=0):
    return dict(name=name, lsize=lsize, halo=halo, thermo=thermo, tfrz=tfrz, surface=surface, rotate=rotate,
                ponds=ponds, column_sw=column_sw, bergs=bergs, aerosols=aerosols, modal=modal, zaer=zaer,
                couple_bgc=couple_bgc, column_bgc=column_bgc, wav=wav, nfreq=nfreq, column_waves=column_waves,
                n_frequencies=n_frequencies, fsd=fsd, aborts=aborts, x2i_drop=tuple(x2i_drop),
                i2x_drop=tuple(i2x_drop), i2x_extra=tuple(i2x_extra), radius=radius, seed=seed)


SPECS = (
    # E3SM's default: free surface, mushy thermodynamics and freezing, the rotated grid, column shortwave
    _spec("mushy_free", lsize=29, seed=1),
    # BL99 under a mushy tfrz_option (icepack_liquidus_temperature falls back to the linear liquidus),
    # an unrotated grid, topographic ponds, no column shortwave, icebergs
    _spec("bl99_mushy_tfrz", lsize=23, thermo="BL99", tfrz="mushy", rotate=False, ponds=True, column_sw=False,
          bergs=True, seed=2),
    # the slab ocean: no frazil mass (frazilMassAdjust left alone); linear_salt
    _spec("som_linear_salt", lsize=17, thermo="mushy", tfrz="linear_salt", surface="mixed_layer", seed=3),
    # constant and -1.8 freezing points; BL99's frazil enthalpy
    _spec("bl99_constant", lsize=13, halo=0, thermo="BL99", tfrz="constant", seed=4),
    _spec("zero_layer_minus1p8", lsize=11, thermo="zero layer", tfrz="minus1p8", ponds=True, seed=5),
    # the aerosols: bulk, then modal with the z-aerosols and their coupled dust iron, and the column BGC
    # (uncoupled: only the z-aerosol concentrations zeroed)
    _spec("aerosols_bulk", lsize=19, aerosols=True, seed=6),
    _spec("aerosols_modal_zaer", lsize=21, aerosols=True, modal=True, zaer=True, couple_bgc=True,
          i2x_extra=["Fioi_dust1"], seed=7),
    _spec("zaer_bulk_column_bgc", lsize=15, zaer=True, column_bgc=True, seed=8),
    # the waves two ways: the column waves and the floe sizes, then neither (Si_ifloe stays 0)
    _spec("waves_twoway", lsize=25, wav="twoway", nfreq=7, column_waves=True, n_frequencies=7, fsd=True,
          i2x_extra=["Si_ifloe"], seed=9),
    _spec("waves_twoway_off", lsize=9, wav="twoway", nfreq=3, i2x_extra=["Si_ifloe"], seed=10),
    _spec("waves_column_only", lsize=9, column_waves=True, n_frequencies=4, seed=11),
    # the stops
    _spec("abort_missing_x2i", lsize=5, aborts=True, x2i_drop=["So_dhdy"]),
    _spec("abort_missing_aerosol", lsize=5, aborts=True, x2i_drop=["Faxa_dstwet4"]),
    _spec("abort_missing_i2x", lsize=5, aborts=True, i2x_drop=["Si_u10withgusts"]),
    _spec("abort_missing_ifloe", lsize=5, aborts=True, wav="twoway", nfreq=2),
    _spec("abort_missing_wavespec", lsize=5, aborts=True, wav="twoway", nfreq=12, i2x_extra=["Si_ifloe"],
          x2i_drop=["Sw_wavespec11"]),
)


def _x2i_list(spec) -> list[str]:
    names = X2I_BASE[:9] + AEROSOL_X2I[:7] + X2I_BASE[9:] + AEROSOL_X2I[7:]
    if spec["wav"] == "twoway":
        names += ["Sw_Hs"] + [f"Sw_wavespec{k:02d}" for k in range(1, spec["nfreq"] + 1)]
    return [n for n in names if n not in spec["x2i_drop"]]


def _i2x_list(spec) -> list[str]:
    return [n for n in I2X_BASE if n not in spec["i2x_drop"]] + list(spec["i2x_extra"])


def _uniform(rng, low: float, high: float, n) -> np.ndarray:
    """``rng.random`` scaled with separate numpy operations (no contraction)."""
    return (rng.random(n) * (high - low)) + low


def _mesh(rng, n: int, radius: float):
    lat = _uniform(rng, -1.5, 1.5, n)
    lon = _uniform(rng, -3.1, 6.2, n)
    lat[0], lon[0] = 0.0, 0.0  # the rotated grid's pole: atan2(0, -0)
    if n > 2:
        lat[1], lon[1] = math.pi / 2, 1.0  # the geographic pole
        lat[2], lon[2] = -0.7, math.pi
    cl = np.cos(lat)
    x = (radius * cl) * np.cos(lon)
    y = (radius * cl) * np.sin(lon)
    z = radius * np.sin(lat)
    if n > 2:
        x[0], y[0], z[0] = radius, 0.0, 0.0
        x[1], y[1], z[1] = 0.0, 0.0, radius
    return lat, lon, x, y, z


def _build(spec) -> dict:
    rng = np.random.default_rng(1000 + spec["seed"])
    ns, nh = spec["lsize"], spec["lsize"] + spec["halo"]
    x2i_names, i2x_names = _x2i_list(spec), _i2x_list(spec)
    pools: dict[str, dict] = {p: {} for p in ("configs", "dimensions", "mesh", "ocean_coupling", "atmos_coupling",
                                              "wave_coupling", "aerosols", "biogeochemistry", "tracers_aggregate",
                                              "velocity_solver", "shortwave", "atmos_fluxes", "ocean_fluxes",
                                              "floe_size_distribution", "berg_fluxes")}
    c = pools["configs"]
    for name, value in (("config_use_aerosols", spec["aerosols"]), ("config_use_modal_aerosols", spec["modal"]),
                        ("config_use_zaerosols", spec["zaer"]),
                        ("config_use_column_biogeochemistry", spec["column_bgc"]),
                        ("config_couple_biogeochemistry_fields", spec["couple_bgc"]),
                        ("config_use_column_waves", spec["column_waves"]),
                        ("config_rotate_cartesian_grid", spec["rotate"]),
                        ("config_use_topo_meltponds", spec["ponds"]),
                        ("config_use_column_shortwave", spec["column_sw"]),
                        ("config_use_data_icebergs", spec["bergs"]),
                        ("config_use_floe_size_distribution", spec["fsd"])):
        c[name] = (1, bool(value))
    for name, value in (("config_column_physics_type", "icepack"), ("config_thermodynamics_type", spec["thermo"]),
                        ("config_ocean_surface_type", spec["surface"])):
        c[name] = (3, value)
    pools["dimensions"]["nFrequencies"] = (4, spec["n_frequencies"])
    lat, lon, x, y, z = _mesh(rng, nh, spec["radius"])
    m = pools["mesh"]
    m["nCellsSolve"] = (4, ns)
    m["sphere_radius"] = (2, spec["radius"])
    for name, a in (("latCell", lat), ("lonCell", lon), ("xCell", x), ("yCell", y), ("zCell", z)):
        m[name] = (5, a)

    def noise(pool, names, low=-3.0, high=3.0):
        for name in names:
            pools[pool][name] = (5, _uniform(rng, low, high, nh))

    noise("ocean_coupling", OCEAN_COUPLING)
    noise("atmos_coupling", ATMOS_COUPLING)
    noise("tracers_aggregate", TRACERS_AGGREGATE, 0.0, 2.0)
    noise("velocity_solver", VELOCITY_SOLVER)
    noise("shortwave", SHORTWAVE, 0.0, 1.0)
    noise("atmos_fluxes", ATMOS_FLUXES, -200.0, 200.0)
    noise("ocean_fluxes", OCEAN_FLUXES, -50.0, 50.0)
    area = pools["tracers_aggregate"]["iceAreaCell"][1]
    for k, v in enumerate((0.0, -0.0, 1.0, 1.25, -1e-300, 3e-300, 0.5)):
        if k < ns:
            area[k] = v
    pools["tracers_aggregate"]["surfaceTemperatureCell"][1][:] = _uniform(rng, -40.0, 0.5, nh)
    if spec["column_waves"]:
        pools["wave_coupling"]["significantWaveHeight"] = (5, _uniform(rng, 0, 9, nh))
        pools["wave_coupling"]["waveSpectra"] = (6, _uniform(rng, 0, 2, (nh, max(spec["n_frequencies"], 1))))
    if spec["aerosols"]:
        pools["aerosols"]["atmosAerosolFlux"] = (6, _uniform(rng, -1, 1, (nh, 3)))
    if spec["zaer"] or spec["column_bgc"]:
        for name, k in ZAEROSOL:
            if spec["zaer"] or name == "oceanZAerosolConc":
                pools["biogeochemistry"][name] = (6, _uniform(rng, -1, 1, (nh, k)))
    if spec["zaer"] and spec["couple_bgc"]:
        pools["biogeochemistry"]["oceanDustIronFlux"] = (5, _uniform(rng, -1, 1, nh))
    if spec["fsd"]:
        pools["floe_size_distribution"]["floeSizeDiameter"] = (5, _uniform(rng, 0, 500, nh))
    if spec["bergs"]:
        noise("berg_fluxes", ("bergFreshwaterFlux", "bergLatentHeatFlux"))

    # x2i: noise, then the frazil and freezing branches
    x2i = _uniform(rng, -2.0, 2.0, (ns, len(x2i_names)))

    def col(name):
        return x2i_names.index(name) if name in x2i_names else None

    if col("So_s") is not None and col("Fioo_q") is not None:
        sss = _uniform(rng, 0.0, 40.0, ns)
        q = _uniform(rng, -30.0, 30.0, ns)
        for k, (s, f) in enumerate(((6.0, 10.0), (5.9, 10.0), (6.1, 10.0), (0.0, 10.0), (100.0, 10.0),
                                    (34.0, 0.0), (34.0, -0.0), (34.0, -10.0), (96.0, 1.0))):
            if k < ns:
                sss[k], q[k] = s, f
        x2i[:, col("So_s")], x2i[:, col("Fioo_q")] = sss, q
        x2i[:, col("So_t")] = _uniform(rng, 270.0, 300.0, ns)
    i2x = _uniform(rng, -9.0, 9.0, (ns, len(i2x_names)))
    return dict(name=spec["name"], lsize=ns, halo=spec["halo"], aborts=spec["aborts"], spec=spec,
                lists={"x2i": x2i_names, "i2x": i2x_names}, x2i=x2i, i2x=i2x, pools=pools,
                wav=spec["wav"], nfreq=spec["nfreq"], ktherm=KTHERM[spec["thermo"]], tfrz=spec["tfrz"])


def cases() -> list[dict]:
    return [_build(s) for s in SPECS]


_POOL_ORDER = ("configs", "dimensions", "mesh", "ocean_coupling", "atmos_coupling", "wave_coupling", "aerosols",
               "biogeochemistry", "tracers_aggregate", "velocity_solver", "shortwave", "atmos_fluxes",
               "ocean_fluxes", "floe_size_distribution", "berg_fluxes")


def _entries(case):
    for pool in _POOL_ORDER[2:]:
        yield pool, "", 7, None  # every subpool exists, as in E3SM
    for pool in _POOL_ORDER:
        for name, (kind, value) in case["pools"][pool].items():
            yield pool, name, kind, value


def input_bytes(case: dict) -> bytes:
    """run_seaice_coupling.F90's stream."""
    parts = [struct.pack("<2i", case["nfreq"], case["ktherm"]), case["wav"].ljust(8).encode(),
             case["tfrz"].ljust(16).encode()]
    for key in ("x2i", "i2x"):
        rlist = ":".join(case["lists"][key]).encode()
        parts.append(struct.pack("<i", len(rlist)) + rlist)
    parts.append(struct.pack("<3i", case["lsize"], len(case["lists"]["x2i"]), len(case["lists"]["i2x"])))
    parts.append(np.ascontiguousarray(case["x2i"], dtype="<f8").tobytes())
    parts.append(np.ascontiguousarray(case["i2x"], dtype="<f8").tobytes())
    entries = list(_entries(case))
    parts.append(struct.pack("<i", len(entries)))
    for pool, name, kind, value in entries:
        parts.append(pool.ljust(32).encode() + struct.pack("<i", kind) + name.ljust(64).encode())
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
    return b"".join(parts)


def input_digest(case: dict) -> str:
    return hashlib.sha256(input_bytes(case)).hexdigest()


def array_entries(case: dict) -> list[tuple[str, str, int, object]]:
    """The arrays the oracle writes back, in its order."""
    return [(p, n, k, v) for p, n, k, v in _entries(case) if k in (5, 6)]


def read_outputs(raw: bytes, case: dict) -> dict[str, np.ndarray]:
    """``i2x``, ``<pool>/<name>`` for every array and ``trig``, from the oracle's output."""
    out, at = {}, 0
    ns, ni = case["lsize"], len(case["lists"]["i2x"])
    out["i2x"] = np.frombuffer(raw, dtype="<f8", count=ns * ni, offset=at).reshape(ns, ni).copy()
    at += 8 * ns * ni
    for pool, name, _, value in array_entries(case):
        out[f"{pool}/{name}"] = np.frombuffer(raw, dtype="<f8", count=value.size, offset=at).reshape(value.shape).copy()
        at += 8 * value.size
    out["trig"] = np.frombuffer(raw, dtype="<f8", count=8 * ns, offset=at).reshape(ns, 8).copy()
    at += 64 * ns
    assert at == len(raw), (case["name"], at, len(raw))
    return out


def load_fixture() -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    import json
    data = np.load(FIXTURE_DIR / "oracle-r8.npz")
    outputs: dict[str, dict[str, np.ndarray]] = {}
    for key in data.files:
        case, _, rest = key.partition("/")
        outputs.setdefault(case, {})[rest] = data[key]
    aborts = json.loads((FIXTURE_DIR / "aborts.json").read_text(encoding="utf-8"))
    return outputs, aborts


# --- the port ------------------------------------------------------------------------------------------------


def indices_for(case: dict):
    from woof.hex.coupled import ice_coupling_e3sm as icc
    return icc.indices(case["lists"]["x2i"], case["lists"]["i2x"], wav_ice_coup=case["wav"],
                       wav_nfreq=case["nfreq"])


def options_for(case: dict):
    from woof.hex.coupled import ice_coupling_e3sm as icc
    s = case["spec"]
    return icc.Options(
        config_thermodynamics_type=s["thermo"], config_sea_freezing_temperature_type=s["tfrz"],
        config_ocean_surface_type=s["surface"], config_rotate_cartesian_grid=s["rotate"],
        config_use_topo_meltponds=s["ponds"], config_use_column_shortwave=s["column_sw"],
        config_use_data_icebergs=s["bergs"], config_use_aerosols=s["aerosols"],
        config_use_modal_aerosols=s["modal"], config_use_zaerosols=s["zaer"],
        config_use_column_biogeochemistry=s["column_bgc"],
        config_couple_biogeochemistry_fields=s["couple_bgc"], config_use_column_waves=s["column_waves"],
        config_use_floe_size_distribution=s["fsd"])


def run_gpu(case: dict, trig=None) -> dict[str, np.ndarray]:
    """The port: the import then the export, outputs as the oracle writes them.
    ``trig`` is the rotation's words (the oracle's recorded ones by default)."""
    import cupy as cp
    from woof.hex.coupled import ice_coupling_e3sm as icc
    idx, opts = indices_for(case), options_for(case)
    if trig is None:
        trig = load_fixture()[0][case["name"]]["trig"]
    dev = {}
    for pool, name, kind, value in _entries(case):
        if kind in (5, 6):
            dev[pool, name] = cp.asarray(np.ascontiguousarray(value, dtype=np.float64))
    pools = {}
    for (pool, name), a in dev.items():
        pools.setdefault(pool, {})[name] = a
    ns = case["lsize"]
    x2i = cp.asarray(case["x2i"].astype(np.float64))
    i2x = cp.asarray(case["i2x"].astype(np.float64))
    n_freq = case["pools"]["dimensions"]["nFrequencies"][1]
    icc.import_x2i(idx, opts, x2i, n_cells_solve=ns, pools=pools, n_frequencies=n_freq)
    icc.export_i2x(idx, opts, i2x, n_cells_solve=ns, pools=pools,
                   trig=cp.asarray(np.ascontiguousarray(trig, dtype=np.float64)))
    out = {"i2x": cp.asnumpy(i2x)}
    for (pool, name), array in dev.items():
        out[f"{pool}/{name}"] = cp.asnumpy(array)
    return out


def measure() -> dict[str, dict[str, int]]:
    """case -> output (``i2x`` column or array) -> words that differ from the oracle."""
    outputs, _ = load_fixture()
    report = {}
    for case in cases():
        if case["aborts"]:
            continue
        got, want = run_gpu(case), outputs[case["name"]]
        fields = {}
        for k, name in enumerate(case["lists"]["i2x"]):
            fields[f"i2x/{name}"] = word_mismatches(got["i2x"][:, k], want["i2x"][:, k])
        for key in want:
            if key not in ("i2x", "trig"):
                fields[key] = word_mismatches(got[key], want[key])
        report[case["name"]] = fields
    return report

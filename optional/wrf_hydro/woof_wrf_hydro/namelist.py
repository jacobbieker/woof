# WOOF-OPTIN: wrf-hydro
"""``hydro.namelist`` mapped to the WRF-Hydro opt-in, refusing what is not ported.

Reads the ``&HYDRO_nlist`` group WRF-Hydro reads (read_rt_nlst,
hydro/OrchestratorLayer/config.F90 in WRF 4.7.1's hydro/, :621-638) and,
when lake_option = 3, the ``&reservoir_nlist`` group (:640-644, read at
:711-714), and maps them to :class:`HydroConfig`, the switches and constants
the ported pieces take.  An option whose value selects something not ported
is refused by name with the reason; a key WRF-Hydro does not read is refused
as unknown.  File names and output switches are recorded for the caller and
not acted on here.  A ``&NUDGING_nlist`` group (streamflow nudging) is
refused: nudging is not ported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Every key of read_rt_nlst's HYDRO_nlist (config.F90 :621-638), lower case.
HYDRO_NLIST_KEYS = frozenset(k.lower() for k in """
NSOIL ZSOIL8 RESTART_FILE SPLIT_OUTPUT_COUNT IGRID geo_static_flnm land_spatial_meta_flnm
out_dt rst_dt DEEPGWSPIN SOLVEG_INITSWC RT_OPTION CHANRTSWCRT channel_option SUBRTSWCRT
OVRTSWCRT AGGFACTRT dtrt_ter dtrt_ch dxrt GwSpinCycles GwPreCycles GwSpinUp GwPreDiag
GwPreDiagInterval gwIhShift GWBASESWCRT gwChanCondSw gwChanCondConstIn gwChanCondConstOut
route_topo_f route_chan_f route_link_f compound_channel channel_loss_option lake_option
route_lake_f route_direction_f route_order_f gwbasmskfil geo_finegrid_flnm gwstrmfil
GW_RESTART RSTRT_SWC TERADJ_SOLAR sys_cpl order_to_write rst_typ rst_bi_in rst_bi_out
gwsoilcpl CHRTOUT_DOMAIN CHANOBS_DOMAIN CHRTOUT_GRID LSMOUT_DOMAIN RTOUT_DOMAIN output_gw
outlake frxst_pts_out udmap_file UDMP_OPT GWBUCKPARM_file bucket_loss io_config_outputs
io_form_outputs hydrotbl_f t0OutputFlag output_channelBucket_influx imperv_adj""".split())

#: reservoir_nlist (config.F90 :640-644), read only when lake_option = 3.
RESERVOIR_NLIST_KEYS = frozenset(k.lower() for k in """
reservoir_persistence_usgs reservoir_persistence_usace reservoir_parameter_file
reservoir_usgs_timeslice_path reservoir_usace_timeslice_path
reservoir_observation_lookback_hours reservoir_observation_update_time_interval_seconds
reservoir_rfc_forecasts reservoir_rfc_forecasts_time_series_path
reservoir_rfc_forecasts_lookback_hours reservoir_type_specified""".split())

#: read_rt_nlst's defaults (config.F90 :659-682) for the keys mapped below.
DEFAULTS = {"udmp_opt": 0, "compound_channel": False, "bucket_loss": 0, "lake_option": -99,
            "reservoir_persistence_usgs": False, "reservoir_persistence_usace": False,
            "reservoir_rfc_forecasts": False, "reservoir_type_specified": False,
            "imperv_adj": 0, "teradj_solar": 0, "channel_loss_option": 0}

#: Keys that name files or set output and restart I/O: recorded, not acted on.
IO_KEYS = frozenset(k for k in HYDRO_NLIST_KEYS if k.endswith(("_f", "_file", "_flnm", "fil",
                                                                  "_path")) or k in {
    "restart_file", "split_output_count", "igrid", "out_dt", "rst_dt", "order_to_write",
    "rst_typ", "rst_bi_in", "rst_bi_out", "chrtout_domain", "chanobs_domain", "chrtout_grid",
    "lsmout_domain", "rtout_domain", "output_gw", "outlake", "frxst_pts_out",
    "io_config_outputs", "io_form_outputs", "t0outputflag", "output_channelbucket_influx",
    "gw_restart", "rstrt_swc",
    # channel_loss_option only selects NWM output variables (module_NWM_io.F90)
    "channel_loss_option"})


class UnportedOption(NotImplementedError):
    pass


@dataclass(frozen=True)
class HydroConfig:
    nsoil: int
    zsoil: tuple[float, ...]
    dxrt: float
    aggfactrt: int
    dtrt_ter: float
    dtrt_ch: float
    subsurface: bool          # SUBRTSWCRT = 1 -> SubsurfaceRouting
    overland: bool            # OVRTSWCRT = 1 -> OverlandRouting
    rt_option: int
    channel: bool             # CHANRTSWCRT = 1 -> ReachRouting
    channel_option: int
    gwbaseswcrt: int          # 0 none, 1/2 -> GroundwaterBucket
    compound_channel: bool
    lake_option: int = 0      # 0 no lakes, 1 level pool, 2 pass-through (LevelPool lake_opt)
    gwsoilcpl: int = 0        # GWBASESWCRT = 3: gwSoilFlux each call when > 0 (Gw2d)
    gw_pre_cycles: int = 0    # GWBASESWCRT = 3: gw2d_ini's gwstep pre-cycles
    gw_chan_cond_sw: int = 0  # GWBASESWCRT = 3, channel_option 3: groundwater-channel exchange
    gw_chan_cond_const_in: float = 0.0
    gw_chan_cond_const_out: float = 0.0
    io: dict = field(default_factory=dict)


def _value(text: str):
    t = text.strip().rstrip(",").strip()
    low = t.lower()
    if low in (".true.", "t", ".t."):
        return True
    if low in (".false.", "f", ".f."):
        return False
    if t[:1] in "'\"":
        return t[1:-1]
    try:
        return int(t)
    except ValueError:
        return float(t.replace("d", "e").replace("D", "e"))


def parse(text: str) -> dict[str, dict]:
    """A Fortran namelist file -> {group: {key: value}}; ``ZSOIL8(2) = x``
    lands in a list.  Enough of the namelist grammar for hydro.namelist."""
    groups: dict[str, dict] = {}
    current = None
    for raw in text.splitlines():
        line = re.sub(r"!.*$", "", raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            current = line[1:].strip().lower()
            groups[current] = {}
            continue
        if line == "/":
            current = None
            continue
        if current is None:
            continue
        for part in re.findall(r"([A-Za-z_][\w]*(?:\(\s*\d+\s*\))?)\s*=\s*('[^']*'|\"[^\"]*\"|[^,]+)", line):
            key, val = part
            m = re.match(r"([A-Za-z_]\w*)\(\s*(\d+)\s*\)", key)
            if m:
                name, idx = m.group(1).lower(), int(m.group(2))
                lst = groups[current].setdefault(name, [])
                while len(lst) < idx:
                    lst.append(None)
                lst[idx - 1] = _value(val)
            else:
                groups[current][key.lower()] = _value(val)
    return groups


def map_hydro_nlist(groups: dict[str, dict]) -> HydroConfig:
    """Map a parsed hydro.namelist; raise UnportedOption naming the first
    option that selects something not ported."""
    if "nudging_nlist" in groups:
        raise UnportedOption("&NUDGING_nlist: streamflow nudging is not ported")
    if "hydro_nlist" not in groups:
        raise ValueError("hydro.namelist has no &HYDRO_nlist group")
    nl = {**DEFAULTS, **groups["hydro_nlist"]}
    unknown = sorted(set(groups["hydro_nlist"]) - HYDRO_NLIST_KEYS)
    if unknown:
        raise ValueError(f"keys WRF-Hydro's HYDRO_nlist does not read: {unknown}")
    lake_option = int(nl["lake_option"])
    if lake_option == 3:                              # :711-714, then :762-769
        reservoir = groups.get("reservoir_nlist")
        if reservoir is None:
            raise ValueError("lake_option = 3 reads a &reservoir_nlist group; there is none")
        unknown = sorted(set(reservoir) - RESERVOIR_NLIST_KEYS)
        if unknown:
            raise ValueError(f"keys WRF-Hydro's reservoir_nlist does not read: {unknown}")
        nl.update(reservoir)
    elif lake_option not in (-99, 0, 1, 2):
        raise ValueError(f"lake_option = {lake_option}: must be 0, 1, 2 or 3 (config.F90 :478)")

    def refuse(key, value, why):
        raise UnportedOption(f"{key} = {value!r}: {why}")

    if nl.get("sys_cpl", 1) != 1:
        refuse("sys_cpl", nl["sys_cpl"], "only sys_cpl = 1 (coupled to WRF) is ported")
    if nl.get("udmp_opt") == 1:
        refuse("UDMP_OPT", 1, "the NWM user-defined mapping (UDMP) is not ported")
    for key in ("reservoir_persistence_usgs", "reservoir_persistence_usace",
                "reservoir_rfc_forecasts", "reservoir_type_specified"):
        if nl.get(key):
            refuse(key, nl[key], "the persistence and RFC-forecast reservoir types are not "
                                 "ported (only the level pool and the pass-through)")
    if lake_option == 3:
        lake_option = 1                               # :768 "set to 1 either way"
    if lake_option == -99:                            # :771-778
        lake_option = 1 if nl.get("route_lake_f", "") != "" else 0
    if nl.get("bucket_loss") == 1:
        refuse("bucket_loss", 1, "bucket loss (UDMP only) is not ported")
    if nl.get("teradj_solar", 0) != 0:
        refuse("TERADJ_SOLAR", nl["teradj_solar"], "terrain-adjusted solar radiation is not ported")
    rt_option = int(nl.get("rt_option", 1))
    if rt_option != 1:
        refuse("rt_option", rt_option, "only rt_option = 1 (steepest descent) is ported")
    channel = int(nl.get("chanrtswcrt", 0)) == 1
    channel_option = int(nl.get("channel_option", 3))
    if channel and channel_option not in (1, 2, 3):
        raise ValueError(f"channel_option = {channel_option} is not a WRF-Hydro option")
    gw = int(nl.get("gwbaseswcrt", 0))
    if gw == 4:
        refuse("GWBASESWCRT", 4, "the area-normalised UDMP bucket is not ported")
    if gw not in (0, 1, 2, 3):
        raise ValueError(f"GWBASESWCRT = {gw} is not a WRF-Hydro option")
    nsoil = int(nl["nsoil"])
    zsoil = tuple(float(z) for z in nl["zsoil8"])
    if len(zsoil) != nsoil or any(z is None for z in zsoil):
        raise ValueError(f"ZSOIL8 needs {nsoil} values, got {zsoil}")
    return HydroConfig(
        nsoil=nsoil, zsoil=zsoil, dxrt=float(nl["dxrt"]), aggfactrt=int(nl["aggfactrt"]),
        dtrt_ter=float(nl["dtrt_ter"]), dtrt_ch=float(nl["dtrt_ch"]),
        subsurface=int(nl.get("subrtswcrt", 0)) == 1, overland=int(nl.get("ovrtswcrt", 0)) == 1,
        rt_option=rt_option, channel=channel, channel_option=channel_option, gwbaseswcrt=gw,
        compound_channel=bool(nl.get("compound_channel")), lake_option=lake_option,
        gwsoilcpl=int(nl.get("gwsoilcpl", 0)), gw_pre_cycles=int(nl.get("gwprecycles", 0)),
        gw_chan_cond_sw=int(nl.get("gwchancondsw", 0)),
        gw_chan_cond_const_in=float(nl.get("gwchancondconstin", 0.0)),
        gw_chan_cond_const_out=float(nl.get("gwchancondconstout", 0.0)),
        io={k: v for k, v in groups["hydro_nlist"].items() if k in IO_KEYS
            and not (lake_option == 0 and k in ("route_lake_f", "outlake"))})


def read(path) -> HydroConfig:
    from pathlib import Path

    return map_hydro_nlist(parse(Path(path).read_text(encoding="utf-8", errors="replace")))

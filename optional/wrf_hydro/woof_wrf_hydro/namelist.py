# WOOF-OPTIN: wrf-hydro
"""``hydro.namelist`` mapped to the WRF-Hydro opt-in, refusing what is not ported.

Reads the ``&HYDRO_nlist`` group WRF-Hydro reads (hydro/Data_Rec/module_namelist.F90
in WRF 4.6.1's hydro/, :119-139) and maps it to :class:`HydroConfig`, the
switches and constants the ported pieces take.  An option whose value selects
something not ported is refused by name with the reason; a key WRF-Hydro does
not read is refused as unknown.  File names and output switches are recorded
for the caller and not acted on here.  A ``&NUDGING_nlist`` group (streamflow
nudging) is refused: nudging is not ported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Every key of module_namelist.F90's HYDRO_nlist (:119-139), lower case.
HYDRO_NLIST_KEYS = frozenset(k.lower() for k in """
NSOIL ZSOIL8 RESTART_FILE SPLIT_OUTPUT_COUNT IGRID geo_static_flnm land_spatial_meta_flnm
out_dt rst_dt DEEPGWSPIN SOLVEG_INITSWC RT_OPTION CHANRTSWCRT channel_option SUBRTSWCRT
OVRTSWCRT AGGFACTRT dtrt_ter dtrt_ch dxrt GwSpinCycles GwPreCycles GwSpinUp GwPreDiag
GwPreDiagInterval gwIhShift GWBASESWCRT gwChanCondSw gwChanCondConstIn gwChanCondConstOut
route_topo_f route_chan_f route_link_f compound_channel route_lake_f
reservoir_persistence_usgs reservoir_persistence_usace reservoir_parameter_file
reservoir_usgs_timeslice_path reservoir_usace_timeslice_path
reservoir_observation_lookback_hours reservoir_observation_update_time_interval_seconds
reservoir_rfc_forecasts reservoir_rfc_forecasts_time_series_path
reservoir_rfc_forecasts_lookback_hours reservoir_type_specified route_direction_f
route_order_f gwbasmskfil geo_finegrid_flnm gwstrmfil GW_RESTART RSTRT_SWC TERADJ_SOLAR
sys_cpl order_to_write rst_typ rst_bi_in rst_bi_out gwsoilcpl CHRTOUT_DOMAIN CHANOBS_DOMAIN
CHRTOUT_GRID LSMOUT_DOMAIN RTOUT_DOMAIN output_gw outlake frxst_pts_out udmap_file UDMP_OPT
GWBUCKPARM_file bucket_loss io_config_outputs io_form_outputs hydrotbl_f t0OutputFlag
output_channelBucket_influx imperv_adj""".split())

#: module_namelist.F90's defaults (:154-174) for the keys mapped below.
DEFAULTS = {"udmp_opt": 0, "compound_channel": False, "bucket_loss": 0,
            "reservoir_persistence_usgs": False, "reservoir_persistence_usace": False,
            "reservoir_rfc_forecasts": False, "reservoir_type_specified": False,
            "imperv_adj": 0, "teradj_solar": 0}

#: Keys that name files or set output and restart I/O: recorded, not acted on.
IO_KEYS = frozenset(k for k in HYDRO_NLIST_KEYS if k.endswith(("_f", "_file", "_flnm", "fil",
                                                                  "_path")) or k in {
    "restart_file", "split_output_count", "igrid", "out_dt", "rst_dt", "order_to_write",
    "rst_typ", "rst_bi_in", "rst_bi_out", "chrtout_domain", "chanobs_domain", "chrtout_grid",
    "lsmout_domain", "rtout_domain", "output_gw", "outlake", "frxst_pts_out",
    "io_config_outputs", "io_form_outputs", "t0outputflag", "output_channelbucket_influx",
    "gw_restart", "rstrt_swc", "reservoir_observation_lookback_hours",
    "reservoir_observation_update_time_interval_seconds",
    "reservoir_rfc_forecasts_lookback_hours"})


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

    def refuse(key, value, why):
        raise UnportedOption(f"{key} = {value!r}: {why}")

    if nl.get("sys_cpl", 1) != 1:
        refuse("sys_cpl", nl["sys_cpl"], "only sys_cpl = 1 (coupled to WRF) is ported")
    if nl.get("udmp_opt") == 1:
        refuse("UDMP_OPT", 1, "the NWM user-defined mapping (UDMP) is not ported")
    for key in ("reservoir_persistence_usgs", "reservoir_persistence_usace",
                "reservoir_rfc_forecasts", "reservoir_type_specified"):
        if nl.get(key):
            refuse(key, nl[key], "reservoirs (level pool and its data assimilation) are not ported")
    if nl.get("bucket_loss") == 1:
        refuse("bucket_loss", 1, "bucket loss (UDMP only) is not ported")
    if nl.get("teradj_solar", 0) != 0:
        refuse("TERADJ_SOLAR", nl["teradj_solar"], "terrain-adjusted solar radiation is not ported")
    rt_option = int(nl.get("rt_option", 1))
    if rt_option != 1:
        refuse("rt_option", rt_option, "only rt_option = 1 (steepest descent) is ported")
    channel = int(nl.get("chanrtswcrt", 0)) == 1
    channel_option = int(nl.get("channel_option", 3))
    if channel and channel_option == 3:
        refuse("channel_option", 3, "the diffusive wave on nodes is not ported (1 and 2 are)")
    if channel and channel_option not in (1, 2):
        raise ValueError(f"channel_option = {channel_option} is not a WRF-Hydro option")
    gw = int(nl.get("gwbaseswcrt", 0))
    if gw == 3:
        refuse("GWBASESWCRT", 3, "the 2-D groundwater model (gw2d) is not ported")
    if gw == 4:
        refuse("GWBASESWCRT", 4, "the area-normalised UDMP bucket is not ported")
    if gw not in (0, 1, 2):
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
        compound_channel=bool(nl.get("compound_channel")),
        io={k: v for k, v in groups["hydro_nlist"].items() if k in IO_KEYS})


def read(path) -> HydroConfig:
    from pathlib import Path

    return map_hydro_nlist(parse(Path(path).read_text(encoding="utf-8", errors="replace")))

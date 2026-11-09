"""A synthetic WRF-Hydro domain for the coupled-forecast tests (no real data).

Writes, into a directory, the files a coupled run's hydro.namelist names:
GEO_STATIC_FLNM (dimensions and land-use indices only), Fulldom_hires.nc
(north-up, as WPS's GIS pre-processing writes it), hydro2dtbl.nc,
CHANPARM.TBL, LAKEPARM.nc, GWBASINS.nc and GWBUCKPARM.nc.  The terrain is a
valley sloping east with one channel along its floor that runs into a lake
and out of it to the east edge, so every routing piece has work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHANPARM = """Channel Parameters
StreamOrder
10,1,  'Bw     HLINK   ChSSlp   MannN'
1,     1.6,    0.02,    0.03,      0.09
2,     2.4,    0.02,    0.03,      0.07
3,     3.5,    0.02,    0.03,      0.06
4,     5.3,    0.03,    0.04,      0.05
5,     7.4,    0.03,    0.04,      0.04
6,     11.,    0.03,    0.04,      0.03
7,     14.,    0.03,    0.04,      0.03
8,     16.,    0.10,    0.04,      0.02
9,     26.,    0.30,    0.05,      0.02
10,    110.,    0.30,    0.10,      0.02
"""


def namelist(*, dxrt=250.0, agg=4, dtrt_ter=10.0, dtrt_ch=10.0, sub=1, ov=1, chan=1,
             rt_option=1, gw=1, lake_option=1, out_dt=1, channel_option=3, extra=""):
    return f"""&HYDRO_nlist
sys_cpl = 2
GEO_STATIC_FLNM = "geo_em.d01.nc"
GEO_FINEGRID_FLNM = "Fulldom_hires.nc"
HYDROTBL_F = "hydro2dtbl.nc"
RESTART_FILE = ''
IGRID = 1
rst_dt = 120
rst_typ = 0
rst_bi_in = 0
rst_bi_out = 0
RSTRT_SWC = 0
GW_RESTART = 0
TERADJ_SOLAR = 0
output_channelBucket_influx = 0
out_dt = {out_dt}
SPLIT_OUTPUT_COUNT = 1
order_to_write = 1
io_form_outputs = 0
io_config_outputs = 0
t0OutputFlag = 1
CHRTOUT_DOMAIN = 1
CHANOBS_DOMAIN = 0
CHRTOUT_GRID = 0
LSMOUT_DOMAIN = 0
RTOUT_DOMAIN = 1
output_gw = 0
outlake = 0
frxst_pts_out = 1
NSOIL = 4
ZSOIL8(1) = -0.10
ZSOIL8(2) = -0.40
ZSOIL8(3) = -1.00
ZSOIL8(4) = -2.00
DXRT = {dxrt}
AGGFACTRT = {agg}
DTRT_CH = {dtrt_ch}
DTRT_TER = {dtrt_ter}
SUBRTSWCRT = {sub}
OVRTSWCRT = {ov}
rt_option = {rt_option}
CHANRTSWCRT = {chan}
channel_option = {channel_option}
route_lake_f = "LAKEPARM.nc"
lake_option = {lake_option}
GWBASESWCRT = {gw}
GWBUCKPARM_file = "GWBUCKPARM.nc"
gwbasmskfil = "GWBASINS.nc"
UDMP_OPT = 0
{extra}/
"""


def write(directory, *, nx, ny, agg=4, water_columns=1, lake=True, **nl):
    """Write the domain for an ``nx`` x ``ny`` mass grid whose last
    ``water_columns`` columns are water (vegtyp 17, soil 14)."""
    import netCDF4

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ixrt, jxrt = nx * agg, ny * agg
    with netCDF4.Dataset(d / "geo_em.d01.nc", "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("west_east", nx)
        ds.createDimension("south_north", ny)
        ds.createDimension("land_cat", 21)
        ds.createDimension("soil_cat", 16)
        ds.ISWATER, ds.ISLAKE, ds.ISURBAN, ds.ISOILWATER = 17, 21, 13, 14
        water = np.zeros((ny, nx), bool)
        water[:, nx - water_columns:] = True
        dims = ("Time", "south_north", "west_east")
        for name, value in (("LU_INDEX", np.where(water, 17.0, 10.0)),
                            ("SCT_DOM", np.where(water, 14.0, 6.0)),
                            ("HGT_M", np.full((ny, nx), 300.0)),
                            ("XLAT_M", 40.0 + 0.008 * np.arange(ny)[:, None] + 0 * water),
                            ("XLONG_M", -100.0 + 0.012 * np.arange(nx)[None, :] + 0 * water)):
            ds.createVariable(name, "f4", dims)[0] = np.asarray(value, np.float32)
    # south-up working arrays; flipped north-up when written
    jj, ii = np.meshgrid(np.arange(jxrt), np.arange(ixrt), indexing="ij")
    valley = jxrt // 2
    topo = (300.0 - 1.5 * ii + 2.0 * np.abs(jj - valley)).astype(np.float32)
    chan = np.full((jxrt, ixrt), -9999, np.int32)
    lakegrid = np.full((jxrt, ixrt), -9999, np.int32)
    direction = np.where(jj > valley, 4, np.where(jj < valley, 64, 1)).astype(np.int32)
    chan[valley, 1:] = 0
    lake_cols = slice(ixrt // 2 - 1, ixrt // 2 + 2) if lake else slice(0, 0)
    if lake:
        lakegrid[valley - 1:valley + 2, lake_cols] = 1
        chan[valley, lake_cols] = -9999
        chan[valley, lake_cols.stop] = 1              # the lake's outlet node
        topo[valley - 1:valley + 2, lake_cols] = topo[valley, lake_cols.start] - 1.0
    order = np.where(chan >= 0, np.where(ii < ixrt // 2, 1, 2), -9999).astype(np.int32)
    frxst = np.full((jxrt, ixrt), -9999, np.int32)
    frxst[valley, ixrt // 4] = 1
    frxst[valley, ixrt - agg * water_columns - 2] = 2
    lat = (40.0 + 0.002 * jj).astype(np.float32)
    lon = (-100.0 + 0.003 * ii).astype(np.float32)
    with netCDF4.Dataset(d / "Fulldom_hires.nc", "w") as ds:
        ds.createDimension("y", jxrt)
        ds.createDimension("x", ixrt)
        for name, arr, t in (("TOPOGRAPHY", topo, "f4"), ("CHANNELGRID", chan, "i4"),
                             ("FLOWDIRECTION", direction, "i2"), ("LAKEGRID", lakegrid, "i4"),
                             ("STREAMORDER", order, "i1"), ("frxst_pts", frxst, "i4"),
                             ("LATITUDE", lat, "f4"), ("LONGITUDE", lon, "f4"),
                             ("LKSATFAC", np.full_like(topo, 1000.0), "f4"),
                             ("RETDEPRTFAC", np.ones_like(topo), "f4"),
                             ("OVROUGHRTFAC", np.ones_like(topo), "f4")):
            ds.createVariable(name, t, ("y", "x"))[:] = arr[::-1]
    with netCDF4.Dataset(d / "hydro2dtbl.nc", "w") as ds:
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        land = np.ones((ny, nx), bool)
        land[:, nx - water_columns:] = False
        for name, value in (("SMCMAX1", 0.439), ("SMCREF1", 0.33), ("SMCWLT1", 0.066),
                            ("OV_ROUGH2D", 0.1), ("LKSAT", 5.0e-6), ("NEXP", 1.0)):
            v = ds.createVariable(name, "f4", ("south_north", "west_east"))
            v[:] = np.where(land, value, {"SMCMAX1": 1.0, "SMCREF1": 1.0}.get(name, value))
    (d / "CHANPARM.TBL").write_text(CHANPARM)
    with netCDF4.Dataset(d / "LAKEPARM.nc", "w") as ds:
        ds.createDimension("feature_id", 1)
        for name, value, t in (("lake_id", 1, "i8"), ("LkArea", 0.56, "f4"),
                               ("LkMxE", topo[valley, lake_cols.start] + 3.0 if lake else 0.0,
                                "f4"),
                               ("WeirE", topo[valley, lake_cols.start] + 1.0 if lake else 0.0,
                                "f4"),
                               ("WeirC", 0.4, "f4"), ("WeirL", 10.0, "f4"),
                               ("Dam_Length", 10.0, "f4"), ("OrificeC", 0.1, "f4"),
                               ("OrificeA", 1.0, "f4"),
                               ("OrificeE", topo[valley, lake_cols.start] - 0.5 if lake else 0.0,
                                "f4"),
                               ("lat", 40.0, "f4"), ("lon", -100.0, "f4"), ("ifd", 0.6, "f4")):
            ds.createVariable(name, t, ("feature_id",))[:] = value
    basins = np.ones((ny, nx), np.int32)
    basins[:, nx // 2:] = 2
    with netCDF4.Dataset(d / "GWBASINS.nc", "w") as ds:
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        ds.createVariable("BASIN", "i4", ("y", "x"))[:] = basins[::-1]
    with netCDF4.Dataset(d / "GWBUCKPARM.nc", "w") as ds:
        ds.createDimension("BasinDim", 2)
        for name, value, t in (("Basin", [1, 2], "i4"), ("Coeff", [1.0, 1.0], "f4"),
                               ("Expon", [3.0, 3.0], "f4"), ("Zmax", [50.0, 50.0], "f4"),
                               ("Zinit", [10.0, 10.0], "f4"),
                               ("Area_sqkm", [1.0, 1.0], "f4")):
            ds.createVariable(name, t, ("BasinDim",))[:] = value
    (d / "hydro.namelist").write_text(namelist(agg=agg, **nl))
    return d / "hydro.namelist"


def write_reservoir_da(directory, *, start, update_seconds=120, lookback_hours=2):
    """Turn a written domain's lake into lake_option 3's USGS persistence /
    level-pool hybrid (type 2): the reservoir parameter file, 15-minute USGS
    timeslices around ``start`` (one missing, one of low quality) and a
    &reservoir_nlist group."""
    from datetime import timedelta

    import netCDF4

    d = Path(directory)
    (d / "usgs").mkdir(exist_ok=True)

    def chars(values, width, pad):
        return np.array([[c.encode() for c in v.ljust(width, pad)] for v in values], "S1")

    with netCDF4.Dataset(d / "reservoir_parm.nc", "w") as ds:
        ds.createDimension("lake", 1)
        ds.createDimension("nchar15", 15)
        ds.createDimension("nweights", 2)
        ds.createVariable("lake_id", "i8", ("lake",))[:] = [1]
        ds.createVariable("reservoir_type", "i4", ("lake",))[:] = [2]
        ds.createVariable("usgs_lake_id", "i4", ("lake",))[:] = [1]
        ds.createVariable("usgs_gage_id", "S1", ("lake", "nchar15"))[:] = chars(["01234567"], 15,
                                                                                 "\0")
        ds.createVariable("usgs_persistence_coefficients", "f4",
                          ("lake", "nweights"))[:] = [[0.75, 0.5]]
    t = start - timedelta(hours=lookback_hours)
    k = 0
    while t <= start + timedelta(hours=1):
        k += 1
        if k != 4:                                         # one file missing
            name = f"{t.strftime('%Y-%m-%d_%H:%M:%S')}.15min.usgsTimeSlice.ncdf"
            with netCDF4.Dataset(d / "usgs" / name, "w") as ds:
                ds.createDimension("stationIdInd", 2)
                ds.createDimension("stationIdStrLen", 15)
                ds.createVariable("stationId", "S1", ("stationIdInd", "stationIdStrLen"))[:] = \
                    chars(["01234567", "07654321"], 15, " ")
                ds.createVariable("discharge", "f4", ("stationIdInd",))[:] = [0.5 + 0.01 * k, 9.0]
                ds.createVariable("discharge_quality", "f4", ("stationIdInd",))[:] = \
                    [50.0 if k % 3 == 0 else 100.0, 100.0]
        t += timedelta(minutes=15)
    nl = d / "hydro.namelist"
    text = nl.read_text().replace("lake_option = 1", "lake_option = 3")
    nl.write_text(text + f"""&reservoir_nlist
reservoir_persistence_usgs = .TRUE.
reservoir_persistence_usace = .FALSE.
reservoir_parameter_file = "reservoir_parm.nc"
reservoir_usgs_timeslice_path = "usgs"
reservoir_usace_timeslice_path = "usace"
reservoir_observation_lookback_hours = {lookback_hours}
reservoir_observation_update_time_interval_seconds = {update_seconds}
reservoir_rfc_forecasts = .FALSE.
reservoir_rfc_forecasts_time_series_path = "rfc"
reservoir_rfc_forecasts_lookback_hours = 28
/
""")


def write_gw2d(directory, *, ih_shift=0.25, pre_cycles=2, soil_cpl=1, chan_cond=(1.0e-4, 5.0e-5)):
    """Turn a written domain's groundwater into GWBASESWCRT = 3, the 2-D
    aquifer: ``gwhires.nc`` (HC, IHEAD, BOTELV, POR, LTYPE on the routing
    grid; readGW2d reads it without the north-up flip Fulldom gets, so it is
    written south-up), CHAN_DEPTH in Fulldom_hires.nc (READ_CHROUTING1 takes
    ZELEV from TOPOGRAPHY - CHAN_DEPTH) and the namelist's gw keys.  The
    head sits in the soil column in some cells and below it in others, above
    the ground around the lake (surface excess), one conductivity is the
    100 m/s sentinel and one cell is inactive (LTYPE 0)."""
    import netCDF4

    d = Path(directory)
    with netCDF4.Dataset(d / "Fulldom_hires.nc", "a") as ds:
        topo = np.asarray(ds.variables["TOPOGRAPHY"][:], np.float32)[::-1]
        chan = np.asarray(ds.variables["CHANNELGRID"][:])[::-1]
        depth = np.where(chan >= 0, 1.0, 0.0).astype(np.float32)
        ds.createVariable("CHAN_DEPTH", "f4", ("y", "x"))[:] = depth[::-1]
    jxrt, ixrt = topo.shape
    jj, ii = np.meshgrid(np.arange(jxrt), np.arange(ixrt), indexing="ij")
    f = np.float32
    hc = (f(2.0e-4) * (f(1.0) + (jj % 3).astype(f))).astype(f)
    hc[jxrt // 2 + 2, 2] = f(100.0)
    ihead = (topo - f(3.0) + f(0.5) * ((ii + 2 * jj) % 5).astype(f)).astype(f)
    ihead[jxrt - 2, 1] = topo[jxrt - 2, 1] + f(0.3)
    ring = (slice(jxrt // 2 - 2, jxrt // 2 + 3), slice(ixrt // 2 - 2, ixrt // 2 + 3))
    ihead[ring] = topo[ring] + f(0.5)                # seeps into the lake hollow
    bot = (topo - f(20.0)).astype(f)
    por = (f(0.25) + f(0.01) * (ii % 3).astype(f)).astype(f)
    ltype = np.ones((jxrt, ixrt), np.int32)
    ltype[1, 2] = 0
    with netCDF4.Dataset(d / "gwhires.nc", "w") as ds:
        ds.createDimension("y", jxrt)
        ds.createDimension("x", ixrt)
        for name, arr, t in (("HC", hc, "f4"), ("IHEAD", ihead, "f4"), ("BOTELV", bot, "f4"),
                             ("POR", por, "f4"), ("LTYPE", ltype, "i4")):
            ds.createVariable(name, t, ("y", "x"))[:] = arr
    nl = d / "hydro.namelist"
    text = nl.read_text()
    text = text.replace("GWBASESWCRT = 1\n", "GWBASESWCRT = 3\n").replace("GWBASESWCRT = 2\n",
                                                                       "GWBASESWCRT = 3\n")
    text = text.replace("UDMP_OPT = 0\n", "UDMP_OPT = 0\n" + f"""gwsoilcpl = {soil_cpl}
GwPreCycles = {pre_cycles}
GwSpinCycles = 0
GwPreDiag = .FALSE.
GwPreDiagInterval = 1
GwSpinUp = .FALSE.
gwIhShift = {ih_shift}
gwChanCondSw = 1
gwChanCondConstIn = {chan_cond[0]}
gwChanCondConstOut = {chan_cond[1]}
""")
    nl.write_text(text)


def write_route_link(directory, *, channel_option=2):
    """Turn a written domain's channel into Route_Link reaches
    (channel_option 1 or 2): Route_Link.nc and a LINKID grid in
    Fulldom_hires.nc.  The valley's channel is two reaches above the lake
    (the second flows into it), the lake holds an internal reach (fed by the
    last pixels above it) and its outlet reach (NHDWaterbodyComID = the
    lake's id), and below it one
    reach takes the outlet's water and two pixel-less tributaries' into the
    last reach, which leaves the domain.  Link ids stay at or below the node
    count (LandRT_ini indexes a node-sized buffer with them); some Qi are
    zero (readLinkSL's order-based start) and one So is below the floor.
    Every reach Muskingum-Cunge routes always has water: with none,
    SUBMUSKINGCUNGE computes QLOSS from WPk, D and C4 it never set (whatever
    the previous call left on the stack)."""
    import netCDF4

    d = Path(directory)
    with netCDF4.Dataset(d / "Fulldom_hires.nc", "a") as ds:
        chan = np.asarray(ds.variables["CHANNELGRID"][:])[::-1]
        jxrt, ixrt = chan.shape
        valley = jxrt // 2
        lake_start = ixrt // 2 - 1
        link = np.full((jxrt, ixrt), -9999, np.int32)
        cols = np.arange(ixrt)
        on = chan[valley] >= 0
        quarter = ixrt // 4
        row = np.where(cols < quarter, 1, np.where(cols < lake_start - 3, 2,
                       np.where(cols < lake_start, 3, np.where(cols < ixrt - quarter, 5, 6))))
        link[valley, on] = row[on]
        ds.createVariable("LINKID", "i4", ("y", "x"))[:] = link[::-1]
    f = np.float32
    rows = dict(
        link=[1, 2, 3, 4, 5, 6, 7, 8], to=[2, 3, 4, 5, 6, 0, 6, 6],
        NHDWaterbodyComID=[-9999, -9999, 1, 1, -9999, -9999, -9999, -9999],
        order=[1, 1, 2, 2, 2, 3, 1, 1],
        Qi=[0.0, 0.5, 0.1, 0.0, 0.2, 0.0, 0.3, 0.05],
        MusK=[900.0, 1200.0, 600.0, 600.0, 1500.0, 1800.0, 900.0, 900.0],
        MusX=[0.2, 0.25, 0.2, 0.2, 0.3, 0.2, 0.1, 0.2],
        Length=[1750.0, 1500.0, 500.0, 600.0, 1750.0, 1750.0, 2000.0, 1200.0],
        n=[0.06, 0.055, 0.05, 0.05, 0.05, 0.045, 0.07, 0.06],
        So=[0.006, 0.004, 0.001, 0.001, 0.0000005, 0.002, 0.01, 0.008],
        ChSlp=[0.8, 0.7, 0.5, 0.5, 0.6, 0.4, 0.9, 0.8],
        BtmWdth=[2.0, 3.0, 5.0, 5.0, 4.0, 6.0, 1.5, 1.0],
        Kchan=[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        alt=[300.0, 290.0, 280.0, 280.0, 270.0, 260.0, 295.0, 285.0],
        lat=[40.02 + 0.001 * k for k in range(8)], lon=[-99.95 + 0.01 * k for k in range(8)])
    with netCDF4.Dataset(d / "Route_Link.nc", "w") as ds:
        ds.createDimension("feature_id", 8)
        ds.createDimension("IDLength", 15)
        for name, values in rows.items():
            t = "i8" if name in ("link", "to", "NHDWaterbodyComID") else (
                "i4" if name == "order" else "f4")
            ds.createVariable(name, t, ("feature_id",))[:] = np.asarray(
                values, np.int64 if t != "f4" else f)
        gages = ["", "07654321", "", "", "", "01234567", "", ""]
        ds.createVariable("gages", "S1", ("feature_id", "IDLength"))[:] = np.array(
            [[c.encode() for c in g.rjust(15)] for g in gages], "S1")
    nl = d / "hydro.namelist"
    text = nl.read_text().replace("channel_option = 3\n",
                                  f"channel_option = {channel_option}\n")
    nl.write_text(text.replace('route_lake_f = "LAKEPARM.nc"\n',
                               'route_lake_f = "LAKEPARM.nc"\nroute_link_f = "Route_Link.nc"\n'))

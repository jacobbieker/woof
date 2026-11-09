# WOOF-OPTIN: wrf-hydro
"""WRF-Hydro's routing-domain initialisation on the host (HYDRO_ini's reads).

What ``HYDRO_ini`` (hydro/HYDRO_drv/module_HYDRO_drv.F90 :1349-1626 in WRF
4.7.1's hydro/) and the routines it calls build before the first coupling
call, for the configurations the coupled driver runs (MPP_LAND on one
process, UDMP_OPT = 0, gridded channels):

* ``get_file_globalatts`` (module_HYDRO_io.F90 :234-297): ISWATER, ISLAKE,
  ISURBAN and ISOILWATER from GEO_STATIC_FLNM's global attributes, with
  WRF-Hydro's USGS defaults when one is missing.
* ``lsm_input`` (module_HYDRO_drv.F90 :1628-1792) with WRF's IVGTYP/ISLTYP
  (sys_cpl = 2), soil layers from WRF's ZS, and ``hdtbl_in_nc``
  (module_HYDRO_io.F90 :11122-11137): SMCMAX1, SMCREF1, SMCWLT1, LKSAT,
  NEXP on the LSM grid and OV_ROUGH2D replicated onto the routing grid.
* ``get_dxdy_mp`` (module_HYDRO_utils.F90 :217-243) for both grids
  (``DXRT > 0`` is required by read_rt_nlst).
* ``READ_ROUTING_seq`` (module_HYDRO_io.F90 :5611-5701): TOPOGRAPHY,
  CHANNELGRID, LKSATFAC, RETDEPRTFAC, OVROUGHRTFAC from GEO_FINEGRID_FLNM,
  flipped north-up to WRF's south-up as nreadRT2d does.
* ``MPP_READ_ROUTEDIM`` / ``READ_ROUTEDIM`` (:5523-5604, :693-1007): every
  CHANNELGRID >= 0 pixel is a node; one process renumbers them in raster
  order (j outer, i inner), so CH_NETLNK is that rank.  A node whose
  FLOWDIRECTION is none of the eight codes stops WRF-Hydro ("READ_ROUTEDIM").
* ``READ_CHROUTING1`` through ``MPP_READ_CHROUTING_new`` (:7364-8276,
  :8639-8813), channel_option = 3: FROM_NODE/TO_NODE/CHANLEN along the flow
  direction, pour points TYPEL = 1 (or 2 with LAKENODE when they drain into
  a LAKEGRID lake), ZELEV, ORDER, frxst_pts, and LAKEPARM's lakes
  (``read_route_lake_netcdf``, :9028-9097).
* ``LandRT_ini`` (module_RT.F90 :642-1682): the eight terrain slopes and the
  steepest-descent index, sox/soy, the retention depth reduced by slope for
  the rt_option, RETDEPRTFAC/OVROUGHRTFAC, 5 mm on channel pixels;
  CHANPARM.TBL's widths, side slopes, Manning's n and initial heads by
  stream order; the initial channel volume and lake elevation;
  ``READ_SIMP_GW``, ``SIMP_GW_IND``, ``get_gw_strm_msk_lind`` and
  ``read_GWBUCKPARM`` (module_HYDRO_io.F90 :1344-1588) for the bucket;
  ``readGW2d`` (:1593-1737) for GWBASESWCRT = 3's aquifer, with READ_CHROUTING1
  taking ZELEV from TOPOGRAPHY - CHAN_DEPTH (:7510-7518).
* ``get_basn_area`` and ``get_node_area`` (module_HYDRO_utils.F90 :329-418).
* channel_option 1 and 2 (Route_Link reaches): the same nodes, Fulldom's
  LINKID (fatal-required, READ_ROUTEDIM :733-737), ``get_NLINKSL``,
  ``read_NSIMLAKES``, ``readLinkSL`` / ``read_route_link_netcdf``
  (:8372-8630, :8919-9014), ``nhdLakeMap_scan`` / ``nhdLakeMap_mpp``
  (:10405-10916: the lake outlet, inflow and internal reaches, the inflow
  reaches' TO_NODE moved to the outlet, LAKEIDX) and LandRT_ini's
  CH_LNKRT_SL (module_RT.F90 :990-1060), as the one-process MPP_LAND build
  computes them.

Every array is float32/int on the host, in WRF's (j, i) as numpy (row, col)
with row 0 the south edge.  Files are read with netCDF4; each one a
configuration needs and does not have is refused by name, as is the text
(non-NetCDF) form of a parameter file, which is not ported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

F = np.float32
#: FLOWDIRECTION code -> (di, dj, 1-based dist index), READ_CHROUTING1's order
DIRECTIONS = {64: (0, 1, 1), 128: (1, 1, 2), 1: (1, 0, 3), 2: (1, -1, 4),
              4: (0, -1, 5), 8: (-1, -1, 6), 16: (-1, 0, 7), 32: (-1, 1, 8)}


class MissingInput(FileNotFoundError):
    """A file or variable a WRF-Hydro configuration reads is absent."""


def _open(path, what):
    import netCDF4

    p = Path(path)
    if not p.is_file():
        raise MissingInput(f"{what}: {p} does not exist")
    try:
        return netCDF4.Dataset(str(p), "r")
    except OSError as exc:
        raise MissingInput(f"{what}: {p} is not a NetCDF file ({exc}); the text "
                           "form of this file is not ported") from exc


def _var(ds, name, shape=None, *, dtype=F, fill=None, flip=False, what=""):
    """get2d_real/get2d_int: the variable (or ``NAME_M``), else ``fill``."""
    for key in (name, name + "_M"):
        if key in ds.variables:
            v = ds.variables[key]
            v.set_auto_maskandscale(False)
            a = np.asarray(v[...]).squeeze()
            if shape is not None and a.shape != shape:
                raise ValueError(f"{what} {name}: shape {a.shape}, expected {shape} "
                                 "(north-up rows, columns)")
            a = a.astype(dtype, copy=False)
            return np.ascontiguousarray(a[::-1] if flip else a)
    if fill is None:
        raise MissingInput(f"{what} has no variable {name}")
    return np.full(shape, fill, dtype)


def dxdy(nx, ny, dx, dy):
    """get_dxdy_mp: (9, ny, nx), the eight neighbour distances and the area."""
    dx, dy = F(dx), F(dy)
    v1 = np.sqrt(F(dx * dx + dy * dy)).astype(F)
    d = np.full((9, ny, nx), F(-1), F)
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    lt_j, lt_i, gt_j, gt_i = j < ny - 1, i < nx - 1, j > 0, i > 0
    for k, (mask, val) in enumerate(((lt_j, dy), (lt_j & lt_i, v1), (lt_i, dx),
                                     (gt_j & lt_i, v1), (gt_j, dy), (gt_j & gt_i, v1),
                                     (gt_i, dx), (lt_j & gt_i, v1))):
        d[k][np.broadcast_to(mask, (ny, nx))] = val
    d[8] = F(dx * dy)
    return d


@dataclass
class RoutingDomain:
    """HYDRO_ini's static state, host side."""

    ix: int
    jx: int
    aggfactrt: int
    nsoil: int
    zsoil: np.ndarray
    sldpth: np.ndarray
    iswater: int
    islake: int
    isurban: int
    isoilwater: int
    vegtyp: np.ndarray
    soiltyp: np.ndarray
    dist_lsm: np.ndarray
    dist: np.ndarray
    smcmax1: np.ndarray
    smcref1: np.ndarray
    smcwlt1: np.ndarray
    lksat: np.ndarray
    nexp: np.ndarray
    elrt: np.ndarray
    ch_netrt: np.ndarray
    lksatfac: np.ndarray
    roughness: np.ndarray
    retention: np.ndarray
    slope: np.ndarray
    sdir: np.ndarray
    sox: np.ndarray
    soy: np.ndarray
    lake_mask: np.ndarray
    latval: np.ndarray
    lonval: np.ndarray
    network: dict | None = None
    reaches: dict | None = None
    lakes: dict | None = None
    buckets: dict | None = None
    gw2d: dict | None = None
    notes: list = field(default_factory=list)

    @property
    def ixrt(self):
        return self.ix * self.aggfactrt

    @property
    def jxrt(self):
        return self.jx * self.aggfactrt


def _globalatts(path):
    """get_file_globalatts with WRF-Hydro's defaults."""
    with _open(path, "GEO_STATIC_FLNM") as ds:
        def att(name, default):
            return int(ds.getncattr(name)) if name in ds.ncattrs() else default
        dims = ds.dimensions
        nx = len(dims["west_east"]) if "west_east" in dims else None
        ny = len(dims["south_north"]) if "south_north" in dims else None
        return (att("ISWATER", 16), att("ISLAKE", -1), att("ISURBAN", 1),
                att("ISOILWATER", 14), nx, ny)


def _chanparm(path):
    """CHAN_PARM_INIT: CHANPARM.TBL's StreamOrder rows (Bw, HLINK, ChSSlp, MannN);
    entries no row sets stay zero (WRF's are uninitialised; none is read)."""
    p = Path(path)
    if not p.is_file():
        raise MissingInput(f"CHANPARM.TBL: {p} does not exist (channel_option = 3 reads it)")
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    botwid, hlink, chan_ss, chmann = (np.zeros(50, F) for _ in range(4))
    if len(lines) < 3 or lines[1][:11].strip() != "StreamOrder":
        return botwid, hlink, chan_ss, chmann
    cats = int(lines[2].split(",")[0])
    for row in lines[3:3 + cats]:
        vals = [v for v in row.replace(",", " ").split() if v]
        order = int(vals[0])
        botwid[order - 1], hlink[order - 1], chan_ss[order - 1], chmann[order - 1] = (
            F(float(v)) for v in vals[1:5])
    return botwid, hlink, chan_ss, chmann


def build(cfg, groups, *, base: Path, ix, jx, zs, vegtyp, soltyp):
    """HYDRO_ini for the coupled driver.  ``cfg`` is the mapped
    :class:`~woof_wrf_hydro.namelist.HydroConfig`, ``groups`` the parsed
    namelist, ``base`` the directory its relative paths resolve against, and
    ``zs``/``vegtyp``/``soltyp`` WRF's ZS (layer centres, m) and IVGTYP/ISLTYP."""
    nl = groups["hydro_nlist"]

    def path(key, default=""):
        value = str(nl.get(key, default) or "")
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else (base / p)

    agg = int(cfg.aggfactrt)
    ixrt, jxrt = ix * agg, jx * agg
    geo_static = path("geo_static_flnm")
    if geo_static is None:
        raise MissingInput("GEO_STATIC_FLNM is empty: HYDRO_ini reads the LSM grid's size "
                           "and land-use indices from it")
    iswater, islake, isurban, isoilwater, gnx, gny = _globalatts(geo_static)
    if (gnx, gny) != (ix, jx):
        raise ValueError(f"GEO_STATIC_FLNM is {gnx} x {gny}; the forecast's mass grid is "
                         f"{ix} x {jx} (wrf_cpl_HYDRO couples them cell for cell)")
    # wrf_cpl_HYDRO :94-99 -- ZSOIL8 is WRF's ZS, not the namelist's
    # (read_rt_nlst's assignment is commented out, config.F90 :741-744).
    zs = np.asarray(zs, F)
    zsoil = (zs if zs[0] < 0 else -zs).astype(F)
    nsoil = len(zsoil)
    sldpth = np.empty(nsoil, F)
    sldpth[0] = abs(zsoil[0])
    for k in range(1, nsoil):
        sldpth[k] = F(zsoil[k - 1] - zsoil[k])
    # lsm_input :1676-1688 with sys_cpl = 2.  WRF passes IVGTYP/ISLTYP by
    # array section, so these edits reach WRF's own arrays; refused when they
    # would change anything rather than change the LSM's inputs here.
    veg = np.asarray(vegtyp, np.int32).copy()
    sol = np.asarray(soltyp, np.int32).copy()
    sol = np.where((veg == iswater) | (veg == islake), isoilwater, sol).astype(np.int32)
    veg = np.where(sol == isoilwater, iswater, veg).astype(np.int32)
    if (veg != np.asarray(vegtyp)).any() or (sol != np.asarray(soltyp)).any():
        raise NotImplementedError(
            "lsm_input rewrites IVGTYP/ISLTYP where they disagree on water (a lake land use, "
            "or water soil under land); WRF's copy-back into the LSM's own arrays is not "
            "ported, so the inputs must already agree")
    hydrotbl = path("hydrotbl_f")
    if hydrotbl is None:
        raise MissingInput("hydrotbl_f is empty (FATAL ERROR in lsm_input: hydrotbl_f is "
                           "empty)")
    with _open(hydrotbl, "hydrotbl_f (the hydro2dtbl.nc spatial parameters; the "
                         "HYDRO.TBL fallback is not ported)") as ds:
        tbl = {k: _var(ds, k, (jx, ix), what="hydrotbl_f")
               for k in ("SMCMAX1", "SMCREF1", "SMCWLT1", "OV_ROUGH2D", "LKSAT")}
        nexp = (_var(ds, "NEXP", (jx, ix), what="hydrotbl_f") if ("NEXP" in ds.variables
                or "NEXP_M" in ds.variables) else np.full((jx, ix), F(1.0), F))
    smcmax1 = np.where(tbl["SMCMAX1"] > F(1.0), F(1.0), tbl["SMCMAX1"]).astype(F)
    smcref1 = np.maximum(np.minimum(tbl["SMCREF1"], smcmax1 - F(0.01)), F(0.0)).astype(F)
    smcwlt1 = np.maximum(np.minimum(tbl["SMCWLT1"], smcref1 - F(0.01)), F(0.0)).astype(F)
    roughness = np.repeat(np.repeat(tbl["OV_ROUGH2D"], agg, axis=0), agg, axis=1)
    dxrt = F(cfg.dxrt)
    dist_lsm = dxdy(ix, jx, F(dxrt * F(agg)), F(dxrt * F(agg)))
    dist = dxdy(ixrt, jxrt, dxrt, dxrt)

    fine = path("geo_finegrid_flnm")
    if fine is None:
        raise MissingInput("GEO_FINEGRID_FLNM is empty: the routing grid comes from "
                           "Fulldom_hires.nc")
    shape = (jxrt, ixrt)
    with _open(fine, "GEO_FINEGRID_FLNM (Fulldom_hires.nc)") as ds:
        what = f"GEO_FINEGRID_FLNM {fine.name}"
        elrt = _var(ds, "TOPOGRAPHY", shape, flip=True, what=what)
        ch_netrt = _var(ds, "CHANNELGRID", shape, dtype=np.int32, flip=True, what=what)
        lksatfac = _var(ds, "LKSATFAC", shape, flip=True, fill=F(-9999.9), what=what)
        retdeprtfac = _var(ds, "RETDEPRTFAC", shape, flip=True, fill=F(-9999.9), what=what)
        ovroughrtfac = _var(ds, "OVROUGHRTFAC", shape, flip=True, fill=F(-9999.9), what=what)
        direction = _var(ds, "FLOWDIRECTION", shape, dtype=np.int32, flip=True,
                         fill=-9999, what=what)
        lakegrid = _var(ds, "LAKEGRID", shape, dtype=np.int32, flip=True, fill=-9999,
                        what=what)
        gorder = _var(ds, "STREAMORDER", shape, dtype=np.int32, flip=True, fill=-9999,
                      what=what)
        frxst = _var(ds, "frxst_pts", shape, dtype=np.int32, flip=True, fill=-9999, what=what)
        latval = _var(ds, "LATITUDE", shape, flip=True, fill=F(-9999.9), what=what)
        lonval = _var(ds, "LONGITUDE", shape, flip=True, fill=F(-9999.9), what=what)
        chan_depth = _var(ds, "CHAN_DEPTH", shape, flip=True, fill=F(-9999.9), what=what)
        ch_lnkrt = None
        if cfg.channel and cfg.channel_option != 3:
            # READ_ROUTING_seq's nreadRT2d_int8, fatal for channel_option 1/2
            ch_lnkrt = _var(ds, "LINKID", shape, dtype=np.int64, flip=True, what=what)
    # READ_ROUTING_seq :5662-5690
    lksatfac = np.where(lksatfac == F(-9999.9), F(1000.0), lksatfac).astype(F)
    retdeprtfac = np.where(retdeprtfac < F(0.0), F(1.0), retdeprtfac).astype(F)
    ovroughrtfac = np.where(ovroughrtfac <= F(0.0), F(1.0), ovroughrtfac).astype(F)

    rt_option = int(cfg.rt_option)
    slope, sdir, retention, sox, soy = _terrain(elrt, dist, rt_option)
    retention = (retention * retdeprtfac).astype(F)
    roughness = (roughness * ovroughrtfac).astype(F)
    retention[ch_netrt >= 0] = F(5.0)

    lake_mask = np.full(shape, -9999, np.int32)
    dom = RoutingDomain(
        ix=ix, jx=jx, aggfactrt=agg, nsoil=nsoil, zsoil=zsoil, sldpth=sldpth,
        iswater=iswater, islake=islake, isurban=isurban, isoilwater=isoilwater, vegtyp=veg,
        soiltyp=sol, dist_lsm=dist_lsm, dist=dist, smcmax1=smcmax1, smcref1=smcref1,
        smcwlt1=smcwlt1, lksat=tbl["LKSAT"], nexp=nexp, elrt=elrt, ch_netrt=ch_netrt,
        lksatfac=lksatfac, roughness=roughness, retention=retention, slope=slope, sdir=sdir,
        sox=sox, soy=soy, lake_mask=lake_mask, latval=latval, lonval=lonval)
    if cfg.channel and cfg.channel_option != 3:
        # the node network READ_ROUTEDIM and READ_CHROUTING1 still build, then
        # the reaches from Route_Link
        dom.network, dom.lake_mask = _network(ch_netrt, direction, lakegrid, gorder, frxst,
                                              elrt, dist, latval, lonval)
        jj, ii = dom.network["chanyj"] - 1, dom.network["chanxi"] - 1
        dom.network["node_area"] = (dom.dist[8, jj, ii] / F(1)).astype(F)
        _reaches(dom, cfg, ch_lnkrt, path("route_link_f"), path("route_lake_f"))
    elif cfg.channel:
        # READ_CHROUTING1 :7510-7518: the channel bed is CHAN_DEPTH below the
        # terrain for GWBASESWCRT = 3 (ZELEV only; LandRT_ini's ELRT stays)
        zelev = (elrt - chan_depth).astype(F) if cfg.gwbaseswcrt == 3 else elrt
        dom.network, dom.lake_mask = _network(ch_netrt, direction, lakegrid, gorder, frxst,
                                              zelev, dist, latval, lonval)
        nlakes = int(max(0, lakegrid.max()))
        if cfg.lake_option == 0 and nlakes > 0:
            raise NotImplementedError(
                "lake_option = 0 with LAKEGRID lakes: getChanDim sets NLAKES = 0 while the "
                "network still drains into lake pixels")
        dom.lakes = _lakes(path("route_lake_f"), nlakes, cfg.lake_option) if nlakes else None
        _channel_params(dom, path("route_chan_f"), base)
        if dom.lakes is not None:
            # LandRT_ini :743-746
            L = dom.lakes
            L["resht"] = (L["orifice_elevation"] + ((L["max_depth"] - L["orifice_elevation"])
                                                    * L["ifd"])).astype(F)
    if cfg.gwbaseswcrt in (1, 2):
        dom.buckets = _buckets(dom, path("gwbasmskfil"), path("gwbuckparm_file"))
    elif cfg.gwbaseswcrt == 3:
        dom.gw2d = _gw2d(dom, cfg, path("gwbasmskfil"), base)
    return dom


def _reaches(dom, cfg, ch_lnkrt, route_link_f, route_lake_f):
    """channel_option 1/2 (UDMP_OPT = 0, one MPP_LAND process): Route_Link's
    reaches, their lakes and the routing-pixel -> reach map."""
    if route_link_f is None:
        raise MissingInput("route_link_f is empty (channel_option 1/2 reads Route_Link)")
    if not str(route_link_f).endswith(".nc"):
        raise MissingInput(f"route_link_f {route_link_f.name}: the text form of Route_Link "
                           "is not ported (only a name ending .nc is read as NetCDF)")
    with _open(route_link_f, "route_link_f (Route_Link.nc)") as ds:
        what = f"route_link_f {route_link_f.name}"
        dim = "feature_id" if "feature_id" in ds.dimensions else "linkDim"
        if dim not in ds.dimensions:
            raise ValueError(f"{what}: Could not find either feature_id or linkDim")
        nl_ = len(ds.dimensions[dim])
        r = {}
        for key, var, t in (("linkid", "link", np.int64), ("to", "to", np.int64),
                            ("chlon", "lon", F), ("chlat", "lat", F), ("zelev", "alt", F),
                            ("order", "order", np.int64), ("qi", "Qi", F), ("musk", "MusK", F),
                            ("musx", "MusX", F), ("chanlen", "Length", F), ("mannn", "n", F),
                            ("so", "So", F), ("chsslp", "ChSlp", F), ("bw", "BtmWdth", F),
                            ("channk", "Kchan", F)):
            # get_1d_netcdf_*: no _M fallback, every one fatal-required
            if var not in ds.variables:
                raise MissingInput(f"{what} has no variable {var}")
            v = ds.variables[var]
            v.set_auto_maskandscale(False)
            r[key] = np.asarray(v[...]).reshape(-1).astype(t)
        lakeida = None
        if "NHDWaterbodyComID" in ds.variables:
            v = ds.variables["NHDWaterbodyComID"]
            v.set_auto_maskandscale(False)
            lakeida = np.asarray(v[...]).reshape(-1).astype(np.int64)
        gages = [""] * nl_
        if "gages" in ds.variables:
            v = ds.variables["gages"]
            v.set_auto_maskandscale(False)
            raw = np.asarray(v[...])
            gages = [b"".join(row).decode("ascii", "replace") for row in raw.reshape(nl_, -1)]
    gages = [g.replace("\0", " ")[:15].ljust(15) for g in gages]
    n = len(r["linkid"])
    if len(set(r["linkid"].tolist())) != n:
        raise ValueError(f"{what}: duplicate link ids (the link -> index hash keeps one)")
    # NLAKES: read_NSIMLAKES (a .nc route_lake_f, else 0); lake_option 0
    # cannot zero it for reaches (getChanDim reads it after zeroing)
    nlakes = 0
    if route_lake_f is not None and str(route_lake_f).endswith(".nc"):
        with _open(route_lake_f, "route_lake_f (LAKEPARM.nc)") as ds:
            d = "feature_id" if "feature_id" in ds.dimensions else "nlakes"
            if d not in ds.dimensions:
                raise ValueError("Could not find either feature_id or nlakes in LAKEPARM")
            nlakes = len(ds.dimensions[d])
    if nlakes and cfg.lake_option == 0:
        raise NotImplementedError("lake_option = 0 with a LAKEPARM.nc for channel_option 1/2: "
                                  "read_NSIMLAKES restores NLAKES after getChanDim zeroes it, "
                                  "and the reservoirs run with lake_option 0")
    if nlakes and lakeida is None:
        raise MissingInput(f"{what} has no NHDWaterbodyComID: readLinkSL leaves LAKEIDA "
                           "unset and nhdLakeMap reads it for every lake")
    if lakeida is None:
        lakeida = np.full(n, -9999, np.int64)
    dom.lakes = _lakes(route_lake_f, nlakes, cfg.lake_option) if nlakes else None
    if dom.lakes is not None:
        L = dom.lakes
        L["resht"] = (L["orifice_elevation"] + ((L["max_depth"] - L["orifice_elevation"])
                                                * L["ifd"])).astype(F)
    # read_route_link_netcdf's So floor, then readLinkSL's
    so = np.where(r["so"] < F(0.00001), F(0.00001), r["so"]).astype(F)
    so = np.maximum(so, F(0.00001)).astype(F)
    # readLinkSL (MPP_LAND, NetCDF): MAXORDER is still READ_CHROUTING1's
    # -9999 (only the text reader updates it); a missing Qi starts at
    # 20 (1 / (MAXORDER + 1 - ORDER))**3 in both QLINK slots, a given one in
    # QLINK(:,1) only
    q1 = r["qi"].copy()
    q2 = np.zeros(n, F)
    low = q1 <= F(1e-3)
    x = (F(1.0) / (F(-9998.0) - r["order"].astype(F))).astype(F)
    x3 = (x * x * x).astype(F)
    q1 = np.where(low, (F(20.0) * x3).astype(F), q1).astype(F)
    q2 = np.where(low, q1, q2).astype(F)
    typel, to_node, lakeidx = _nhd_lake_map(r["linkid"], r["to"].copy(), lakeida,
                                            None if dom.lakes is None
                                            else dom.lakes["lake_id"].copy())
    # LandRT_ini :990-1050 (MPP_LAND, one process): the link ids on the grid,
    # ascending (LLINKID; they index buf(GNLINKS), GNLINKS the node count),
    # and CH_LNKRT_SL the local position of each pixel's id; getLocalIndx's
    # LLINKIDINDX takes a local position to its Route_Link index, which is
    # where updateLinkV puts drive_CHANNEL's lateral inflow
    nnodes = len(dom.network["chanxi"])
    llinkid = sorted({int(v) for v in ch_lnkrt[ch_lnkrt > 0].tolist()})
    if llinkid and llinkid[-1] > nnodes:
        raise NotImplementedError(
            f"a Fulldom LINKID above the node count {nnodes}: LandRT_ini indexes buf(GNLINKS) "
            "with it (out of bounds); only link ids up to the node count index that buffer")
    index = {int(k): i + 1 for i, k in enumerate(r["linkid"])}
    missing = [v for v in llinkid if v not in index]
    if missing:
        raise ValueError(f"Fulldom LINKID {missing[0]} is not a Route_Link link: "
                         "LLINKIDINDX is 0 for it and updateLinkV writes out of bounds")
    llinkidindx = np.asarray([0] + [index[v] for v in llinkid], np.int64)
    sl = np.full(ch_lnkrt.shape, -99, np.int64)
    for local, v in enumerate(llinkid, start=1):
        sl[ch_lnkrt == v] = local
    # what ReachRouting reads: each node pixel's Route_Link reach, through
    # LLINKIDINDX for both drive_CHANNEL's branches -- a channel pixel's
    # CH_LNKRT_SL, and a lake pixel's LAKE_MSKRT (its lake number taken as a
    # local position)
    jj, ii = dom.network["chanyj"] - 1, dom.network["chanxi"] - 1
    reach_sl = np.where(sl > 0, llinkidindx[np.maximum(sl, 0)], -99)
    lake_nodes = (sl[jj, ii] <= 0) & (dom.lake_mask[jj, ii] > 0)
    if (dom.lake_mask[jj, ii][lake_nodes] > len(llinkid)).any():
        raise NotImplementedError("a node on a lake pixel whose lake number exceeds the grid's "
                                  "link count: drive_CHANNEL indexes LQLateral(LNLINKSL) with it")
    lake_reach = np.where(dom.lake_mask > 0,
                          llinkidindx[np.clip(dom.lake_mask, 0, len(llinkid))], -9999)
    dom.reaches = dict(
        nlinksl=n, linkid=r["linkid"], to_node=to_node, typel=typel, lake_lookup=lakeidx,
        order=r["order"], chlon=r["chlon"], chlat=r["chlat"], zelev=r["zelev"],
        musk=r["musk"], musx=r["musx"], chanlen=r["chanlen"], mannn=r["mannn"], so=so,
        chsslp=r["chsslp"], bw=r["bw"], tw=np.zeros(n, F), tw_cc=np.zeros(n, F),
        n_cc=np.zeros(n, F), channk=r["channk"], qlink1=q1, qlink2=q2,
        hlink=np.full(n, F(-999.0), F), ch_lnkrt_sl=sl, llinkidindx=llinkidindx,
        reach_sl=reach_sl, lake_reach=lake_reach, gages=gages, lakeida=lakeida)


def _nhd_lake_map(linkid, to_node, lakeida, lakeidm):
    """nhdLakeMap_scan then nhdLakeMap_mpp on one process (module_HYDRO_io.F90
    :10405-10916): TYPEL (1 a lake's outlet reach, 2 inside a lake, 3 flowing
    into one, -999 otherwise), TO_NODE with each inflow reach moved to its
    lake's outlet, and LAKEIDX (the 1-based lake an outlet reach routes,
    else -999)."""
    n = len(linkid)
    nlakes = 0 if lakeidm is None else len(lakeidm)
    index = {int(k): i for i, k in enumerate(linkid)}
    # links that drain into a link of the network, in link order (ind), and
    # each link's upstream positions in that list (gToNodeOut)
    ind = [it for it in range(n) if int(to_node[it]) in index]
    ups = [[] for _ in range(n)]
    for kk, it in enumerate(ind):
        ups[index[int(to_node[it])]].append(kk)
    up_lake = [lakeida[it] for it in ind]
    up_to = [int(to_node[it]) for it in ind]
    # nhdLakeMap_scan: one outlet per lake, else the lake is dropped
    lakelinkid = np.zeros(nlakes, np.int64)
    for i in range(nlakes):
        for k in range(n):
            for j in ups[k]:
                if up_to[j] == linkid[k] and lakeida[k] < 0 and up_lake[j] == lakeidm[i]:
                    if lakelinkid[i] > 0:
                        lakelinkid[i] = -999
                    if lakelinkid[i] == 0:
                        lakelinkid[i] = ind[j] + 1
    down_lake = np.full(n, -999, np.int64)
    for it in range(n):
        if int(to_node[it]) in index:
            down_lake[it] = lakeida[index[int(to_node[it])]]
    lakemask = np.zeros(nlakes, np.int64)
    for k in range(n):
        if lakeida[k] > 0:
            if any(down_lake[k] == lakeidm[i] for i in range(nlakes)):
                continue
            for i in range(nlakes):
                if lakeida[k] == lakeidm[i]:
                    lakemask[i] += 1
                    break
    for i in range(nlakes):
        if lakemask[i] != 1:
            lakelinkid[i] = -999
    for i in range(nlakes):
        if lakelinkid[i] == -999:
            lakeidm[i] = -999
    # nhdLakeMap_mpp
    tmp_typel = [-999] * len(ind)
    tmp_out = list(up_to)
    for i in range(nlakes):
        for k in range(n):
            for j in ups[k]:
                if up_to[j] == linkid[k] and lakeida[k] < 0 and up_lake[j] == lakeidm[i]:
                    tmp_typel[j] = 1
                    tmp_out[j] = up_lake[j]
                    lakelinkid[i] = ind[j] + 1
                elif (up_to[j] == linkid[k] and up_lake[j] < 0 and lakeida[k] > 0
                      and lakeida[k] == lakeidm[i]):
                    tmp_typel[j] = 3
                    tmp_out[j] = lakeidm[i]
                elif up_lake[j] == lakeidm[i] and tmp_typel[j] != 1:
                    tmp_typel[j] = 2
    lakeidx = np.full(n, -999, np.int64)
    for i in range(nlakes):
        if lakelinkid[i] > 0:
            lakeidx[lakelinkid[i] - 1] = i + 1
    typel = np.full(n, -999, np.int64)
    outlake = np.asarray(to_node, np.int64).copy()
    for kk, it in enumerate(ind):          # TONODE2RSL / TONODE2RSL8, flag -999
        if tmp_typel[kk] != -999:
            typel[it] = tmp_typel[kk]
        if tmp_out[kk] != -999:
            outlake[it] = tmp_out[kk]
    for i in range(n):
        for j in range(n):
            if typel[i] == 3 and typel[j] == 1 and outlake[j] == outlake[i]:
                to_node[i] = linkid[j]
    return typel, to_node, lakeidx


def _terrain(elrt, dist, rt_option):
    """LandRT_ini :1051-1186 (the slope block) and :1213-1216 (sox/soy edges)."""
    jxrt, ixrt = elrt.shape
    slope = np.full((8, jxrt, ixrt), F(-999), F)
    sdir = np.zeros((3, jxrt, ixrt), np.int32)
    retention = np.full((jxrt, ixrt), F(0.001), F)
    sox = np.zeros((jxrt, ixrt), F)
    soy = np.zeros((jxrt, ixrt), F)
    nb = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
    if jxrt < 3 or ixrt < 3:
        raise ValueError("the routing grid needs at least 3 x 3 pixels")
    inner = (slice(1, jxrt - 1), slice(1, ixrt - 1))
    e = elrt[inner]

    def shifted(di, dj):
        return elrt[1 + dj:jxrt - 1 + dj, 1 + di:ixrt - 1 + di]

    # every operation below is one IEEE binary32 operation, element by element
    sox[inner] = (e - shifted(1, 0)) / dist[2][inner]
    soy[inner] = (e - shifted(0, 1)) / dist[0][inner]

    def reduce_by(vmax, limit):
        frac = (vmax / F(limit)).astype(F)
        r = (retention[inner] * (F(1.0) - frac)).astype(F)
        r = np.where(r < F(0.0), F(0.0), r)
        return np.where(vmax > F(limit), F(0.0), r).astype(F)

    if rt_option == 2:
        retention[inner] = reduce_by(np.where(sox[inner] > soy[inner], sox[inner],
                                              soy[inner]).astype(F), 0.1)
    s = np.stack([((e - shifted(di, dj)) / dist[k][inner]).astype(F)
                  for k, (di, dj) in enumerate(nb)])
    slope[(slice(None),) + inner] = s
    best = np.argmax(s, axis=0)          # the first maximum, as the strict '>' keeps
    vmax = np.take_along_axis(s, best[None], axis=0)[0]
    jj, ii = np.meshgrid(np.arange(1, jxrt - 1), np.arange(1, ixrt - 1), indexing="ij")
    di = np.array([d[0] for d in nb])[best]
    dj = np.array([d[1] for d in nb])[best]
    sdir[0][inner] = ii + di + 1
    sdir[1][inner] = jj + dj + 1
    sdir[2][inner] = best + 1
    if rt_option == 1:
        retention[inner] = reduce_by(vmax, 0.75)
    sox[:, ixrt - 1] = sox[:, ixrt - 2]
    sox[:, 0] = sox[:, 1]
    soy[jxrt - 1, :] = soy[jxrt - 2, :]
    soy[0, :] = soy[1, :]
    return slope, sdir, retention, sox, soy


def _network(ch_netrt, direction, lakegrid, gorder, frxst, elrt, dist, latval, lonval):
    """READ_ROUTEDIM's node count, MPP_READ_ROUTEDIM's raster renumbering and
    READ_CHROUTING1's channel_option = 3 links (UDMP_OPT = 0)."""
    jxrt, ixrt = ch_netrt.shape
    direction = np.where(direction == -128, 128, direction)
    is_node = ch_netrt >= 0
    bad = is_node & ~np.isin(direction, list(DIRECTIONS))
    if bad.any():
        j, i = np.argwhere(bad)[0]
        raise ValueError(f"FLOWDIRECTION {int(direction[j, i])} at routing pixel "
                         f"({i + 1}, {j + 1}) of a CHANNELGRID node: WRF-Hydro stops in "
                         "READ_ROUTEDIM (cnt .ne. NLINKS)")
    netlnk = np.full((jxrt, ixrt), -9999, np.int64)
    jj, ii = np.nonzero(is_node)              # row-major: j outer, i inner
    n = len(jj)
    netlnk[jj, ii] = np.arange(1, n + 1)
    net = dict(
        ch_netlnk=netlnk, chanxi=(ii + 1).astype(np.int64), chanyj=(jj + 1).astype(np.int64),
        from_node=np.arange(1, n + 1, dtype=np.int64), to_node=np.full(n, -9999, np.int64),
        typel=np.zeros(n, np.int64), lakenode=np.zeros(n, np.int64),
        order=gorder[jj, ii].astype(np.int64), frxst=frxst[jj, ii].astype(np.int64),
        zelev=elrt[jj, ii].astype(F), chanlen=np.zeros(n, F), chlat=latval[jj, ii].astype(F),
        chlon=lonval[jj, ii].astype(F))
    for k in range(n):
        j, i = int(jj[k]), int(ii[k])
        di, dj, d = DIRECTIONS[int(direction[j, i])]
        ti, tj = i + di, j + dj
        inside = 0 <= ti < ixrt and 0 <= tj < jxrt
        net["chanlen"][k] = dist[d - 1, j, i]
        if inside and ch_netrt[tj, ti] >= 0:
            net["to_node"][k] = netlnk[tj, ti]
        elif inside and lakegrid[tj, ti] > 0:
            net["typel"][k] = 2
            net["lakenode"][k] = lakegrid[tj, ti]
        else:
            net["typel"][k] = 1
    return net, lakegrid.astype(np.int32)


def _lakes(path, nlakes, lake_option):
    """read_route_lake_netcdf (LAKEPARM.nc), one row per LAKEGRID lake."""
    if path is None:
        raise MissingInput("route_lake_f is empty but LAKEGRID has lakes (Fatal error: "
                           "route_lake_f must be specified in the hydro.namelist)")
    with _open(path, "route_lake_f (LAKEPARM.nc)") as ds:
        what = f"route_lake_f {Path(path).name}"
        names = dict(lake_area="LkArea", max_depth="LkMxE", weir_elevation="WeirE",
                     weir_coefficient="WeirC", weir_length="WeirL", dam_length="Dam_Length",
                     orifice_coefficient="OrificeC", orifice_area="OrificeA",
                     orifice_elevation="OrificeE")
        out = {}
        for key, var in names.items():
            a = _var(ds, var, what=what)
            if a.shape != (nlakes,):
                raise ValueError(f"{what} {var} has {a.shape[0] if a.ndim else 1} lakes; "
                                 f"LAKEGRID numbers {nlakes}")
            out[key] = a
        out["lake_id"] = _var(ds, "lake_id", dtype=np.int64, what=what)
        # 'ifd' is read non-fatally; rt_allocate's ELEVLAKE = 0 stands otherwise
        out["ifd"] = (_var(ds, "ifd", what=what) if "ifd" in ds.variables
                      else np.zeros(nlakes, F))
    out["lake_opt"] = int(lake_option)
    return out


def _channel_params(dom, route_chan_f, base):
    """LandRT_ini :1338-1550 for channel_option = 3: CHANPARM.TBL by stream
    order where Fulldom left zero (every node: READ_CHROUTING1 zeroes them),
    HLINK_INIT, and CVOL; So stays rt_allocate's 0.01."""
    botwid, hlink_init, chan_ss, chmann = _chanparm(base / "CHANPARM.TBL")
    net = dom.network
    order = net["order"]
    idx = np.where((order >= 1) & (order <= 6), order, np.where(order >= 7, order, 5)) - 1
    if (order > 50).any():
        raise ValueError("STREAMORDER above 50: CHANPARM.TBL holds 50 orders (NCHANTYPES)")
    net["bw"] = botwid[idx].astype(F)
    net["chsslp"] = chan_ss[idx].astype(F)
    net["mannn"] = chmann[idx].astype(F)
    net["hlink"] = hlink_init[idx].astype(F)
    net["so"] = np.full(len(order), F(0.01), F)
    bw, ss, h, ln = net["bw"], net["chsslp"], net["hlink"], net["chanlen"]
    net["cvol"] = (F(1.0) / ss).astype(F)
    net["cvol"] = ((bw + net["cvol"] * h) * h * ln).astype(F)
    # get_node_area -> get_area_g8: the routing cell area under each node
    jj, ii = net["chanyj"] - 1, net["chanxi"] - 1
    net["node_area"] = (dom.dist[8, jj, ii] / F(1)).astype(F)
    if route_chan_f is not None:
        dom.notes.append("route_chan_f is not read for channel_option = 3")


def _gw2d(dom, cfg, basins, base):
    """readGW2d (module_HYDRO_io.F90 :1593-1737) as LandRT_ini calls it
    (module_RT.F90 :1389-1396) for GWBASESWCRT = 3: gwhires.nc in the run
    directory, read as stored (no north-up flip, unlike Fulldom), LTYPE 2
    around the edge, IHEAD + gwIhShift, no conductivity and a huge porosity
    where LTYPE = 0; the aquifer's surface is LandRT_ini's ELRT."""
    if basins is None or not Path(basins).is_file():
        # read_rt_nlst's check (config.F90 :437-442) for any GWBASESWCRT > 0
        raise MissingInput("gwbasmskfil is empty or missing (hydro.namelist ERROR: Please "
                           "specify a gwbasmskfil file; read for every GWBASESWCRT > 0)")
    if dom.aggfactrt != 1:
        raise NotImplementedError(
            f"GWBASESWCRT = 3 with AGGFACTRT = {dom.aggfactrt}: under MPP_LAND readGW2d "
            "reads each gwhires.nc field into a routing-grid array through get2d_real / "
            "get2d_int with the LSM grid's size (global_nx x global_ny), so only the first "
            "IX*JX values are read (out of place) and the rest are left unset; the file "
            "reads whole only with AGGFACTRT = 1")
    gwhires = base / "gwhires.nc"
    shape = (dom.jxrt, dom.ixrt)
    with _open(gwhires, "gwhires.nc (readGW2d reads ./gwhires.nc for GWBASESWCRT = 3)") as ds:
        what = "gwhires.nc"
        # get2d_real leaves the array unset when a variable is missing
        fields = {k: _var(ds, k, shape, what=what) for k in ("HC", "IHEAD", "BOTELV", "POR")}
        if "LTYPE" not in ds.variables:          # get2d_int has no _M fallback
            raise MissingInput(f"{what} has no variable LTYPE")
        ltype = _var(ds, "LTYPE", shape, dtype=np.int32, what=what)
    ltype = ltype.copy()
    ltype[0, :] = ltype[-1, :] = 2
    ltype[:, 0] = ltype[:, -1] = 2
    ihead = (fields["IHEAD"] + F(cfg.gw_ih_shift)).astype(F)
    hc = np.where(ltype == 0, F(0.0), fields["HC"]).astype(F)
    por = np.where(ltype == 0, F(10e21), fields["POR"]).astype(F)
    return dict(ltype=ltype, hycond=hc, ho=ihead, bot=fields["BOTELV"].astype(F), poros=por,
                elev=dom.elrt.copy())


def _buckets(dom, basins, parm):
    """READ_SIMP_GW, SIMP_GW_IND, get_gw_strm_msk_lind, read_GWBUCKPARM and
    get_basn_area (GWBASESWCRT 1 and 2, UDMP_OPT = 0)."""
    if basins is None:
        raise MissingInput("gwbasmskfil is empty (GWBASESWCRT 1/2 reads the basin mask)")
    if parm is None:
        raise MissingInput("GWBUCKPARM_file is empty (GWBASESWCRT 1/2 reads the bucket "
                           "parameters)")
    ix, jx, agg = dom.ix, dom.jx, dom.aggfactrt
    with _open(basins, "gwbasmskfil (GWBASINS.nc)") as ds:
        mask = _var(ds, "BASIN", (jx, ix), dtype=np.int32, flip=True,
                    what=f"gwbasmskfil {Path(basins).name}")
    gnumbasns = int(max(0, mask.max()))
    strm = np.full((dom.jxrt, dom.ixrt), -9999, np.int32)
    fine = np.repeat(np.repeat(mask, agg, axis=0), agg, axis=1)
    on = dom.ch_netrt >= 0
    strm[on] = fine[on]
    present = sorted({int(b) for b in mask.ravel() if b > 0})
    basnsind = np.asarray(present, np.int64)
    lind = np.full(strm.shape, -999, np.int32)
    for k, b in enumerate(present):
        lind[strm == b] = k + 1
    with _open(parm, "GWBUCKPARM_file (GWBUCKPARM.nc)") as ds:
        what = f"GWBUCKPARM_file {Path(parm).name}"
        tab = {k: _var(ds, k, what=what) for k in ("Coeff", "Expon", "Zmax", "Zinit")}
        bas_id = _var(ds, "Basin", dtype=np.int64, what=what)
    if any(len(v) < gnumbasns for v in tab.values()):
        raise ValueError(f"{what} holds {len(tab['Coeff'])} basins; the basin mask numbers "
                         f"{gnumbasns}")
    sel = basnsind - 1
    # get_area_g: the mean LSM cell area of each basin (it overwrites
    # GWBUCKPARM's Area_sqkm), summed in j-outer, i-inner REAL order
    area = np.zeros(gnumbasns, F)
    count = np.zeros(gnumbasns, np.int64)
    for j in range(jx):
        for i in range(ix):
            b = mask[j, i]
            if b > 0:
                area[b - 1] = F(area[b - 1] + dom.dist_lsm[8, j, i])
                count[b - 1] += 1
    area = np.where(count > 0, area / np.maximum(count, 1).astype(F), area).astype(F)
    # the bucket's pixel count is module_RT's REAL count over gw_strm_msk,
    # indexed by basin number as read (one process: basnsInd(k) = k when
    # every basin is present)
    if len(present) != gnumbasns:
        raise NotImplementedError("a basin number absent from the basin mask: the "
                                  "one-process bucket indexing of pix_ct_1 is not ported")
    return dict(gwsubbasmsk=mask, basnsind=basnsind, gw_strm_msk=strm, gw_strm_msk_lind=lind,
                c=tab["Coeff"][sel].astype(F), ex=tab["Expon"][sel].astype(F),
                z_mx=tab["Zmax"][sel].astype(F), z_init=tab["Zinit"][sel].astype(F),
                bas_id=bas_id[sel], basns_area=area[sel].astype(F))

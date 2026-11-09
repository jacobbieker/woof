# WOOF-OPTIN: wrf-hydro
"""WRF's side of the WRF/WRF-Hydro coupling, around the ported routing units.

``wrf_cpl_HYDRO`` (hydro/CPL/WRF_cpl/module_wrf_HYDRO.F90 in WRF 4.7.1) and
the ``HYDRO_exe`` it calls (hydro/HYDRO_drv/module_HYDRO_drv.F90 :560-751),
on device, for one domain (WRF calls it only when ``num_nests < 1``):

* WRF sets ``HYDRO_dt = DTBL`` whenever the surface driver runs the LSM
  (module_surface_driver.F :2647 Noah, :3125 Noah-MP) and, right after
  surface_driver, calls ``wrf_drv_HYDRO`` when ``HYDRO_dt > 1``
  (dyn_em/module_first_rk_step_part1.F :1093-1097).  So the coupling
  interval is the LSM step.  DTRT_TER and DTRT_CH are snapped to divide it
  (module_wrf_HYDRO.F90 :133-176); DTCT starts at the namelist DTRT_CH
  (config.F90 :828).
* :meth:`WrfHydroCoupled.exchange` is one ``wrf_cpl_HYDRO`` call after HYDRO_ini:
  SMOIS, SH2O, INFXSRT and SOLDRAIN copied into the routing domain
  (:187-195), ``HYDRO_exe``, then SMOIS, SH2O and SFCHEADRT copied back
  (:204-210).
* ``HYDRO_exe`` in its order: disaggregate (SUBRTSWCRT or OVRTSWCRT),
  subsurface routing, overland routing (or, without it, RunoffAggregate of
  the subsurface spill onto INFXSRT), QSTRMVOLRT/LAKE_INFLORT for the step,
  the groundwater bucket, channel routing, aggregate, then SOLDRAIN and
  QSUBRT zeroed.

The static state is :mod:`woof_wrf_hydro.routing_domain`.  Ported for the
coupled run: SUBRTSWCRT and OVRTSWCRT (rt_option 1 and 2), channel_option 3
with level-pool or pass-through lakes, GWBASESWCRT 0, 1 and 2.  Refused by
name here or in the namelist mapper: Route_Link reaches (channel_option 1
and 2), UDMP_OPT = 1, GWBASESWCRT = 3, IMPERV_ADJ, restarts (RESTART_FILE,
rst_typ = 1), sys_cpl other than 2, rtFlag = 0 (no SUBRTSWCRT, OVRTSWCRT
or GWBASESWCRT), channel-only modes.

Output, at out_dt as HYDRO_out schedules it (with the t0 frame when
t0OutputFlag = 1): ``frxst_pts_out.txt`` in output_chrt's channel_option 3
format; a CHRTOUT-like NetCDF per frame (streamflow and head at every node,
WRF-Hydro's legacy names); an RTOUT-like NetCDF of the routing-grid fields
(zwattablrt, sfcheadsubrt, QSTRMVOLRT, SOIL_M).  The LSM-grid exchange fields
(SFCHEADRT, INFXSRT, SOLDRAIN) are the forecast's own history rows.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import grid_transform, namelist, routing_domain
from ._compile import raw_module

F = np.float32
_MODULE = None


def _k(name):
    global _MODULE
    if _MODULE is None:
        source = (Path(__file__).resolve().parent / "coupled.cu").read_text(encoding="utf-8")
        _MODULE = raw_module(source, ("--fmad=false", "--ftz=false"))
    return _MODULE.get_function(name)


def snap(routing_dt, hydro_dt):
    """module_wrf_HYDRO.F90 :133-152: a routing step that divides HYDRO_dt."""
    routing_dt, hydro_dt = F(routing_dt), F(hydro_dt)
    if routing_dt >= hydro_dt:
        return hydro_dt
    mm = int(hydro_dt / routing_dt)
    if F(F(mm) * routing_dt) < hydro_dt:
        return F(hydro_dt / F(mm))
    return routing_dt


def refusals(cfg, groups):
    """What the coupled driver does not run, by name (before any file is read)."""
    nl = groups["hydro_nlist"]
    if int(nl.get("sys_cpl", 1)) != 2:
        raise namelist.UnportedOption(
            f"sys_cpl = {nl.get('sys_cpl', 1)!r}: wrf_cpl_HYDRO stops unless sys_cpl = 2 "
            "(\"sys_cpl should be 2.  Check hydro.namelist file.\")")
    if not (cfg.subsurface or cfg.overland or cfg.gwbaseswcrt):
        raise namelist.UnportedOption(
            "SUBRTSWCRT = OVRTSWCRT = GWBASESWCRT = 0: rtFlag is 0 (config.F90 :933) and "
            "wrf_cpl_HYDRO returns without routing anything")
    if int(nl.get("imperv_adj", 0)) != 0:
        raise namelist.UnportedOption("IMPERV_ADJ: the impervious-fraction adjustment of "
                                      "disaggregateDomain and the retention depth is not ported")
    if str(nl.get("restart_file", "") or "") or int(nl.get("rst_typ", 0)) == 1:
        raise namelist.UnportedOption("RESTART_FILE / rst_typ = 1: restarting WRF-Hydro's "
                                      "routing state is not ported")
    if cfg.channel and cfg.channel_option != 3:
        raise namelist.UnportedOption(
            f"channel_option = {cfg.channel_option}: Route_Link reach routing is not wired "
            "into the coupled driver (the gridded network, channel_option = 3, is)")
    if cfg.compound_channel:
        raise namelist.UnportedOption("compound_channel: Muskingum-Cunge's compound channel "
                                      "is a Route_Link option")


class WrfHydroCoupled:
    """HYDRO_ini, then one :meth:`exchange` per LSM step."""

    def __init__(self, namelist_path, *, ix, jx, zs, vegtyp, soltyp, hydro_dt, start,
                 output_dir=None):
        import cupy as cp

        from . import baseflow, diffusive, levelpool, overland, subsurface

        path = Path(namelist_path).resolve()
        self.groups = namelist.parse(path.read_text(encoding="utf-8", errors="replace"))
        self.cfg = cfg = namelist.map_hydro_nlist(self.groups)
        refusals(cfg, self.groups)
        nl = self.groups["hydro_nlist"]
        self.dom = dom = routing_domain.build(cfg, self.groups, base=path.parent, ix=ix,
                                              jx=jx, zs=zs, vegtyp=vegtyp, soltyp=soltyp)
        hydro_dt = F(hydro_dt)
        if hydro_dt <= F(1):
            raise ValueError("HYDRO_dt <= 1 s: WRF calls wrf_drv_HYDRO only when HYDRO_dt > 1")
        self.dt = hydro_dt
        self.dtrt_ter = snap(cfg.dtrt_ter, hydro_dt)
        self.dtrt_ch = snap(cfg.dtrt_ch, hydro_dt)
        self.start = start
        self.elapsed = 0
        self.out_dt = int(nl.get("out_dt", 60)) * 60
        self.t0_output = int(nl.get("t0outputflag", 1)) == 1
        self.igrid = int(nl.get("igrid", 1))
        self.flags = {k: int(nl.get(k, 0)) for k in ("chrtout_domain", "rtout_domain",
                                                      "frxst_pts_out")}
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.out_counts = 0
        self.routing = cfg.subsurface or cfg.overland

        d = lambda a, t=cp.float32: cp.asarray(np.ascontiguousarray(a, t))  # noqa: E731
        ixrt, jxrt, nsoil = dom.ixrt, dom.jxrt, dom.nsoil
        self.shape, self.rt_shape = (jx, ix), (jxrt, ixrt)
        z = lambda shape=(jxrt, ixrt): cp.zeros(shape, cp.float32)  # noqa: E731
        # the LSM-side copies (RT_DOMAIN's SMC, SH2OX, INFXSRT, SOLDRAIN, SICE)
        self.smc, self.sh2ox, self.sice = (z((nsoil, jx, ix)) for _ in range(3))
        self.infxsrt, self.soldrain, self.head_lsm = z((jx, ix)), z((jx, ix)), z((jx, ix))
        # disaggregateDomain's weights: LandRT_ini :1640-1641
        self.infxswgt = cp.full((jxrt, ixrt), F(1.0) / F(dom.aggfactrt * dom.aggfactrt),
                                cp.float32)
        self.sh2owgt = cp.ones((nsoil, jxrt, ixrt), cp.float32)
        self.static = dict(
            dist=d(dom.dist), dist9=d(dom.dist[8]), area_lsm=d(dom.dist_lsm[8]),
            smcmax1=d(dom.smcmax1), smcref1=d(dom.smcref1), smcwlt1=d(dom.smcwlt1),
            vegtyp=d(dom.vegtyp, cp.int32), soiltyp=d(dom.soiltyp, cp.int32),
            lksat=d(dom.lksat), nexp=d(dom.nexp), lksatfac=d(dom.lksatfac),
            ch_netrt=d(dom.ch_netrt, cp.int32), elrt=d(dom.elrt))
        # disaggregateDomain clears the lake mask under LSM water cells off
        # the channel network on every call (Noah_distr_routing.F90 :1162-1169);
        # the edit is static, so the routing units are built on its result.
        lake_mask = dom.lake_mask.copy()
        if self.routing:
            water = np.repeat(np.repeat(dom.vegtyp == dom.iswater, dom.aggfactrt, 0),
                              dom.aggfactrt, 1)
            lake_mask[water & (dom.ch_netrt <= 0)] = -9999
        self.lake_mask = d(lake_mask, cp.int32)
        self.gt_out = {k: z((nsoil, jxrt, ixrt)) for k in ("smcrefrt", "smcmaxrt", "smcwltrt",
                                                          "smcrt")}
        self.gt_out.update(infxsubrt=z(), lksatrt=z(), nexprt=z(),
                           soiltyprt=cp.zeros((jxrt, ixrt), cp.int32))
        self.sub = self.sstate = None
        if cfg.subsurface:
            self.sub = subsurface.SubsurfaceRouting(
                dt=hydro_dt, sldpth=dom.sldpth, zsoil=dom.zsoil, lksat=self.gt_out["lksatrt"],
                nexp=self.gt_out["nexprt"],
                soldep=cp.full((jxrt, ixrt), F(-1.0) * dom.zsoil[-1], cp.float32),
                slope=d(dom.slope), sdir=d(dom.sdir, cp.int32), dist=self.static["dist"],
                rt_option=cfg.rt_option)
        self.sstate = subsurface.SubsurfaceState.zeros(ixrt, jxrt)
        self.ostate = overland.OverlandState(
            head=z(), retention=d(dom.retention), to_channel=z(), to_lake=z(),
            boundary_flux=z(), totals=cp.zeros(3, cp.float32))
        self.qx, self.qy = z(), z()
        self.ov = None
        if cfg.overland:
            self.ov = overland.OverlandRouting(
                dist=self.static["dist"], slope=d(dom.slope), sdir=d(dom.sdir, cp.int32),
                roughness=d(dom.roughness), ch_netrt=self.static["ch_netrt"],
                lake_mask=self.lake_mask, rt_option=cfg.rt_option, sox=d(dom.sox),
                soy=d(dom.soy))
        self.qstrm_ts, self.qstrm_acc, self.lake_ts = z(), z(), z()
        self.qinflowbase = z()
        self.bucket = None
        if cfg.gwbaseswcrt in (1, 2):
            b = dom.buckets
            self.bucket = baseflow.GroundwaterBucket(
                gwbaseswcrt=cfg.gwbaseswcrt, gwsubbasmsk=b["gwsubbasmsk"],
                basnsind=b["basnsind"], gw_strm_msk_lind=b["gw_strm_msk_lind"],
                basns_area=d(b["basns_area"]), c=d(b["c"]), ex=d(b["ex"]), z_mx=d(b["z_mx"]),
                pixct=d(baseflow.pixel_count(b["gw_strm_msk"], len(b["basnsind"]))),
                dist9=self.static["dist9"])
            nb = len(b["basnsind"])
            self.bucket_cells = d([np.float32((b["gwsubbasmsk"] == k).sum())
                                   for k in b["basnsind"]])
            self.z_gw = d(b["z_init"])
            self.qin_gw, self.qout_gw = cp.zeros(nb, cp.float32), cp.zeros(nb, cp.float32)
        self.channel = None
        if cfg.channel:
            net = dom.network
            lakes = levelpool.LevelPool.empty()
            if dom.lakes is not None:
                L = dom.lakes
                lakes = levelpool.LevelPool(
                    water_elevation=L["resht"], lake_opt=L["lake_opt"],
                    **{k: L[k] for k in levelpool._PROPS})
            self.channel = diffusive.DiffusiveWaveRouting(
                ch_netrt=dom.ch_netrt, lake_mskrt=lake_mask, ch_netlnk=net["ch_netlnk"],
                chanxi=net["chanxi"], chanyj=net["chanyj"], node_area=net["node_area"],
                from_node=net["from_node"],
                # pour points keep READ_CHROUTING1's -9999, which only TYPEL = 0
                # links read; the port's convention for "none" is 0
                to_node=np.where(net["to_node"] < 0, 0, net["to_node"]), typel=net["typel"],
                zelev=net["zelev"], chanlen=net["chanlen"], mannn=net["mannn"], so=net["so"],
                chsslp=net["chsslp"], bw=net["bw"], retdep_chan=F(0.001),
                dtrt_ch=self.dtrt_ch, dtct=F(cfg.dtrt_ch), gwbaseswcrt=0,
                lakenode=net["lakenode"], lakes=lakes)
            n = len(net["chanxi"])
            self.hlink, self.cvol = d(net["hlink"]), d(net["cvol"])
            self.qlink1, self.qlateral = cp.zeros(n, cp.float32), cp.zeros(n, cp.float32)
        self._boundary_prev = None
        self.budget_log = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if self.t0_output:
                self._write_outputs()
            self.out_counts = 1

    # -- the coupling call ---------------------------------------------------
    def exchange(self, *, smois, sh2o, infxsrt, soldrain, sfcheadrt) -> None:
        """One wrf_cpl_HYDRO call; ``smois``/``sh2o`` (nsoil, jx, ix) and
        ``sfcheadrt`` (jx, ix) are written in place."""
        import cupy as cp

        cp.copyto(self.smc, smois)
        cp.copyto(self.sh2ox, sh2o)
        cp.copyto(self.infxsrt, infxsrt)
        cp.copyto(self.soldrain, soldrain)
        before = self._budget_state() if self.budget_log is not None else None
        self._hydro_exe()
        cp.copyto(smois, self.smc)
        cp.copyto(sh2o, self.sh2ox)
        cp.copyto(sfcheadrt, self.head_lsm)
        if before is not None:
            self.budget_log.append(self._budget_terms(before))
        self.soldrain.fill(0)            # HYDRO_exe :746-747
        self.sstate.end_hydro_step()

    def _hydro_exe(self) -> None:
        import cupy as cp

        cfg, s, o = self.cfg, self.static, self.ostate
        o.to_channel.fill(0)
        lake_dum = o.to_lake.copy()
        if self.routing:
            grid_transform.disaggregate(
                agg=self.dom.aggfactrt, iswater=self.dom.iswater, sldpth=self.dom.sldpth,
                sice=self.sice, smc=self.smc, sh2ox=self.sh2ox, infxsrt=self.infxsrt,
                area_lsm=s["area_lsm"], smcmax1=s["smcmax1"], smcref1=s["smcref1"],
                smcwlt1=s["smcwlt1"], vegtyp=s["vegtyp"], lksat=s["lksat"], nexp=s["nexp"],
                dist9=s["dist9"], infxswgt=self.infxswgt, lksatfac=s["lksatfac"],
                ch_netrt=s["ch_netrt"], sh2owgt=self.sh2owgt, elrt=s["elrt"],
                soiltyp=s["soiltyp"], lake_mskrt=self.lake_mask, out=self.gt_out)
        infx = self.gt_out["infxsubrt"]
        if self.sub is not None:
            g = self.gt_out
            self.sub.route(self.sstate, smc=g["smcrt"], smcmax=g["smcmaxrt"],
                           smcref=g["smcrefrt"], smcwlt=g["smcwltrt"], infiltration_excess=infx)
        if self.ov is not None:
            self.ov.route(o, infx, self.qx, self.qy, dt=self.dt, dtrt_ter=self.dtrt_ter)
            infx.fill(0)
        else:
            if self.sub is not None:
                ix, jx = self.dom.ix, self.dom.jx
                _k("hyd_runoff_aggregate")((((ix * jx) + 127) // 128,), (128,), (
                    infx, self.infxsrt, np.int32(ix), np.int32(jx),
                    np.int32(self.dom.aggfactrt)))
            infx.fill(0)
            o.head.fill(0)
        cp.copyto(self.qstrm_ts, o.to_channel)
        self.qstrm_acc += self.qstrm_ts
        cp.subtract(o.to_lake, lake_dum, out=self.lake_ts)
        if self.bucket is not None:
            self.bucket.step(z=self.z_gw, qin=self.qin_gw, qout=self.qout_gw,
                             qinflowbase=self.qinflowbase, infxsrt=self.infxsrt,
                             soldrain=self.soldrain, dt=self.dt,
                             ovrtswcrt=int(bool(cfg.overland)))
        if self.channel is not None:
            self.channel.route(dt=self.dt, qstrmvolrt=self.qstrm_ts,
                               qinflowbase=self.qinflowbase, hlink=self.hlink, cvol=self.cvol,
                               qlink1=self.qlink1, qlateral=self.qlateral,
                               lakeinflort=self.lake_ts)
        if self.routing:
            g = self.gt_out
            grid_transform.aggregate(
                agg=self.dom.aggfactrt, head_rt=o.head, dist9=s["dist9"], smcrt=g["smcrt"],
                smcmaxrt=g["smcmaxrt"], sice=self.sice, head_lsm=self.head_lsm,
                sh2ox=self.sh2ox, smc=self.smc, infxswgt=self.infxswgt, sh2owgt=self.sh2owgt)
        # HYDRO_time_adv and HYDRO_out
        self.elapsed += int(round(float(self.dt)))
        if self.output_dir is not None and self.elapsed == self._next_output():
            self._write_outputs()
            self.out_counts += 1

    def _next_output(self):
        step = int(round(float(self.dt))) if float(self.dt) > self.out_dt else self.out_dt
        return step * self.out_counts

    # -- state for digests and the water budget -----------------------------
    def digest_arrays(self) -> dict:
        out = {"wrf_hydro/elapsed": np.asarray(self.elapsed)}
        named = dict(smc=self.smc, sh2ox=self.sh2ox, head_lsm=self.head_lsm,
                     sh2owgt=self.sh2owgt, infxswgt=self.infxswgt, head_rt=self.ostate.head,
                     to_lake=self.ostate.to_lake, boundary_flux=self.ostate.boundary_flux,
                     zwattablrt=self.sstate.zwattablrt, qsubbdryrt=self.sstate.qsubbdryrt,
                     qstrm_acc=self.qstrm_acc, qinflowbase=self.qinflowbase)
        if self.bucket is not None:
            named.update(z_gw=self.z_gw, qin_gw=self.qin_gw, qout_gw=self.qout_gw)
        if self.channel is not None:
            named.update(hlink=self.hlink, cvol=self.cvol, qlink1=self.qlink1,
                         qlateral=self.qlateral, dtct=self.channel.dtct,
                         resht=self.channel.lakes.resht, qlakeo=self.channel.lakes.qlakeo)
        out.update({f"wrf_hydro/{k}": v for k, v in named.items()})
        return out

    def _sum(self, a, area=None):
        import cupy as cp

        a = a.astype(cp.float64)
        if area is not None:
            a = a * area.astype(cp.float64)
        return float(a.sum())

    def _budget_state(self):
        """Water on both grids in m3 (float64 sums), before HYDRO_exe."""
        import cupy as cp

        dz = cp.asarray(self.dom.sldpth.astype(np.float64)).reshape(-1, 1, 1)
        area = self.static["area_lsm"]
        soil = self._sum((self.smc.astype(cp.float64) * dz).sum(axis=0), area)
        st = dict(soil=soil, infxsrt=self._sum(self.infxsrt, area) / 1000.0,
                  soldrain=self._sum(self.soldrain, area) / 1000.0,
                  boundary_ov=self._sum(self.ostate.boundary_flux, self.static["dist9"]) / 1000.0,
                  boundary_sub=self._sum(self.sstate.qsubbdryrt) * float(self.dt),
                  to_lake=self._sum(self.ostate.to_lake, self.static["dist9"]) / 1000.0)
        if self.bucket is not None:
            # a bucket holds z (mm) over its basin's cells, ct * the mean cell area
            st["bucket"] = self._sum(self.z_gw * self.bucket_cells, self.bucket.area) / 1000.0
        if self.channel is not None:
            st["channel"] = self._sum(self.cvol)
        return st

    def _budget_terms(self, before):
        """One cycle's water budget: what the LSM handed over (soil water,
        ponded INFXSRT, SOLDRAIN) against what came back (soil water, SFCHEADRT)
        plus what left to the channel, lakes, domain edges and the bucket."""
        after = self._budget_state()
        area = self.static["area_lsm"]
        terms = dict(
            lsm_in=before["soil"] + before["infxsrt"],
            lsm_out=after["soil"] + self._sum(self.head_lsm, area) / 1000.0,
            to_channel=self._sum(self.qstrm_ts, self.static["dist9"]) / 1000.0,
            to_lake=after["to_lake"] - before["to_lake"],
            boundary_ov=-(after["boundary_ov"] - before["boundary_ov"]),
            boundary_sub=-(after["boundary_sub"] - before["boundary_sub"]),
            soldrain=before["soldrain"])
        if self.bucket is not None:
            terms["bucket_in"] = self._sum(self.qin_gw) * float(self.dt)
            terms["bucket_out"] = self._sum(self.qout_gw) * float(self.dt)
            terms["bucket_change"] = after["bucket"] - before["bucket"]
            terms["qinflowbase"] = self._sum(self.qinflowbase, self.static["dist9"]) / 1000.0
        if self.channel is not None:
            terms["channel_change"] = after["channel"] - before["channel"]
        return terms

    # -- output --------------------------------------------------------------
    def history_planes(self) -> dict:
        """Routing-grid planes for the RTOUT-like file (host copies)."""
        return dict(zwattablrt=self.sstate.zwattablrt.get(), sfcheadsubrt=self.ostate.head.get(),
                    QSTRMVOLRT=self.qstrm_acc.get(), SOIL_M=self.gt_out["smcrt"].get())

    def streamflow(self):
        """(frxst id, lon, lat, q m3/s, head m) at every forecast point, in node order."""
        if self.channel is None:
            return []
        net = self.dom.network
        q, h = self.qlink1.get(), self.hlink.get()
        return [(int(net["frxst"][k]), float(net["chlon"][k]), float(net["chlat"][k]),
                 float(q[k]), float(h[k])) for k in range(len(q)) if net["frxst"][k] != -9999]

    def _write_outputs(self):
        when = self.start + timedelta(seconds=self.elapsed)
        stamp = when.strftime("%Y%m%d%H%M")
        if self.channel is not None and self.flags["frxst_pts_out"] == 1:
            hydro_time = when.strftime("%Y-%m-%d_%H:%M:%S")
            with (self.output_dir / "frxst_pts_out.txt").open("a") as fh:
                for pid, lon, lat, q, h in self.streamflow():
                    # output_chrt's FORMAT 117
                    fh.write(f"{self.elapsed:8d},{hydro_time[:10]} {hydro_time[11:19]},"
                             f"{pid:7d},{lon:10.5f},{lat:8.5f},{q:9.3f},"
                             f"{q * 35.314666711511576:12.3f},{h:6.3f}\n")
        if self.channel is not None and self.flags["chrtout_domain"] == 1:
            self._write_chrtout(self.output_dir / f"{stamp}.CHRTOUT_DOMAIN{self.igrid}", when)
        if self.routing and self.flags["rtout_domain"] == 1:
            self._write_rtout(self.output_dir / f"{stamp}.RTOUT_DOMAIN{self.igrid}", when)

    def _write_chrtout(self, path, when):
        import netCDF4

        net = self.dom.network
        with netCDF4.Dataset(str(path), "w") as ds:
            ds.createDimension("station", len(net["chanxi"]))
            ds.createDimension("time", 1)
            ds.model_output_valid_time = when.strftime("%Y-%m-%d_%H:%M:%S")
            ds.title = "CHRTOUT-like channel output (WOOF's WRF-Hydro opt-in, channel_option 3)"
            t = ds.createVariable("time", "i4", ("time",))
            t.units = f"seconds since {self.start:%Y-%m-%d %H:%M:%S}"
            t[:] = self.elapsed
            for name, data, dt, units in (
                    ("streamflow", self.qlink1.get(), "f4", "m3 s-1"),
                    ("head", self.hlink.get(), "f4", "m"),
                    ("order", net["order"], "i4", "1"),
                    ("latitude", net["chlat"], "f4", "degrees_north"),
                    ("longitude", net["chlon"], "f4", "degrees_east"),
                    ("frxst_pts", net["frxst"], "i4", "1")):
                v = ds.createVariable(name, dt, ("station",))
                v.units = units
                v[:] = data

    def _write_rtout(self, path, when):
        import netCDF4

        planes = self.history_planes()
        jxrt, ixrt = self.rt_shape
        with netCDF4.Dataset(str(path), "w") as ds:
            ds.createDimension("time", 1)
            ds.createDimension("y", jxrt)
            ds.createDimension("x", ixrt)
            ds.createDimension("depth", self.dom.nsoil)
            ds.model_output_valid_time = when.strftime("%Y-%m-%d_%H:%M:%S")
            ds.title = "RTOUT-like routing-grid output (WOOF's WRF-Hydro opt-in)"
            t = ds.createVariable("time", "i4", ("time",))
            t.units = f"seconds since {self.start:%Y-%m-%d %H:%M:%S}"
            t[:] = self.elapsed
            # north-up rows, as Fulldom_hires.nc and WRF-Hydro's RTOUT files are
            for name, units in (("zwattablrt", "m"), ("sfcheadsubrt", "mm"),
                                ("QSTRMVOLRT", "mm")):
                v = ds.createVariable(name, "f4", ("time", "y", "x"))
                v.units = units
                v[0] = planes[name][::-1]
            v = ds.createVariable("SOIL_M", "f4", ("time", "depth", "y", "x"))
            v.units = "m3 m-3"
            v[0] = planes["SOIL_M"][:, ::-1]


def start_time(value) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))

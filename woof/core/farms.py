"""GPU column interface to public WRF 4.7.1 Farms_driver.

All arrays remain on the device. The last model level is excluded from
the path integrals, exactly as Integrate_1var/Integrate_2var do in WRF.
"""
from __future__ import annotations
import os
import numpy as np

FARMS_FIELDS = ("swdown2", "swddir2", "swddni2", "swddif2", "swdownc2", "swddnic2")
#: ``WOOF_FARMS_FUSED=0`` keeps the per-step glue (host cloud-fraction
#: launch, MYNN merge ufuncs, radius multiplies, output copies) for A/B and
#: byte-identity checks; the fused path is one launch.
FARMS_FUSED_ENABLED = os.environ.get("WOOF_FARMS_FUSED", "1") != "0"


def farms_surface_columns(p8w, rho, dz, albedo, ssa, asymmetry, aod550,
                          angstrom, coszen, qv, qi, qs, qc,
                          re_cloud, re_ice, re_snow, cldfra, *,
                          julian, has_reqc=0, has_reqi=0, has_reqs=0, out=None,
                          outs=None, re_scale=1.0, cloud=None, mynn=None,
                          cf_out=None):
    """Six surface fluxes for (nlev,ncol) columns, radii in metres.

    Inputs must be contiguous float32 CuPy buffers. p8w can include the
    final interface; its surface row is the only pressure row FARMS reads.
    The output can be a persistent driver allocation of shape (6,ncol), or
    ``outs``, six contiguous float32 (ncol,) arrays written directly.

    The fused options (kernels/farms.cu ``farms_cell_inputs``):
    ``re_scale`` multiplies the radii in the kernel (1.e-6 for microns, the
    same word as the host multiply); ``cloud=(fc, fi, fs, ferrier, t, p)``
    composes the cloud fraction per cell from the layer temperature and
    pressure instead of reading ``cldfra``; ``mynn=(replace, qc_bl, qi_bl,
    cldfra_bl)`` merges the carried MYNN cloud as
    :func:`woof.core.mynn_radiation.merge_mynn_bl_clouds`; ``cf_out`` takes
    the (merged) fraction at every level of every column.
    """
    import cupy as cp
    from woof.core.kernels import get_kernel

    if len(qv.shape) != 2 or qv.shape[0] < 2:
        raise ValueError("FARMS needs (nlev,ncol), at least two levels")
    nz, nc = qv.shape
    volume = [rho, dz, qv, qi, qs, qc, re_cloud, re_ice, re_snow, cldfra]
    surface = (albedo, ssa, asymmetry, aod550, angstrom, coszen)
    if cloud is not None:
        volume += [cloud[4], cloud[5]]
    if mynn is not None:
        volume += list(mynn[1:])
    if cf_out is not None:
        volume.append(cf_out)
    for value in (p8w, *volume, *surface):
        if (not isinstance(value, cp.ndarray) or value.dtype != cp.float32
                or not value.flags.c_contiguous):
            raise TypeError("FARMS inputs must be contiguous float32 device arrays")
    if any(value.shape != (nz, nc) for value in volume):
        raise ValueError("FARMS volume buffers must match qv.shape")
    if p8w.shape not in ((nz, nc), (nz + 1, nc)):
        raise ValueError("FARMS interface pressure has the wrong shape")
    if any(value.shape != (nc,) for value in surface):
        raise ValueError("FARMS surface buffers must have shape (ncol,)")
    flags = (has_reqc, has_reqi, has_reqs)
    if any(value not in (0, 1) for value in flags):
        raise ValueError("FARMS has_re* flags must be 0 or 1")
    if outs is None:
        if out is None:
            out = cp.empty((6, nc), dtype=cp.float32)
        if (not isinstance(out, cp.ndarray) or out.shape != (6, nc)
                or out.dtype != cp.float32 or not out.flags.c_contiguous):
            raise TypeError("FARMS output must be contiguous float32 (6,ncol)")
        outs = tuple(out[a] for a in range(6))
    elif out is not None:
        raise ValueError("FARMS takes out or outs, not both")
    if len(outs) != 6 or any(
            not isinstance(value, cp.ndarray) or value.shape != (nc,)
            or value.dtype != cp.float32 or not value.flags.c_contiguous
            for value in outs):
        raise TypeError("FARMS outs must be six contiguous float32 (ncol,) arrays")
    # CUDA arguments cannot be null: an unread pointer is a harmless stand-in.
    fc, fi, fs, ferrier, t, pres = (cloud if cloud is not None
                                    else (0, 0, 0, 0, qv, qv))
    replace, qc_bl, qi_bl, cldfra_bl = (mynn if mynn is not None
                                        else (0, qv, qv, qv))
    from woof.wrf_exact import ENABLED
    entry = "farms_column" if ENABLED else "farms_column_fast"
    grid = ((nc + 63) // 64,) if ENABLED else ((nc + 31) // 32,)
    block = (64,) if ENABLED else (32, 8)
    get_kernel("farms", entry)(
        grid, block,
        (np.int32(nc), np.int32(nz - 1), *(np.int32(value) for value in flags),
         np.float32(julian), p8w, rho, dz, albedo, ssa, asymmetry, aod550,
         angstrom, coszen, qv, qi, qs, qc, re_cloud, re_ice, re_snow, cldfra,
         *outs, np.float32(re_scale), np.int32(nz),
         np.int32(cloud is not None), np.int32(fc), np.int32(fi),
         np.int32(fs), np.int32(ferrier), t, pres,
         np.int32(mynn is not None), np.int32(replace), qc_bl, qi_bl,
         cldfra_bl, np.int32(cf_out is not None),
         cf_out if cf_out is not None else outs[0]))
    return out


from woof.core.swint import ShortwaveInterpolation


class FastSurfaceShortwave(ShortwaveInterpolation):
    """FARMS on every Solar_step, with this domain's transported geography.

    Fluxes live in driver.fields for history, checkpoint, nest transport
    and each ensemble member. Only cosine scratch is rebuilt on restart.
    """
    def every_step(self, driver, atmosphere, state, cfg):
        import cupy as cp
        from woof.core.rrtmg_legacy import legacy_scheme_has_req, legacy_cloud_fraction_flags
        from woof.core.swint import calendar_scalars
        nz, ny, nx = atmosphere['qv'].shape
        nc = ny * nx
        if FARMS_FUSED_ENABLED and self._fused_inputs(driver, atmosphere, state, cfg) is not None:
            return self._every_step_fused(driver, atmosphere, state, cfg)
        column = lambda value: cp.ascontiguousarray(value, dtype=cp.float32).reshape(-1, nc)
        surface = lambda value: cp.ascontiguousarray(value, dtype=cp.float32).reshape(nc)
        cf = cp.zeros((nz,ny,nx), cp.float32)
        qc, qi = atmosphere['qc'], atmosphere['qi']
        diagnosed = False
        if int(cfg.icloud) == 3 or cfg.cu_rad_feedback:
            from woof.core.cloud_fraction import prepare_radiation_clouds
            view = prepare_radiation_clouds(atmosphere, driver.fields, state, cfg)
            atmosphere = view.atmosphere
            qc, qi = atmosphere['qc'], atmosphere['qi']
            if view.fraction is not None:
                cf, diagnosed = view.fraction, True
        if not diagnosed and int(cfg.icloud) == 2:
            from woof.core.rrtmgp import cal_cldfra2
            fqi, _ = legacy_cloud_fraction_flags(int(cfg.mp_physics))
            cf = cal_cldfra2(column(qc), column(qi),
                f_qc=int(cfg.mp_physics)!=0, f_qi=fqi).reshape(nz,ny,nx)
        if int(cfg.icloud) == 1:
            from woof.core.uwpbl import uwpbl_radiation_cloud_fraction
            from woof.core.mynn_radiation import merge_mynn_bl_clouds, mynn_bl_cloud_active, wrf_itimestep
            if not diagnosed:
                if int(cfg.mp_physics) == 95:
                    # Eta carries its saturation ice weight as F_ICE_PHY.
                    uwpbl_radiation_cloud_fraction(atmosphere, cfg, cf, state=state)
                else:
                    from woof.core.kernels import get_kernel
                    fqi, fqs = legacy_cloud_fraction_flags(int(cfg.mp_physics))
                    values = tuple(cp.ascontiguousarray(atmosphere[key], dtype=cp.float32)
                                   for key in ('qv','qc','qi','qs','temperature','pressure'))
                    get_kernel('farms','farms_cloud_fraction')(
                        ((cf.size+127)//128,), (128,),
                        (*values, cf, np.int32(cf.size),
                         np.int32(bool(cfg.moist) and int(cfg.mp_physics)!=0),
                         np.int32(fqi), np.int32(fqs), np.int32(int(cfg.mp_physics) in (5,15))))
            if mynn_bl_cloud_active(cfg.bl_pbl_physics, cfg.icloud_bl):
                qc, qi, cf = merge_mynn_bl_clouds(qc.copy(), qi.copy(), cf,
                    qc_bl=driver.fields.get('qc_bl'), qi_bl=driver.fields.get('qi_bl'),
                    cldfra_bl=driver.fields.get('cldfra_bl'),
                    bl_pbl_physics=cfg.bl_pbl_physics, icloud_bl=cfg.icloud_bl,
                    itimestep=wrf_itimestep(state.elapsed_seconds,cfg.dt),
                    bl_mynn_version=cfg.bl_mynn_version,
                    temperature=atmosphere['temperature'])
        if int(cfg.shcu_physics) == 5:
            from woof.core.column_cumulus import merge_column_cumulus_clouds
            qc, qi, cf = merge_column_cumulus_clouds(
                qc, qi, cf, fields=driver.fields, cfg=cfg, columns=lambda value: value)
        # WRF Solar_step updates CLDFRA on every FARMS step, including
        # steps between full radiation calls. Publish that same fraction
        # for history and consumers holding the radiation cloud field.
        if driver.cldfra is not None:
            driver.cldfra[...] = cf
        if 'uw_cldfra' in driver.fields:
            driver.fields['uw_cldfra'][...] = cf
        flags = legacy_scheme_has_req(cfg.mp_physics, cfg.use_mp_re)
        zero = cp.zeros((nz,nc), cp.float32)
        radii = []
        for name, flag in zip(('effc','effi','effs'), flags):
            value = getattr(state, name, None)
            if flag and value is None:
                raise ValueError(f'FARMS microphysics radius state.{name} is missing')
            radii.append(column(value) * np.float32(1.e-6) if flag else zero)
        if cfg.aer_opt == 0:
            aer = cp.zeros(nc, cp.float32)
            ssa, asy, aod, ang = aer, aer, aer, aer
        elif cfg.aer_opt == 1:
            if driver.aodtot is None:
                raise ValueError('FARMS aer_opt=1 needs the radiation AODTOT producer')
            aod = surface(driver.aodtot)
            ssa, asy, ang = (cp.full(nc,np.float32(x),cp.float32) for x in
                             (cfg.aer_ssa_val,cfg.aer_asy_val,cfg.aer_angexp_val))
        elif cfg.aer_opt == 2 or (cfg.aer_opt == 3 and cfg.rrtmg_aer3_source == "wrf_461"):
            ssa, asy, aod, ang = (surface(driver.aerosol2d[name]) for name in
                                  ('AERSSA2D','AERASY2D','AOD5502D','ANGEXP2D'))
        else:
            raise ValueError('FARMS aer_opt=3 needs stock WRF gt_aod diagnostics')
        julian, _, _ = calendar_scalars(self.start_time,state.elapsed_seconds,generation=getattr(cfg,'solar_generation','wrf_480'))
        fluxes = farms_surface_columns(
            column(atmosphere['p_interface']), column(atmosphere['rho']),
            column(atmosphere['dz']), surface(driver.fields['albedo']),
            ssa, asy, aod, ang, self.coszen_loc(state.elapsed_seconds),
            column(atmosphere['qv']), column(qi), column(atmosphere['qs']),
            column(qc), *radii, column(cf), julian=julian,
            has_reqc=int(flags[0]), has_reqi=int(flags[1]), has_reqs=int(flags[2]))
        for name, value in zip(FARMS_FIELDS,fluxes):
            driver.fields[name][...] = value.reshape(ny,nx)
        if cfg.couple_farms:
            driver.fields['swdown'][...] = driver.fields['swdown2']
            from woof.core.radiation_carriers import CARRIER_SOURCE_RADIATION_SCHEME
            driver.carriers.declare('swdown', source=CARRIER_SOURCE_RADIATION_SCHEME,
                                    model_time=float(state.elapsed_seconds))

    # -- the one-launch path -------------------------------------------------

    def _fused_inputs(self, driver, atmosphere, state, cfg):
        """The fused launch's inputs, or None when this configuration or
        these arrays need the per-step glue above: the cloud fraction is
        the shared composition (icloud = 1, no Eta, no cumulus or
        shallow-cumulus cloud, no icloud = 3 producer) with at most the
        wrf_461 MYNN merge, and every array is a contiguous float32 device
        array of this domain's shape."""
        import cupy as cp
        from woof.core.mynn_radiation import mynn_bl_cloud_active
        from woof.core.rrtmg_legacy import legacy_scheme_has_req
        if int(cfg.icloud) != 1 or cfg.cu_rad_feedback or int(cfg.mp_physics) == 95:
            return None
        if int(getattr(cfg, 'shcu_physics', 0)) == 5:
            return None
        mynn = mynn_bl_cloud_active(cfg.bl_pbl_physics, cfg.icloud_bl)
        if mynn and getattr(cfg, 'bl_mynn_version', 'wrf_461') == 'gsd_41':
            return None
        shape = atmosphere['qv'].shape
        nz, ny, nx = shape

        def volume(value):
            return (isinstance(value, cp.ndarray) and value.dtype == cp.float32
                    and value.flags.c_contiguous and value.shape == shape)

        def plane(value):
            return (isinstance(value, cp.ndarray) and value.dtype == cp.float32
                    and value.flags.c_contiguous and value.shape == (ny, nx))

        keys = ('qv', 'qc', 'qi', 'qs', 'temperature', 'pressure', 'rho', 'dz')
        if any(key not in atmosphere or not volume(atmosphere[key]) for key in keys):
            return None
        p8w = atmosphere.get('p_interface')
        if (not isinstance(p8w, cp.ndarray) or p8w.dtype != cp.float32
                or not p8w.flags.c_contiguous or p8w.shape not in (shape, (nz + 1, ny, nx))):
            return None
        bl = None
        if mynn:
            bl = tuple(driver.fields.get(key) for key in ('qc_bl', 'qi_bl', 'cldfra_bl'))
            if any(value is None or not volume(value) for value in bl):
                return None
        flags = legacy_scheme_has_req(cfg.mp_physics, cfg.use_mp_re)
        radii = []
        for name, flag in zip(('effc', 'effi', 'effs'), flags):
            value = getattr(state, name, None) if flag else None
            if flag and (value is None or not volume(value)):
                return None
            radii.append(value)
        if not plane(driver.fields.get('albedo')):
            return None
        if any(not plane(driver.fields.get(name)) for name in FARMS_FIELDS):
            return None
        cf_out = driver.cldfra
        if cf_out is None and 'uw_cldfra' in driver.fields:
            cf_out = driver.fields['uw_cldfra']
        if cf_out is not None and not volume(cf_out):
            return None
        return dict(mynn=bl, radii=radii, flags=flags, p8w=p8w, cf_out=cf_out)

    def _aerosol_surface(self, driver, cfg, nc):
        """ssa, asy, aod, ang on (ncol,): the constant ones are built once."""
        import cupy as cp
        if cfg.aer_opt == 0:
            key = ('zero', nc)
            if getattr(self, '_aer_cache', None) is None or self._aer_cache[0] != key:
                aer = cp.zeros(nc, cp.float32)
                self._aer_cache = (key, (aer, aer, aer, aer))
            return self._aer_cache[1]
        if cfg.aer_opt == 1:
            if driver.aodtot is None:
                raise ValueError('FARMS aer_opt=1 needs the radiation AODTOT producer')
            key = ('const', nc, float(cfg.aer_ssa_val), float(cfg.aer_asy_val),
                   float(cfg.aer_angexp_val))
            if getattr(self, '_aer_cache', None) is None or self._aer_cache[0] != key:
                ssa, asy, ang = (cp.full(nc, np.float32(x), cp.float32) for x in
                                 (cfg.aer_ssa_val, cfg.aer_asy_val, cfg.aer_angexp_val))
                self._aer_cache = (key, (ssa, asy, ang))
            ssa, asy, ang = self._aer_cache[1]
            return ssa, asy, cp.ascontiguousarray(driver.aodtot, dtype=cp.float32).reshape(nc), ang
        if cfg.aer_opt == 2 or (cfg.aer_opt == 3 and cfg.rrtmg_aer3_source == "wrf_461"):
            return tuple(cp.ascontiguousarray(driver.aerosol2d[name], dtype=cp.float32).reshape(nc)
                         for name in ('AERSSA2D', 'AERASY2D', 'AOD5502D', 'ANGEXP2D'))
        raise ValueError('FARMS aer_opt=3 needs stock WRF gt_aod diagnostics')

    def _every_step_fused(self, driver, atmosphere, state, cfg):
        """FARMS in one launch: the cloud fraction composed per cell, the
        MYNN cloud merged in registers, the radii scaled in the kernel, the
        six fluxes written straight into the driver's fields and the merged
        fraction into ``driver.cldfra`` (the words of the per-step glue in
        :meth:`every_step`)."""
        from woof.core.mynn_radiation import wrf_itimestep
        from woof.core.rrtmg_legacy import legacy_cloud_fraction_flags
        from woof.core.swint import calendar_scalars
        inputs = self._fused_inputs(driver, atmosphere, state, cfg)
        nz, ny, nx = atmosphere['qv'].shape
        nc = ny * nx
        column = lambda value: value.reshape(-1, nc)
        fqi, fqs = legacy_cloud_fraction_flags(int(cfg.mp_physics))
        cloud = (int(bool(cfg.moist) and int(cfg.mp_physics) != 0), int(fqi), int(fqs),
                 int(int(cfg.mp_physics) in (5, 15)),
                 column(atmosphere['temperature']), column(atmosphere['pressure']))
        mynn = None
        if inputs['mynn'] is not None:
            # WRF applies the mass merge on timestep one too; the fraction
            # replacement waits for a carried previous interval.
            replace = wrf_itimestep(state.elapsed_seconds, cfg.dt) != 1
            mynn = (int(replace), *(column(value) for value in inputs['mynn']))
        flags = inputs['flags']
        radii = tuple(column(value) if value is not None else column(atmosphere['qv'])
                      for value in inputs['radii'])
        ssa, asy, aod, ang = self._aerosol_surface(driver, cfg, nc)
        julian, _, _ = calendar_scalars(self.start_time, state.elapsed_seconds)
        cf_out = inputs['cf_out']
        farms_surface_columns(
            column(inputs['p8w']), column(atmosphere['rho']),
            column(atmosphere['dz']), driver.fields['albedo'].reshape(nc),
            ssa, asy, aod, ang, self.coszen_loc(state.elapsed_seconds),
            column(atmosphere['qv']), column(atmosphere['qi']), column(atmosphere['qs']),
            column(atmosphere['qc']), *radii, column(atmosphere['qv']), julian=julian,
            has_reqc=int(flags[0]), has_reqi=int(flags[1]), has_reqs=int(flags[2]),
            outs=tuple(driver.fields[name].reshape(nc) for name in FARMS_FIELDS),
            re_scale=1.e-6, cloud=cloud, mynn=mynn,
            cf_out=column(cf_out) if cf_out is not None else None)
        if cf_out is not None and driver.cldfra is not None and 'uw_cldfra' in driver.fields:
            driver.fields['uw_cldfra'][...] = driver.cldfra
        if cfg.couple_farms:
            driver.fields['swdown'][...] = driver.fields['swdown2']
            from woof.core.radiation_carriers import CARRIER_SOURCE_RADIATION_SCHEME
            driver.carriers.declare('swdown', source=CARRIER_SOURCE_RADIATION_SCHEME,
                                    model_time=float(state.elapsed_seconds))

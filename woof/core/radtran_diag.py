"""WRF-side carrier of the woof.radtran spectral diagnostic (RunConfig radtran_*).

Output-only.  At the configured cadence (the first step, then every step
that crosses a multiple of ``radtran_interval_s``, the hail-diagnostic rule)
the physics driver hands this carrier the time-t atmosphere it prepared for
radiation; the carrier builds woof.radtran columns on the model's interface
levels (woof.radtran.model), evaluates the sun at the step's own time
(woof.core.swint.calendar_scalars + woof.core.solar.solar_geometry_device,
the per-step zenith cosine WRF's swint uses), solves on the device and
writes the physics fields RT_EDIR, RT_EDN, RT_EUP, RT_EUP_TOA (and RT_RAD).
No prognostic or other diagnostic word changes.

Restart: rebuilt by initialize_physics from the resumed RunConfig and the
domain's latitude, longitude and start time; its outputs live in
``fields`` and are serialized with them (woof/io/restart.py).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from woof.radtran import model


class RadtranRun:
    def __init__(self, cfg, *, start_time, latitude_deg, longitude_deg, shape, fields):
        self.diag = model.diagnostic_from_config(cfg)
        model.validate(self.diag)
        self.start_time = start_time
        self.latitude = np.asarray(np.broadcast_to(latitude_deg, shape), dtype=np.float32)
        self.longitude = np.asarray(np.broadcast_to(longitude_deg, shape), dtype=np.float32)
        self.shape = tuple(shape)
        self.generation = None
        self.calls = 0
        # the axis variables are constants of the run
        import cupy as cp
        centre, lo, hi = model.band_table(self.diag)
        fields["rt_wavelength"][...] = cp.asarray(np.asarray(self.diag.wavelengths, dtype=np.float64))
        fields["rt_band_min"][...] = cp.asarray(lo)
        fields["rt_band_max"][...] = cp.asarray(hi)
        if self.diag.umu:
            dirs = self.diag.directions()
            fields["rt_umu"][...] = cp.asarray(np.array([d[0] for d in dirs]))
            fields["rt_phi"][...] = cp.asarray(np.array([d[1] for d in dirs]))

    def due(self, state, cfg) -> bool:
        epoch = float(getattr(state, "domain_start_offset", 0.0) or 0.0)
        return model.due(float(state.elapsed_seconds) - epoch, float(cfg.dt), self.diag.interval_s)

    def coszen(self, elapsed_seconds: float, generation: str) -> np.ndarray:
        import cupy as cp
        from woof.core.solar import solar_geometry_device
        from woof.core.swint import calendar_scalars
        julian, xtime, gmt = calendar_scalars(self.start_time, elapsed_seconds)
        cz = solar_geometry_device(julian, xtime, gmt, cp.asarray(self.latitude), cp.asarray(self.longitude),
                                   generation=generation)[1]
        return np.asarray(cp.asnumpy(cz), dtype=np.float64)

    def run(self, driver, state, cfg, atmosphere) -> None:
        import cupy as cp
        elapsed = float(state.elapsed_seconds)
        valid = self.start_time + timedelta(seconds=elapsed)
        day = int(valid.timetuple().tm_yday)
        mu0 = self.coszen(elapsed, getattr(cfg, "solar_generation", "wrf_480")).reshape(-1)
        nz = int(atmosphere["temperature"].shape[0])
        cols = lambda a, n: np.asarray(cp.asnumpy(a), dtype=np.float64).reshape(n, -1).T
        f = driver.fields
        alb_name = "albsol" if int(getattr(cfg, "alb_sol", 0) or 0) == 1 and "albsol" in f else "albedo"
        albedo = np.clip(np.asarray(cp.asnumpy(f[alb_name]), dtype=np.float64).reshape(-1), 0.0, 1.0)
        qv = cols(atmosphere["qv"], nz)
        rho_dry = cols(atmosphere["rho"], nz) / (1.0 + qv)
        eff = lambda name: (None if getattr(state, name, None) is None else cols(getattr(state, name), nz))
        out = model.compute(
            self.diag, z_w_m=cols(atmosphere["z_interface"], nz + 1), p_w_pa=cols(atmosphere["p_interface"], nz + 1),
            t_mass=cols(atmosphere["temperature"], nz), qv_mass=qv, umu0=mu0, albedo=albedo,
            qc_mass=cols(atmosphere["qc"], nz), qi_mass=cols(atmosphere["qi"], nz), rho_dry=rho_dry,
            re_cloud_um=eff("effc"), re_ice_um=eff("effi"), day_of_year=day)
        ny, nx = self.shape
        for name, key in (("rt_edir", "edir"), ("rt_edn", "edn"), ("rt_eup", "eup"), ("rt_eup_toa", "eup_toa")):
            f[name][...] = cp.asarray(out[key].reshape(-1, ny, nx))
        if "rad" in out:
            f["rt_rad"][...] = cp.asarray(out["rad"].reshape(out["rad"].shape[0], out["rad"].shape[1], ny, nx))
        self.calls += 1

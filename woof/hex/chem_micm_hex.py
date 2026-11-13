"""MICM gas-phase chemistry on the hex WOOF route: a chemistry config's ``micm`` block (CheMPAS-A's coupling).

``--chemistry-config FILE`` whose ``gpuwm-hex.chemistry/v1`` document carries a ``micm`` block runs a
MechanismConfiguration mechanism through WOOF's MICM port (woof.chem_micm) on every hex grid point, coupled
the way NCAR CheMPAS-A couples MUSICA to MPAS-Atmosphere (v2026.08.01, mpas_musica.F / mpas_atm_chemistry.F):

* every state species of the mechanism is an appended hex scalar holding a dry-air mass mixing ratio
  (kg kg-1), transported by the hex monotonic scalar FCT; its name is the species name in lower case;
* after each step's microphysics (or every ``chemistry_interval`` seconds, on the accumulated time) each
  level of each cell is one MICM grid point: concentration = max(q, 0) * rho_dry / M (mol m-3, M the
  species' molecular weight), temperature and pressure from the post-step columns, air density
  P / (R T) with R = 8.314462618 J mol-1 K-1; after the solve q = c M / rho_dry;
* photolysis rates are CheMPAS-A's no-TUV-x form, ``j_max * max(0, cos(SZA))`` per rate with its solar
  geometry (mpas_solar_geometry.F, the NOAA declination and equation of time at each cell's latitude and
  longitude), unless the block sets ``"photolysis": "tuvx"`` (below); emission, first-order-loss and
  user-defined rate parameters are constants from the block.

``"photolysis": "tuvx"`` takes the rates from WOOF's TUV-x port (woof.chem_tuvx, graded against NCAR TUV-x
v0.17.0) instead, with a ``"tuvx"`` block: ``config`` (a TUV-x configuration, e.g. the repository's
tools/tuvx_oracle/photolysis.json; relative paths are against the chemistry config), ``reactions`` (each MICM
photolysis rate's short name -> a TUV-x reaction name of that configuration) and optional ``data`` (a TUV-x data
root; by default the pinned files are fetched on demand, woof.chem_tuvx_data).  At the start of each chemistry
interval every cell is one TUV-x column on the host: the model's interface heights and edge temperatures
(layer temperatures, the mean of neighbours inside, the nearest layer at the ends), air from P / (k T), O2 as
0.2095 of air, O3 from the mechanism's O3, and above the model top (to 120 km, ~3 km layers) the US Standard
Atmosphere 1976 temperature, air falling off with a 7 km scale height from the top layer, and a fixed ozone
profile; the solar zenith angle from the same solar geometry (cos SZA unclamped) and TUV-x's Earth-Sun distance
factor for the date.  Each layer's rate is the mean of its two edges'.  The calculation runs in binary64 numpy
on the host, in batches of columns: about 15 s per thousand cells of ~90 TUV-x layers on an M3 Max, once per
chemistry interval, so large meshes want a long ``chemistry_interval`` (a device port is future work).

Every grid point is its own solve, so (unlike CheMPAS-A, which hands MICM all of a rank's points as one
state with one shared step size) a column's chemistry does not depend on the decomposition.  Keys follow
CheMPAS-A's namelist: ``micm_file``, ``chemistry_interval``, ``relative_tolerance``,
``absolute_tolerance``, plus ``photolysis`` (j_max per rate, or ``"tuvx"``), ``tuvx`` and ``rate_parameters``.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np

MODE = "micm"
#: CheMPAS-A mpas_chem_constants.F CHEM_GAS_R.
GAS_R = 8.314462618
KEYS = frozenset(("micm_file", "chemistry_interval", "relative_tolerance", "absolute_tolerance",
                  "photolysis", "rate_parameters", "initial_mass_mixing_ratio", "tuvx"))
TUVX_KEYS = frozenset(("config", "reactions", "data"))
#: Species molecular-weight keys: MechanismConfiguration's, then CheMPAS-A's comment property.
MOLAR_MASS_KEYS = ("molecular weight [kg mol-1]", "__molar mass")
_UNITS = {}


def _number(value, what, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"micm {what} must be a finite number, got {value!r}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"micm {what} must be {'positive' if positive else 'nonnegative'}, got {value!r}")
    return float(value)


def load(path, raw):
    """(mechanism, rows, settings) for a micm config at ``path``, refusing what the hex route cannot run."""
    from woof.chem_micm import _read, load_mechanism
    block = raw["micm"]
    if not isinstance(block, dict):
        raise ValueError("micm must be an object")
    unknown = sorted(set(block) - KEYS)
    if unknown:
        raise ValueError(f"micm keys {unknown} are not carried; carried: {sorted(KEYS)}")
    if raw.get("inputs"):
        raise ValueError("micm starts from zero composition; an inputs block is not remapped onto MICM rows yet")
    if "micm_file" not in block:
        raise ValueError("micm needs micm_file, the MechanismConfiguration mechanism")
    mech_path = (Path(path).parent / block["micm_file"]).resolve()
    document = _read(mech_path)
    mech = load_mechanism(document)
    species = {row["name"]: row for row in document["species"]}
    rows = []
    for name in mech.species:
        row_name = name.lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", row_name):
            raise ValueError(f"micm species {name!r} has no hex scalar name (lower case must be an identifier)")
        mass = next((species[name][k] for k in MOLAR_MASS_KEYS if k in species[name]), None)
        if mass is None:
            raise ValueError(f"micm species {name!r} declares no molecular weight ({MOLAR_MASS_KEYS[0]!r}); "
                             "its mass mixing ratio cannot become a concentration")
        rows.append(SimpleNamespace(name=row_name, species=name, units="kg kg-1", phase="gas",
                                    molar_mass=_number(float(mass), f"{name} molecular weight", positive=True)))
    if len({r.name for r in rows}) != len(rows):
        raise ValueError("micm species names collide in lower case")
    photolysis = block.get("photolysis", {})
    constants = block.get("rate_parameters", {})
    settings = {"photolysis": {}, "constants": {}}
    tuvx = None
    if photolysis == "tuvx":
        tuvx = _load_tuvx(path, block.get("tuvx"))
        settings["tuvx"] = tuvx
        photolysis = tuvx["reactions"]
    elif "tuvx" in block:
        raise ValueError('micm tuvx is read only with "photolysis": "tuvx"')
    for label in mech.parameters:
        kind, _, short = label.partition(".")
        if kind == "PHOTO" and tuvx is not None:
            if short not in photolysis:
                raise ValueError(f"micm photolysis rate {short!r} names no TUV-x reaction in tuvx.reactions")
            settings["photolysis"][label] = photolysis[short]
        elif kind == "PHOTO":
            entry = photolysis.get(short)
            if not isinstance(entry, dict) or set(entry) != {"j_max"}:
                raise ValueError(f"micm photolysis rate {short!r} needs {{\"j_max\": s-1}}; TUV-x photolysis "
                                 "is not ported, so every rate takes CheMPAS-A's j_max * max(0, cos SZA)")
            settings["photolysis"][label] = _number(entry["j_max"], f"photolysis {short} j_max")
        else:
            if label not in constants:
                raise ValueError(f"micm rate parameter {label!r} has no value in rate_parameters")
            settings["constants"][label] = _number(constants[label], f"rate parameter {label}")
    extra = (set(photolysis) - {p.partition('.')[2] for p in settings["photolysis"]}) | (
        set(constants) - set(settings["constants"]))
    if extra:
        raise ValueError(f"micm names rate parameters the mechanism does not have: {sorted(extra)}")
    settings["interval"] = _number(block.get("chemistry_interval", 0.0), "chemistry_interval")
    settings["rtol"] = _number(block.get("relative_tolerance", 1.0e-6), "relative_tolerance", positive=True)
    settings["atol"] = _number(block.get("absolute_tolerance", 1.0e-3), "absolute_tolerance", positive=True)
    initial = block.get("initial_mass_mixing_ratio", {})
    names = {r.species for r in rows}
    if not isinstance(initial, dict) or set(initial) - names:
        raise ValueError(f"micm initial_mass_mixing_ratio names species the mechanism does not transport: "
                         f"{sorted(set(initial) - names) if isinstance(initial, dict) else initial!r}")
    settings["initial"] = {k: _number(v, f"initial mass mixing ratio of {k}") for k, v in initial.items()}
    settings["mechanism"] = str(mech_path)
    settings["mechanism_sha256"] = hashlib.sha256(mech_path.read_bytes()).hexdigest()
    if tuvx is not None and "O3" not in mech.species:
        raise ValueError("micm TUV-x photolysis takes its ozone from the mechanism, which has no O3 species")
    return mech, rows, settings


def _load_tuvx(path, block):
    """The ``tuvx`` block, checked against its TUV-x configuration's reactions."""
    if not isinstance(block, dict) or set(block) - TUVX_KEYS or not {"config", "reactions"} <= set(block):
        raise ValueError(f'micm "photolysis": "tuvx" needs a tuvx block with config and reactions '
                         f"(keys {sorted(TUVX_KEYS)})")
    config_path = (Path(path).parent / block["config"]).resolve()
    document = json.loads(config_path.read_text())
    known = [r["name"] for r in document["photolysis"]["reactions"]]
    reactions = block["reactions"]
    if not isinstance(reactions, dict) or not all(isinstance(v, str) for v in reactions.values()):
        raise ValueError("micm tuvx.reactions maps each photolysis rate's short name to a TUV-x reaction name")
    unknown = sorted(set(reactions.values()) - set(known))
    if unknown:
        raise ValueError(f"micm tuvx.reactions names {unknown}, not reactions of {config_path} ({known})")
    data = block.get("data")
    return {"config": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "reactions": dict(reactions),
            "data": None if data is None else str((Path(path).parent / data).resolve())}


def declare(path, raw):
    _mech, rows, _settings = load(path, raw)
    for r in rows:
        _UNITS[r.name] = "kg kg-1"
        _UNITS[r.name + "_column_mass"] = "kg m-2"
    return rows


def history_units(name):
    return _UNITS.get(name)


def history_metadata(path, raw, digest):
    mech, rows, settings = load(path, raw)
    return json.dumps({"schema": "gpuwm-hex.chemistry/v1", "mode": MODE, "config_sha256": digest,
                       "mechanism": settings["mechanism"], "mechanism_sha256": settings["mechanism_sha256"],
                       "rows": [{"name": r.name, "species": r.species, "units": r.units} for r in rows],
                       "solver": "MICM three-stage Rosenbrock, one grid point per solve (woof.chem_micm)",
                       "photolysis": ("TUV-x v0.17.0 (woof.chem_tuvx), config sha256 "
                                      + settings["tuvx"]["config_sha256"] if "tuvx" in settings else
                                      "j_max * max(0, cos SZA) (CheMPAS-A without TUV-x)")}, sort_keys=True)


def product_names(rows):
    return tuple(r.name + "_column_mass" for r in rows)


def cos_sza(lat_deg, lon_deg, day_of_year, hour_utc):
    """CheMPAS-A mpas_solar_geometry.F solar_cos_sza, on the device in binary64."""
    import cupy as cp
    gamma = 2.0 * math.pi * ((day_of_year - 1) + (hour_utc - 12.0) / 24.0) / 365.0
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2.0 * gamma) + 0.000907 * math.sin(2.0 * gamma)
            - 0.002697 * math.cos(3.0 * gamma) + 0.001480 * math.sin(3.0 * gamma))
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2.0 * gamma) - 0.040849 * math.sin(2.0 * gamma))
    deg = math.pi / 180.0
    hour_angle = deg * ((hour_utc * 60.0 + eqtime + 4.0 * lon_deg) / 4.0 - 180.0)
    lat = lat_deg * deg
    return cp.sin(lat) * math.sin(decl) + cp.cos(lat) * math.cos(decl) * cp.cos(hour_angle)


class HexMicmChemistry:
    """MICM on hex grid points (see module docstring)."""

    def __init__(self, path, raw, digest, *, nlev, ncells, offset, start_time, dt, lat_deg=None, lon_deg=None):
        import cupy as cp
        self.path, self.raw, self.digest = Path(path), raw, digest
        self.mech, self.rows, self.settings = load(path, raw)
        self.nlev, self.ncells, self.offset = int(nlev), int(ncells), int(offset)
        self.start_time, self.dt = start_time, float(dt)
        interval = self.settings["interval"]
        if interval and (interval < self.dt or not float(interval / self.dt).is_integer()):
            raise ValueError("micm chemistry_interval must be zero or a whole number of model steps")
        self.lat = self.lon = None
        if lat_deg is not None:
            self.set_geometry(lat_deg, lon_deg)
        self.molar_mass = cp.asarray([r.molar_mass for r in self.rows], dtype=cp.float64)
        self.accumulated = 0.0
        self.pending = None
        self.steps = 0
        self.driver = None
        self.provider_receipt = None
        self.last_status = None
        self._tuvx = None

    def set_geometry(self, lat_deg, lon_deg):
        import cupy as cp
        self.lat = cp.asarray(lat_deg, dtype=cp.float64).reshape(self.ncells)
        self.lon = cp.asarray(lon_deg, dtype=cp.float64).reshape(self.ncells)

    def _params(self, elapsed_end, chem_dt, columns=None, conc=None):
        """Rate parameters per grid point at the start of the chemistry interval being solved."""
        import cupy as cp
        from datetime import timedelta
        npt = self.nlev * self.ncells
        labels = self.mech.parameters
        prm = cp.zeros((max(len(labels), 1), self.nlev, self.ncells), dtype=cp.float64)
        when = self.start_time + timedelta(seconds=float(elapsed_end) - float(chem_dt))
        hour = when.hour + when.minute / 60.0 + when.second / 3600.0
        cosz = None
        tuvx_rates = None
        if "tuvx" in self.settings and any(label in self.settings["photolysis"] for label in labels):
            if self.lat is None:
                raise RuntimeError("micm photolysis needs the cell latitudes and longitudes (attach)")
            cosz = cos_sza(self.lat, self.lon, when.timetuple().tm_yday, hour)
            tuvx_rates = self.tuvx_rates(when, cosz, columns, conc)
        for i, label in enumerate(labels):
            if tuvx_rates is not None and label in self.settings["photolysis"]:
                prm[i] = cp.asarray(tuvx_rates[self.settings["photolysis"][label]])
            elif label in self.settings["photolysis"]:
                if cosz is None:
                    if self.lat is None:
                        raise RuntimeError("micm photolysis needs the cell latitudes and longitudes (attach)")
                    cosz = cp.maximum(cos_sza(self.lat, self.lon, when.timetuple().tm_yday, hour), 0.0)
                prm[i] = self.settings["photolysis"][label] * cosz
            else:
                prm[i] = self.settings["constants"][label]
        return cp.ascontiguousarray(prm.reshape(prm.shape[0], npt).T)

    def tuvx_core(self):
        if self._tuvx is None:
            from woof import chem_tuvx, chem_tuvx_data
            tuvx = self.settings["tuvx"]
            root = chem_tuvx_data.ensure(root=tuvx["data"])
            self._tuvx = chem_tuvx.Tuvx(json.loads(Path(tuvx["config"]).read_text()), root=root)
        return self._tuvx

    def tuvx_rates(self, when, cosz, columns, conc):
        """``{TUV-x reaction: rates (nlev, ncells), s-1}`` at layer mid-points (see the module docstring)."""
        import cupy as cp
        from woof import chem_tuvx
        if columns is None or conc is None:
            raise RuntimeError("micm TUV-x photolysis needs the step's columns and concentrations")
        n, nc = self.nlev, self.ncells
        host = lambda a: np.asarray(cp.asnumpy(a), dtype=np.float64)  # noqa: E731
        z = host(columns["z_interface"]).reshape(n + 1, nc)
        o3 = host(conc[:, self.mech.species.index("O3")]).reshape(n, nc)
        hour = when.hour + when.minute / 60.0 + when.second / 3600.0
        inputs = tuvx_columns(z, host(columns["temperature"]).reshape(n, nc),
                              host(columns["pressure"]).reshape(n, nc), o3)
        sza = np.degrees(np.arccos(np.clip(host(cosz).reshape(nc), -1.0, 1.0)))
        esd = chem_tuvx.earth_sun_distance(when.year, when.month, when.day, hour)
        core = self.tuvx_core()
        out = {name: np.empty((n, nc)) for name in core.reactions}
        for start in range(0, nc, TUVX_BATCH):
            part = slice(start, min(nc, start + TUVX_BATCH))
            rates = core.run(*(a[part] for a in inputs), sza[part], esd)
            for name, edge in rates.items():
                out[name][:, part] = (0.5 * (edge[:, :n] + edge[:, 1:n + 1])).T
        return out

    def columns(self, prepared, phase):
        return {"temperature": prepared.t_p, "pressure": prepared.pres_p, "rho_dry": prepared.rho_dry,
                "z_interface": prepared.z_p}

    def step(self, scalars, columns, *, dt, elapsed_seconds, physics_fields=None, microphysics=None):
        """Solve on the post-step columns (``temperature``, ``pressure``, ``rho_dry``, [level, cell])."""
        import cupy as cp
        from woof.chem_micm import solve
        self.pending = {"accumulated": self.accumulated}
        self.accumulated += float(dt)
        interval = self.settings["interval"] or float(dt)
        if self.accumulated + 1e-9 * interval < interval:
            return
        chem_dt, self.accumulated = self.accumulated, 0.0
        n, nc = self.nlev, self.ncells
        rows = scalars[self.offset:self.offset + len(self.rows)]
        if tuple(rows.shape) != (len(self.rows), n, nc):
            raise ValueError("micm rows and hex columns differ in shape")
        rho = columns["rho_dry"].astype(cp.float64).reshape(n * nc)
        T = columns["temperature"].astype(cp.float64).reshape(n * nc)
        P = columns["pressure"].astype(cp.float64).reshape(n * nc)
        q = cp.maximum(rows.astype(cp.float64).reshape(len(self.rows), n * nc), 0.0)
        conc = cp.ascontiguousarray((q * rho / self.molar_mass[:, None]).T)
        params = (self._params(elapsed_seconds, chem_dt, columns=columns, conc=conc)
                  if self.mech.parameters else None)
        out, status, _steps = solve(self.mech, conc, temperature=T, pressure=P, air_density=P / (GAS_R * T),
                                    params=params, time_step=chem_dt, rtol=self.settings["rtol"],
                                    atol=self.settings["atol"])
        bad = int(cp.count_nonzero(status != 2))
        if bad:
            raise FloatingPointError(f"MICM did not converge at {bad} hex grid points "
                                     f"(states {sorted(set(cp.asnumpy(status).tolist()))})")
        rows[...] = (out.T * self.molar_mass[:, None] / rho).reshape(len(self.rows), n, nc).astype(cp.float32)

    def commit(self):
        self.pending = None
        self.steps += 1

    def abort(self):
        if self.pending is not None:
            self.accumulated = self.pending["accumulated"]
        self.pending = None

    def products(self, scalars, columns):
        import cupy as cp
        rows = scalars[self.offset:self.offset + len(self.rows)].astype(cp.float64)
        layer = columns["rho_dry"].astype(cp.float64) * (columns["z_interface"][1:] - columns["z_interface"][:-1])
        return {r.name + "_column_mass": cp.sum(rows[i] * layer, axis=0).astype(cp.float32)
                for i, r in enumerate(self.rows)}

    def receipt(self):
        return {"schema": "gpuwm-hex.chemistry/v1", "mode": MODE, "config": str(self.path),
                "config_sha256": self.digest, "mechanism": self.settings["mechanism"],
                "mechanism_sha256": self.settings["mechanism_sha256"],
                "rows": [{"name": r.name, "species": r.species, "molar_mass": r.molar_mass} for r in self.rows],
                "products": list(product_names(self.rows)), "committed_steps": self.steps,
                "solver": "MICM three-stage Rosenbrock, one grid point per solve (woof.chem_micm)",
                "photolysis": self.settings["photolysis"], "rate_parameters": self.settings["constants"]}

    def restart_payload(self):
        return {"mode": MODE, "config_sha256": self.digest, "steps": self.steps,
                "accumulated": self.accumulated, "mechanism_sha256": self.settings["mechanism_sha256"]}

    def restore_payload(self, payload):
        if payload is None or payload.get("mode") != MODE or payload.get("config_sha256") != self.digest:
            raise ValueError("chemistry restart config identity differs")
        # The config names its mechanism by path: an edited mechanism file under an unchanged config would
        # resume checkpointed species under other reactions.  (A payload written before the digest was
        # recorded carries none.)
        recorded = payload.get("mechanism_sha256")
        if recorded is not None and recorded != self.settings["mechanism_sha256"]:
            raise ValueError("chemistry restart mechanism identity differs: the MICM mechanism file changed")
        self.steps = int(payload["steps"])
        self.accumulated = float(payload["accumulated"])


#: columns per TUV-x batch (host memory: about 30 MB per (column, edge, bin) array at 120 edges)
TUVX_BATCH = 128
#: physical constants for the TUV-x columns
BOLTZMANN = 1.380649e-23
AVOGADRO = 6.02214076e23
O2_FRACTION = 0.2095
#: US Standard Atmosphere 1976 temperature break points [km, K], extended to 120 km (above the model top)
_US76_Z = (0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 86.0, 100.0, 120.0)
_US76_T = (288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.87, 195.08, 360.0)
TUVX_TOP_KM = 120.0
TUVX_EXTENSION_KM = 3.0
AIR_SCALE_HEIGHT_KM = 7.0


def standard_ozone(z_km):
    """The fixed ozone profile above the model top [molecule cm-3]: a layer peaking at 23 km."""
    z = np.asarray(z_km, dtype=np.float64)
    return 4.5e12 * np.exp(-((z - 23.0) / 7.5) ** 2) + 5.0e11 * np.exp(-z / 4.0)


def tuvx_columns(z_m, temperature, pressure, o3_mol_m3):
    """(edges km, edge temperatures K, air, O2, O3 layer columns molecule cm-2), each (ncells, ...), from model
    columns [level, cell] (``z_m`` interfaces [m], the rest layers), completed above the model top."""
    zkm = (z_m / 1000.0).T
    t = temperature.T
    nc, n = t.shape
    dz_cm = np.diff(zkm, axis=1) * 1.0e5
    air = pressure.T / (BOLTZMANN * t) * 1.0e-6
    o3 = o3_mol_m3.T * AVOGADRO * 1.0e-6
    tedge = np.empty((nc, n + 1))
    tedge[:, 0], tedge[:, n] = t[:, 0], t[:, n - 1]
    tedge[:, 1:n] = 0.5 * (t[:, :-1] + t[:, 1:])
    top = zkm[:, n]
    next_ = int(np.ceil(float((TUVX_TOP_KM - top.min()) / TUVX_EXTENSION_KM))) if top.min() < TUVX_TOP_KM else 0
    if next_ <= 0:
        return zkm, tedge, air * dz_cm, O2_FRACTION * air * dz_cm, o3 * dz_cm
    frac = np.arange(1, next_ + 1) / next_
    ext = top[:, None] + (TUVX_TOP_KM - top)[:, None] * frac[None, :]
    edges = np.concatenate([zkm, ext], axis=1)
    ext_mid = 0.5 * (edges[:, n:-1] + edges[:, n + 1:])
    ext_dz = np.diff(edges[:, n:], axis=1) * 1.0e5
    top_mid = 0.5 * (zkm[:, n - 1] + zkm[:, n])
    ext_air = air[:, n - 1:n] * np.exp(-(ext_mid - top_mid[:, None]) / AIR_SCALE_HEIGHT_KM)
    t_all = np.concatenate([tedge, np.interp(ext, _US76_Z, _US76_T)], axis=1)
    air_layer = np.concatenate([air * dz_cm, ext_air * ext_dz], axis=1)
    o3_layer = np.concatenate([o3 * dz_cm, standard_ozone(ext_mid) * ext_dz], axis=1)
    return edges, t_all, air_layer, O2_FRACTION * air_layer, o3_layer


def product_slugs(receipt):
    return tuple("var:wrf_" + name for name in receipt["products"])


def build_column_backend(row, config, *, constructor, prep_geometry, kernel_cache, gwdo_static=None,
                         gwdo_kernel_cache=None, checkout=None, cell_area_m2=None, rollback_snapshot=True):
    from . import chem_backend
    path, raw, digest, species_row = config

    def runtime(seam):
        made = HexMicmChemistry(
            path, raw, digest, nlev=constructor.n_levels, ncells=constructor.n_columns,
            offset=len(species_row.species) - len(declare(path, raw)), start_time=seam._start_time,
            dt=float(seam._cfg.dt))
        made.species_row = species_row
        return made

    return chem_backend.build(runtime, constructor=constructor, prep_geometry=prep_geometry,
                              kernel_cache=kernel_cache, gwdo_static=gwdo_static,
                              gwdo_kernel_cache=gwdo_kernel_cache, checkout=checkout,
                              rollback_snapshot=rollback_snapshot)


def attach(backend, host, driver):
    """Bind the committed atmosphere and the mesh geometry; the rows start from zero (CheMPAS-A seeds
    its tracers from the mechanism's initial values only on request)."""
    import cupy as cp
    runtime = backend.chemistry
    if host.get("regional") is not None:
        raise ValueError("micm on a regional mesh needs chemistry lateral boundaries, which the hex route "
                         "does not drive yet")
    arrays = host["prepared"].mesh.arrays
    n = runtime.ncells
    runtime.set_geometry(np.degrees(np.asarray(arrays["latCell"], dtype=np.float64))[:n],
                         np.degrees(np.asarray(arrays["lonCell"], dtype=np.float64))[:n])
    runtime.driver = driver
    if runtime.steps == 0:
        seed(runtime, driver.atmosphere.state.scalars)


def seed(runtime, scalars):
    """Rows at the first step: zero, then the block's uniform initial mass mixing ratios."""
    import cupy as cp
    rows = scalars[runtime.offset:runtime.offset + len(runtime.rows)]
    if bool(cp.any(rows != 0)):
        raise ValueError("micm rows carry mass before the first step; initial composition comes from "
                         "initial_mass_mixing_ratio")
    for i, r in enumerate(runtime.rows):
        rows[i] = np.float32(runtime.settings["initial"].get(r.species, 0.0))

"""WRF's lightning diagnostics (lightning_option 1, 2, 3) on the MPAS-A v8.4.2 suite route.

A WOOF-only output; MPAS-A v8.4.2 has none.  The WOOF namelist group ``&diagnostics`` takes WRF's &physics lightning
selectors under MPAS style names (each absent = WRF's default)::

    config_lightning_option         integer  0 (off), 1 (PR92, maximum w), 2 (PR92, cloud-top height), 3 (LPI)
    config_lightning_dt             real     s; 0 = every step (WRF lightning_dt)
    config_lightning_start_seconds  real     s; first activation (WRF lightning_start_seconds)
    config_flashrate_factor         real     WRF flashrate_factor (1.)
    config_iccg_method              integer  WRF iccg_method 0..3 (0 = 2: Boccippio's CONUS climatology, else the
                                             prescribed ratio; 1 prescribed; 3 Price & Rind 1993 cold-cloud depth)
    config_iccg_prescribed_num      real     WRF iccg_prescribed_num (0.)
    config_iccg_prescribed_den      real     WRF iccg_prescribed_den (1.)
    config_cellcount_method         integer  WRF cellcount_method 0..2

The fields are ``diag`` fields [nCells] in the ``diagnostics`` stream, packaged by WRF's package names
(Registry.EM_COMMON ``ltng_crm_PR92w``, ``ltng_crm_PR92z``, ``ltng_lpi``) with WRF's names, units and text
(woof.lightning_contract.FIELD_METADATA)::

    ic_flashcount cg_flashcount  #      accumulated intra-cloud / cloud-to-ground flashes (the whole run, as WRF's)
    ic_flashrate  cg_flashrate   # s-1  the last activation's rates (WRF keeps them out of history; written here)
    lpi                          J kg-1 lightning potential index (option 3 only; WRF's unit text "m^2 s-2")

Option 3 is WRF's ``ltng_lpi`` package: it computes LPI only, and its flash fields stay 0, as WRF's do.  With the
option off nothing here is allocated, constructed or compiled, and the Registry, the default streams and the driver
receipt are the ones a run without this module has.

Nothing is re-derived.  The columns run through the WRF route's own launcher (woof.core.lightning.launch_lightning,
woof/core/kernels/lightning.cu: WRF module_lightning_driver.F, module_ltng_crmpr92.F, module_ltng_lpi.F and
module_ltng_iccg.F) and its activation clock (woof.core.lightning.next_activation), with one mesh cell as one WRF
column (``ny = 1``, ``nx = nCells``).  The mesh column is handed over the way woof.core.lightning.update_lightning
hands over a WRF column, every field with WRF's extra top memory level:

* ``t`` = ``theta * exner``, ``theta``, ``pi`` = ``exner``, the mid-layer heights
  ``z = (zgrid(k) + zgrid(k+1)) / 2``, ``dz`` = the layer depth (0 on the top memory level), ``w`` the state's
  ``w`` at the layer faces, ``ht`` = ``zgrid(1)``;
* ``qc qr qi qs qg`` from the run's scalars (zeros where the scheme carries none, as WRF's absent species);
* ``land`` = ``xland`` (WRF's 1 land / 2 water), ``lat``/``lon`` = ``latCell``/``lonCell`` in degrees on
  (-180, 180] (the IC:CG climatology box of iccg_method 0/2);
* options 1 and 2 read the 10 cm reflectivity of the step's end state, formed by the scheme's own radar operator
  (thompson_refl_start_v842.radar: MPAS module_mp_thompson calc_refl10cm, the kernel the microphysics' diagnostics
  step launches), top memory level -35 dBZ, as the WRF route forms refl_10cm on every lightning step.

Timing.  The WRF route folds lightning after each step's microphysics, on the step's end state, when the activation
clock (``ltngacttime``) falls in [step start, step start + dt]; the clock then advances by ``lightning_dt`` (every
step when 0).  The same here: :meth:`LightningModule.update` runs after every step (step start = (step - 1) * dt).
The flash counts accumulate for the whole run and the rates and LPI are the last activation's, as WRF's; they and
the clock ride the checkpoint's accumulator dict.

Mesh notes.  The convective-cell statistics of options 1 and 2 (WRF's per-level counts of reflectivity above 20 dBZ
and the patch's maximum w) reduce over every solve cell of the mesh at once: the mesh is one WRF patch, so
``cellcount_method`` 1 (patch) and 2 (domain) are the same reduction.  A multi-rank mesh is refused (the patch
reduction would be per rank).  The PR92 flash rate is per cell of the identified convective region, as WRF's is per
grid column (WRF's own convention, independent of the column's area).

Refused by name: ``config_lightning_option = 11`` (PR92 on a cumulus scheme's deep-cloud top: the route's Grell-Freitas
does not publish ``ktop_deep``); options 1 and 2 with microphysics whose radar operator this module does not carry
(only mp_thompson and mp_thompson_aerosols here; mp_wsm6's operator needs its native constants); option 3 without
graupel (mp_kessler, off); ``config_iccg_method = 4`` (held on the WRF route too: WRF's Registry allocates
ICCG_IN_NUM/DEN with one plane while module_ltng_iccg.F reads twelve); ``config_cldtop_adjustment`` (read only by
option 11).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields as _dc_fields
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from woof.lightning_contract import FIELD_METADATA, LIGHTNING_OPTIONS, validate_lightning

from .utils import Direction

GROUP = "diagnostics"
#: config name -> (LightningConfig attribute, kind, WRF default)
_OPTIONS = {
    "config_lightning_option": ("option", int, 0),
    "config_lightning_dt": ("lightning_dt", float, 0.0),
    "config_lightning_start_seconds": ("start_seconds", float, 0.0),
    "config_flashrate_factor": ("flashrate_factor", float, 1.0),
    "config_iccg_method": ("iccg_method", int, 0),
    "config_iccg_prescribed_num": ("iccg_prescribed_num", float, 0.0),
    "config_iccg_prescribed_den": ("iccg_prescribed_den", float, 1.0),
    "config_cellcount_method": ("cellcount_method", int, 0),
}
CLDTOP_OPTION = "config_cldtop_adjustment"
OPTIONS = (*_OPTIONS, CLDTOP_OPTION)
#: WRF's package per option (Registry.EM_COMMON), one Registry package each
PACKAGES = {option: LIGHTNING_OPTIONS[option]["package"] for option in (1, 2, 3)}
SUPPORTED = tuple(PACKAGES)
FLASH_NAMES = ("ic_flashcount", "cg_flashcount", "ic_flashrate", "cg_flashrate")
NAMES = (*FLASH_NAMES, "lpi")
#: MPAS config_microp_scheme -> WRF mp_physics (woof.lightning_contract's admission reads it)
MP_PHYSICS = {"mp_thompson": 8, "mp_thompson_aerosols": 28, "mp_wsm6": 6, "mp_kessler": 1}
#: the schemes whose radar operator options 1 and 2 read (thompson_refl_start_v842.SCHEMES)
REFLECTIVITY_SCHEMES = ("mp_thompson", "mp_thompson_aerosols")
#: the schemes that carry graupel for option 3 (woof.lightning_contract.GRAUPEL_PROVIDERS on this route)
GRAUPEL_SCHEMES = ("mp_thompson", "mp_thompson_aerosols", "mp_wsm6")
SPECIES = ("qc", "qr", "qi", "qs", "qg")
STATE_PREFIX = "lightning:"
ACTIVATION_KEY = STATE_PREFIX + "ltngacttime"


@dataclass(frozen=True)
class LightningConfig:
    option: int = 0
    lightning_dt: float = 0.0
    start_seconds: float = 0.0
    flashrate_factor: float = 1.0
    iccg_method: int = 0
    iccg_prescribed_num: float = 0.0
    iccg_prescribed_den: float = 1.0
    cellcount_method: int = 0

    @property
    def package(self) -> str:
        return PACKAGES[self.option]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(LIGHTNING_OPTIONS[self.option]["fields"])

    def receipt(self) -> dict:
        return {f.name: getattr(self, f.name) for f in _dc_fields(self)}


def _field(name: str) -> dict[str, Any]:
    meta = FIELD_METADATA[name]
    gates = PACKAGES[3] if name == "lpi" else ";".join(PACKAGES.values())
    return {"type": "real", "dimensions": ["nCells", "Time"],
            "attrs": {"units": meta["units"], "long_name": meta["description"], "wrf_name": name.upper()},
            "gates": [gates]}


FIELDS = {name: _field(name) for name in NAMES}


def field_definition(name: str) -> dict[str, Any]:
    row = FIELDS[name]
    return {"type": row["type"], "dimensions": list(row["dimensions"]),
            "attrs": dict(row["attrs"]), "gates": list(row["gates"])}


def _value(key: str, value: Any, kind: type) -> Any:
    if isinstance(value, bool):
        raise ValueError(f"{key} is {'an integer' if kind is int else 'real'}, not {value!r}")
    if kind is int:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{key} is an integer, not {value!r}")
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} is real and finite, not {value!r}")
    return float(value)


def resolve(groups: Mapping[str, Mapping[str, Any]]) -> LightningConfig | None:
    """The options from the namelist's &diagnostics group (None when lightning is off)."""
    supplied = {str(k).casefold(): v for k, v in groups.get(GROUP, {}).items()}
    values = {attr: _value(key, supplied[key], kind) if key in supplied else default
              for key, (attr, kind, default) in _OPTIONS.items()}
    option = values["option"]
    if option == 0:
        extra = [k for k in OPTIONS if k in supplied and k != "config_lightning_option"]
        if extra:
            raise ValueError(f"&{GROUP} {extra[0]} is set while config_lightning_option is 0; it would be read by nothing")
        return None
    if option == 11:
        raise ValueError(f"&{GROUP} config_lightning_option = 11 is not available on the MPAS-A v8.4.2 suite route: "
                         "WRF's PR92 on the cumulus scheme's deep-cloud top reads ktop_deep, which the route's "
                         "Grell-Freitas does not publish; use 1, 2 (PR92 on the resolved storms) or 3 (LPI)")
    if option not in SUPPORTED:
        raise ValueError(f"&{GROUP} config_lightning_option = {option} has no flash or LPI producer; use 0, 1, 2 or 3")
    if CLDTOP_OPTION in supplied:
        raise ValueError(f"&{GROUP} {CLDTOP_OPTION} is read only by config_lightning_option = 11, which this route "
                         "does not run; it would be read by nothing")
    config = LightningConfig(**values)
    # WRF's own admission (woof.lightning_contract.validate_lightning), the microphysics checked by check()
    probe = SimpleNamespace(lightning_option=option, moist=True, mp_physics=8, do_radar_ref=1, cu_physics=0,
                            lightning_dt=config.lightning_dt, lightning_start_seconds=config.start_seconds,
                            flashrate_factor=config.flashrate_factor, iccg_method=config.iccg_method,
                            iccg_prescribed_num=config.iccg_prescribed_num,
                            iccg_prescribed_den=config.iccg_prescribed_den,
                            cellcount_method=config.cellcount_method, cldtop_adjustment=0.0)
    try:
        validate_lightning(probe)
    except ValueError as error:
        raise ValueError(f"&{GROUP} lightning: {error}") from None
    if option != 3 and config.iccg_method == 4:      # validate_lightning holds it too; named here for the route
        raise ValueError(f"&{GROUP} config_iccg_method = 4 is held")
    if config.lightning_dt < 0:
        raise ValueError(f"&{GROUP} config_lightning_dt must be >= 0 (0: every step)")
    return config


def check(config: Any) -> None:
    """Refuse at configuration what the run's microphysics cannot feed."""
    ltng = getattr(config, "lightning_diagnostics", None)
    if ltng is None:
        return
    scheme = config.selectors.get("microphysics") if config.active("microphysics") else "off"
    if ltng.option in (1, 2) and scheme not in REFLECTIVITY_SCHEMES:
        raise ValueError(f"&{GROUP} config_lightning_option = {ltng.option} (PR92) identifies the storms by their 10 cm "
                         f"reflectivity, which this module forms with the radar operator of "
                         f"{', '.join(REFLECTIVITY_SCHEMES)}; this run's microphysics is {scheme!r} (option 3, LPI, "
                         f"runs on {', '.join(GRAUPEL_SCHEMES)})")
    if ltng.option == 3 and scheme not in GRAUPEL_SCHEMES:
        raise ValueError(f"&{GROUP} config_lightning_option = 3 (LPI) needs a graupel mass carrier; on this route "
                         f"{', '.join(GRAUPEL_SCHEMES)} carry one, and this run's microphysics is {scheme!r}")


def packages(config: Any) -> set[str]:
    ltng = getattr(config, "lightning_diagnostics", None)
    return {ltng.package} if ltng is not None else set()


def extend_registry(root, active: set[str]) -> None:
    """Add the option's fields to the ``diag`` struct and the ``diagnostics`` stream of a parsed Registry tree."""
    import xml.etree.ElementTree as ET
    struct = root.find("./var_struct[@name='diag']")
    stream = root.find("./streams/stream[@name='diagnostics']")
    for name, row in FIELDS.items():
        gates = [p for p in row["gates"][0].split(";") if p in active]
        if not gates:
            continue
        package = row["gates"][0]
        if struct.find(f"./var[@name='{name}']") is None:
            ET.SubElement(struct, "var", {
                "name": name, "type": "real", "dimensions": " ".join(row["dimensions"]),
                "units": row["attrs"]["units"], "description": row["attrs"]["long_name"],
                "default_value": "0.0", "packages": package})
        if stream is not None and stream.find(f"./var[@name='{name}']") is None:
            ET.SubElement(stream, "var", {"name": name, "packages": package})


class LightningModule:
    """Fold the flash rates/counts or LPI after every due step; publish them for a due frame.  ``owner`` is the
    route's ResidentOutputDiagnostics (its device pools, buffers and dycore)."""

    def __init__(self, owner, config: LightningConfig):
        self.owner, self.config = owner, config
        self._step = None

    def setup(self, utils):
        cp = self.cp = self.owner.cp
        dims = self.owner.session.layout.dimensions
        self.nc, self.nl = int(dims["nCellsSolve"]), int(dims["nVertLevels"])
        self.utils = utils
        if getattr(self.owner, "halo", None) is not None:
            raise ValueError(f"&{GROUP} config_lightning_option is not available on a multi-rank mesh: WRF's PR92 "
                             "convective-cell statistics reduce over the whole patch, here every rank's cells")
        options = getattr(self.owner.session.config, "execution_options", None) or {}
        self.scheme = str(options.get("config_microp_scheme", "")).strip()
        self.names = self.config.names
        self.carried = {n for n in self.names if utils.stream_inclusion_count(n, Direction.OUTPUT) > 0}
        self.dt = float(np.float32(self.owner.dycore.config.config_dt))
        self.interval = self.config.lightning_dt if self.config.lightning_dt > 0 else self.dt
        self.activation = float(np.float32(self.config.start_seconds))
        self.out = {name: cp.zeros((1, self.nc), dtype=cp.float32) for name in NAMES}
        self.activations = 0

    # -- inputs ----------------------------------------------------------------------------------------
    def _get(self, pool, name):
        return self.owner.authority(pool, name)

    def _levels(self, pool, name, levels):
        return self.cp.ascontiguousarray(self._get(pool, name)[:levels, :self.nc], dtype=self.cp.float32)

    def _plane(self, pool, name):
        return self.cp.ascontiguousarray(self._get(pool, name)[:self.nc], dtype=self.cp.float32)

    def _scalar(self, name):
        indices = self.owner.session.layout.indices
        key = ("state", "index_" + name)
        if key not in indices or int(indices[key]) < 1:
            return None
        row = int(indices[key]) - 1
        return self.cp.ascontiguousarray(self._get("state", "scalars")[row, :self.nl, :self.nc],
                                         dtype=self.cp.float32)

    def columns(self) -> dict[str, Any]:
        """The mesh columns as woof.core.lightning.update_lightning hands WRF's over: (nl + 1, 1, nCells) float32
        words with WRF's top memory level, planes (1, nCells)."""
        cp, nc, nl = self.cp, self.nc, self.nl
        f32 = np.float32

        def extend(value, top=None):
            value = value.reshape(nl, 1, nc)
            last = value[-1:] if top is None else cp.full((1, 1, nc), top, dtype=cp.float32)
            return cp.ascontiguousarray(cp.concatenate((value, last)))

        zgrid = self._levels("mesh", "zgrid", nl + 1)
        theta = self._levels("diag", "theta", nl)
        exner = self._levels("diag", "exner", nl)
        z = cp.multiply(f32(0.5), cp.add(zgrid[:-1], zgrid[1:]))
        dz = cp.subtract(zgrid[1:], zgrid[:-1])
        fields = {"t": extend(cp.multiply(theta, exner)), "theta": extend(theta), "pi": extend(exner),
                  "z": extend(z), "dz": extend(dz, 0.0),
                  "w": cp.ascontiguousarray(self._levels("state", "w", nl + 1).reshape(nl + 1, 1, nc)),
                  "ht": cp.ascontiguousarray(zgrid[0].reshape(1, nc)),
                  "land": self._plane("sfc_input", "xland").reshape(1, nc)}
        fields["lat"], fields["lon"] = self._geography()
        zero = cp.zeros((nl + 1, 1, nc), dtype=cp.float32)
        for name in SPECIES:
            value = self._scalar(name)
            fields[name] = zero if value is None else extend(value, 0.0)
        if self.config.option in (1, 2):
            fields["refl"] = extend(self.reflectivity(theta, zgrid), -35.0)
        else:
            fields["refl"] = zero
        fields["top"] = cp.ones((1, nc), dtype=cp.int32)
        fields["inputnum"] = fields["inputden"] = cp.zeros((12, 1, nc), dtype=cp.float32)
        return fields

    def _geography(self):
        """latCell/lonCell in degrees, longitude on (-180, 180], (1, nCells) float32 (static: formed once on the host)."""
        if getattr(self, "_latlon", None) is None:
            cp = self.cp
            lat = np.degrees(cp.asnumpy(self._plane("mesh", "latCell")).astype(np.float64))
            lon = np.degrees(cp.asnumpy(self._plane("mesh", "lonCell")).astype(np.float64))
            lon = np.where(lon > 180.0, lon - 360.0, lon)
            self._latlon = tuple(cp.asarray(np.ascontiguousarray(v.astype(np.float32).reshape(1, self.nc)))
                                 for v in (lat, lon))
            self._spacing = float(np.sqrt(np.mean(cp.asnumpy(self._plane("mesh", "areaCell")).astype(np.float64))))
        return self._latlon

    def reflectivity(self, theta, zgrid):
        """The step's 10 cm reflectivity (nl, nCells), dBZ: the scheme's own calc_refl10cm on the end state."""
        from woof.hex.mpasphys.thompson_refl_start_v842 import device_math_source, radar
        cp = self.cp
        if getattr(self, "_maths", None) is None:
            self._maths = device_math_source()
        zero = cp.zeros((self.nl, self.nc), dtype=cp.float32)
        state = {n: (self._scalar(n) if self._scalar(n) is not None else zero) for n in ("qv", "qc", "qr", "nr",
                                                                                        "qs", "qg")}
        state["theta"] = theta
        state["p"] = cp.add(self._levels("diag", "pressure_base", self.nl), self._levels("diag", "pressure_p", self.nl))
        return radar(state, zgrid, maths=self._maths, cp=cp)["refl10cm"]

    # -- the per-step fold ------------------------------------------------------------------------------
    def begin(self, step, when):
        """The step the next :meth:`update` folds (steps <= 0 fold nothing)."""
        self._step = int(step) if int(step) > 0 else None

    def due(self) -> float | None:
        """The activation clock's next value when the step that :meth:`begin` named is due, else None."""
        from woof.core.lightning import next_activation
        if self._step is None:
            return None
        return next_activation(self.activation, (self._step - 1) * self.dt, self.dt, self.interval)

    def update(self):
        if self._step is None or not self.nc:
            return
        following = self.due()
        if following is None:
            return
        from woof.core.lightning import launch_lightning
        c = self.config
        fields = self.columns()
        spacing = self._spacing
        output = {name: self.out[name] for name in self.names}
        output.setdefault("lpi", self.cp.zeros((1, self.nc), dtype=self.cp.float32))
        launch_lightning(fields, output, option=c.option, method=c.iccg_method, cellcount_method=c.cellcount_method,
                         dx=spacing, dy=spacing, interval=self.interval, factor=c.flashrate_factor,
                         num=c.iccg_prescribed_num, den=c.iccg_prescribed_den, adjust=0.0, month=1)
        self.activation = following
        self.activations += 1

    # -- the due frame ----------------------------------------------------------------------------------
    def compute(self):
        """Publish what a due stream will write."""
        for name in self.carried:
            if self.utils.field_will_be_written(name):
                self.owner.buffer("diag", name)[:self.nc] = self.out[name][0]

    # -- checkpoint continuation -----------------------------------------------------------------------
    def state(self):
        cp = self.cp
        state = {STATE_PREFIX + n: cp.asnumpy(self.out[n]) for n in self.names}
        state[ACTIVATION_KEY] = np.array([self.activation], dtype=np.float64)
        return state

    def resume(self, state):
        for key, value in state.items():
            if key.startswith(STATE_PREFIX) and key[len(STATE_PREFIX):] in self.names:
                target = self.out[key[len(STATE_PREFIX):]]
                target[...] = self.cp.asarray(value).reshape(target.shape)
        if ACTIVATION_KEY in state:
            self.activation = float(np.asarray(state[ACTIVATION_KEY]).reshape(-1)[0])

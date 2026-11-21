"""One E3SM v3.2.0 coupler interval on the GPU, opt-in: the ported coupler
routines chained in the driver's order.

:class:`CouplerInterval` holds the coupler's side of an ocean and a sea ice
on one grid with an atmosphere (aoflux_grid ``'ocn'``, ``cpl_seq_option``
``RASM_OPTION1``, one instance of each, one task). Its phases are
``cime_comp_mod``'s, each running the routines already ported and held word
for word to E3SM on their own:

- :meth:`~CouplerInterval.atmocn_setup` (``cime_run_atmocn_setup``):
  ``prep_ocn_calc_i2x_ox`` (``ice_c2_ocn``), ``seq_flux_atmocn_mct``
  (:mod:`.flux_atmocn_e3sm`), ``prep_ocn_mrg`` (:mod:`.prep_ocn_merge_e3sm`)
  and ``prep_ocn_accum`` (:mod:`.ocn_accum_e3sm`) when the ocean is
  prognostic, then ``seq_flux_ocnalb_mct`` (:mod:`.ocnalb_e3sm`);
- :meth:`~CouplerInterval.ocn_setup_send` (``cime_run_ocn_setup_send``):
  ``prep_ocn_accum_avg`` into x2o;
- :meth:`~CouplerInterval.ice_setup_send` (``cime_run_ice_setup_send``):
  ``prep_ice_calc_o2x_ix`` (``ocn_c2_ice``), ``prep_ice_calc_a2x_ix``
  (``atm_c2_ice``) and ``prep_ice_mrg`` (:mod:`.prep_ice_merge_e3sm`) when
  the sea ice is prognostic;
- :meth:`~CouplerInterval.ice_recv_post` and
  :meth:`~CouplerInterval.ocn_recv_post`: the components' new exports;
- :meth:`~CouplerInterval.update_fractions` (``cime_run_update_fractions``):
  ``seq_frac_set`` (:mod:`.seq_frac_e3sm`);
- :meth:`~CouplerInterval.step`: one coupler step in the driver's order,
  given the run alarms.

The ice-to-ocean and ocean-to-ice mappers (``seq_map_init_rearrolap``) are
``copy_only`` on one grid: ``seq_map_map`` copies the shared fields
(``mct_aVect_copy``), so i2x_ox, a2x_ix and o2x_ix are copies of their
sources, taken when E3SM takes them and zero until then (``prep_ocn_init``
and ``prep_ice_init`` zero them). The ice-to-atmosphere mapper is
:class:`.seq_frac_e3sm.Mapper` (copy, rearrange or a matrix). The flux grid's
mask is ``seq_flux_init_mct``'s, taken from the fractions at construction.
The first merge does each merge's first-call bookkeeping, as E3SM does.

``seq_flux_atmocn_mct`` calls glibc's binary64 ``atan``: like
:mod:`.flux_atmocn_e3sm`, an interval runs only with the opt-in
transcription (``WOOF_GLIBC239_LIBM64=1``) and is refused by name without
it. This is not called by any default path; the chain is held word for word
to E3SM's own routines, cut verbatim, linked and driven in this order
(``tools/e3sm_coupler_oracle/run_coupler_interval.F90``,
``tests/test_e3sm_coupler_interval_parity.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

from types import SimpleNamespace

import numpy as np

E3SM_TAG = "v3.2.0"
E3SM_COMMIT = "648b167b7f95267a159a9fdee25faf604c2bfc4f"
SOURCE_FILES = ("driver-mct/main/cime_comp_mod.F90:cime_run_atmocn_setup,cime_run_ocn_setup_send,"
                "cime_run_ice_setup_send,cime_run_update_fractions,cime_run_atmocn_fluxes,cime_run_ocn_albedos",)

#: the vectors an interval holds and the grid each is on ("o": the ocean's and sea ice's; "a": the atmosphere's)
VECTORS = {"dom": "o", "a2x": "o", "o2x": "o", "i2x": "o", "xao": "o", "fractions_o": "o", "fractions_i": "o",
           "fractions_a": "a", "r2x_o": "o", "w2x_o": "o", "g2x_o": "o", "r2x_i": "o", "g2x_i": "o", "w2x_i": "o",
           "x2o": "o", "x2i": "o"}


def _ported() -> SimpleNamespace:
    """The ported routines, imported when an interval is used (so importing
    the coupled package's other modules loads none of them)."""
    from woof.hex.coupled import flux_atmocn_e3sm, ocn_accum_e3sm, ocnalb_e3sm, prep_ice_merge_e3sm
    from woof.hex.coupled import prep_ocn_merge_e3sm, seq_frac_e3sm
    return SimpleNamespace(flux=flux_atmocn_e3sm, accum=ocn_accum_e3sm, alb=ocnalb_e3sm, imerge=prep_ice_merge_e3sm,
                           omerge=prep_ocn_merge_e3sm, frac=seq_frac_e3sm)


class IntervalRefusal(ValueError):
    """A coupler configuration this port does not hold to E3SM."""


@dataclass(frozen=True)
class Switches:
    """The driver's switches an interval reads: ``ocn_prognostic``,
    ``ice_prognostic``, ``ice_c2_ocn``, ``ocn_c2_ice`` and ``atm_c2_ice``
    (cime_comp_mod's), infodata's ``flux_albav``, ``nextsw_cday`` and
    ``flux_epbalfact``, the namelist's ``seq_flux_mct_albdif`` and
    ``_albdir``, seq_frac_mct's ``seq_frac_dead``, and ``rof2ocn_nutrients``,
    ``seq_flds_i2o_per_cat``, ``wav_ocn_coup`` and ``wav_ice_coup``
    (seq_flds_mod's)."""

    ocn_prognostic: bool = True
    ice_prognostic: bool = True
    ice_c2_ocn: bool = True
    ocn_c2_ice: bool = True
    atm_c2_ice: bool = True
    flux_albav: bool = False
    nextsw_cday: float = -1.0
    flux_epbalfact: float = 1.0
    albdif: float = 0.06
    albdir: float = 0.07
    frac_dead: bool = False
    rof2ocn_nutrients: bool = False
    i2o_per_cat: bool = False
    wav_ocn_coup: str = "none"
    wav_ice_coup: str = "none"

    def check(self) -> None:
        for name in ("ocn_prognostic", "ice_prognostic", "ice_c2_ocn", "ocn_c2_ice", "atm_c2_ice", "flux_albav",
                     "frac_dead", "rof2ocn_nutrients", "i2o_per_cat"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool")
        for name in ("nextsw_cday", "flux_epbalfact", "albdif", "albdir"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
                raise TypeError(f"{name} must be a real")
        for name in ("wav_ocn_coup", "wav_ice_coup"):
            if getattr(self, name) != "none":
                raise IntervalRefusal(f"{name} = {getattr(self, name)!r} is not ported in an interval: "
                                      "there is no wave model to map from (prep_ocn_calc_w2x_ox and "
                                      "prep_ice_calc_w2x_ix are not chained)")


class CouplerInterval:
    """The coupler's vectors and state for one ocean, sea ice and atmosphere.

    ``lists`` maps each of :data:`VECTORS` to its field names in order
    (MCT's ``rList``; ``dom`` its other fields after E3SM's coordinates,
    so it must include ``lat``, ``lon`` and ``hgt`` first as the coupler
    holds it). ``values`` maps each but ``a2x`` (it comes per step) to its
    starting ``rAttr`` as a ``(lsize, nflds)`` float64 array (host or cupy;
    copied). The interval's arrays are its own device arrays, read with
    :meth:`vector`."""

    def __init__(self, lists: dict, values: dict, *, n: int, n_a: int, mapper_i2a, flux,
                 switches: Switches = Switches()):
        import cupy as cp
        m = self._m = _ported()
        _flux, _accum, _frac = m.flux, m.accum, m.frac
        if not isinstance(switches, Switches):
            raise TypeError("switches must be interval_e3sm.Switches")
        switches.check()
        if not isinstance(flux, _flux.FluxOptions):
            raise TypeError("flux must be a flux_atmocn_e3sm.FluxOptions")
        flux.check()
        if flux.wav_atm_coup != "none":
            raise IntervalRefusal("wav_atm_coup = 'twoway' is not ported in an interval: there is no wave model")
        if not isinstance(mapper_i2a, _frac.Mapper):
            raise TypeError("mapper_i2a must be a seq_frac_e3sm.Mapper")
        for name, size in (("n", n), ("n_a", n_a)):
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
                raise ValueError(f"{name} must be a positive integer")
        missing = set(VECTORS) - set(lists)
        if missing:
            raise ValueError(f"lists has no {', '.join(sorted(missing))}")
        self.switches, self.flux, self.mapper_i2a = switches, flux, mapper_i2a
        self.n, self.n_a = int(n), int(n_a)
        self.lists = {k: tuple(lists[k]) for k in VECTORS}
        for k, names in self.lists.items():
            if len(set(names)) != len(names):
                raise ValueError(f"{k} repeats a field name; MCT's lists hold each once")
        self._v = {}
        for k, grid in VECTORS.items():
            shape = (self.n if grid == "o" else self.n_a, len(self.lists[k]))
            if k == "a2x":
                self._v[k] = cp.zeros(shape, dtype=np.float64)
                continue
            if k not in values:
                raise ValueError(f"values has no {k}")
            v = values[k]
            if tuple(v.shape) != shape or v.dtype != np.float64:
                raise ValueError(f"{k} must be float64 of shape {shape}")
            self._v[k] = cp.array(v, dtype=np.float64, order="C", copy=True)
        # prep_ocn_init's and prep_ice_init's own: the mapped copies, zeroed
        self._v["i2x_ox"] = cp.zeros_like(self._v["i2x"])
        self._v["a2x_ix"] = cp.zeros_like(self._v["a2x"])
        self._v["o2x_ix"] = cp.zeros_like(self._v["o2x"])
        self.accumulator = _accum.OceanAccumulator(self.n, self.lists["x2o"])
        self._ocn_plan = self._ice_plan = None
        self.frac_errors: list[str] = []
        # seq_flux_init_mct: the flux grid's mask from the domain and the starting fractions
        self.mask = _flux.flux_mask(dom=self._v["dom"], dom_fields=self.lists["dom"],
                                    fractions=self._v["fractions_o"], fractions_fields=self.lists["fractions_o"])

    def vector(self, name: str):
        """The interval's device array for ``name`` (one of :data:`VECTORS`,
        ``i2x_ox``, ``a2x_ix`` or ``o2x_ix``), or ``acc`` (the accumulator)."""
        return self.accumulator.acc if name == "acc" else self._v[name]

    def _set(self, name: str, array) -> None:
        import cupy as cp
        dst = self._v[name]
        if tuple(array.shape) != dst.shape or array.dtype != np.float64:
            raise ValueError(f"{name} must be float64 of shape {dst.shape}")
        dst[...] = cp.asarray(array)

    # -- cime_run_atmocn_setup ------------------------------------------------
    def atmocn_setup(self, a2x) -> None:
        """``cime_run_atmocn_setup`` with this step's atmosphere export on the ocean grid."""
        v, s, m = self._v, self.switches, self._m
        _flux, _omerge, _alb = m.flux, m.omerge, m.alb
        self._set("a2x", a2x)
        if s.ocn_prognostic and s.ice_c2_ocn:
            v["i2x_ox"][...] = v["i2x"]                      # prep_ocn_calc_i2x_ox: seq_map_map, copy_only
        _flux.seq_flux_atmocn(a2x=v["a2x"], a2x_fields=self.lists["a2x"], o2x=v["o2x"],
                              o2x_fields=self.lists["o2x"], xao=v["xao"], xao_fields=self.lists["xao"],
                              mask=self.mask, options=self.flux)
        if s.ocn_prognostic:
            if self._ocn_plan is None:
                self._ocn_plan = _omerge.plan(
                    {"a2x": self.lists["a2x"], "i2x": self.lists["i2x"], "r2x": self.lists["r2x_o"],
                     "w2x": self.lists["w2x_o"], "g2x": self.lists["g2x_o"], "xao": self.lists["xao"],
                     "fractions": self.lists["fractions_o"], "x2o": self.lists["x2o"]},
                    nutrients=s.rof2ocn_nutrients, per_cat=s.i2o_per_cat, wav_ocn_coup=s.wav_ocn_coup)
            _omerge.merge(self._ocn_plan, {"a2x": v["a2x"], "i2x": v["i2x_ox"], "r2x": v["r2x_o"],
                                           "w2x": v["w2x_o"], "g2x": v["g2x_o"], "xao": v["xao"],
                                           "fractions": v["fractions_o"]},
                          v["x2o"], flux_epbalfact=float(s.flux_epbalfact))
            self.accumulator.accum(v["x2o"])
        _alb.seq_flux_ocnalb(a2x=v["a2x"], a2x_fields=self.lists["a2x"], fractions=v["fractions_o"],
                             fractions_fields=self.lists["fractions_o"], xao=v["xao"], xao_fields=self.lists["xao"],
                             flux_albav=bool(s.flux_albav), albdif=float(s.albdif), albdir=float(s.albdir),
                             nextsw_cday=float(s.nextsw_cday))

    # -- cime_run_ocn_setup_send ----------------------------------------------
    def ocn_setup_send(self) -> None:
        """``cime_run_ocn_setup_send``: the window's average into x2o."""
        if self.switches.ocn_prognostic:
            self.accumulator.accum_avg(self._v["x2o"])

    # -- cime_run_ice_setup_send ----------------------------------------------
    def ice_setup_send(self) -> None:
        """``cime_run_ice_setup_send``: x2i from the ocean's and atmosphere's exports."""
        v, s, _imerge = self._v, self.switches, self._m.imerge
        if not s.ice_prognostic:
            return
        if s.ocn_c2_ice:
            v["o2x_ix"][...] = v["o2x"]                      # prep_ice_calc_o2x_ix: seq_map_map, copy_only
        if s.atm_c2_ice:
            v["a2x_ix"][...] = v["a2x"]                      # prep_ice_calc_a2x_ix: seq_map_map, copy_only
        if self._ice_plan is None:
            self._ice_plan = _imerge.plan({"a2x": self.lists["a2x"], "o2x": self.lists["o2x"],
                                           "r2x": self.lists["r2x_i"], "g2x": self.lists["g2x_i"],
                                           "w2x": self.lists["w2x_i"], "x2i": self.lists["x2i"]},
                                          wav_ice_coup=s.wav_ice_coup)
        _imerge.merge(self._ice_plan, {"a2x": v["a2x_ix"], "o2x": v["o2x_ix"], "r2x": v["r2x_i"],
                                       "g2x": v["g2x_i"], "w2x": v["w2x_i"]},
                      v["x2i"], flux_epbalfact=float(s.flux_epbalfact))

    def ice_recv_post(self, i2x) -> None:
        """The sea ice's new export (``cime_run_ice_recv_post``)."""
        self._set("i2x", i2x)

    def ocn_recv_post(self, o2x) -> None:
        """The ocean's new export (``cime_run_ocn_recv_post``)."""
        self._set("o2x", o2x)

    # -- cime_run_update_fractions --------------------------------------------
    def update_fractions(self) -> None:
        """``cime_run_update_fractions``: ``seq_frac_set`` from the sea ice's export."""
        v, _frac = self._v, self._m.frac
        self.frac_errors += _frac.seq_frac_set(
            i2x=v["i2x"], i2x_fields=self.lists["i2x"], dom=v["dom"], dom_fields=self.lists["dom"],
            fractions_i=v["fractions_i"], fi_fields=self.lists["fractions_i"], fractions_o=v["fractions_o"],
            fo_fields=self.lists["fractions_o"], fractions_a=v["fractions_a"], fa_fields=self.lists["fractions_a"],
            mapper_i2o=_frac.Mapper("copy"), mapper_i2a=self.mapper_i2a, dead=bool(self.switches.frac_dead))

    def step(self, a2x, *, ocn_run: bool, ice_run: bool, ocn_next: bool, i2x=None, o2x=None) -> None:
        """One coupler step in RASM_OPTION1's order. ``i2x`` is the sea ice's
        export after it runs (required with ``ice_run``), ``o2x`` the
        ocean's when its run ends (required with ``ocn_next``)."""
        for name, value in (("ocn_run", ocn_run), ("ice_run", ice_run), ("ocn_next", ocn_next)):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be a bool")
        if ice_run and i2x is None or ocn_next and o2x is None:
            raise ValueError("a component that runs this step needs its new export")
        self.atmocn_setup(a2x)
        if ocn_run:
            self.ocn_setup_send()
        if ice_run:
            self.ice_setup_send()
            self.ice_recv_post(i2x)
        self.update_fractions()
        if ocn_next:
            self.ocn_recv_post(o2x)

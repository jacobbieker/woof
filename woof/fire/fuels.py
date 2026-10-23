"""Standard surface fire behaviour fuel models and WRF-SFIRE fuel categories.

Two tables live here, and they answer different questions.

* :data:`STANDARD_FUEL_MODELS` is the set of 53 standard fire behaviour fuel
  models consumed by the full Rothermel (1972) spread model with the Albini
  (1976) corrections (:mod:`woof.fire.rothermel`): the original 13 of
  Albini (1976) / Anderson (1982) and the 40 of Scott & Burgan (2005).
  Each model carries the oven-dry load of the five particle classes (1-h,
  10-h, 100-h dead; live herbaceous; live woody), the surface-area-to-volume
  ratio (SAV) of each, fuel bed depth, dead fuel moisture of extinction, heat
  content, particle density, mineral contents and the dynamic flag.

* :data:`WRF_SFIRE_FUEL_CATEGORIES` is WRF-SFIRE's ``namelist.fire``
  ``&fuel_categories`` default for its 13 Anderson categories plus the
  no-fuel category 14, as a record per category.  That model lumps each
  category into one fuel particle (one ``savr``, one ``fgi``) and adds the
  quantities the coupled fire-atmosphere model needs beyond the spread
  rate: the burn time constant ``weight``, the canopy fuel
  ``fci_d``/``fct``, the wind reduction factor ``windrf`` and the
  chaparral switch ``ichap``.
* :data:`WRF471_DATA` / :data:`WRF471_NAMELIST` are the same tables as the
  literal text WRF v4.7.1 compiles and reads (54 categories of compiled
  DATA, the namelist's 14 over them), from which
  :mod:`woof.fire.wrf_sfire` rounds each real(4) value exactly as gfortran
  does.

Units.  Rothermel and every fuel model table are published in US customary
units (lb ft-2, ft, ft-1, Btu lb-1).  The dataclasses store SI -- kg m-2,
m, m-1, kJ kg-1, kg m-3 -- and the source rows keep the published numbers
verbatim so a reader can check them against the paper; the conversion
happens once, at import, through the factors defined below.

Provenance.

* Anderson 13, loads in lb ft-2: Albini, F. A. (1976) *Computer-based
  models of wildland fire behavior: a user's manual*, USDA FS
  Intermountain Station, Table 4, as carried by BehavePlus (firelab/behave
  ``src/behave/fuelModels.cpp``).  Anderson (1982) GTR INT-122 Table 1
  lists the same models in tons acre-1 (0.74 t/ac = 0.034 lb ft-2).  Live
  woody SAV of model 7 is 1500 ft-1 here, as in BehavePlus and in Andrews
  (2018) RMRS-GTR-371 Table 8a, whose characteristic SAV of 1552 ft-1 it
  reproduces; Anderson (1982) prints 1550.
* Scott & Burgan 40, loads in tons acre-1: Scott, J. H. & Burgan, R. E.
  (2005) *Standard fire behavior fuel models*, RMRS-GTR-153, Table 7,
  cross-checked row by row against Andrews (2018) Table 8b-8e and the WRF
  ``module_fr_fire_phys.F`` DATA statements.
* Constants common to all 53 (Andrews 2018 Table 8 footnote a): 10-h SAV
  109 ft-1, 100-h SAV 30 ft-1, heat content 8000 Btu lb-1, particle density
  32 lb ft-3, total mineral content 0.0555, effective mineral content 0.010.
* WRF-SFIRE categories: WRF v4.7.1 (commit f52c197)
  ``test/em_fire/namelist.fire`` ``&fuel_categories`` and ``&fuel_scalars``,
  and the DATA statements of ``phys/module_fr_fire_phys.F``, pinned by
  sha256 in ``tools/fire_ros_wrf471_oracle/SOURCES.sha256``.  The compiled
  DATA differ from the namelist in ``fgi`` (category 4 compiles 3.591
  kg m-2, the dead 1/10/100-h sum, where the namelist has 2.468), and the
  namelist accepts only the Anderson 13.  ``itree``, ``fcwh``, ``fcz0`` and
  ``ffw`` are openwfm WRF-SFIRE (``phys/module_fr_sfire_phys.F``) additions
  that WRF v4.7.1 does not carry.
"""

from __future__ import annotations

from dataclasses import dataclass


# --------------------------------------------------------------------------
# unit conversions (exact definitions of the international foot and pound)
# --------------------------------------------------------------------------

FT = 0.3048                         # m per ft
LB = 0.45359237                     # kg per lb
BTU = 1.05505585262                 # kJ per (International Table) Btu
#: lb ft-2 -> kg m-2 (4.8824...)
LB_PER_FT2 = LB / FT**2
#: short tons acre-1 -> lb ft-2: 2000 lb over 43 560 ft2, as BehavePlus spells it
TONS_PER_ACRE_IN_LB_PER_FT2 = 2000.0 / 43560.0
#: lb ft-3 -> kg m-3 (16.018...)
LB_PER_FT3 = LB / FT**3
#: ft-1 -> m-1 (3.2808...)
PER_FT = 1.0 / FT
#: Btu lb-1 -> kJ kg-1 (2.326 within the IT Btu's definition)
BTU_PER_LB = BTU / LB
#: ft min-1 -> m s-1, WRF-SFIRE's ``.00508``
FT_PER_MIN = FT / 60.0
#: Btu ft-2 min-1 -> kW m-2
BTU_PER_FT2_MIN = BTU / FT**2 / 60.0
#: Btu ft-2 -> kJ m-2
BTU_PER_FT2 = BTU / FT**2
#: Btu ft-1 s-1 -> kW m-1
BTU_PER_FT_S = BTU / FT

#: Fixed SAV of the 10-h and 100-h dead classes for every standard model, ft-1.
SAV_10H_PER_FT = 109.0
SAV_100H_PER_FT = 30.0
#: Particle density, heat content and mineral contents shared by all 53.
PARTICLE_DENSITY_LB_FT3 = 32.0
HEAT_CONTENT_BTU_LB = 8000.0
TOTAL_MINERAL_CONTENT = 0.0555
EFFECTIVE_MINERAL_CONTENT = 0.010


@dataclass(frozen=True)
class FuelModel:
    """One standard fire behaviour fuel model, in SI units.

    Loads are oven-dry, kg m-2; SAV ratios m-1; depth m; heat content
    kJ kg-1; particle density kg m-3; moisture of extinction and mineral
    contents are fractions.  ``dynamic`` models transfer live herbaceous
    load to a dead herbaceous class according to curing (Scott & Burgan
    2005); static models do not.
    """

    number: int
    code: str
    name: str
    dynamic: bool
    load_1h: float
    load_10h: float
    load_100h: float
    load_live_herb: float
    load_live_woody: float
    sav_1h: float
    sav_live_herb: float
    sav_live_woody: float
    depth: float
    moisture_extinction_dead: float
    heat_content_dead: float = HEAT_CONTENT_BTU_LB * BTU_PER_LB
    heat_content_live: float = HEAT_CONTENT_BTU_LB * BTU_PER_LB
    sav_10h: float = SAV_10H_PER_FT * PER_FT
    sav_100h: float = SAV_100H_PER_FT * PER_FT
    particle_density: float = PARTICLE_DENSITY_LB_FT3 * LB_PER_FT3
    total_mineral_content: float = TOTAL_MINERAL_CONTENT
    effective_mineral_content: float = EFFECTIVE_MINERAL_CONTENT

    def __post_init__(self):
        # The spread model divides by depth, SAV, particle density and the
        # dead moisture of extinction; a zero there is a malformed model,
        # not a firebreak (non-burnable models carry zero *load* instead).
        positive = ("sav_1h", "sav_10h", "sav_100h", "sav_live_herb",
                    "sav_live_woody", "depth", "moisture_extinction_dead",
                    "particle_density")
        for name in positive:
            value = getattr(self, name)
            if not (value > 0.0 and value < float("inf")):
                raise ValueError(f"fuel model {self.code}: {name} must be "
                                 f"positive and finite, got {value!r}")
        for name in ("load_1h", "load_10h", "load_100h", "load_live_herb",
                     "load_live_woody", "heat_content_dead",
                     "heat_content_live", "total_mineral_content",
                     "effective_mineral_content"):
            value = getattr(self, name)
            if not (value >= 0.0 and value < float("inf")):
                raise ValueError(f"fuel model {self.code}: {name} must be "
                                 f"non-negative and finite, got {value!r}")

    @property
    def burnable(self) -> bool:
        """Whether any class carries fuel."""
        return self.total_load > 0.0

    @property
    def total_load(self) -> float:
        """Oven-dry load of all five classes, kg m-2."""
        return (self.load_1h + self.load_10h + self.load_100h
                + self.load_live_herb + self.load_live_woody)

    @property
    def dead_load(self) -> float:
        """Oven-dry 1-h + 10-h + 100-h load, kg m-2."""
        return self.load_1h + self.load_10h + self.load_100h


def _model(number, code, name, dynamic, loads, savs, depth_ft, mx_percent,
           *, load_unit):
    """Build a :class:`FuelModel` from one published US-customary row.

    ``loads`` is (1-h, 10-h, 100-h, live herb, live woody) in ``load_unit``
    lb ft-2 per published unit; ``savs`` is (1-h, live herb, live woody) in
    ft-1; ``depth_ft`` in ft; ``mx_percent`` the dead moisture of extinction
    in percent.
    """
    w1, w10, w100, wh, ww = (x * load_unit * LB_PER_FT2 for x in loads)
    s1, sh, sw = (x * PER_FT for x in savs)
    return FuelModel(
        number=number, code=code, name=name, dynamic=dynamic,
        load_1h=w1, load_10h=w10, load_100h=w100,
        load_live_herb=wh, load_live_woody=ww,
        sav_1h=s1, sav_live_herb=sh, sav_live_woody=sw,
        depth=depth_ft * FT, moisture_extinction_dead=mx_percent / 100.0)


# Albini (1976) / Anderson (1982) 13, as BehavePlus carries them.
# number, code, name, dynamic,
#   (1-h, 10-h, 100-h, live herb, live woody) lb ft-2,
#   (1-h, live herb, live woody) SAV ft-1, depth ft, dead Mx %.
# A live SAV whose class carries no load is never read; BehavePlus fills
# them with 1500 and so does this table.
_ANDERSON_ROWS = (
    (1, "FM1", "Short grass", False,
     (0.034, 0.0, 0.0, 0.0, 0.0), (3500, 1500, 1500), 1.0, 12),
    (2, "FM2", "Timber grass and understory", False,
     (0.092, 0.046, 0.023, 0.023, 0.0), (3000, 1500, 1500), 1.0, 15),
    (3, "FM3", "Tall grass", False,
     (0.138, 0.0, 0.0, 0.0, 0.0), (1500, 1500, 1500), 2.5, 25),
    (4, "FM4", "Chaparral", False,
     (0.230, 0.184, 0.092, 0.0, 0.230), (2000, 1500, 1500), 6.0, 20),
    (5, "FM5", "Brush", False,
     (0.046, 0.023, 0.0, 0.0, 0.092), (2000, 1500, 1500), 2.0, 20),
    (6, "FM6", "Dormant brush, hardwood slash", False,
     (0.069, 0.115, 0.092, 0.0, 0.0), (1750, 1500, 1500), 2.5, 25),
    (7, "FM7", "Southern rough", False,
     (0.052, 0.086, 0.069, 0.0, 0.017), (1750, 1500, 1500), 2.5, 40),
    (8, "FM8", "Short needle litter", False,
     (0.069, 0.046, 0.115, 0.0, 0.0), (2000, 1500, 1500), 0.2, 30),
    (9, "FM9", "Long needle or hardwood litter", False,
     (0.134, 0.019, 0.007, 0.0, 0.0), (2500, 1500, 1500), 0.2, 25),
    (10, "FM10", "Timber litter and understory", False,
     (0.138, 0.092, 0.230, 0.0, 0.092), (2000, 1500, 1500), 1.0, 25),
    (11, "FM11", "Light logging slash", False,
     (0.069, 0.207, 0.253, 0.0, 0.0), (1500, 1500, 1500), 1.0, 15),
    (12, "FM12", "Medium logging slash", False,
     (0.184, 0.644, 0.759, 0.0, 0.0), (1500, 1500, 1500), 2.3, 20),
    (13, "FM13", "Heavy logging slash", False,
     (0.322, 1.058, 1.288, 0.0, 0.0), (1500, 1500, 1500), 3.0, 25),
)

# Scott & Burgan (2005) Table 7: loads in tons acre-1.  A live SAV whose
# class carries no load is printed "--" there and is filled with the
# BehavePlus value; it is never read.
_SCOTT_BURGAN_ROWS = (
    # grass (GR), all dynamic
    (101, "GR1", "Short, sparse, dry climate grass", True,
     (0.10, 0.0, 0.0, 0.30, 0.0), (2200, 2000, 1500), 0.4, 15),
    (102, "GR2", "Low load, dry climate grass", True,
     (0.10, 0.0, 0.0, 1.00, 0.0), (2000, 1800, 1500), 1.0, 15),
    (103, "GR3", "Low load, very coarse, humid climate grass", True,
     (0.10, 0.40, 0.0, 1.50, 0.0), (1500, 1300, 1500), 2.0, 30),
    (104, "GR4", "Moderate load, dry climate grass", True,
     (0.25, 0.0, 0.0, 1.90, 0.0), (2000, 1800, 1500), 2.0, 15),
    (105, "GR5", "Low load, humid climate grass", True,
     (0.40, 0.0, 0.0, 2.50, 0.0), (1800, 1600, 1500), 1.5, 40),
    (106, "GR6", "Moderate load, humid climate grass", True,
     (0.10, 0.0, 0.0, 3.40, 0.0), (2200, 2000, 1500), 1.5, 40),
    (107, "GR7", "High load, dry climate grass", True,
     (1.00, 0.0, 0.0, 5.40, 0.0), (2000, 1800, 1500), 3.0, 15),
    (108, "GR8", "High load, very coarse, humid climate grass", True,
     (0.50, 1.00, 0.0, 7.30, 0.0), (1500, 1300, 1500), 4.0, 30),
    (109, "GR9", "Very high load, humid climate grass", True,
     (1.00, 1.00, 0.0, 9.00, 0.0), (1800, 1600, 1500), 5.0, 40),
    # grass-shrub (GS), all dynamic
    (121, "GS1", "Low load, dry climate grass-shrub", True,
     (0.20, 0.0, 0.0, 0.50, 0.65), (2000, 1800, 1800), 0.9, 15),
    (122, "GS2", "Moderate load, dry climate grass-shrub", True,
     (0.50, 0.50, 0.0, 0.60, 1.00), (2000, 1800, 1800), 1.5, 15),
    (123, "GS3", "Moderate load, humid climate grass-shrub", True,
     (0.30, 0.25, 0.0, 1.45, 1.25), (1800, 1600, 1600), 1.8, 40),
    (124, "GS4", "High load, humid climate grass-shrub", True,
     (1.90, 0.30, 0.10, 3.40, 7.10), (1800, 1600, 1600), 2.1, 40),
    # shrub (SH)
    (141, "SH1", "Low load, dry climate shrub", True,
     (0.25, 0.25, 0.0, 0.15, 1.30), (2000, 1800, 1600), 1.0, 15),
    (142, "SH2", "Moderate load, dry climate shrub", False,
     (1.35, 2.40, 0.75, 0.0, 3.85), (2000, 1800, 1600), 1.0, 15),
    (143, "SH3", "Moderate load, humid climate shrub", False,
     (0.45, 3.00, 0.0, 0.0, 6.20), (1600, 1800, 1400), 2.4, 40),
    (144, "SH4", "Low load, humid climate timber-shrub", False,
     (0.85, 1.15, 0.20, 0.0, 2.55), (2000, 1800, 1600), 3.0, 30),
    (145, "SH5", "High load, dry climate shrub", False,
     (3.60, 2.10, 0.0, 0.0, 2.90), (750, 1800, 1600), 6.0, 15),
    (146, "SH6", "Low load, humid climate shrub", False,
     (2.90, 1.45, 0.0, 0.0, 1.40), (750, 1800, 1600), 2.0, 30),
    (147, "SH7", "Very high load, dry climate shrub", False,
     (3.50, 5.30, 2.20, 0.0, 3.40), (750, 1800, 1600), 6.0, 15),
    (148, "SH8", "High load, humid climate shrub", False,
     (2.05, 3.40, 0.85, 0.0, 4.35), (750, 1800, 1600), 3.0, 40),
    (149, "SH9", "Very high load, humid climate shrub", True,
     (4.50, 2.45, 0.0, 1.55, 7.00), (750, 1800, 1500), 4.4, 40),
    # timber-understory (TU)
    (161, "TU1", "Low load, dry climate timber-grass-shrub", True,
     (0.20, 0.90, 1.50, 0.20, 0.90), (2000, 1800, 1600), 0.6, 20),
    (162, "TU2", "Moderate load, humid climate timber-shrub", False,
     (0.95, 1.80, 1.25, 0.0, 0.20), (2000, 1800, 1600), 1.0, 30),
    (163, "TU3", "Moderate load, humid climate timber-grass-shrub", True,
     (1.10, 0.15, 0.25, 0.65, 1.10), (1800, 1600, 1400), 1.3, 30),
    (164, "TU4", "Dwarf conifer with understory", False,
     (4.50, 0.0, 0.0, 0.0, 2.00), (2300, 1800, 2000), 0.5, 12),
    (165, "TU5", "Very high load, dry climate timber-shrub", False,
     (4.00, 4.00, 3.00, 0.0, 3.00), (1500, 1800, 750), 1.0, 25),
    # timber litter (TL), all static and dead only
    (181, "TL1", "Low load compact conifer litter", False,
     (1.00, 2.20, 3.60, 0.0, 0.0), (2000, 1800, 1600), 0.2, 30),
    (182, "TL2", "Low load broadleaf litter", False,
     (1.40, 2.30, 2.20, 0.0, 0.0), (2000, 1800, 1600), 0.2, 25),
    (183, "TL3", "Moderate load conifer litter", False,
     (0.50, 2.20, 2.80, 0.0, 0.0), (2000, 1800, 1600), 0.3, 20),
    (184, "TL4", "Small downed logs", False,
     (0.50, 1.50, 4.20, 0.0, 0.0), (2000, 1800, 1600), 0.4, 25),
    (185, "TL5", "High load conifer litter", False,
     (1.15, 2.50, 4.40, 0.0, 0.0), (2000, 1800, 1600), 0.6, 25),
    (186, "TL6", "Moderate load broadleaf litter", False,
     (2.40, 1.20, 1.20, 0.0, 0.0), (2000, 1800, 1600), 0.3, 25),
    (187, "TL7", "Large downed logs", False,
     (0.30, 1.40, 8.10, 0.0, 0.0), (2000, 1800, 1600), 0.4, 25),
    (188, "TL8", "Long-needle litter", False,
     (5.80, 1.40, 1.10, 0.0, 0.0), (1800, 1800, 1600), 0.3, 35),
    (189, "TL9", "Very high load broadleaf litter", False,
     (6.65, 3.30, 4.15, 0.0, 0.0), (1800, 1800, 1600), 0.6, 35),
    # slash-blowdown (SB), all static and dead only
    (201, "SB1", "Low load activity fuel", False,
     (1.50, 3.00, 11.00, 0.0, 0.0), (2000, 1800, 1600), 1.0, 25),
    (202, "SB2", "Moderate load activity fuel or low load blowdown", False,
     (4.50, 4.25, 4.00, 0.0, 0.0), (2000, 1800, 1600), 1.0, 25),
    (203, "SB3", "High load activity fuel or moderate load blowdown", False,
     (5.50, 2.75, 3.00, 0.0, 0.0), (2000, 1800, 1600), 1.2, 25),
    (204, "SB4", "High load blowdown", False,
     (5.25, 3.50, 5.25, 0.0, 0.0), (2000, 1800, 1600), 2.7, 25),
)

ANDERSON_13: tuple[FuelModel, ...] = tuple(
    _model(*row, load_unit=1.0) for row in _ANDERSON_ROWS)
SCOTT_BURGAN_40: tuple[FuelModel, ...] = tuple(
    _model(*row, load_unit=TONS_PER_ACRE_IN_LB_PER_FT2)
    for row in _SCOTT_BURGAN_ROWS)

#: All 53 standard models keyed by their BehavePlus/LANDFIRE number.
STANDARD_FUEL_MODELS: dict[int, FuelModel] = {
    m.number: m for m in ANDERSON_13 + SCOTT_BURGAN_40}

# Scott & Burgan (2005) Table 6: the five defined non-burnable models (94-97
# are reserved and undefined).  They carry no fuel, so nothing but their
# zero loads is ever read; depth, SAV and extinction take placeholder
# values only so the record is well formed.
_NON_BURNABLE_ROWS = (
    (91, "NB1", "Urban, developed"), (92, "NB2", "Snow, ice"),
    (93, "NB3", "Agricultural"), (98, "NB8", "Open water"),
    (99, "NB9", "Bare ground"),
)
#: Non-burnable models keyed by number: zero load, zero spread.
NON_BURNABLE_FUEL_MODELS: dict[int, FuelModel] = {
    number: _model(number, code, name, False, (0.0,) * 5, (1500,) * 3, 1.0,
                   25, load_unit=1.0)
    for number, code, name in _NON_BURNABLE_ROWS}

_ALL_MODELS = {**STANDARD_FUEL_MODELS, **NON_BURNABLE_FUEL_MODELS}
_BY_CODE = {m.code: m for m in _ALL_MODELS.values()}


def fuel_model(key: int | str | FuelModel) -> FuelModel:
    """Look up a standard or non-burnable fuel model by number or code.

    Numbers are ``1``..``13``, the Scott & Burgan ``101``..``204`` and the
    non-burnable ``91``, ``92``, ``93``, ``98``, ``99``; codes are
    ``"FM1"`` .. ``"FM13"``, the Scott & Burgan codes (``"GR2"``,
    ``"TL8"``, ...) and ``"NB1"`` .. ``"NB9"``, case-insensitive.  A
    :class:`FuelModel` passes through unchanged, so custom models are
    accepted wherever a standard one is.  A number must be integral.
    """
    if isinstance(key, FuelModel):
        return key
    if isinstance(key, str):
        try:
            return _BY_CODE[key.strip().upper()]
        except KeyError:
            raise KeyError(f"unknown fuel model code {key!r}") from None
    number = float(key)
    if not number.is_integer():
        raise ValueError(f"fuel model number must be integral, got {key!r}")
    try:
        return _ALL_MODELS[int(number)]
    except KeyError:
        raise KeyError(f"unknown fuel model number {key!r}") from None


# --------------------------------------------------------------------------
# WRF-SFIRE namelist.fire defaults
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WrfSfireFuelScalars:
    """WRF-SFIRE ``&fuel_scalars``: constants shared by every category.

    ``cmbcnst`` is the heat of combustion of dry fuel (J kg-1), ``hfgl`` the
    surface heat flux that ignites the canopy (W m-2), ``fuelmc_g`` and
    ``fuelmc_c`` the default surface and canopy fuel moisture (fractions).
    """

    cmbcnst: float = 17.433e6
    hfgl: float = 17.0e4
    fuelmc_g: float = 0.08
    fuelmc_c: float = 1.00
    nfuelcats: int = 13
    no_fuel_cat: int = 14

    @property
    def fuelheat_btu_lb(self) -> float:
        """WRF's ``fuelheat = cmbcnst * 4.30e-04`` (Btu lb-1), as WRF folds it.

        WRF's constant is a rounded J kg-1 -> Btu lb-1 factor (exactly
        4.2992e-4), and the spread rate reads this product, not 8000.
        """
        return self.cmbcnst * 4.30e-04


@dataclass(frozen=True)
class WrfSfireFuelCategory:
    """One WRF-SFIRE ``&fuel_categories`` column, with namelist names.

    The field names are the namelist's, because they are the coupling
    contract with ``namelist.fire``.  Units are WRF's own:

    ``windrf``      wind reduction factor from ``fcwh`` (20 ft) to mid-flame
    ``fgi``         initial total mass of surface fuel, kg m-2
    ``fueldepthm``  fuel bed depth, m
    ``savr``        fuel particle SAV, ft-1 (US customary, as WRF uses it)
    ``fuelmce``     dead moisture of extinction, fraction
    ``fueldens``    oven-dry particle density, lb ft-3 (32 solid, 19 rotten)
    ``st``, ``se``  total and effective mineral content, fractions
    ``weight``      burn-out weighting parameter, s; the fuel fraction left
                    decays as ``exp(-t / fuel_time)`` with
                    ``fuel_time = weight / 0.85`` (s)
    ``fci_d``       initial dry canopy fuel mass, kg m-2
    ``fct``         canopy fuel burn-out time after drying, s
    ``ichap``       1: Clark et al. (2004) chaparral spread rate
    ``itree``       tree-presence class used by the canopy model
    ``fcwh``        height of the wind fed to the spread rate, m
    ``fcz0``        roughness length of the fuel bed, m
    ``ffw``         fraction of the fuel consumed in the flaming front
    """

    category: int
    name: str
    windrf: float
    fgi: float
    fueldepthm: float
    savr: float
    fuelmce: float
    fueldens: float
    st: float
    se: float
    weight: float
    fci_d: float
    fct: float
    ichap: int
    itree: int
    fcwh: float
    fcz0: float
    ffw: float

    @property
    def fuel_time(self) -> float:
        """Burn time constant, s: ``fuel_time = weight / 0.85`` (set_fire_params)."""
        return self.weight / 0.85

    @property
    def fcbr(self) -> float:
        """Canopy fuel burn rate ``fci_d / fct`` (kg m-2 s-1), 0 when ``fct`` is 0."""
        return self.fci_d / self.fct if self.fct != 0.0 else 0.0


_WRF_NAMES = (
    "Short grass (1 ft)", "Timber (grass and understory)",
    "Tall grass (2.5 ft)", "Chaparral (6 ft)", "Brush (2 ft)",
    "Dormant brush, hardwood slash", "Southern rough",
    "Closed timber litter", "Hardwood litter",
    "Timber (litter + understory)", "Light logging slash",
    "Medium logging slash", "Heavy logging slash", "no fuel")
# test/em_fire/namelist.fire &fuel_categories (the openwfm WRF-SFIRE DATA
# carry the same values), 1..14.
_WRF_WINDRF = (0.36, 0.36, 0.44, 0.55, 0.42, 0.44, 0.44,
               0.36, 0.36, 0.36, 0.36, 0.43, 0.46, 1e-7)
_WRF_FGI = (0.166, 0.897, 0.675, 2.468, 0.785, 1.345, 1.092,
            1.121, 0.780, 2.694, 2.582, 7.749, 13.024, 1.e-7)
_WRF_FUELDEPTHM = (0.305, 0.305, 0.762, 1.829, 0.61, 0.762, 0.762,
                   0.0610, 0.0610, 0.305, 0.305, 0.701, 0.914, 0.305)
_WRF_SAVR = (3500., 2784., 1500., 1739., 1683., 1564., 1562.,
             1889., 2484., 1764., 1182., 1145., 1159., 3500.)
_WRF_FUELMCE = (0.12, 0.15, 0.25, 0.20, 0.20, 0.25, 0.40,
                0.30, 0.25, 0.25, 0.15, 0.20, 0.25, 0.12)
# "(4) - best fit of Latham data; (5)-(7) could be 60-120; (8)-(10) could
# be 300-1600; (11)-(13) could be 300-1600"
_WRF_WEIGHT = (7., 7., 7., 180., 100., 100., 100.,
               900., 900., 900., 900., 900., 900., 7.)
# "1.12083 is 5 tons/acre.  5-50 tons/acre orig., 100-300 after blowdown"
_WRF_FCI_D = (0., 0., 0., 1.123, 0., 0., 0.,
              1.121, 1.121, 1.121, 1.121, 1.121, 1.121, 0.)
_WRF_FCT = (60., 60., 60., 60., 60., 60., 60.,
            60., 120., 180., 180., 180., 180., 60.)
_WRF_ICHAP = (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
_WRF_ITREE = (0, 5, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0)
# "consistent with BEHAVE": the 20 ft reference height
_WRF_FCWH = (6.096,) * 14
# "roughness length 0.13*fueldepthm except cat 3 fz0=0.1 for consistency
# with landuse"
_WRF_FCZ0 = (0.0396, 0.0396, 0.1000, 0.2378, 0.0793, 0.0991, 0.0991,
             0.0079, 0.0079, 0.0396, 0.0396, 0.0911, 0.1188, 0.0396)

WRF_SFIRE_FUEL_SCALARS = WrfSfireFuelScalars()

#: WRF-SFIRE categories 1..14 (14 = no fuel), keyed by category number.
WRF_SFIRE_FUEL_CATEGORIES: dict[int, WrfSfireFuelCategory] = {
    k + 1: WrfSfireFuelCategory(
        category=k + 1, name=_WRF_NAMES[k], windrf=_WRF_WINDRF[k],
        fgi=_WRF_FGI[k], fueldepthm=_WRF_FUELDEPTHM[k], savr=_WRF_SAVR[k],
        fuelmce=_WRF_FUELMCE[k], fueldens=32.0, st=0.0555, se=0.010,
        weight=_WRF_WEIGHT[k], fci_d=_WRF_FCI_D[k], fct=_WRF_FCT[k],
        ichap=_WRF_ICHAP[k], itree=_WRF_ITREE[k], fcwh=_WRF_FCWH[k],
        fcz0=_WRF_FCZ0[k], ffw=0.9)
    for k in range(14)
}


# --------------------------------------------------------------------------
# WRF v4.7.1 phys/module_fr_fire_phys.F, as literal text
# --------------------------------------------------------------------------
#
# The real(4) values WRF computes with are the nearest floats to these
# decimal strings (Fortran rounds a literal, and list-directed input, once
# to the declared kind), so the strings are kept exactly as WRF spells them
# and woof.fire.wrf_sfire rounds each once, correctly.  Two configurations:
#
# * "defaults" -- the module's DATA statements, read when no namelist.fire
#   exists: 54 categories (13 Anderson, 14 no fuel, 15-54 the Scott &
#   Burgan 40, lumped to one particle), nfuelcats = 53.
# * "namelist" -- test/em_fire/namelist.fire over those DATA: categories
#   1-14 replaced, nfuelcats = 13, so only the Anderson 13 are accepted.
#
# fueldens, st, se are 32., 0.0555, 0.010 for every category in both
# (DATA nf* and the namelist alike); fgi_lh is not in the namelist, so its
# DATA values (zero for 1-14) stand in both.

WRF471_DATA: dict[str, tuple[str, ...]] = {
    "fgi": ('0.166', '0.896', '0.674', '3.591', '0.784', '1.344', '1.091',
            '1.120', '0.780', '2.692', '2.582', '7.749', '13.024', '1.e-7',
            '0.0224', '0.0224', '0.1121', '0.0560', '0.0897', '0.0224',
            '0.2242', '0.3363', '0.4483', '0.0448', '0.2242', '0.1233',
            '0.5156', '0.1121', '1.0088', '0.7734', '0.4932', '1.2778',
            '0.9751', '2.4659', '1.4123', '1.5580', '0.5828', '0.8967',
            '0.3363', '1.0088', '2.4659', '1.5244', '1.3226', '1.2329',
            '1.3899', '1.8046', '1.0760', '2.1969', '1.8606', '3.1608',
            '3.4746', '2.8582', '2.5219', '3.1384'),
    "fgi_lh": ('0.00',) * 14 + (
        '0.0673', '0.2242', '0.3363', '0.4259', '0.5604', '0.7622', '1.2105',
        '1.6364', '2.0175', '0.1121', '0.1345', '0.3250', '0.7622', '0.0336',
        '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000',
        '0.3475', '0.0448', '0.0000', '0.1457') + ('0.0000',) * 15,
    "fueldepthm": (
        '0.305', '0.305', '0.762', '1.829', '0.61', '0.762', '0.762', '0.0610',
        '0.0610', '0.305', '0.305', '0.701', '0.914', '0.305', '0.1219',
        '0.3048', '0.6096', '0.6096', '0.4572', '0.4572', '0.9144', '1.2192',
        '1.5240', '0.2743', '0.4572', '0.5486', '0.6401', '0.3048', '0.3048',
        '0.7315', '0.9144', '1.8288', '0.6096', '1.8288', '0.9144', '1.3411',
        '0.1829', '0.3048', '0.3962', '0.1524', '0.3048', '0.0610', '0.0610',
        '0.0914', '0.1219', '0.1829', '0.0914', '0.1219', '0.0914', '0.1829',
        '0.3048', '0.3048', '0.3658', '0.8230'),
    "savr": (
        '3500.', '2784.', '1500.', '1739.', '1683.', '1564.', '1562.', '1889.',
        '2484.', '1764.', '1182.', '1145.', '1159.', '3500.', '2200.', '2000.',
        '1500.', '2000.', '1800.', '2200.', '2000.', '1500.', '1800.', '2000.',
        '2000.', '1800.', '1800.', '2000.', '2000.', '1600.', '2000.', '750.',
        '750.', '750.', '750.', '750.', '2000.', '2000.', '1800.', '2300.',
        '1500.', '2000.', '2000.', '2000.', '2000.', '2000.', '2000.', '2000.',
        '1800.', '1800.', '2000.', '2000.', '2000.', '2000.'),
    "fuelmce": (
        '0.12', '0.15', '0.25', '0.20', '0.20', '0.25', '0.40', '0.30', '0.25',
        '0.25', '0.15', '0.20', '0.25', '0.12', '0.15', '0.15', '0.30', '0.15',
        '0.40', '0.40', '0.15', '0.30', '0.40', '0.15', '0.15', '0.40', '0.40',
        '0.15', '0.15', '0.40', '0.30', '0.15', '0.30', '0.15', '0.40', '0.40',
        '0.20', '0.30', '0.30', '0.12', '0.25', '0.30', '0.25', '0.20', '0.25',
        '0.25', '0.25', '0.25', '0.35', '0.35', '0.25', '0.25', '0.25', '0.25'),
    "weight": (
        '7.', '7.', '7.', '180.', '100.', '100.', '100.', '900.', '900.',
        '900.', '900.', '900.', '900.', '7.') + ('7.',) * 13 + (
        '100.', '100.', '100.', '100.', '180.', '100.', '180.', '100.',
        '100.') + ('900.',) * 18,
    "ichap": ('0', '0', '0', '1') + ('0',) * 50,
    "fueldens": ('32.',) * 54,
    "st": ('0.0555',) * 54,
    "se": ('0.010',) * 54,
}

#: test/em_fire/namelist.fire &fuel_categories, categories 1-14.
WRF471_NAMELIST: dict[str, tuple[str, ...]] = {
    "fgi": ('0.166', '0.897', '0.675', '2.468', '0.785', '1.345', '1.092',
            '1.121', '0.780', '2.694', '2.582', '7.749', '13.024', '1.e-7'),
    "fueldepthm": ('0.305', '0.305', '0.762', '1.829', '0.61', '0.762',
                   '0.762', '0.061', '0.061', '0.305', '0.305', '0.701',
                   '0.914', '0.305'),
    "savr": ('3500.', '2784.', '1500.', '1739.', '1683.', '1564.', '1562.',
             '1889.', '2484.', '1764.', '1182.', '1145.', '1159.', '3500.'),
    "fuelmce": ('0.12', '0.15', '0.25', '0.20', '0.20', '0.25', '0.40',
                '0.30', '0.25', '0.25', '0.15', '0.20', '0.25', '0.12'),
    "fueldens": ('32.',) * 14,
    "st": ('0.0555',) * 14,
    "se": ('0.010',) * 14,
    "weight": ('7.', '7.', '7.', '180.', '100.', '100.', '100.', '900.',
               '900.', '900.', '900.', '900.', '900.', '7.'),
}

#: &fuel_scalars as literal text: the DATA defaults, which namelist.fire
#: repeats except for nfuelcats.
WRF471_SCALARS: dict[str, str] = {
    "cmbcnst": "17.433e+06", "hfgl": "17.e4", "fuelmc_g": "0.08",
    "fuelmc_g_lh": "1.20", "fuelmc_c": "1.00",
}
WRF471_NFUELCATS = {"defaults": 53, "namelist": 13}
WRF471_NO_FUEL_CAT = 14

#: set_fire_params' ksb map, fuel number -> WRF category (1-based); every
#: other number in 1..204 maps to the no-fuel category 14.
WRF471_KSB: dict[int, int] = {
    **{k: k for k in range(1, 14)},
    **{101 + i: 15 + i for i in range(9)},      # GR1-GR9
    **{121 + i: 24 + i for i in range(4)},      # GS1-GS4
    **{141 + i: 28 + i for i in range(9)},      # SH1-SH9
    **{161 + i: 37 + i for i in range(5)},      # TU1-TU5
    **{181 + i: 42 + i for i in range(9)},      # TL1-TL9
    **{201 + i: 51 + i for i in range(4)},      # SB1-SB4
}
WRF471_KSB_SIZE = 204

"""uvspec-like options for woof.radtran (a documented subset).

The names and meanings follow libRadtran's uvspec input options so a woof
case can be written as a uvspec input file for the oracle
(:meth:`Options.to_uvspec`) and a uvspec input in this subset can be read
(:meth:`Options.from_uvspec`).  Supported:

    atmosphere_file <afgl name or libRadtran atmmod path>
    mol_abs_param reptran [coarse|medium|fine]
    wavelength <lo> [<hi>]                     nm; every overlapping band
    rte_solver disort | twostr
    number_of_streams <n>                      default 6 (16 with umu)
    sza <deg>   phi0 <deg>   umu <...>   phi <...>
    albedo <A>
    pseudospherical   earth_radius <km>
    deltam on|off                              default on (disort: f = chi_2N; twostr: f = g^2)
    disort_intcor moments|off                  woof: TMS only (no IMS)
    zout <km ...|sur|toa>                      levels of the column only
    output_quantity transmittance              default: absolute irradiance
    day_of_year <n>                            Spencer (1971) Sun-Earth distance
    wc_file 1D <file>  wc_properties hu
    ic_file 1D <file>  ic_properties fu  ic_fu reff_def on|off
    no_absorption [mol]  no_scattering [mol]

Ignored (accepted for oracle inputs): data_files_path, quiet, verbose,
source solar (without file), output_user, mol_file (the oracle writes these).
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class Options:
    wavelength: tuple = (550.0, 550.0)
    mol_abs_param: str = "reptran coarse"
    rte_solver: str = "disort"
    number_of_streams: int | None = None
    sza: float = 0.0
    phi0: float = 0.0
    umu: tuple = ()
    phi: tuple = ()
    albedo: float = 0.0
    pseudospherical: bool = False
    earth_radius: float = 6370.0
    deltam: bool | None = None
    disort_intcor: str = "moments"
    zout: tuple = ("sur",)
    output_quantity: str = "irradiance"
    day_of_year: int | None = None
    atmosphere_file: str = "us"
    wc_file: str | None = None
    wc_properties: str = "hu"
    ic_file: str | None = None
    ic_properties: str = "fu"
    ic_fu_reff_def: bool = False
    no_absorption: bool = False
    no_scattering: bool = False

    # ---- derived ----
    @property
    def streams(self) -> int:
        if self.rte_solver == "twostr":
            return 2
        if self.number_of_streams is not None:
            return int(self.number_of_streams)
        return 16 if self.umu else 6

    @property
    def reptran_resolution(self) -> str:
        parts = self.mol_abs_param.split()
        if parts[0] != "reptran":
            raise ValueError("woof.radtran supports mol_abs_param reptran [coarse|medium|fine]")
        return parts[1] if len(parts) > 1 else "coarse"

    @property
    def use_deltam(self) -> bool:
        # on by default for both solvers (uvspec's behaviour: disort truncates
        # at moment 2N, twostr with f = g^2)
        return True if self.deltam is None else bool(self.deltam)

    def validate(self) -> "Options":
        if self.rte_solver not in ("disort", "twostr"):
            raise ValueError("rte_solver must be disort or twostr")
        if self.rte_solver == "twostr" and self.umu:
            raise ValueError("twostr computes irradiances only (no umu)")
        if self.disort_intcor not in ("moments", "off"):
            raise ValueError("disort_intcor must be moments or off (woof implements the TMS correction)")
        if self.output_quantity not in ("irradiance", "transmittance"):
            raise ValueError("output_quantity must be irradiance (default) or transmittance")
        lo, hi = self.wavelength
        if hi < lo:
            raise ValueError("wavelength range must be increasing")
        if not (0.0 <= self.albedo <= 1.0):
            raise ValueError("albedo must be within [0, 1]")
        if not (0.0 <= self.sza < 90.0):
            raise ValueError("sza must be within [0, 90) degrees")
        self.reptran_resolution
        return self

    # ---- uvspec text ----
    @classmethod
    def from_uvspec(cls, text: str, *, base: Path | None = None) -> "Options":
        kw = {}
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tok = shlex.split(line)
            key, args = tok[0], tok[1:]
            if key in ("data_files_path", "quiet", "verbose", "output_user", "mol_file",
                       "print_disort_info"):
                continue
            if key == "source":
                if args[0] != "solar" or len(args) > 1:
                    raise ValueError("woof.radtran: source solar (REPTRAN's own extraterrestrial flux) only")
                continue
            if key == "atmosphere_file":
                kw["atmosphere_file"] = args[0]
            elif key == "mol_abs_param":
                kw["mol_abs_param"] = " ".join(args)
            elif key == "wavelength":
                lo = float(args[0])
                kw["wavelength"] = (lo, float(args[1]) if len(args) > 1 else lo)
            elif key == "rte_solver":
                kw["rte_solver"] = args[0]
            elif key == "number_of_streams":
                kw["number_of_streams"] = int(args[0])
            elif key in ("sza", "phi0", "albedo", "earth_radius"):
                kw[key] = float(args[0])
            elif key in ("umu", "phi"):
                kw[key] = tuple(float(a) for a in args)
            elif key == "pseudospherical":
                kw["pseudospherical"] = True
            elif key == "deltam":
                kw["deltam"] = args[0] == "on"
            elif key == "disort_intcor":
                kw["disort_intcor"] = args[0]
            elif key == "zout":
                kw["zout"] = tuple(a if a in ("sur", "toa") else float(a) for a in args)
            elif key == "output_quantity":
                kw["output_quantity"] = args[0]
            elif key == "day_of_year":
                kw["day_of_year"] = int(args[0])
            elif key in ("wc_file", "ic_file"):
                if args[0] != "1D":
                    raise ValueError(f"{key}: only the 1D file type is supported")
                p = Path(args[1])
                kw[key] = str(p if p.is_absolute() or base is None else base / p)
            elif key == "wc_properties":
                kw["wc_properties"] = args[0]
            elif key == "ic_properties":
                kw["ic_properties"] = args[0]
            elif key == "ic_fu":
                if args[0] == "reff_def":
                    kw["ic_fu_reff_def"] = args[1] == "on"
                elif args[0] == "deltascaling" and args[1] == "on":
                    raise ValueError("ic_fu deltascaling on is not supported")
            elif key == "no_absorption":
                kw["no_absorption"] = True
            elif key == "no_scattering":
                kw["no_scattering"] = True
            else:
                raise ValueError(f"woof.radtran does not support uvspec option {key!r}")
        return cls(**kw).validate()

    def to_uvspec(self, *, data_path: str, atmosphere_path: str | None = None,
                  extra: tuple = ()) -> str:
        lines = [f"data_files_path {data_path}"]
        atm = atmosphere_path or f"{data_path}/atmmod/{_afgl_stem(self.atmosphere_file)}.dat"
        lines.append(f"atmosphere_file {atm}")
        lines.append("source solar")
        lines.append(f"mol_abs_param {self.mol_abs_param}")
        lo, hi = self.wavelength
        lines.append(f"wavelength {lo!r} {hi!r}")
        lines.append(f"rte_solver {self.rte_solver}")
        if self.rte_solver != "twostr":
            lines.append(f"number_of_streams {self.streams}")
        lines.append(f"sza {self.sza!r}")
        lines.append(f"phi0 {self.phi0!r}")
        if self.umu:
            lines.append("umu " + " ".join(repr(u) for u in self.umu))
        if self.phi:
            lines.append("phi " + " ".join(repr(p) for p in self.phi))
        lines.append(f"albedo {self.albedo!r}")
        if self.pseudospherical:
            lines.append("pseudospherical")
            lines.append(f"earth_radius {self.earth_radius!r}")
        if self.deltam is not None:
            lines.append(f"deltam {'on' if self.deltam else 'off'}")
        if self.rte_solver == "disort":
            lines.append(f"disort_intcor {self.disort_intcor}")
        lines.append("zout " + " ".join(str(z) for z in self.zout))
        if self.output_quantity == "transmittance":
            lines.append("output_quantity transmittance")
        if self.day_of_year is not None:
            lines.append(f"day_of_year {self.day_of_year}")
        if self.wc_file:
            lines.append(f"wc_file 1D {self.wc_file}")
            lines.append(f"wc_properties {self.wc_properties}")
        if self.ic_file:
            lines.append(f"ic_file 1D {self.ic_file}")
            lines.append(f"ic_properties {self.ic_properties}")
            lines.append(f"ic_fu reff_def {'on' if self.ic_fu_reff_def else 'off'}")
        if self.no_absorption:
            lines.append("no_absorption")
        if self.no_scattering:
            lines.append("no_scattering")
        lines.extend(extra)
        return "\n".join(lines) + "\n"

    def with_(self, **kw) -> "Options":
        return replace(self, **kw).validate()


def _afgl_stem(name: str) -> str:
    from woof.radtran.atmosphere import AFGL
    return AFGL.get(name, name)

"""Cut MPAS-Model v8.4.1's own routines, byte for byte, into oracle modules.

    python3 extract.py MPAS_SOURCE_ROOT BUILD_DIR [--print-pins]

Each routine is found by its own header and ``end`` line and its text is
checked against the sha256 pinned below before anything is written; no
statement is edited.  The routines go into three wrapper modules whose USE
lines the stubs (stubs.F90) satisfy:

* ``mpas_vector_operations`` -- ``mpas_cross_product_in_r3``,
  ``mpas_vec_mag_in_r3`` (src/operators/mpas_vector_operations.F);
* ``pn_mpas_geometry`` -- the barycentric chain of
  src/operators/mpas_geometry_utils.F: ``mpas_arc_length``,
  ``mpas_distance_plane``, ``mpas_distance``, ``mpas_triangle_signed_area``,
  ``mpas_triangle_signed_area_plane``, ``mpas_triangle_signed_area_sphere``,
  ``mpas_point_in_polygon``, ``mpas_calculate_barycentric_weights``,
  ``mpas_calculate_barycentric_weights_for_points``,
  ``mpas_cells_to_points_using_baryweights``;
* ``pn_mpas_relax`` -- ``atm_bdy_adjust_dynamics_relaxzone_tend``
  (src/core_atmosphere/dynamics/mpas_atm_time_integration.F).
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

GEOMETRY = "src/operators/mpas_geometry_utils.F"
VECTOR = "src/operators/mpas_vector_operations.F"
DYNAMICS = "src/core_atmosphere/dynamics/mpas_atm_time_integration.F"

#: (file, routine, sha256 of the cut text).
BLOCKS = (
    (VECTOR, "mpas_cross_product_in_r3", "b44611b12c1b59d3ab4a537036e1b7f6b9de5d8cea781b12588255bbe89fd6ea"),
    (VECTOR, "mpas_vec_mag_in_r3", "7a063ab40832d55b72353553bfbe70b023a3db4d79615dfe13911bfe1896b4e5"),
    (GEOMETRY, "mpas_arc_length", "551c4e53f57235ffe641fd9a61eeea211e382e144fcb53e74ab273520165259b"),
    (GEOMETRY, "mpas_distance_plane", "9a5540808c91d9382ebc63acc1a320a0e1338415ae94ddf70f88ce518c7f33df"),
    (GEOMETRY, "mpas_distance", "d0b74b8c322ada9be881e513e915a835029fd4acdde481ae259ebb12efa5a3e5"),
    (GEOMETRY, "mpas_calculate_barycentric_weights", "ccb140447b2221da3c2e53b4ef4091d7601331f5ab03d3a28ebaf4eb1dfde95b"),
    (GEOMETRY, "mpas_calculate_barycentric_weights_for_points", "86e97d44acf3d6bc4170972f082f20b3a258069df213382f91f1adae5af5df15"),
    (GEOMETRY, "mpas_triangle_signed_area", "5273f013812ca4f6ba51250bac03bf660696231085d8b021984df8c50e784878"),
    (GEOMETRY, "mpas_triangle_signed_area_plane", "559e96f3ac16c055f74b9349503ec592cad0e1bb990257f553f56353efbf9273"),
    (GEOMETRY, "mpas_triangle_signed_area_sphere", "4799fb0796948bfb8509475d3586f6b33dba585980d7e3b84f1efd5dade6aa88"),
    (GEOMETRY, "mpas_point_in_polygon", "dc5ee673fdaaad4e8a8bb4065bb3c518707987e375b95f4997a6540436cbecad"),
    (GEOMETRY, "mpas_cells_to_points_using_baryweights", "e3c22e7afca641bbcb6985780c490fe34d3db04815ca2a7329849d92bb8f70fc"),
    (DYNAMICS, "atm_bdy_adjust_dynamics_relaxzone_tend", "2bc796b3335671fc4887cdb4b62ff3020020cf1e5ece01f2421e842bac0a47ae"),
)

MODULES = {
    "mpas_vector_operations": (VECTOR, ("use mpas_kind_types",)),
    "pn_mpas_geometry": (GEOMETRY, ("use mpas_kind_types", "use mpas_derived_types",
                                    "use mpas_pool_routines", "use mpas_vector_operations",
                                    "use mpas_log")),
    "pn_mpas_relax": (DYNAMICS, ("use mpas_kind_types", "use mpas_derived_types",
                                 "use mpas_pool_routines", "use pn_relax_constants")),
}


def cut(text: str, name: str) -> str:
    """The routine ``name`` from its header line to its ``end`` line."""
    lines = text.splitlines(keepends=True)
    head = re.compile(r"^\s*(?:[\w() =,]*\s)?(?:function|subroutine)\s+" + re.escape(name)
                      + r"\s*\(", re.IGNORECASE)
    tail = re.compile(r"^\s*end\s+(?:function|subroutine)\s+" + re.escape(name) + r"\b",
                      re.IGNORECASE)
    start = next(i for i, line in enumerate(lines) if head.match(line))
    stop = next(i for i in range(start, len(lines)) if tail.match(lines[i]))
    return "".join(lines[start:stop + 1])


def main(argv: list[str]) -> int:
    root, build = Path(argv[1]), Path(argv[2])
    show = "--print-pins" in argv
    texts = {f: (root / f).read_text() for f in {b[0] for b in BLOCKS}}
    cuts: dict[str, str] = {}
    bad = []
    for path, name, pin in BLOCKS:
        block = cut(texts[path], name)
        digest = hashlib.sha256(block.encode()).hexdigest()
        if show:
            print(f'    ({path!r}, "{name}", "{digest}"),')
        elif digest != pin:
            bad.append(f"{path}:{name} sha256 {digest} != pin {pin}")
        cuts[name] = (path, block)
    if bad:
        print("\n".join(bad), file=sys.stderr)
        return 3
    if show:
        return 0
    build.mkdir(parents=True, exist_ok=True)
    for module, (path, uses) in MODULES.items():
        body = "".join(block for p, block in cuts.values() if p == path)
        text = (f"! Cut from MPAS-Model v8.4.1 {path} by extract.py; statements unedited.\n"
                f"module {module}\n" + "".join(f"   {u}\n" for u in uses)
                + "   implicit none\ncontains\n" + body + f"end module {module}\n")
        (build / f"{module}.F90").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

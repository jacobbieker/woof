"""The pinned MPAS-Model oracle for the coupled cores.

The atmosphere (hex) and the ocean, sea-ice and land-ice cores are all pinned
to one MPAS-Model snapshot so they are mutually consistent and the atmosphere
matches what the hex port already proved. That snapshot is v8.4.2, whose tag is
the ``master`` head and the exact commit the hex (MPAS-Atmosphere) port pins.
It carries all four cores (``core_atmosphere``, ``core_ocean``, ``core_seaice``,
``core_landice``) and the coupled ``driver`` in one unified tree.

The frozen 2021 per-component develop branches (``ocean/develop``,
``seaice/develop``, ``landice/develop``) are legacy, not the oracle.
"""

#: The public oracle.
ORACLE_REPO = "MPAS-Dev/MPAS-Model"
ORACLE_TAG = "v8.4.2"
ORACLE_PIN = "f34984b2c89353cf31c235b08d3b4acd96b37bb3"

#: The strict reference build recipe the bit-exact proofs are measured against,
#: mirroring the hex (MPAS-Atmosphere) pin: single precision, no FP contraction,
#: no vectorised reassociation. See components/hex/README.md.
STRICT_RECIPE = {
    "compiler": "GNU Fortran",
    "precision": "single (real32)",
    "fp_contraction": "off",
    "reassociation": "off",
    "note": "the same strict recipe the hex port pins MPAS-A v8.4.2 to",
}

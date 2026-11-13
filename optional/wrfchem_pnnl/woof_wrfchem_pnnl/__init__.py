# WOOF-OPTIN: wrfchem_pnnl
"""WRF-Chem's Battelle/PNNL routines for WOOF: an explicit opt-in.

See ../LICENSE and ../NOTICE.md.  This package is in no wheel and no default
build; it is reached only through ``woof.optin.load("wrfchem_pnnl")`` with
``WOOF_ENABLE_WRFCHEM_PNNL=1`` in a source checkout.
"""

from __future__ import annotations

from pathlib import Path

#: optional/wrfchem_pnnl, the directory holding LICENSE, NOTICE.md and oracle/.
ROOT = Path(__file__).resolve().parents[1]
#: WRF v4.7.1 (f52c197e) files this opt-in transcribes, by sha256.
WRF_PINS = {
    "phys/module_mixactivate.F":
        "d5192f37a45a2b4084165789bdff827813043dd8d16db0958d9c652132c6e9a1",
    "share/module_model_constants.F":
        "5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062",
}

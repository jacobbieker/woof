# WOOF-OPTIN: wrfchem_pnnl
"""Compile an opt-in CUDA unit with IEEE subnormals kept.

CuPy appends ``-ftz=true`` after a RawModule's own options, so a unit asking
for ``--ftz=false`` still flushed binary32 subnormals to zero, which the
Fortran oracles these units are graded against do not do.  On the CUDA
backend the source carries ``woof.cuda_subnormals``' marker, whose hook
rewrites CuPy's final options to ``--ftz=false``; other backends compile
the source as given.
"""

from __future__ import annotations


def raw_module(source: str, options):
    import cupy as cp

    from woof import backend

    if backend.active() == "cuda":
        from woof.cuda_subnormals import MARKER, install

        install()
        source = MARKER + source
    options = tuple(o for o in options if o.lstrip("-").split("=", 1)[0] != "ftz")
    return cp.RawModule(code=source, options=options + ("--ftz=false",))

"""Expand tools/vendor_libm64.cuh.in into woof/backend/translate/vendor_libm64.cuh.

The same D()/F() expansion as tools/portable_libm64_header.py (one RN
intrinsic per AST node); helper and constant names take the ``wlm_`` prefix
so the header can sit beside portable_libm64.cuh in one unit.
This is a development tool.  The generated header is the shipped source.
"""
import re
from pathlib import Path

from portable_libm64_header import expand

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    template = (ROOT / "tools/vendor_libm64.cuh.in").read_text(encoding="utf-8")
    code_start = template.index("__device__")
    body = template[code_start:].replace("plm_", "wlm_")
    names = {n for n in set(re.findall(r"\b[A-Z][A-Z0-9_]*\b", body)) - {"D", "F", "WOOF_VENDOR_LIBM64"}
             if not n.startswith("WCR_")}          # f64cr.cuh's own macros, which this header calls
    body = re.sub(r"\b[A-Z][A-Z0-9_]*\b", lambda m: "wlm_" + m[0] if m[0] in names else m[0], body)
    out = ROOT / "woof/backend/translate/vendor_libm64.cuh"
    # expand() spells its negation helper plm_neg*; this header's is wlm_.
    text = expand(template[:code_start] + body).replace("plm_neg", "wlm_neg")
    out.write_text(text, encoding="utf-8", newline="\n")

"""Generate utan_tbl.rs from glibc 2.39 sysdeps/ieee754/dbl-64/utan.tbl.

Reads the LITTLE_ENDI block; each mynumber {{lo, hi}} becomes the u64
hi << 32 | lo.  Usage: python utan.py <utan.tbl> <out.rs> <header.txt>, where header.txt is
the opening comment of the committed utan_tbl.rs.
"""
import re, sys
src = open(sys.argv[1]).read()
block = src[src.index("#ifdef LITTLE_ENDI"):]
block = block[:block.index("#endif")]
pairs = re.findall(r"\{\{\s*0x([0-9a-fA-F]+)\s*,\s*0x([0-9a-fA-F]+)\s*\}\s*\}", block)
assert len(pairs) == 186 * 4, len(pairs)
words = [(int(hi, 16) << 32) | int(lo, 16) for lo, hi in pairs]
out = [open(sys.argv[3]).read().rstrip("\n"), "",
       "/// xfg[i] = (xi, Fi, Gi, FFi), i = 16..201 in glibc's numbering.",
       "pub(super) static XFG: [[u64; 4]; 186] = ["]
for i in range(186):
    w = words[4 * i:4 * i + 4]
    out.append("    [" + ", ".join(f"0x{v:016x}" for v in w) + "],")
out.append("];")
open(sys.argv[2], "w").write("\n".join(out) + "\n")

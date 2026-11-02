"""Convert glibc 2.39 uatan.tbl (LITTLE_ENDI block) into uatan_tbl.rs's table.

Usage: python uatan.py <uatan.tbl>; prints the table."""
import re, sys
src = open(sys.argv[1]).read()
le = src[src.index("#ifdef LITTLE_ENDI"):]
pairs = re.findall(r"\{\{\s*0[xX]([0-9a-fA-F]{8}),\s*0[xX]([0-9a-fA-F]{8})\s*\}\s*\}", le)
assert len(pairs) == 241 * 7, len(pairs)
words = [(int(hi, 16) << 32) | int(lo, 16) for lo, hi in pairs]
# cross-check against the BIG_ENDI block (hi, lo)
be = src[src.index("#ifdef BIG_ENDI"):src.index("#ifdef LITTLE_ENDI")]
bp = re.findall(r"\{\{\s*0[xX]([0-9a-fA-F]{8}),\s*0[xX]([0-9a-fA-F]{8})\s*\}\s*\}", be)
assert [(int(h, 16) << 32) | int(l, 16) for h, l in bp] == words
out = []
for r in range(241):
    row = ", ".join(f"0x{w:016x}" for w in words[r * 7:(r + 1) * 7])
    out.append(f"    [{row}],")
print("\n".join(out))

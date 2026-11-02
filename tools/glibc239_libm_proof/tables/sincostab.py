"""Generate sincostab.rs's table from glibc 2.39 sysdeps/ieee754/dbl-64/sincostab.c
(usage: python sincostab.py <sincostab.c>; prints the table).

The LITTLE_ENDI block lists int pairs (low word, high word); each pair is
one double, emitted as its u64 bit pattern hi << 32 | lo.
"""
import re, sys
src = open(sys.argv[1]).read()
block = src[src.index("#ifdef LITTLE_ENDI"):]
block = block[:block.index("} };")]
words = [int(w, 16) for w in re.findall(r"0x([0-9A-Fa-f]{8})", block)]
assert len(words) == 880, len(words)
vals = [(words[2*i+1] << 32) | words[2*i] for i in range(440)]
out = ["pub(super) static SINCOSTAB: [u64; 440] = ["]
for i in range(0, 440, 4):
    out.append("    " + " ".join(f"0x{v:016x}," for v in vals[i:i+4]))
out.append("];")
print("\n".join(out))

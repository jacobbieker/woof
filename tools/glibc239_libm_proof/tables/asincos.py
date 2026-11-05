"""Generate asincos_tbl.rs from glibc 2.39's asincos.tbl, root.tbl
and powtwo.tbl (sysdeps/ieee754/dbl-64).  asincos.tbl's LITTLE_ENDI block
holds (low, high) 32-bit word pairs; each becomes one u64 bit pattern.
root.tbl and powtwo.tbl are decimal literals, copied verbatim.  Usage:
python asincos.py <dbl-64 dir> <out.rs> <header.txt>, where header.txt is the
opening comment of the committed asincos_tbl.rs."""
import re, sys
src = sys.argv[1]
out = sys.argv[2]
text = open(f"{src}/asincos.tbl").read()
le = text[text.index("#ifdef LITTLE_ENDI"):]
le = le[le.index(".i = {"):le.index("} };")]
words = [int(w, 16) for w in re.findall(r"0x([0-9A-Fa-f]{8})", le)]
assert len(words) == 5136, len(words)
asncs = [(hi << 32) | lo for lo, hi in zip(words[0::2], words[1::2])]
# cross-check against the BIG_ENDI block
be = text[text.index("#ifdef BIG_ENDI"):text.index("#ifdef LITTLE_ENDI")]
bw = [int(w, 16) for w in re.findall(r"0x([0-9A-Fa-f]{8})", be)]
assert [(h << 32) | l for h, l in zip(bw[0::2], bw[1::2])] == asncs
def decimals(name, path):
    t = open(path).read()
    body = t[t.index(name):]
    body = body[body.index("{") + 1:body.index("}")]
    return [v.strip() for v in body.replace("\n", " ").split(",") if v.strip()]
inroot = decimals("inroot[128]", f"{src}/root.tbl")
powtwo = decimals("powtwo[]", f"{src}/powtwo.tbl")
assert len(inroot) == 128 and len(powtwo) == 28
header = open(sys.argv[3]).read()
with open(out, "w") as f:
    f.write(header)
    f.write("\n// asincos.tbl `asncs` (LITTLE_ENDI word pairs as binary64 bit patterns),\n"
            "// root.tbl `inroot` and powtwo.tbl `powtwo`.  Generated mechanically.\n\n")
    f.write(f"pub(super) static ASNCS: [u64; {len(asncs)}] = [\n")
    for i in range(0, len(asncs), 4):
        f.write("    " + " ".join(f"0x{v:016x}," for v in asncs[i:i + 4]) + "\n")
    f.write("];\n\n")
    f.write("pub(super) static INROOT: [f64; 128] = [\n")
    for i in range(0, 128, 4):
        f.write("    " + " ".join(f"{v}," for v in inroot[i:i + 4]) + "\n")
    f.write("];\n\n")
    f.write("pub(super) static POWTWO: [f64; 28] = [\n")
    for i in range(0, 28, 4):
        f.write("    " + " ".join(f"{v}," for v in powtwo[i:i + 4]) + "\n")
    f.write("];\n")

"""Emit the exp, log and pow tables of glibc239-libm as u64 bit patterns.

Usage: python arm_data.py <dir> exp|log|pow, where <dir> holds e_exp_data.c,
e_log_data.c and e_pow_log_data.c: glibc 2.39's sysdeps/ieee754/dbl-64, or
the Arm upstream files in tools/uwpbl_wrf471_oracle/libm64/upstream run
through `gcc -E -P` (the committed tables; both give the same words)."""
import re, struct, sys
D = sys.argv[1]  # glibc-2.39/sysdeps/ieee754/dbl-64
NUM = re.compile(r'-?0x[0-9a-fA-F]*\.?[0-9a-fA-F]*p[+-]?\d+(?:\s*\*\s*-?\d+)?|-?0x[0-9a-fA-F]+')
def val(tok):
    tok = tok.replace(' ', '')
    if 'p' in tok:
        mult = 1
        if '*' in tok:
            tok, m = tok.split('*'); mult = int(m)
        return float.fromhex(tok) * mult
    return int(tok, 16)
def bits(v):
    if isinstance(v, int): return v
    return struct.unpack('<Q', struct.pack('<d', v))[0]
def block(text, start, end='},'):
    i = text.index(start); j = text.index(end, i + len(start)); body = text[i + len(start):j]
    body = '\n'.join(l for l in body.splitlines() if not l.strip().startswith(('//', '#')))
    body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
    return [val(t) for t in NUM.findall(body)]
def emit(name, values, per=2):
    out = [f"pub(super) static {name}: [u64; {len(values)}] = ["]
    for k in range(0, len(values), per):
        out.append("    " + " ".join(f"{bits(v):#018x}," for v in values[k:k + per]))
    out.append("];")
    return "\n".join(out)
def rd(p, m):
    t = open(p).read()
    return t[t.index(m):] if m in t else t
exp = rd(f"{D}/e_exp_data.c", "const struct exp_data")
log = rd(f"{D}/e_log_data.c", "const struct log_data")
powd = rd(f"{D}/e_pow_log_data.c", "const struct pow_log_data")
which = sys.argv[2]
if which == "exp":
    poly = block(exp, ".poly = {"); assert len(poly) == 4
    tab = block(exp, ".tab = {", "#endif" if "#endif" in exp else "},"); assert len(tab) == 256, len(tab)
    print(emit("POLY", poly, 1)); print(emit("TAB", tab))
elif which == "log":
    p1 = block(log, ".poly1 = {"); assert len(p1) == 11
    p = block(log, ".poly = {"); assert len(p) == 5
    t = log[log.index(".tab = {"):]; t = t[:t.index("#endif")] if "#endif" in t else t[:t.index("};")]
    t = t.split(".tab2")[0]
    pairs = re.findall(r'\{(' + NUM.pattern + r'),\s*(' + NUM.pattern + r')\}', t)
    tab = [val(x) for pr in pairs for x in pr]; assert len(tab) == 256, len(tab)
    print(emit("POLY1", p1, 1)); print(emit("POLY", p, 1)); print(emit("TAB", tab))
elif which == "pow":
    p = block(powd, ".poly = {"); assert len(p) == 7, p
    t = powd[powd.index(".tab = {"):]; t = t[:t.index("#endif")] if "#endif" in t else t[:t.index("};")]
    rows = re.findall(r"^A\((.*)\)\s*$", t, flags=re.M) or [r.replace(", 0,", ",", 1) for r in re.findall(r"^\{(.*)\},\s*$", t, flags=re.M)]
    tab = []
    for r in rows:
        a, b, c = [val(x) for x in NUM.findall(r)]; tab += [a, b, c]
    assert len(tab) == 384, len(tab)
    print(emit("POLY", p, 1)); print(emit("TAB", tab, 3))

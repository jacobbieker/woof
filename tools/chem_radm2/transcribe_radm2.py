"""Transcribe WRF-Chem's RADM2 solver statements into chem_radm2.cu.

    pixi run -e default python tools/chem_radm2/transcribe_radm2.py WRF_SRC [--check]

WRF_SRC is a WRF v4.6.1 tree; chem/module_radm.F and chem/module_data_radm2.F
must match their pins in tools/chem_wrf461_oracle/radm2/SOURCES.sha256.  The
script rewrites the block between ``// BEGIN GENERATED`` and ``// END
GENERATED`` in woof/core/kernels/chem_radm2.cu (``--check`` only compares).

What is transcribed, statement for statement and in source order, is the
per-level arithmetic of ``radm`` (rate constants, module_radm.F:396-492),
``integ1n`` (591-843), ``predraten`` (865-1137) and ``producn`` (1167-1633),
plus the DATA tables of module_data_radm2.F.  The vertical ``DO j = jcs,
jce`` loops become the body for the one level a thread owns: every statement
inside them reads and writes only level j.  ``setdtc`` (the only cross-level
step) and the driver are hand-written in the .cu.

Arithmetic: gfortran -O0 evaluates left to right with no reassociation or
contraction, so every float operation becomes FADD/FSUB/FMUL/FDIV (rounded,
never fused) in the source's own association.  An operand of kind 8 (the one
``2.D0`` at module_radm.F:1288) promotes the rest of that expression to
binary64 exactly as Fortran does (DADD/DMUL), rounded to float on
assignment.  Integer powers of 2 are a multiply (gfortran expands them);
real powers call WOOF's glibc powf, EXP glibc expf, LOG10 glibc log10f.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "woof/core/kernels/chem_radm2.cu"
PINS = ROOT / "tools/chem_wrf461_oracle/radm2/SOURCES.sha256"
RANGES = {"rates": (396, 492), "integ1n": (591, 843),
          "predraten": (865, 1137), "producn": (1167, 1633)}

# Arrays with the level index j first (dropped: a thread owns one level).
LEVEL_ARRAYS = {"rk", "crk", "rj", "crj", "dvc", "loss", "prod", "lossl",
                "prodl", "vcl", "prdrog", "vdrog"}
# Level scalars written as x(j) or x(j,1).
LEVEL_SCALARS = {"h2o", "t", "p", "tin", "pot", "to300", "patmot1", "patmot2",
                 "patmot3", "aquad", "bquad", "dum", "rh", "wlc", "tauinv",
                 "taulinv", "h2o2sv", "hoxsv", "jb", "jbb", "jc", "jloss",
                 "jprod"}
LOCAL_ARRAYS = {"crji", "crki"}
INT_VARS = {"l", "ir", "irdum", "niter", "iter", "inumho", "inumhox", "j",
            "k", "i", "kdum", "iaerosol_sorgam"}
REAL_SCALARS = {"tsqrd", "xk0", "xk2", "xk3", "eqno", "alow", "dtc", "r"}
DATA_ARRAYS = {"thafac": "R2_THAFAC", "eor": "R2_EOR", "xk0300": "R2_XK0300",
               "xntroe": "R2_XNTROE", "xkf300": "R2_XKF300",
               "xmtroe": "R2_XMTROE", "afac": "R2_AFAC", "bfac": "R2_BFAC",
               "const": "R2_CONST", "cmin": "R2_CMIN"}
INT_DATA = {"itroe": "R2_ITROE", "intgrt": "R2_INTGRT", "qdtc": "R2_QDTC"}
REAL_PARAMS = {"ch4": "1.7", "co2": "350.", "n2": "7.81E5", "o2": "2.09E5",
               "epsilc": "1.E-16", "c302": "5417.4", "c303": "19.83"}
FUNCS = {"exp": "gfk_exp", "log10": "gfk_log10"}


def read_params(data_text):
    """Integer PARAMETERs of module_data_radm2.F (species and loop extents)."""
    params = {}
    for name, value in re.findall(r"INTEGER,\s*PARAMETER\s*::\s*(\w+)\s*=\s*(\d+)\s*$",
                                  data_text, re.M | re.I):
        params[name.lower()] = int(value)
    block = re.search(r"INTEGER, PARAMETER :: ldiag.*?numchem_radm = 41", data_text, re.S)
    for name, value in re.findall(r"(\w+)\s*=\s*(\d+)", block.group(0)):
        params[name.lower()] = int(value)
    params["lspec"] = params["lpred"] + params["lss"]
    return params


def _values(text):
    out = []
    for item in text.replace("&", " ").replace("\n", " ").split(","):
        item = item.strip()
        if not item:
            continue
        if "*" in item:
            n, v = item.split("*")
            out += [v.strip()] * int(n)
        else:
            out.append(item)
    return out


def read_data(data_text, params):
    tables = {}
    for name, body in re.findall(r"DATA\s+(\w+)\s*/(.*?)/", data_text, re.S | re.I):
        tables[name.lower()] = _values(body)
    for name in ("itroe", "intgrt", "qdtc"):
        body = re.search(rf"::\s*{name}\s*=\s*\(/(.*?)/\)", data_text, re.S | re.I).group(1)
        tables[name] = _values(body)
    cmin = re.search(r"cmin\s*=\s*\(/\((.*?),itrdu=1,lspec\)/\)", data_text, re.I).group(1)
    tables["cmin"] = [cmin] * params["lspec"]
    return tables


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------
TOKEN = re.compile(r"""\s*(?:
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?(?:_8)?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<op>\*\*|>=|<=|==|/=|[-+*/(),<>])
)""", re.X)


def tokenize(text):
    pos, out = 0, []
    text = text.rstrip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m or m.end() == pos:
            raise SyntaxError(f"cannot tokenize {text[pos:]!r} in {text!r}")
        pos = m.end()
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
    return out


class Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, value=None):
        tok = self.peek()
        if value is not None and tok[1] != value:
            raise SyntaxError(f"expected {value!r}, got {tok!r}")
        self.i += 1
        return tok

    def compare(self):
        left = self.expr()
        if self.peek()[1] in (">=", "<=", "==", "/=", "<", ">"):
            op = self.take()[1]
            return ("cmp", op, left, self.expr())
        return left

    def expr(self):
        if self.peek()[1] in ("+", "-"):
            sign = self.take()[1]
            node = self.term()
            if sign == "-":
                node = ("neg", node)
        else:
            node = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.factor()
        while self.peek()[1] in ("*", "/"):
            op = self.take()[1]
            node = ("bin", op, node, self.factor())
        return node

    def factor(self):
        base = self.primary()
        if self.peek()[1] == "**":
            self.take()
            # ** is right-associative and binds tighter than unary minus.
            if self.peek()[1] == "-":
                self.take()
                return ("pow", base, ("neg", self.factor()))
            return ("pow", base, self.factor())
        return base

    def primary(self):
        kind, value = self.take()
        if kind == "num":
            return ("num", value)
        if value == "(":
            node = self.expr()
            self.take(")")
            return node
        if kind == "name":
            name = value.lower()
            if self.peek()[1] == "(":
                self.take("(")
                args = [self.expr()]
                while self.peek()[1] == ",":
                    self.take(",")
                    args.append(self.expr())
                self.take(")")
                return ("ref", name, args)
            return ("ref", name, [])
        raise SyntaxError(f"unexpected {value!r}")


class Emitter:
    def __init__(self, params):
        self.params = params

    def index(self, node):
        code, kind = self.emit(node)
        if kind != "int":
            raise TypeError(f"non-integer index {node}")
        try:
            return str(eval(f"({code})-1", {}))
        except Exception:
            return f"({code})-1"

    def number(self, text):
        if text.endswith("_8"):
            return text[:-2] + ("" if re.search(r"[.Ee]", text[:-2]) else ".0"), "double"
        if re.search(r"[Dd]", text):
            return re.sub(r"[Dd]", "E", text), "double"
        if re.search(r"[.Ee]", text):
            return text + "f", "float"
        return text, "int"

    def emit(self, node):
        tag = node[0]
        if tag == "num":
            return self.number(node[1])
        if tag == "neg":
            code, kind = self.emit(node[1])
            return f"(-{code})", kind
        if tag == "pow":
            base, kb = self.emit(node[1])
            if node[2] == ("num", "2"):
                return self.arith("*", (base, kb), (base, kb))
            exp, ke = self.emit(node[2])
            assert kb == "float" and ke == "float", node
            return f"gfk_pow({base},{exp})", "float"
        if tag == "bin":
            return self.arith(node[1], self.emit(node[2]), self.emit(node[3]))
        if tag == "cmp":
            a, b = self.emit(node[2]), self.emit(node[3])
            op = {"/=": "!="}.get(node[1], node[1])
            return f"({self.coerce(a, b)[0]} {op} {self.coerce(b, a)[0]})", "bool"
        if tag == "ref":
            return self.ref(node[1], node[2])
        raise ValueError(node)

    @staticmethod
    def coerce(a, b):
        code, kind = a
        other = b[1]
        if kind == "int" and other in ("float", "double"):
            return (code + ".f" if code.isdigit() and other == "float" else
                    code + ".0" if code.isdigit() else f"({other}){code}"), other
        if kind == "float" and other == "double":
            return f"(double){code}", "double"
        return code, kind

    def arith(self, op, a, b):
        ka, kb = a[1], b[1]
        if ka == kb == "int":
            return f"({a[0]}{op}{b[0]})", "int"
        kind = "double" if "double" in (ka, kb) else "float"
        x, _ = self.coerce(a, (None, kind))
        y, _ = self.coerce(b, (None, kind))
        prefix = "D" if kind == "double" else "F"
        name = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV"}[op]
        return f"{prefix}{name}({x},{y})", kind

    def ref(self, name, args):
        if name in FUNCS:
            (code, kind), = [self.emit(a) for a in args]
            assert kind == "float"
            return f"{FUNCS[name]}({code})", "float"
        if name in ("max", "amax1", "min", "amin1"):
            vals = [self.emit(a) for a in args]
            kind = "double" if any(k == "double" for _, k in vals) else "float"
            vals = [self.coerce(v, (None, kind))[0] for v in vals]
            fn = ("fmax" if name in ("max", "amax1") else "fmin") + ("f" if kind == "float" else "")
            code = vals[0]
            for v in vals[1:]:
                code = f"{fn}({code},{v})"
            return code, kind
        if name == "vc":
            return f"vc[{self.index(args[2])}]", "float"
        if name in LEVEL_ARRAYS:
            return f"{name}[{self.index(args[1])}]", "float"
        if name in LEVEL_SCALARS or name in REAL_SCALARS:
            return name, "float"
        if name in LOCAL_ARRAYS:
            return f"{name}[{self.index(args[0])}]", "float"
        if name in DATA_ARRAYS:
            return f"{DATA_ARRAYS[name]}[{self.index(args[0])}]", "float"
        if name in INT_DATA:
            return f"{INT_DATA[name]}[{self.index(args[0])}]", "int"
        if name in REAL_PARAMS:
            return REAL_PARAMS[name] + "f", "float"
        if name in INT_VARS:
            return name, "int"
        if name in self.params:
            return str(self.params[name]), "int"
        raise KeyError(f"unknown name {name!r}")


# --------------------------------------------------------------------------
# Statements
# --------------------------------------------------------------------------
def logical_lines(lines):
    """Join & continuations and drop comments; yield (lineno, text)."""
    parts, start = [], None
    for n, raw in lines:
        text = raw.split("!", 1)[0].strip()
        if not text:
            continue
        if start is None:
            start = n
        if text.startswith("&") and parts:
            text = text[1:].strip()
        cont = text.endswith("&")
        parts.append(text[:-1].strip() if cont else text)
        if not cont:
            yield start, " ".join(parts)
            parts, start = [], None


def split_assign(text):
    depth = 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "=" and depth == 0 and text[i - 1] not in "<>/=" and text[i + 1] != "=":
            return text[:i].strip(), text[i + 1:].strip()
    return None


def translate(lines, params, indent="    "):
    em = Emitter(params)
    out, depth = [], 1
    ignored_int = {"k", "i", "kdum"}

    def put(s):
        out.append(indent * depth + s)

    for n, text in logical_lines(lines):
        low = text.lower()
        m = re.match(r"do\s+(\w+)\s*=\s*(.+?)\s*,\s*(.+)$", low)
        if m:
            var, lo, hi = m.groups()
            if var == "j":
                put(f"{{ // {n}: DO j (this level)")
            else:
                put(f"for (int {var}={em.emit(Parser(tokenize(lo)).expr())[0]}; "
                    f"{var}<={em.emit(Parser(tokenize(hi)).expr())[0]}; ++{var}) {{ // {n}")
            depth += 1
            continue
        if re.fullmatch(r"end\s*do", low):
            depth -= 1
            put("}")
            continue
        m = re.match(r"if\s*\((.*)\)\s*then$", low)
        if m:
            put(f"if {em.emit(Parser(tokenize(m.group(1))).compare())[0]} {{ // {n}")
            depth += 1
            continue
        if low == "else":
            depth -= 1
            put("} else {")
            depth += 1
            continue
        if re.fullmatch(r"end\s*if", low):
            depth -= 1
            put("}")
            continue
        if low == "return":
            continue
        cond = None
        m = re.match(r"if\s*\(", low)
        if m:  # one-line IF (cond) stmt
            d, i = 0, low.index("(")
            for i in range(low.index("("), len(low)):
                d += low[i] == "("
                d -= low[i] == ")"
                if d == 0:
                    break
            cond = em.emit(Parser(tokenize(low[low.index("(") + 1:i])).compare())[0]
            low = low[i + 1:].strip()
        lhs, rhs = split_assign(low)
        lhs_node = Parser(tokenize(lhs)).primary()
        if lhs_node[1] in ignored_int:
            continue
        if lhs_node[1] == "prdrog" and not lhs_node[2]:
            stmt = f"for (int q=0; q<{params['ldrog']}; ++q) prdrog[q] = 0.f;"
        else:
            code, kind = em.emit(Parser(tokenize(rhs)).expr())
            target, tkind = em.emit(lhs_node)
            if tkind == "int":
                stmt = f"{target} = {code};"
            elif kind == "double":
                stmt = f"{target} = (float)({code});"
            elif kind == "int":
                stmt = f"{target} = {code}.f;" if code.isdigit() else f"{target} = (float)({code});"
            else:
                stmt = f"{target} = {code};"
        put((f"if {cond} " if cond else "") + stmt + f" // {n}")
    assert depth == 1, depth
    return out


def generate(src: Path) -> str:
    radm = (src / "chem/module_radm.F").read_text()
    data = "\n".join(line.split("!", 1)[0] for line in
                     (src / "chem/module_data_radm2.F").read_text().splitlines())
    params = read_params(data)
    params["ldrog"] = 17
    # module_data_sorgam.F:417-466, the DROG precursor slots.
    for n, name in enumerate(("pxyl", "ptol", "pcsl1", "pcsl2", "phc8", "poli1",
                              "poli2", "poli3", "polt1", "polt2", "polt3",
                              "papi1", "papi2", "papi3", "plim1", "plim2",
                              "plim3"), 1):
        params[name] = n
    tables = read_data(data, params)
    lines = list(enumerate(radm.splitlines(), 1))
    out = ["// BEGIN GENERATED by tools/chem_radm2/transcribe_radm2.py from",
           "// chem/module_radm.F and chem/module_data_radm2.F (WRF v4.6.1).",
           "// Do not edit by hand; edit the generator and re-run it."]
    for name, cname in {**DATA_ARRAYS, **INT_DATA}.items():
        vals = tables[name]
        if name in INT_DATA:
            body = ", ".join(v.strip() for v in vals)
            out.append(f"__constant__ int {cname}[{len(vals)}] = {{{body}}};")
        else:
            body = ", ".join(v.strip() + ("f" if not v.strip().endswith("f") else "")
                             for v in vals)
            out.append(f"__constant__ float {cname}[{len(vals)}] = {{{body}}};")
    sigs = {
        "rates": "__device__ void r2_rates(float t1, float p1, float rh1, float &h2o, float *rk, float *vc, float *vcl)",
        "predraten": "__device__ void r2_predraten(const float *rj, const float *rk, const float *vc, float *crj, float *crk, float *dvc, float h2o)",
        "producn": "__device__ void r2_producn(const float *crj, const float *crk, float *loss, float *prod, float *prodl, float *lossl, float *prdrog)",
        "integ1n": "__device__ void r2_integ1n(float dtc, float *vc, float *loss, float *prod, float *vcl, float *lossl, float *prodl, const float *rk, const float *dvc, float h2o, const float *rj, float *vdrog, const float *prdrog, int iaerosol_sorgam)",
    }
    locals_ = {
        "rates": ["float t = t1, p = p1, rh = rh1;",
                  "float tin, pot, to300, patmot1, patmot2, patmot3, aquad, bquad, dum, tsqrd, xk0, xk2, xk3;",
                  "int irdum;"],
        "predraten": [],
        "producn": ["float alow;"],
        "integ1n": ["float eqno, alow, tauinv, taulinv, h2o2sv, hoxsv, jb, jbb, jc, jloss, jprod;",
                    "float crji[21], crki[140];", "int inumho, inumhox;"],
    }
    for key in ("rates", "predraten", "producn", "integ1n"):
        lo, hi = RANGES[key]
        out.append(f"// module_radm.F:{lo}-{hi}")
        out.append(sigs[key] + " {")
        out += ["    " + s for s in locals_[key]]
        out += translate(lines[lo - 1:hi], params)
        out.append("}")
    out.append("// END GENERATED")
    return "\n".join(out) + "\n"


def check_pins(src: Path):
    for line in PINS.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split()
        got = hashlib.sha256((src / rel).read_bytes()).hexdigest()
        if got != digest:
            raise SystemExit(f"{rel}: sha256 {got}, pinned {digest}")


def main(argv):
    src = Path(argv[1])
    check_pins(src)
    block = generate(src)
    text = KERNEL.read_text()
    start = text.index("// BEGIN GENERATED")
    end = text.index("// END GENERATED") + len("// END GENERATED\n")
    new = text[:start] + block + text[end:]
    if "--check" in argv:
        if new != text:
            raise SystemExit("chem_radm2.cu's generated block is stale")
        return
    KERNEL.write_text(new)


if __name__ == "__main__":
    main(sys.argv)

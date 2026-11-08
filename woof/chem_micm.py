"""MICM gas-phase chemistry, one grid point per thread: NCAR MICM's Rosenbrock solver as WOOF kernels.

A mechanism is data: an NCAR MechanismConfiguration v1 document (JSON or YAML; the format CheMPAS-A and
MUSICA read).  :func:`load_mechanism` turns it into the solver MICM builds from it -- the same species
order, process order, Jacobian sparsity, LU fill pattern and operation order -- and :func:`kernel_source`
writes that solver as one straight-line CUDA translation unit (binary64, every operation an explicit
round-to-nearest add, multiply or divide; exp, pow and log10 correctly rounded through f64cr.cuh), which
the MLX backend translates for Metal like any other WOOF kernel.

What is transcribed, and from where (NCAR MICM, Apache-2.0, ``licenses/NOTICE-NCAR-MICM.txt``; the
oracle is MUSICA v0.17.2 with MICM e15898ce, ``RosenbrockStandardOrder``):

* musica ``src/configuration/parse.cpp`` ``ConvertChemistry``: species, the gas phase, reactant copies
  per (integral ceiling of) coefficient, the per-type conversion order and the ``PHOTO.``/``EMIS.``/
  ``LOSS.``/``USER.`` rate-parameter labels;
* MICM ``solver_builder.inl``: the stable sort of processes by rate-constant type, the custom parameter
  map, ``DiagonalMarkowitzReorder`` of the species (``linear_solver.inl``, unsigned ``size_t`` cost
  arithmetic included) and the Jacobian (its nonzeros plus the diagonal);
* MICM ``process_set.hpp``: forcing and negative-Jacobian accumulation order;
* MICM ``lu_decomposition_doolittle.inl`` and ``linear_solver.inl``: the fill pattern, the decomposition
  and the triangular solves;
* MICM ``rate_constant_functions.hpp`` and ``reaction_rate_store.hpp``: Arrhenius, Troe, user-defined
  rate constants and the parameterized-reactant multipliers (third bodies, constant species);
* MICM ``rosenbrock.inl`` and ``rosenbrock_solver_parameters.hpp``: the three-stage Rosenbrock method
  (Sandu et al. 1997) with its step-size control, non-in-place LU (the alpha shift carried across
  rejected attempts), and the normalized error.

One deliberate difference: MICM sums the error norm over every grid cell of a state, so its step sizes
-- and its answer -- depend on which cells share a solve.  Here every grid point is its own solve (MICM
with one grid cell), so a column's chemistry is independent of the mesh, the decomposition and the batch.

Reaction types carried: ARRHENIUS (A, B, C or Ea, D, E), TROE, PHOTOLYSIS, EMISSION, FIRST_ORDER_LOSS,
USER_DEFINED.  Every other type is refused by name.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

#: MechanismConfiguration's Boltzmann constant (src/detail/constants.hpp), J K-1.
BOLTZMANN = 1.380649e-23
#: MICM's rate-constant type order (solver_builder.inl stable sort).
TYPE_ORDER = {"ARRHENIUS": 0, "TROE": 1, "USER": 7}
#: musica ConvertChemistry: user-defined families in conversion order, with their label prefixes.
USER_FAMILIES = (("PHOTOLYSIS", "PHOTO."), ("EMISSION", "EMIS."), ("FIRST_ORDER_LOSS", "LOSS."),
                 ("USER_DEFINED", "USER."))
CARRIED = frozenset(("ARRHENIUS", "TROE")) | frozenset(t for t, _ in USER_FAMILIES)
#: ThreeStageRosenbrockParameters (rosenbrock_solver_parameters.hpp) and RosenbrockSolverParameters
#: defaults; DEFAULT_H_MIN / DEFAULT_H_START (rosenbrock.hpp).
ROS3 = {
    "stages": 3,
    "a": (1.0, 1.0, 0.0),
    "c": (-0.10156171083877702091975600115545e+01, 0.40759956452537699824805835358067e+01,
          0.92076794298330791242156818474003e+01),
    "new_function_evaluation": (True, True, False),
    "m": (0.1e+01, 0.61697947043828245592553615689730e+01, -0.42772256543218573326238373806514),
    "e": (0.5, -0.29079558716805469821718236208017e+01, 0.22354069897811569627360909276199),
    "estimator_of_local_order": 3.0,
    "gamma0": 0.43586652150845899941601945119356,
    "max_number_of_steps": 1000,
    "round_off": 2.220446049250313e-16,
    "factor_min": 0.2, "factor_max": 6.0, "rejection_factor_decrease": 0.1, "safety_factor": 0.9,
    "default_h_min": 1.0e-15, "default_h_start": 1.0e-6,
}
#: MICM SolverState values (solver_result.hpp) the kernel reports.
STATES = {"Running": 0, "Converged": 2, "ConvergenceExceededMaxSteps": 3, "StepSizeTooSmall": 4,
          "NaNDetected": 6, "InfDetected": 7}
_SIZE_T = 1 << 64


class MechanismError(ValueError):
    """A mechanism this port does not carry, or a malformed one."""


@dataclass(frozen=True)
class Parameterized:
    """MICM ParameterizedFunction: c0 + c_T T + c_P P + c_rho rho."""
    c0: float = 0.0
    c_T: float = 0.0
    c_P: float = 0.0
    c_rho: float = 0.0


@dataclass
class Process:
    kind: str                     # ARRHENIUS, TROE or USER
    reactants: list               # species names, one per copy
    products: list                # (name, yield)
    params: dict
    label: str | None = None      # user-defined parameter label


@dataclass
class Mechanism:
    name: str
    species: tuple                # state variable names in solver order
    parameterized: dict           # name -> Parameterized (not state variables)
    processes: list               # in MICM order
    parameters: tuple             # user-defined labels in parameter order
    jacobian: tuple               # sorted (row, col) of the Jacobian sparsity, diagonal included
    lower: tuple                  # L fill pattern (row, col), diagonal included
    upper: tuple                  # U fill pattern (row, col)
    fill: object = field(repr=False, default=None)

    @property
    def size(self):
        return len(self.species)


def _read(path):
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        return yaml.safe_load(text)
    return json.loads(text)


def _components(entries, what):
    out = []
    for entry in entries or ():
        name = entry["species name"]
        coefficient = float(entry.get("coefficient", 1.0))
        out.append((name, coefficient))
    return out


def _reactant_copies(components):
    """musica reaction_components_to_reactants: ``for (int i = 0; i < coefficient; i++)``."""
    names = []
    for name, coefficient in components:
        i = 0
        while i < coefficient:
            names.append(name)
            i += 1
    return names


def load_mechanism(path_or_document):
    """The MICM solver structure for a MechanismConfiguration v1 mechanism."""
    raw = path_or_document if isinstance(path_or_document, dict) else _read(path_or_document)
    version = str(raw.get("version", ""))
    if not version.startswith("1."):
        raise MechanismError(f"MechanismConfiguration version {version!r}: this port reads v1 documents")
    species_rows = {row["name"]: row for row in raw["species"]}
    parameterized = {}
    for name, row in species_rows.items():
        p = None
        if "constant concentration [mol m-3]" in row:
            p = Parameterized(c0=float(row["constant concentration [mol m-3]"]))
        if "constant mixing ratio [mol mol-1]" in row:
            p = Parameterized(c_rho=float(row["constant mixing ratio [mol mol-1]"]))
        if row.get("is third body", False):
            p = Parameterized(c_rho=1.0)
        if p is not None:
            parameterized[name] = p
    gas = [phase for phase in raw.get("phases", ()) if phase["name"] == "gas"]
    if len(gas) != 1:
        raise MechanismError("the mechanism needs exactly one phase named 'gas'")
    phase_names = [entry["name"] for entry in gas[0]["species"]]
    for name in phase_names:
        if name not in species_rows:
            raise MechanismError(f"gas phase names undeclared species {name!r}")
    names = [n for n in phase_names if n not in parameterized]

    by_type = {}
    for reaction in raw["reactions"]:
        kind = reaction["type"]
        if kind not in CARRIED:
            raise MechanismError(f"reaction type {kind} is not carried by WOOF's MICM port; carried: "
                                 f"{sorted(CARRIED)}")
        if reaction.get("gas phase", "gas") != "gas":
            raise MechanismError(f"reaction {reaction.get('name')!r} is not in the gas phase")
        by_type.setdefault(kind, []).append(reaction)
    processes = []
    for reaction in by_type.get("ARRHENIUS", ()):
        if "Ea" in reaction and "C" in reaction:
            raise MechanismError("Arrhenius 'Ea' and 'C' are mutually exclusive")
        c = -1 * float(reaction["Ea"]) / BOLTZMANN if "Ea" in reaction else float(reaction.get("C", 0.0))
        processes.append(Process("ARRHENIUS", _reactant_copies(_components(reaction.get("reactants"), "r")),
                                 _components(reaction.get("products"), "p"),
                                 {"A": float(reaction.get("A", 1.0)), "B": float(reaction.get("B", 0.0)),
                                  "C": c, "D": float(reaction.get("D", 300.0)),
                                  "E": float(reaction.get("E", 0.0))}))
    for reaction in by_type.get("TROE", ()):
        processes.append(Process("TROE", _reactant_copies(_components(reaction.get("reactants"), "r")),
                                 _components(reaction.get("products"), "p"),
                                 {k: float(reaction.get(k, d)) for k, d in
                                  (("k0_A", 1.0), ("k0_B", 0.0), ("k0_C", 0.0), ("kinf_A", 1.0),
                                   ("kinf_B", 0.0), ("kinf_C", 0.0), ("Fc", 0.6), ("N", 1.0))}))
    for kind, prefix in USER_FAMILIES:
        for reaction in by_type.get(kind, ()):
            processes.append(Process("USER", _reactant_copies(_components(reaction.get("reactants"), "r")),
                                     _components(reaction.get("products"), "p"),
                                     {"scaling": float(reaction.get("scaling factor", 1.0))},
                                     label=prefix + reaction["name"]))
    # MICM stable-sorts by rate-constant type (Arrhenius, Troe, ..., user-defined).
    processes.sort(key=lambda p: TYPE_ORDER[p.kind])
    labels = []
    for p in processes:
        if p.label is not None:
            if p.label in labels:
                raise MechanismError(f"duplicate rate parameter label {p.label!r}")
            labels.append(p.label)
    for p in processes:
        for name in p.reactants + [n for n, _ in p.products]:
            if name not in names and name not in parameterized:
                raise MechanismError(f"reaction species {name!r} is not a gas-phase species")

    def nonzeros(index):
        ids = set()
        for p in processes:
            react = [index[n] for n in p.reactants if n not in parameterized]
            prods = [index[n] for n, _ in p.products if n not in parameterized]
            for ind in react:
                for dep in react:
                    ids.add((dep, ind))
                for dep in prods:
                    ids.add((dep, ind))
        return ids

    index = {n: i for i, n in enumerate(names)}
    perm = markowitz(nonzeros(index), len(names))
    ordered = tuple(names[perm[i]] for i in range(len(names)))
    index = {n: i for i, n in enumerate(ordered)}
    jacobian = sorted(nonzeros(index) | {(i, i) for i in range(len(ordered))})
    fill = fill_pattern(jacobian, len(ordered))
    return Mechanism(raw.get("name", ""), ordered, parameterized, processes, tuple(labels), tuple(jacobian),
                     tuple(sorted(fill["L"])), tuple(sorted(fill["U"])), fill)


def markowitz(elements, order):
    """MICM DiagonalMarkowitzReorder, with its unsigned 64-bit cost arithmetic."""
    out_n = [set() for _ in range(order)]
    in_n = [set() for _ in range(order)]
    for i, j in elements:
        out_n[i].add(j)
        in_n[j].add(i)
    row = [len(s) for s in out_n]
    col = [len(s) for s in in_n]
    alive = [True] * order
    perm = []
    for _step in range(order):
        pivot, best = order, _SIZE_T - 1
        for v in range(order):
            if not alive[v]:
                continue
            cost = ((row[v] - 1) % _SIZE_T) * ((col[v] - 1) % _SIZE_T) % _SIZE_T
            if pivot == order or cost < best:
                best, pivot = cost, v
        perm.append(pivot)
        alive[pivot] = False
        cols = [c for c in sorted(out_n[pivot]) if c != pivot and alive[c]]
        ins = [i for i in sorted(in_n[pivot]) if i != pivot and alive[i]]
        for i in ins:
            for c in cols:
                if c not in out_n[i]:
                    out_n[i].add(c)
                    row[i] += 1
                    in_n[c].add(i)
                    col[c] += 1
        for c in cols:
            in_n[c].discard(pivot)
            col[c] -= 1
        for i in ins:
            out_n[i].discard(pivot)
            row[i] -= 1
    return perm


def fill_pattern(elements, n):
    """MICM LuDecompositionDoolittle::ComputeFillPattern."""
    present = set(elements)
    arow = [[c for c in range(n) if (r, c) in present] for r in range(n)]
    acol = [[r for r in range(n) if (r, c) in present] for c in range(n)]
    lrow, urow, lcol = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    ucol = [[] for _ in range(n)]
    L, U = set(), set()
    for i in range(n):
        touched = {i}
        touched.update(k for k in arow[i] if k >= i)
        for j in lrow[i]:
            touched.update(k for k in urow[j] if k >= i)
        for k in sorted(touched):
            U.add((i, k))
            urow[i].append(k)
            ucol[k].append(i)
        L.add((i, i))
        touched = {k for k in acol[i] if k > i}
        for j in ucol[i]:
            touched.update(k for k in lcol[j] if k > i)
        for k in sorted(touched):
            L.add((k, i))
            lcol[i].append(k)
            lrow[k].append(i)
    return {"L": L, "U": U, "arow": arow, "acol": acol, "lrow": lrow, "urow": urow, "lcol": lcol}


# -- code generation ---------------------------------------------------------------------------------

def _lit(x):
    """A binary64 literal that round-trips exactly."""
    x = float(x)
    if not math.isfinite(x):
        raise MechanismError(f"non-finite mechanism parameter {x!r}")
    return f"{x.hex()}" if x != int(x) or abs(x) >= 2 ** 53 else f"{x!r}"


def _hexlit(x):
    return f"({float(x).hex()})" if float(x) != 0 else "0.0"


def _rate_constant(p, k):
    """MICM rate_constant_functions.hpp, operation by operation."""
    if p.kind == "ARRHENIUS":
        q = p.params
        # A * exp(C / T) * pow(T / D, B) * (1.0 + E * P), left to right.
        return (f"rc[{k}] = MUL(MUL(MUL({_hexlit(q['A'])}, wcr_exp(DIV({_hexlit(q['C'])}, T))), "
                f"wcr_pow(DIV(T, {_hexlit(q['D'])}), {_hexlit(q['B'])})), "
                f"ADD(1.0, MUL({_hexlit(q['E'])}, P)));")
    if p.kind == "TROE":
        q = {key: _hexlit(v) for key, v in p.params.items()}
        return ("{ double k0 = MUL(MUL(%(k0_A)s, wcr_exp(DIV(%(k0_C)s, T))), wcr_pow(DIV(T, 300.0), %(k0_B)s));"
                " double kinf = MUL(MUL(%(kinf_A)s, wcr_exp(DIV(%(kinf_C)s, T))),"
                " wcr_pow(DIV(T, 300.0), %(kinf_B)s));"
                " double ratio = DIV(MUL(k0, RHO), kinf);"
                " rc[%(k)d] = MUL(DIV(MUL(k0, RHO), ADD(1.0, ratio)),"
                " wcr_pow(%(Fc)s, DIV(%(N)s, ADD(%(N)s, wcr_pow(wcr_log10(ratio), 2.0))))); }"
                % dict(q, k=k))
    return f"rc[{k}] = MUL(prm[{p.param_index}], {_hexlit(p.params['scaling'])});"


def kernel_source(mech, *, name="micm_ros3"):
    """The CUDA translation unit solving ``mech`` (see module docstring)."""
    n = mech.size
    index = {s: i for i, s in enumerate(mech.species)}
    jpos = {e: i for i, e in enumerate(mech.jacobian)}
    lpos = {e: i for i, e in enumerate(mech.lower)}
    upos = {e: i for i, e in enumerate(mech.upper)}
    param_index = {label: i for i, label in enumerate(mech.parameters)}
    lines = []
    emit = lines.append

    # rate constants, then the parameterized multipliers (reaction_rate_store.hpp order)
    emit("__device__ inline void rate_constants(double T, double P, double RHO, const double *prm, double *rc) {")
    for k, p in enumerate(mech.processes):
        p.param_index = param_index.get(p.label)
        emit("  " + _rate_constant(p, k))
    for k, p in enumerate(mech.processes):
        funcs = [mech.parameterized[r] for r in p.reactants if r in mech.parameterized]
        if not funcs:
            continue
        terms = []
        for f in funcs:
            terms.append(f"ADD(ADD(ADD({_hexlit(f.c0)}, MUL({_hexlit(f.c_T)}, T)), MUL({_hexlit(f.c_P)}, P)), "
                         f"MUL({_hexlit(f.c_rho)}, RHO))")
        value = "1.0"
        for t in terms:
            value = f"MUL({value}, {t})"
        emit(f"  rc[{k}] = MUL(rc[{k}], {value});")
    emit("}")

    # forcing (process_set.hpp AddForcingTerms)
    emit("__device__ inline void forcing(const double *y, const double *rc, double *f) {")
    emit(f"  for (int i = 0; i < {n}; ++i) f[i] = 0.0;")
    emit("  double rate;")
    for k, p in enumerate(mech.processes):
        react = [index[r] for r in p.reactants if r not in mech.parameterized]
        emit(f"  rate = rc[{k}];")
        for r in react:
            emit(f"  rate = MUL(rate, y[{r}]);")
        for r in react:
            emit(f"  f[{r}] = SUB(f[{r}], rate);")
        for name, yld in p.products:
            if name in mech.parameterized:
                continue
            emit(f"  f[{index[name]}] = ADD(f[{index[name]}], MUL({_hexlit(yld)}, rate));")
    emit("}")

    # negative Jacobian (process_set.hpp constructor columns + SubtractJacobianTerms)
    columns = []
    for k, p in enumerate(mech.processes):
        for r in p.reactants:
            if r in index:
                columns.append((index[r], k))
    columns.sort(key=lambda c: c[0])   # std::stable_sort on the independent id
    emit("__device__ inline void neg_jacobian(const double *y, const double *rc, double *J) {")
    emit(f"  for (int i = 0; i < {len(mech.jacobian)}; ++i) J[i] = 0.0;")
    emit("  double d;")
    for ind, k in columns:
        p = mech.processes[k]
        found = False
        dependent = []
        for r in p.reactants:
            if r in mech.parameterized:
                continue
            rid = index[r]
            if rid == ind and not found:
                found = True
                continue
            dependent.append(rid)
        emit(f"  d = rc[{k}];")
        for r in dependent:
            emit(f"  d = MUL(d, y[{r}]);")
        for r in dependent:
            e = jpos[(r, ind)]
            emit(f"  J[{e}] = ADD(J[{e}], d);")
        e = jpos[(ind, ind)]
        emit(f"  J[{e}] = ADD(J[{e}], d);")
        for name, yld in p.products:
            if name in mech.parameterized:
                continue
            e = jpos[(index[name], ind)]
            emit(f"  J[{e}] = SUB(J[{e}], MUL({_hexlit(yld)}, d));")
    emit("}")

    emit("__device__ inline void add_diagonal(double *J, double alpha) {")
    for i in range(n):
        e = jpos[(i, i)]
        emit(f"  J[{e}] = ADD(J[{e}], MUL(alpha, 1.0));")
    emit("}")

    # Doolittle decomposition (lu_decomposition_doolittle.inl Initialize + Decompose)
    fp = mech.fill
    present = set(mech.jacobian)
    emit("__device__ inline void decompose(const double *A, double *L, double *U) {")
    for i in range(n):
        for k in fp["urow"][i]:
            u = upos[(i, k)]
            emit(f"  U[{u}] = {f'A[{jpos[(i, k)]}]' if (i, k) in present else '0.0'};")
            for j in fp["lrow"][i]:
                if k not in fp["urow"][j]:
                    continue
                emit(f"  U[{u}] = SUB(U[{u}], MUL(L[{lpos[(i, j)]}], U[{upos[(j, k)]}]));")
        emit(f"  L[{lpos[(i, i)]}] = 1.0;")
        for k in fp["lcol"][i]:
            l = lpos[(k, i)]
            emit(f"  L[{l}] = {f'A[{jpos[(k, i)]}]' if (k, i) in present else '0.0'};")
            for j in fp["lrow"][k]:
                if j >= i:
                    break
                if i not in fp["urow"][j]:
                    continue
                emit(f"  L[{l}] = SUB(L[{l}], MUL(L[{lpos[(k, j)]}], U[{upos[(j, i)]}]));")
            emit(f"  L[{l}] = DIV(L[{l}], U[{upos[(i, i)]}]);")
    emit("}")

    # triangular solves (linear_solver.inl)
    lower, upper = set(mech.lower), set(mech.upper)
    emit("__device__ inline void lu_solve(const double *L, const double *U, double *x) {")
    for i in range(n):
        for j in range(i):
            if (i, j) in lower:
                emit(f"  x[{i}] = SUB(x[{i}], MUL(L[{lpos[(i, j)]}], x[{j}]));")
        emit(f"  x[{i}] = DIV(x[{i}], L[{lpos[(i, i)]}]);")
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if (i, j) in upper:
                emit(f"  x[{i}] = SUB(x[{i}], MUL(U[{upos[(i, j)]}], x[{j}]));")
        emit(f"  x[{i}] = DIV(x[{i}], U[{upos[(i, i)]}]);")
    emit("}")

    body = "\n".join(lines)
    return _TEMPLATE.format(n=n, nnz=len(mech.jacobian), nl=len(mech.lower), nu=len(mech.upper),
                            nr=len(mech.processes), np=max(len(mech.parameters), 1),
                            nparam=len(mech.parameters), body=body, name=name,
                            f64cr=_f64cr(), **{k: _hexlit(v) if isinstance(v, float) else v
                                                for k, v in _ros3_literals().items()})


def _ros3_literals():
    r = ROS3
    return {"c0": r["c"][0], "c1": r["c"][1], "c2": r["c"][2], "a0": r["a"][0], "a1": r["a"][1],
            "m0": r["m"][0], "m1": r["m"][1], "m2": r["m"][2], "e0": r["e"][0], "e1": r["e"][1],
            "e2": r["e"][2], "gamma0": r["gamma0"], "order": r["estimator_of_local_order"],
            "round_off": r["round_off"], "fmin": r["factor_min"], "fmax": r["factor_max"],
            "rdec": r["rejection_factor_decrease"], "safety": r["safety_factor"],
            "dhmin": r["default_h_min"], "dhstart": r["default_h_start"],
            "max_steps": r["max_number_of_steps"]}


def _f64cr():
    return (Path(__file__).resolve().parent / "core" / "kernels" / "f64cr.cuh").read_text()


_TEMPLATE = r"""// SPDX-License-Identifier: Apache-2.0
// Generated by woof.chem_micm from a MechanismConfiguration mechanism: NCAR MICM's three-stage
// Rosenbrock solver (Apache-2.0, licenses/NOTICE-NCAR-MICM.txt) for one grid point per thread.
{f64cr}
#define MUL(a, b) __dmul_rn((a), (b))
#define ADD(a, b) __dadd_rn((a), (b))
#define SUB(a, b) __dsub_rn((a), (b))
#define DIV(a, b) __ddiv_rn((a), (b))
#define NS {n}
#define NNZ {nnz}
#define NL {nl}
#define NU {nu}
#define NR {nr}
{body}

__device__ inline double dmin(double a, double b) {{ return b < a ? b : a; }}
__device__ inline double dmax(double a, double b) {{ return a < b ? b : a; }}

// rosenbrock.inl Solve (non-in-place LU, no constraints) and NormalizedError for one grid cell.
extern "C" __global__ void {name}(
    int ncell, double time_step, const double *temperature, const double *pressure,
    const double *air_density, const double *params, const double *atol, double rtol,
    double *conc, int *status, int *steps) {{
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= ncell) return;
  double Y[NS], Ynew[NS], F0[NS], K0[NS], K1[NS], K2[NS], Yerr[NS];
  double J[NNZ], L[NL], U[NU], rc[NR];
  for (int i = 0; i < NS; ++i) Y[i] = conc[cell * NS + i];
  rate_constants(temperature[cell], pressure[cell], air_density[cell], params + cell * {np}, rc);
  const double h_min = MUL({dhmin}, time_step);
  const double h_max = time_step;
  const double h_start = MUL({dhstart}, time_step);
  double H = dmin(dmax(h_min, fabs(h_start)), fabs(h_max));
  double present = 0.0;
  bool reject_last = false, reject_more = false;
  int state = 0, nsteps = 0, accepted_count = 0;
  while (ADD(SUB(present, time_step), {round_off}) <= 0.0 && state == 0) {{
    if (nsteps > {max_steps}) {{ state = 3; break; }}
    if (ADD(present, MUL(0.1, H)) == present || H <= dmin({round_off}, h_min)) {{ state = 4; break; }}
    H = dmin(H, fabs(SUB(time_step, present)));
    forcing(Y, rc, F0);
    neg_jacobian(Y, rc, J);
    bool accepted = false;
    double last_alpha = 0.0;
    while (!accepted) {{
      double alpha = DIV(1.0, MUL(H, {gamma0}));
      const double cumulative = alpha;
      alpha = SUB(alpha, last_alpha);
      last_alpha = cumulative;
      add_diagonal(J, alpha);
      decompose(J, L, U);
      // stage 0
      for (int i = 0; i < NS; ++i) K0[i] = F0[i];
      lu_solve(L, U, K0);
      // stage 1 (new function evaluation), then K2 takes its raw forcing (no evaluation at stage 2)
      for (int i = 0; i < NS; ++i) Ynew[i] = ADD(Y[i], MUL({a0}, K0[i]));
      forcing(Ynew, rc, K1);
      for (int i = 0; i < NS; ++i) K2[i] = K1[i];
      {{ const double c = DIV({c0}, H); for (int i = 0; i < NS; ++i) K1[i] = ADD(K1[i], MUL(c, K0[i])); }}
      lu_solve(L, U, K1);
      // stage 2
      {{ const double c = DIV({c1}, H); for (int i = 0; i < NS; ++i) K2[i] = ADD(K2[i], MUL(c, K0[i])); }}
      {{ const double c = DIV({c2}, H); for (int i = 0; i < NS; ++i) K2[i] = ADD(K2[i], MUL(c, K1[i])); }}
      lu_solve(L, U, K2);
      for (int i = 0; i < NS; ++i) {{
        double v = ADD(Y[i], MUL({m0}, K0[i]));
        v = ADD(v, MUL({m1}, K1[i]));
        Ynew[i] = ADD(v, MUL({m2}, K2[i]));
        double e = ADD(0.0, MUL({e0}, K0[i]));
        e = ADD(e, MUL({e1}, K1[i]));
        Yerr[i] = ADD(e, MUL({e2}, K2[i]));
      }}
      double error = 0.0;
      for (int i = 0; i < NS; ++i) {{
        const double ymax = fabs(Y[i]) > fabs(Ynew[i]) ? fabs(Y[i]) : fabs(Ynew[i]);
        const double s = DIV(Yerr[i], ADD(atol[i], MUL(rtol, ymax)));
        error = ADD(error, MUL(s, s));
      }}
      error = dmax(__dsqrt_rn(DIV(error, (double)NS)), 1.0e-10);
      const double fac = dmin({fmax}, dmax({fmin}, DIV({safety}, wcr_pow(error, DIV(1.0, {order})))));
      double Hnew = MUL(H, fac);
      nsteps += 1;
      if (isnan(error)) {{ state = 6; break; }}
      if (isinf(error)) {{ state = 7; break; }}
      if (error < 1.0 || H < h_min) {{
        accepted_count += 1;
        present = ADD(present, H);
        for (int i = 0; i < NS; ++i) Y[i] = Ynew[i];
        Hnew = dmax(h_min, dmin(Hnew, h_max));
        if (reject_last) Hnew = dmin(Hnew, H);
        reject_last = false;
        reject_more = false;
        H = Hnew;
        accepted = true;
      }} else {{
        if (reject_more) Hnew = MUL(H, {rdec});
        reject_more = reject_last;
        reject_last = true;
        H = Hnew;
      }}
    }}
  }}
  if (state == 0) state = 2;
  for (int i = 0; i < NS; ++i) conc[cell * NS + i] = Y[i];
  status[cell] = state;
  steps[cell] = nsteps;
}}
"""


def solve(mech, concentrations, *, temperature, pressure, air_density, params=None, time_step,
          rtol=1.0e-6, atol=1.0e-3, cache=None):
    """Advance ``concentrations`` (ncell, nspecies) [mol m-3] in solver species order by ``time_step``.

    Returns (concentrations, status, steps) on the device.  ``params`` is (ncell, nparameters) in
    ``mech.parameters`` order.  ``atol`` is a scalar or one value per species (MICM's default 1e-3)."""
    import cupy as cp
    import numpy as np
    conc = cp.ascontiguousarray(cp.asarray(concentrations, dtype=cp.float64))
    ncell = conc.shape[0]
    if conc.shape != (ncell, mech.size):
        raise ValueError(f"concentrations must be (ncell, {mech.size})")
    col = lambda a: cp.ascontiguousarray(cp.broadcast_to(cp.asarray(a, dtype=cp.float64), (ncell,)))
    T, P, rho = col(temperature), col(pressure), col(air_density)
    nparam = max(len(mech.parameters), 1)
    prm = cp.zeros((ncell, nparam), cp.float64) if params is None else cp.ascontiguousarray(
        cp.asarray(params, dtype=cp.float64).reshape(ncell, len(mech.parameters)))
    tol = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(atol, dtype=cp.float64), (mech.size,)))
    status = cp.zeros(ncell, cp.int32)
    steps = cp.zeros(ncell, cp.int32)
    kernel = _kernel(mech)
    kernel(((ncell + 63) // 64,), (64,), (np.int32(ncell), np.float64(time_step), T, P, rho, prm, tol,
                                           np.float64(rtol), conc, status, steps))
    return conc, status, steps


_KERNELS = {}


def _kernel(mech):
    import cupy as cp
    source = kernel_source(mech)
    kernel = _KERNELS.get(source)
    if kernel is None:
        kernel = _KERNELS[source] = cp.RawKernel(source, "micm_ros3", options=("--fmad=false",))
    return kernel

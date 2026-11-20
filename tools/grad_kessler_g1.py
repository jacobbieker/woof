"""Gate G1 measurements for the differentiable Kessler (plan P1).

    python tools/grad_kessler_g1.py [--device cuda|mps|cpu] [--columns 40000] [--out g1.json]

On the WRF Kessler oracle fixture (``tests/data/oracles/kessler``: 104
columns x 50 levels, 13 regimes, dt 5/18/60/120/300 s), against woof's own
``kessler_column`` kernel run by the production launcher on this host's
backend (CUDA or Metal):

``forward``   one call per dt from the fixture's inputs: the torch module
              (float32 on the device, float64 on the CPU or CUDA) against
              the kernel's words; max |diff| per field, max FP32 ULP and
              the share of words within 4 ULP.
``dot``       the adjoint test <TL dx, dy> = <dx, AD dy> per dt, float64
              (CPU or CUDA; MPS has no float64) and float32 (the device).
``tl``        TL convergence ||f(x + a dx) - f(x)|| / ||a TL dx||, float64.
``gradcheck`` torch.autograd.gradcheck per column (float64, CPU, dt 60 s),
              in scaled controls (theta in K, each q relative to its own
              value), tolerance atol 1e-5 / rtol 1e-3: how many columns
              pass, which fail (and their regimes).
``fd_kernel`` the float64 tangent-linear against central differences of
              woof's float32 KERNEL along a relative direction
              (t +- 0.1 K, q +- 1 % of the value, times eps 0.25 and
              0.05), per output field and per column (a column whose
              perturbation crosses a switch disagrees; the share of
              columns that agree is the measure).
``cost``      the fixture tiled to ``--columns`` columns: the kernel, the
              torch forward, and the torch forward + backward (the
              adjoint), median wall time of a call closed by a device sync.

Prints one JSON document (and writes it with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np

FIELDS = ("t", "qv", "qc", "qr")
SCALE = {"t": 1.0, "qv": 1e-3, "qc": 1e-4, "qr": 1e-4}


def _kernel(inputs: dict, dt: float) -> dict:
    """One ``launch_kessler`` call on ``[nz, ncol]`` host columns."""
    import cupy as cp
    from woof.core.microphysics import launch_kessler
    nz, ncol = inputs["t"].shape
    dev = {k: cp.asarray(np.ascontiguousarray(inputs[k], np.float32).reshape(nz, 1, ncol))
           for k in ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w")}
    rainnc = cp.zeros((1, ncol), cp.float32)
    rainncv = cp.zeros((1, ncol), cp.float32)
    launch_kessler(dev["t"], dev["qv"], dev["qc"], dev["qr"], dev["rho"], dev["pii"], dev["z"],
                   dev["dz8w"], rainnc, rainncv, float(dt))
    out = {k: cp.asnumpy(dev[k]).reshape(nz, ncol) for k in FIELDS}
    out["rainncv"] = cp.asnumpy(rainncv).reshape(ncol)
    out["rainnc"] = cp.asnumpy(rainnc).reshape(ncol)
    return out


def _sync(device: str) -> None:
    import torch
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _tensors(inputs, device, dtype):
    import torch
    from woof.verify.kessler_oracle import INPUT_FIELDS
    return [torch.from_numpy(np.ascontiguousarray(inputs[k], np.float64)).to(dtype).to(device)
            for k in INPUT_FIELDS]


def _torch_out(inputs, dt, device, dtype) -> dict:
    import torch
    from woof.grad.kessler import _kessler
    with torch.no_grad():
        out = _kessler(*_tensors(inputs, device, dtype), dt)
    host = {k: out[i].cpu().double().numpy() for i, k in enumerate(FIELDS)}
    host["rainncv"] = out[4].cpu().double().numpy()
    host["rainnc"] = out[5].cpu().double().numpy()
    return host


def forward(inputs, device) -> list:
    from woof.core.fp32_ulp import fp32_ulp_distance
    import torch
    rows = []
    wide = "cpu" if device == "mps" else device
    for dt in inputs["dt"]:
        dt = float(dt)
        kern = _kernel(inputs, dt)
        for label, dev, dtype in (("float32", device, torch.float32), ("float64", wide, torch.float64)):
            got = _torch_out(inputs, dt, dev, dtype)
            row = {"dt": dt, "torch": f"{label} {dev}"}
            for name in FIELDS + ("rainncv", "rainnc"):
                g = got[name].astype(np.float32)
                w = kern[name]
                ulp = fp32_ulp_distance(np.ascontiguousarray(g), np.ascontiguousarray(w))
                row[name] = {"max_abs": float(np.max(np.abs(got[name] - w.astype(np.float64)))),
                             "max_ulp": int(ulp.max()), "within_4ulp": float(np.mean(ulp <= 4)),
                             "identical": float(np.mean(ulp == 0))}
            rows.append(row)
    return rows


def _direction(x, names, generator, relative: bool):
    import torch
    out = []
    for name, a in zip(names, x):
        noise = torch.randn(a.shape, generator=generator, dtype=torch.float64).to(a.dtype).to(a.device)
        if name == "t":
            out.append(noise * (0.1 if relative else 1.0))
        elif name in ("qv", "qc", "qr"):
            out.append(noise * (0.01 * a.abs() if relative else SCALE[name] * (a != 0)))
        else:
            out.append(noise * 1e-3 * a.abs().mean() if not relative else torch.zeros_like(a))
    return out


def dot(inputs, device) -> list:
    import torch
    from woof.grad import checks
    from woof.grad.kessler import _kessler
    from woof.verify.kessler_oracle import INPUT_FIELDS
    rows = []
    wide = "cpu" if device == "mps" else device
    for dt in inputs["dt"]:
        dt = float(dt)
        f = lambda *a: _kessler(*a, dt)
        for dev, dtype in ((wide, torch.float64), (device, torch.float32)):
            x = _tensors(inputs, dev, dtype)
            g = torch.Generator().manual_seed(0)
            dx = _direction(x, INPUT_FIELDS, g, relative=False)
            nz, ncol = x[0].shape
            dy = [(torch.randn((nz, ncol), generator=g, dtype=torch.float64) / SCALE[k]).to(dtype).to(dev)
                  for k in FIELDS]
            dy += [torch.randn(ncol, generator=g, dtype=torch.float64).to(dtype).to(dev) for _ in range(2)]
            r = checks.dot_product(f, x, dx, dy)
            row = {"dt": dt, "dtype": str(dtype).removeprefix("torch."), "device": dev, "relative": r.relative}
            if dtype == torch.float64:
                row["tl_convergence"] = checks.tl_convergence(f, x, dx, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
            rows.append(row)
    return rows


def gradcheck(inputs) -> dict:
    import torch
    from woof.grad.kessler import _kessler
    dt = 60.0
    full = _tensors(inputs, "cpu", torch.float64)
    passed, failed = [], []
    for col in range(full[0].shape[1]):
        cols = [a[:, col:col + 1].clone() for a in full]
        t0, qv0, qc0, qr0 = cols[:4]
        # Scaled controls: theta in K, each mixing ratio relative to its own
        # value (q = q0 (1 + u)), so a zero word stays zero (the clamp at
        # q = 0 is a kink AT the point, where no derivative exists to check)
        # and every input moves by a comparable amount; outputs are the
        # increments, theta in K and q in g/kg.
        state = tuple(torch.zeros_like(t0).requires_grad_(True) for _ in range(4))

        def f(ut, uqv, uqc, uqr):
            out = _kessler(t0 + ut, qv0 * (1 + uqv), qc0 * (1 + uqc), qr0 * (1 + uqr), *cols[4:], dt)
            return (out[0] - t0, (out[1] - qv0) * 1e3, (out[2] - qc0) * 1e3, (out[3] - qr0) * 1e3,
                    out[4], out[5])

        ok = torch.autograd.gradcheck(f, state, eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=False)
        (passed if ok else failed).append(col)
    regimes = np.asarray(inputs["regime"]).astype(str)
    return {"dt": dt, "columns": int(full[0].shape[1]), "passed": len(passed),
            "failed": {str(c): regimes[c] for c in failed}}


def fd_kernel(inputs) -> list:
    import torch
    from woof.grad import checks
    from woof.grad.kessler import _kessler
    from woof.verify.kessler_oracle import INPUT_FIELDS
    rows = []
    for dt in inputs["dt"]:
        dt = float(dt)
        x = _tensors(inputs, "cpu", torch.float64)
        g = torch.Generator().manual_seed(1)
        dx = _direction(x, INPUT_FIELDS, g, relative=True)
        _, tl = checks.tangent_linear(lambda *a: _kessler(*a, dt), x, dx)
        base = {k: np.asarray(inputs[k], np.float64) for k in INPUT_FIELDS}
        step = {k: d.numpy() for k, d in zip(INPUT_FIELDS, dx)}
        row = {"dt": dt}
        for eps in (0.25, 0.05):
            plus = _kernel({k: base[k] + eps * step[k] for k in INPUT_FIELDS}, dt)
            minus = _kernel({k: base[k] - eps * step[k] for k in INPUT_FIELDS}, dt)
            fd_cols, ad_cols = 0.0, 0.0
            for i, name in enumerate(FIELDS + ("rainncv", "rainnc")):
                fd = (plus[name].astype(np.float64) - minus[name].astype(np.float64)) / (2 * eps)
                ad = tl[i].numpy()
                err = np.linalg.norm(fd - ad) / max(np.linalg.norm(ad), 1e-300)
                row.setdefault(name, {})[f"eps{eps}"] = float(err)
                if name in SCALE:
                    fd_cols = fd_cols + ((fd - ad) / SCALE[name]) ** 2
                    ad_cols = ad_cols + (ad / SCALE[name]) ** 2
            # per column, the four fields scaled (t in K, q in 1e-3/1e-4 kg/kg)
            per_col = np.sqrt(fd_cols.sum(axis=0)) / np.maximum(np.sqrt(ad_cols.sum(axis=0)), 1e-300)
            row.setdefault("columns", {})[f"eps{eps}"] = {
                "median": float(np.median(per_col)), "p90": float(np.quantile(per_col, 0.9)),
                "within_1e-3": float(np.mean(per_col < 1e-3)), "within_1e-2": float(np.mean(per_col < 1e-2)),
                "within_1e-1": float(np.mean(per_col < 1e-1))}
        rows.append(row)
    return rows


def _median_time(fn, device, repeat=10) -> float:
    fn()
    _sync(device)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def cost(inputs, device, columns: int, compiled: bool = False) -> dict:
    import cupy as cp
    import torch
    from woof.core.microphysics import launch_kessler
    from woof.grad.kessler import _kessler, kessler_compiled
    from woof.verify.kessler_oracle import INPUT_FIELDS
    if compiled:
        _kessler = kessler_compiled
    reps = -(-columns // inputs["t"].shape[1])
    tiled = {k: np.tile(np.asarray(inputs[k], np.float32), (1, reps))[:, :columns] for k in INPUT_FIELDS}
    nz = tiled["t"].shape[0]
    dt = 60.0
    out = {"columns": columns, "levels": nz, "dt": dt, "device": device, "compiled": compiled}

    dev = {k: cp.asarray(tiled[k].reshape(nz, 1, columns)) for k in INPUT_FIELDS}
    work = {k: dev[k].copy() for k in FIELDS}
    rain = cp.zeros((1, columns), cp.float32), cp.zeros((1, columns), cp.float32)

    def kernel():
        for k in FIELDS:
            work[k][...] = dev[k]
        launch_kessler(work["t"], work["qv"], work["qc"], work["qr"], dev["rho"], dev["pii"], dev["z"],
                       dev["dz8w"], rain[0], rain[1], dt)
        cp.cuda.Device().synchronize()

    def copies():
        for k in FIELDS:
            work[k][...] = dev[k]
        cp.cuda.Device().synchronize()

    out["kernel_s"] = max(_median_time(kernel, "cpu") - _median_time(copies, "cpu"), 0.0)
    out["kernel_with_copies_s"] = _median_time(kernel, "cpu")

    x = [torch.from_numpy(tiled[k]).to(device) for k in INPUT_FIELDS]

    def fwd():
        with torch.no_grad():
            _kessler(*x, dt)

    leaves = [a.clone().requires_grad_(True) for a in x[:4]]
    seeds = [torch.ones_like(a) for a in x[:4]]

    def fwd_bwd():
        o = _kessler(*leaves, *x[4:], dt)
        torch.autograd.grad(o[:4], leaves, seeds)

    out["torch_forward_s"] = _median_time(fwd, device)
    out["torch_forward_backward_s"] = _median_time(fwd_bwd, device)
    out["torch_backward_s"] = out["torch_forward_backward_s"] - out["torch_forward_s"]
    out["backward_over_kernel"] = out["torch_backward_s"] / max(out["kernel_s"], 1e-12)
    out["backward_over_torch_forward"] = out["torch_backward_s"] / out["torch_forward_s"]
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="torch device (default: cuda on CUDA, mps on Metal)")
    parser.add_argument("--columns", type=int, default=40000)
    parser.add_argument("--skip", default="", help="comma-separated sections to skip")
    parser.add_argument("--compiled", action="store_true",
                        help="time the torch.compile'd stages (kessler_compiled) in the cost section")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    from woof import backend
    from woof.verify.kessler_oracle import load_fixture
    backend.install()
    device = args.device or ("mps" if backend.requested() == "mlx" else "cuda")
    inputs, _ = load_fixture()
    skip = set(filter(None, args.skip.split(",")))
    report = {"backend": backend.requested(), "device": device}
    for name, fn in (("forward", lambda: forward(inputs, device)), ("dot", lambda: dot(inputs, device)),
                     ("gradcheck", lambda: gradcheck(inputs)), ("fd_kernel", lambda: fd_kernel(inputs)),
                     ("cost", lambda: cost(inputs, device, args.columns, args.compiled))):
        if name not in skip:
            report[name] = fn()
            print(f"[g1] {name} done", file=sys.stderr, flush=True)
    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

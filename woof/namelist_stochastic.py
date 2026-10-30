"""Representable WRF v4.6.1 stochastic controls for the ensemble run door."""
from __future__ import annotations

from woof.ensemble.stochastic import StochasticConfig
from woof.ensemble.stochastic_seeds import WRF_SEED_DEFAULTS
from woof.wrf_namelist_registry import wrf_namelist_keys

SUPPORTED_SELECTORS = frozenset({"sppt", "skebs", "stoch_force_opt", "spp",
                               "spp_conv", "spp_pbl", "spp_lsm"})
#: WRF-Solar EPS (woof/core/wrfsolar_eps.py): RunConfig keys of their own,
#: imported by :func:`_import_multi_perturb`, not ensemble-door processes.
EPS_SELECTORS = ("multi_perturb", "pert_farms", "pert_deng", "pert_thom", "pert_cld3")
EPS_FACTORS = ("pert_farms_albedo", "pert_farms_aod", "pert_farms_angexp",
               "pert_farms_aerasy", "pert_farms_qv", "pert_farms_qc",
               "pert_farms_qs", "pert_deng_qv", "pert_deng_qc", "pert_deng_t",
               "pert_deng_w", "pert_thom_qv", "pert_thom_qc", "pert_thom_qi",
               "pert_thom_qs", "pert_thom_ni", "pert_cld3_qv", "pert_cld3_t")
#: Emitted only where they leave WRF's Registry default.
EPS_DEFAULTS = {"multi_perturb": 0, "pert_farms": False, "pert_deng": False,
                "pert_thom": False, "pert_cld3": False,
                **{key: 0.0 for key in EPS_FACTORS}, "spdt": -1.0,
                "num_pert_3d": 15, "nens": 1}
UNSUPPORTED_SELECTORS = frozenset({"rand_perturb", "perturb_bdy",
    "perturb_chem_bdy", "pert_mynn", "pert_noah"})
#: What WRF v4.7.1 does under each selector the engine has no consumer for
#: (Registry/registry.stoch, registry.bdy_perturb), so the refusal says which
#: operation would be lost rather than one sentence for all ten.  A row
#: leaves this table, and UNSUPPORTED_SELECTORS, when its consumer lands.
UNSUPPORTED_SELECTOR_OPERATION = {
    "rand_perturb": "WRF fills the RAND_PERT random-pattern array and "
                    "writes it to history (registry.stoch:15, :205)",
    "perturb_bdy": "WRF perturbs the lateral boundary tendencies of U, V "
                   "and T (registry.bdy_perturb:9: 1 with the SKEBS "
                   "pattern, 2 with a supplied one)",
    "perturb_chem_bdy": "WRF perturbs the lateral boundaries of chemical "
                        "tracers with the RAND_PERT pattern "
                        "(registry.bdy_perturb:10)",
    "pert_mynn": "WRF-Solar EPS perturbs MYNN's qv, qc, theta and qke "
                 "around the PBL scheme (registry.stoch:112, "
                 "module_pbl_driver.F:1647-1725); woof ports the FARMS, "
                 "Deng, Thompson and icloud = 3 hooks only",
    "pert_noah": "WRF-Solar EPS perturbs Noah's qv, theta, soil moisture "
                 "and soil temperature around the land surface "
                 "(registry.stoch:117, module_surface_driver.F:2810-2915); "
                 "woof ports the FARMS, Deng, Thompson and icloud = 3 hooks "
                 "only",
}
assert set(UNSUPPORTED_SELECTOR_OPERATION) == UNSUPPORTED_SELECTORS
SEED_NOTICE = ("Stochastic physics uses engine keyed Philox and cuFFT, not WRF's "
    "date-dependent RANDOM_NUMBER realization. Authored nens and iseed_* labels "
    "are retained and bound to versioned process keys and complete checkpoints. "
    "The imported ensemble has one stochastic forecast; --members N changes its "
    "member count. nens is a seed label and never a member count.")


def _import_multi_perturb(section, *, max_dom, registry, fix, drop, error):
    """WRF-Solar EPS: ``multi_perturb`` and its keys as RunConfig columns.

    Returns {key: per-domain column} when multi_perturb = 1 on some domain,
    else None (the keys are then left for the inert drop below)."""
    raw = section.entries.get("multi_perturb")
    column = ([] if raw is None else list(raw[:max_dom]))
    if any(type(v) is not int or v not in (0, 1) for v in column):
        raise error("stoch", "multi_perturb", raw, "must be integer 0 or 1")
    if not any(column):
        if raw is not None:
            fix("stoch", "multi_perturb", section.take("multi_perturb"), 0,
                "multi_perturb is off: WRF runs no pert_* hook")
        return None
    section.take("multi_perturb")
    out = {"multi_perturb": column + [0] * (max_dom - len(column))}

    def default(key):
        row = registry[key]
        text = row["default"].lower()
        return (text == ".true." if row["type"] == "logical" else
                int(text) if row["type"] == "integer" else
                float(text.replace("d", "e")))

    for key in (*EPS_SELECTORS[1:], *EPS_FACTORS, "spdt"):
        values = section.take(key)
        base = default(key)
        col = [] if values is None else list(values[:max_dom])
        logical = registry[key]["type"] == "logical"
        for v in col:
            if logical and type(v) is not bool:
                raise error("stoch", key, values, "must be a WRF logical")
            if not logical and (isinstance(v, bool) or not isinstance(v, (int, float))):
                raise error("stoch", key, values, "must be a WRF real")
        col = [v if logical else float(v) for v in col]
        out[key] = col + [base] * (max_dom - len(col))
    for key in ("num_pert_3d", "nens"):
        values = section.take(key)
        value = default(key) if values is None else values[0]
        if values is not None and len(values) != 1:
            raise error("stoch", key, values, "WRF declares one scalar value")
        if type(value) is not int:
            raise error("stoch", key, values, "must be a WRF integer")
        out[key] = [value] * max_dom
    # INITIALIZE_STOCH hands SETUP_RAND_PERTURB the namelist forcing band
    # (module_stoch.F:390-404); woof's generator takes WRF's defaults.
    for key in ("kminforct", "lminforct", "kmaxforct", "lmaxforct"):
        values = section.take(key)
        if values is not None and any(v != default(key) for v in values):
            raise error("stoch", key, values,
                        "multi_perturb's patterns take the forcing band's "
                        "Registry default; a moved band is not ported")
        if values is not None:
            fix("stoch", key, values, default(key), "the multi_perturb forcing band")
    for key in ("pert_mynn", "pert_noah"):
        values = section.entries.get(key)
        if values is not None and any(bool(v) for v in values):
            raise error("stoch", key, values,
                        UNSUPPORTED_SELECTOR_OPERATION[key])
    return out


def import_stochastic_section(section, *, max_dom, fix, drop, error):
    """Consume only supported controls, retaining disabled import bytes.

    Returns (ensemble-door controls or None, per-domain RunConfig columns):
    the SPP consumer flags and, with multi_perturb = 1, WRF-Solar EPS's keys
    (woof/core/wrfsolar_eps.py), which run on the ordinary forecast."""
    registry = {key: row for (group, key), row in wrf_namelist_keys().items()
                if group == "stoch"}
    eps_columns = _import_multi_perturb(section, max_dom=max_dom,
                                        registry=registry, fix=fix,
                                        drop=drop, error=error)

    def default(key):
        row = registry[key]
        raw = row["default"].lower()
        return (raw == ".true." if row["type"] == "logical" else
                int(raw) if row["type"] == "integer" else float(raw.replace("d", "e")))

    for key in sorted(UNSUPPORTED_SELECTORS & section.entries.keys()):
        values = section.take(key)
        if any(bool(value) for value in values):
            from woof.ensemble_admission import missing_random_consumer
            raise error("stoch", key, values,
                        f"{UNSUPPORTED_SELECTOR_OPERATION[key]}; "
                        + missing_random_consumer(key))
        fix("stoch", key, values, 0, "selected stochastic consumer is disabled")
    selectors = {}
    for key in sorted(SUPPORTED_SELECTORS):
        values = section.take(key)
        raw = [] if values is None else values[:max_dom]
        if any(type(value) is not int or value not in (0, 1) for value in raw):
            raise error("stoch", key, raw, "must be integer 0 or 1")
        column = raw[:max_dom] + [0] * max(0, max_dom - len(raw))
        selectors[key] = column
        if values is not None and not any(column):
            fix("stoch", key, values, 0, "selected stochastic consumer is disabled")
    selectors["skebs"] = [int(a or b) for a, b in
        zip(selectors["skebs"], selectors["stoch_force_opt"])]
    if any(selectors["spp"]):
        # module_check_a_mundo.F sets the complete arrays, not just domain i.
        for name in ("spp_conv", "spp_pbl", "spp_lsm"):
            selectors[name] = [1] * max_dom
    active = any(any(column) for column in selectors.values())
    if active and eps_columns is not None:
        raise error("stoch", "multi_perturb", eps_columns["multi_perturb"],
                    "WRF-Solar EPS runs on the ordinary forecast and SPPT/"
                    "SKEBS/SPP on the ensemble door; one import cannot carry "
                    "both (run the EPS members as separate forecasts with "
                    "their own nens, as WRF-Solar EPS does)")
    if not active:
        for key in sorted(registry.keys() & section.entries.keys()):
            drop("stoch", key, section.take(key),
                "stochastic parameters and bookkeeping are inert with every &stoch selector off"
                if eps_columns is None else
                "inert beside multi_perturb (WRF-Solar EPS reads its table, "
                "the pert_* keys, spdt, num_pert_3d and nens)")
        section.finish()
        return None, ({} if eps_columns is None else eps_columns)

    def uniform(key, column):
        if len(set(column)) != 1:
            raise error("stoch", key, column,
                "the provider shares this stochastic control across domains; "
                "unequal per-domain forcing would apply a different pattern or enable an off domain")
        return column[0]

    enabled = {key: uniform(key, selectors[key]) for key in ("sppt", "skebs")}

    def value(key, selected=None):
        values = section.take(key)
        base = default(key)
        if values is None:
            return base
        if registry[key]["nentries"] == "max_domains":
            column = values[:max_dom] + [base] * max(0, max_dom - len(values))
            result = uniform(key, column if selected is None else
                             [item for item, flag in zip(column, selected) if flag])
        else:
            if len(values) != 1:
                raise error("stoch", key, values, "WRF declares one scalar value")
            result = values[0]
        if registry[key]["type"] == "integer" and type(result) is not int:
            raise error("stoch", key, values, "must be a WRF integer")
        if registry[key]["type"] == "real" and (type(result) not in (int, float)):
            raise error("stoch", key, values, "must be a finite numeric WRF coefficient")
        return result

    def pin(key, expected, reason):
        actual = value(key)
        if type(actual) is not type(expected) and not (
                type(actual) in (int, float) and type(expected) in (int, float)):
            raise error("stoch", key, actual, reason)
        if actual != expected:
            raise error("stoch", key, actual, reason)
        fix("stoch", key, [actual], expected, reason)

    def parameters(kind, suffix, vertical, selected=None):
        overrides = {field: value(prefix + suffix, selected) for field, prefix in (
            ("stddev", "gridpt_stddev_"), ("cutoff_sigma", "stddev_cutoff_"),
            ("lengthscale_m", "lengthscale_"), ("timescale_s", "timescale_"))}
        pin(vertical, 0, "vertical phase rotation is not implemented; a uniform pattern cannot replace it")
        try:
            StochasticConfig(kind=kind, **overrides)
        except ValueError as problem:
            raise error("stoch", "gridpt_stddev_" + suffix, overrides, str(problem)) from problem
        return overrides

    controls = {"sppt": False, "skebs": False, "spp": False}
    if enabled["sppt"]:
        controls["sppt"] = parameters("sppt", "sppt", "sppt_vertstruc")
    if enabled["skebs"]:
        pin("skebs_vertstruc", 0, "vertical phase rotation is not implemented; a uniform SKEBS field cannot replace it")
        pin("stoch_vertstruc_opt", 0, "the obsolete vertical-phase selector enables an unimplemented SKEBS structure")
        stddev = controls["sppt"]["stddev"] if enabled["sppt"] else value("gridpt_stddev_sppt")
        cutoff = controls["sppt"]["cutoff_sigma"] if enabled["sppt"] else value("stddev_cutoff_sppt")
        skebs = {}
        for kind, suffix, temporal in (("skebs_psi", "", "psi"), ("skebs_theta", "t", "t")):
            bounds = {}
            for field, prefix in (("min_wavenumber", "min"), ("max_wavenumber", "max")):
                k, l = value("k" + prefix + "forc" + suffix), value("l" + prefix + "forc" + suffix)
                if k != l:
                    raise error("stoch", "k" + prefix + "forc" + suffix, [k, l],
                        "k/l SKEBS wavenumber limits differ; the current spectral operator shares both limits")
                bounds[field] = k
            config = dict(stddev=stddev, cutoff_sigma=cutoff, **bounds,
                timescale_s=value("ztau_" + temporal), backscatter=value("tot_backscat_" + temporal),
                spectral_exponent=value("rexponent_" + temporal))
            try:
                StochasticConfig(kind=kind, **config)
            except ValueError as problem:
                raise error("stoch", "tot_backscat_" + temporal, config, str(problem)) from problem
            skebs["psi" if not suffix else "theta"] = config
        controls["skebs"] = skebs
    selected = {name: selectors["spp_" + name] for name in ("conv", "pbl", "lsm")}
    if any(any(column) for column in selected.values()):
        controls["spp"] = True
        controls["spp_configs"] = {name: parameters("spp_" + name, "spp_" + name,
            "vertstruc_spp_" + name, column) for name, column in selected.items() if any(column)}
    if enabled["sppt"] or any(any(column) for column in selected.values()):
        for key in ("kminforct", "lminforct", "kmaxforct", "lmaxforct"):
            if key in section.entries:
                pin(key, default(key), "WRF rejects changed temperature wavenumber limits when SPPT/SPP is enabled")
            elif controls["skebs"] and controls["skebs"]["theta"][
                    "min_wavenumber" if "min" in key else "max_wavenumber"] != default(key):
                raise error("stoch", key, controls["skebs"]["theta"],
                    "WRF rejects changed temperature wavenumber limits when SPPT/SPP is enabled")
    labels = {key: value(key) for key in WRF_SEED_DEFAULTS}
    controls["wrf_seed_labels"] = labels
    from woof.ensemble.stochastic_seeds import normalize_wrf_seed_labels
    try:
        normalize_wrf_seed_labels(labels)
    except ValueError as problem:
        raise error("stoch", "nens", labels, str(problem)) from problem
    if enabled["sppt"] or enabled["skebs"]:
        pin("hrrr_cycling", True, "a restart that reseeds a stochastic process cannot preserve its complete spectral history")
    for key in sorted(registry.keys() & section.entries.keys()):
        if key in ("zsigma2_eps", "zsigma2_eta"):
            reason = "WRF computes the SKEBS noise variance from accepted dt; this registered coefficient has no consumer"
        elif key == "hrrr_cycling":
            reason = "WRF consults this restart reseeding switch for SPPT/SKEBS/random fields, not the selected SPP consumers"
        elif key in ("spdt", "num_pert_3d") or key.startswith("pert_"):
            reason = "multi_perturb and its field consumers are disabled; this perturbation-table coefficient is inert"
        elif "rand_pert" in key:
            reason = "rand_perturb is disabled; its random-field coefficient is inert"
        elif "spp_" in key:
            reason = "the corresponding spp_conv/pbl/lsm consumer is disabled; its parameter-pattern coefficient is inert"
        elif "sppt" in key:
            reason = "SPPT is disabled; this correlation/vertical parameter does not affect the selected SKEBS/SPP processes"
        elif key in {"kminforc", "kmaxforc", "lminforc", "lmaxforc", "rexponent_psi",
                     "rexponent_t", "skebs_vertstruc", "stoch_vertstruc_opt", "tot_backscat_psi",
                     "tot_backscat_t", "ztau_psi", "ztau_t"}:
            reason = "SKEBS is disabled; SETUP_RAND_PERTURB uses these controls only for its W/T backscatter branches"
        else:
            raise error("stoch", key, section.entries[key],
                "this registered parameter has no classified stochastic consumer; refusing to drop an active control")
        drop("stoch", key, section.take(key), reason)
    section.finish()
    return controls, {"spp_" + name: column for name, column in selected.items()}

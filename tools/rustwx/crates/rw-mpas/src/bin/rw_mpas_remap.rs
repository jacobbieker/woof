//! `rw_mpas_remap`: remap an MPAS init or restart state from one mesh onto
//! another, in Rust.
//!
//! Conservative where conservation matters (density, theta, the mixing
//! ratios, the momentum the edge wind is rebuilt from, soil and surface
//! state), barycentric where it does not (diagnostics), conservative in
//! height onto the target's own terrain-following levels, and rebalanced
//! hydrostatically with the same column routine `rw_mpas_init` uses.  See
//! `rw_mpas::remap` for the whole pipeline and what it refuses.
//!
//! The output is laid out from the TARGET template (`--to-vertical`, else
//! `--to-grid` when it carries a vertical grid): every static, mesh and
//! vertical-metric variable is carried from it, every state variable is
//! computed here, and the receipt's emit ledger says which is which.

use std::path::PathBuf;
use std::process::ExitCode;

use rw_mpas::init::dynamics::VirtualFactor;
use rw_mpas::remap::{run, Balance, RemapConfig, DEFAULT_MIN_COVERAGE};

/// `GPUWM_BRIDGE_SOURCE_REV=<40-hex commit>`: see `rw_mpas_init` for why the
/// stamp exists and how the release cut reads it.
pub static GPUWM_BRIDGE_SOURCE_REV_STAMP: &str =
    concat!("GPUWM_BRIDGE_SOURCE_REV=", env!("GPUWM_BRIDGE_SOURCE_REV"));

/// The literal the Python bridge contract handshakes on.
pub const ABI_MARKER: &str = "rw_mpas_remap --from-grid A.grid.nc --from-state A.state.nc \
--to-grid B.grid.nc [--to-static B.static.nc] [--to-vertical B.vertical.nc] --out B.state.nc \
--balance hydrostatic|carry --virtual-factor reproduce-fortran|consistent \
[--min-coverage X] [--receipt JSON]";

fn usage() -> String {
    format!(
        "usage: {ABI_MARKER}\n\n\
         --from-grid        the source mesh (verticesOnCell, latVertex/lonVertex, cellsOnVertex)\n\
         --from-state       the source state: an init, restart or single-frame history on that\n\
        \x20                  mesh, carrying rho, theta, qv and zgrid (or zgrid in --from-grid)\n\
         --to-grid          the target mesh\n\
         --to-static        the target static, for the land mask the soil remap is masked by\n\
         --to-vertical      the target's vertical artifact (zgrid, zz, fzm, fzp, dzu, rdzw and\n\
        \x20                  the init-stream slots).  Without it --to-grid must carry a vertical\n\
        \x20                  grid itself, as an init on the target mesh does\n\
         --balance          hydrostatic rebuilds rho in hydrostatic balance on the target with\n\
        \x20                  rw_mpas_init's own column routine; carry keeps the conservatively\n\
        \x20                  remapped rho, so the mass budget closes and the column is unbalanced\n\
         --virtual-factor   reproduce-fortran (theta_m = theta (1 + 1.61 qv), what the init\n\
        \x20                  writer and the dycore's equation of state use) or consistent\n\
         --min-coverage     the fraction of every target cell the source must cover\n\
        \x20                  (default {DEFAULT_MIN_COVERAGE}); below it the remap is refused with a\n\
        \x20                  coverage report\n\n\
         Neither --balance nor --virtual-factor has a default: each changes the\n\
         numbers in a file that opens cleanly either way."
    )
}

fn run_cli() -> Result<String, (String, Option<String>)> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "--help" || a == "-h") {
        return Err((usage(), None));
    }
    if argv.iter().any(|a| a == "--abi") {
        return Ok(ABI_MARKER.to_string());
    }
    if argv.iter().any(|a| a == "--version") {
        return Ok(format!("rw_mpas_remap {}", env!("CARGO_PKG_VERSION")));
    }
    let mut map = std::collections::BTreeMap::new();
    let mut it = argv.into_iter();
    while let Some(token) = it.next() {
        if !token.starts_with("--") {
            return Err((format!("unexpected argument \"{token}\"\n\n{}", usage()), None));
        }
        let key = token.trim_start_matches("--").to_string();
        let value = it
            .next()
            .ok_or_else(|| (format!("--{key} needs a value\n\n{}", usage()), None))?;
        if map.insert(key.clone(), value).is_some() {
            return Err((format!("--{key} was given twice"), None));
        }
    }
    let known = [
        "from-grid", "from-state", "to-grid", "to-static", "to-vertical", "out", "balance",
        "virtual-factor", "min-coverage", "receipt",
    ];
    if let Some(k) = map.keys().find(|k| !known.contains(&k.as_str())) {
        return Err((format!("unknown option --{k}\n\n{}", usage()), None));
    }
    let need = |key: &str| -> Result<String, (String, Option<String>)> {
        map.get(key)
            .cloned()
            .ok_or_else(|| (format!("--{key} was not given, and it has no default\n\n{}", usage()), None))
    };
    let cfg = RemapConfig {
        from_grid: PathBuf::from(need("from-grid")?),
        from_state: PathBuf::from(need("from-state")?),
        to_grid: PathBuf::from(need("to-grid")?),
        to_static: map.get("to-static").map(PathBuf::from),
        to_vertical: map.get("to-vertical").map(PathBuf::from),
        out: PathBuf::from(need("out")?),
        balance: Balance::parse(&need("balance")?).map_err(|e| (e, None))?,
        virtual_factor: match need("virtual-factor")?.as_str() {
            "reproduce-fortran" => VirtualFactor::ReproduceFortran,
            "consistent" => VirtualFactor::Consistent,
            other => {
                return Err((
                    format!("--virtual-factor takes reproduce-fortran or consistent, not \"{other}\""),
                    None,
                ))
            }
        },
        min_coverage: match map.get("min-coverage") {
            None => DEFAULT_MIN_COVERAGE,
            Some(v) => v
                .parse::<f64>()
                .map_err(|_| (format!("--min-coverage is not a number: {v}"), None))?,
        },
        provenance: format!(
            "rw_mpas_remap from {}",
            std::env::current_exe()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|_| "an unnamed tree".to_string())
        ),
    };
    let receipt = run(&cfg).map_err(|e| (e.to_string(), None))?;
    let json = serde_json::to_string_pretty(&receipt).map_err(|e| (e.to_string(), None))?;
    if let Some(path) = map.get("receipt") {
        std::fs::write(path, &json).map_err(|e| (format!("cannot write {path}: {e}"), None))?;
    }
    if let Some(refusal) = &receipt.refusal {
        return Err((refusal.clone(), Some(json)));
    }
    Ok(json)
}

fn main() -> ExitCode {
    let _ = std::hint::black_box(GPUWM_BRIDGE_SOURCE_REV_STAMP);
    match run_cli() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err((message, json)) => {
            if let Some(json) = json {
                println!("{json}");
            }
            eprintln!("rw_mpas_remap: {message}");
            ExitCode::FAILURE
        }
    }
}

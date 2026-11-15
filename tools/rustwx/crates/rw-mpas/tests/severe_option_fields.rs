//! A hex run's severe-weather option diagnostics (HAILCAST, lightning, AFWA,
//! nwp_diagnostics' interval extremes) reach the render frame under WRF's
//! names, from the diag stream hex writes them to; a run without the options
//! reads exactly as before and names none of them absent.
//!
//! THE GAP (severe-weather inventory, 2026-11-14): hex wrote hailcast_*,
//! the flash counts and rates, lpi and afwa_* to its diag stream and the
//! converter mapped none of them, so `woof hex products` reported every such
//! map "not stored".
use std::path::{Path, PathBuf};

use rw_mpas::fieldmap::{FIELD_MAP, field_set_allows, is_optional};
use rw_mpas::history::{read_history_with_init, Timestamp};
use rw_store::netcdf_classic::{NcClassicWriter, NcData, NcDim, NcFormat, NcType, NcVarDef};

const LAT: [f64; 3] = [0.1, -0.2, 0.3];
const LON: [f64; 3] = [0.5, 1.0, -2.0];
const THETA: [f32; 6] = [300.0, 301.0, 302.0, 303.0, 304.0, 305.0];
const T2M: [f32; 3] = [280.5, 281.5, 282.5];
const SEVERE: [(&str, &str, [f32; 3]); 5] = [
    ("hailcast_diam_max", "HAILCAST_DIAM_MAX", [0.0, 31.25, 52.5]),
    ("lpi", "LPI", [0.0, 12.0, 0.5]),
    ("ic_flashcount", "IC_FLASHCOUNT", [3.0, 0.0, 17.0]),
    ("afwa_hail", "AFWA_HAIL", [1.5, 0.0, 70.0]),
    ("w_dn_max", "W_DN_MAX", [-4.0, -12.5, 0.0]),
];

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rw-mpas-severe-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn dims() -> Vec<NcDim> {
    vec![NcDim::record("Time"), NcDim::fixed("nCells", 3), NcDim::fixed("nVertLevels", 2)]
}

fn write_history(path: &Path) {
    let vars = vec![
        NcVarDef::new("theta", NcType::Float, vec![0, 1, 2]),
        NcVarDef::new("latCell", NcType::Double, vec![1]),
        NcVarDef::new("lonCell", NcType::Double, vec![1]),
    ];
    let mut w = NcClassicWriter::create(path, NcFormat::Data64, dims(), vec![], vars, 1).unwrap();
    w.put_record("theta", 0, NcData::Floats(&THETA)).unwrap();
    w.put("latCell", NcData::Doubles(&LAT)).unwrap();
    w.put("lonCell", NcData::Doubles(&LON)).unwrap();
    w.finish().unwrap();
}

fn write_diag(path: &Path, severe: bool) {
    let mut vars = vec![NcVarDef::new("t2m", NcType::Float, vec![0, 1])];
    if severe {
        vars.extend(SEVERE.iter().map(|(name, _, _)| NcVarDef::new(*name, NcType::Float, vec![0, 1])));
    }
    let mut w = NcClassicWriter::create(path, NcFormat::Data64, dims(), vec![], vars, 1).unwrap();
    w.put_record("t2m", 0, NcData::Floats(&T2M)).unwrap();
    if severe {
        for (name, _, values) in &SEVERE {
            w.put_record(name, 0, NcData::Floats(values)).unwrap();
        }
    }
    w.finish().unwrap();
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn time() -> Option<Timestamp> {
    Some(Timestamp::parse("2024-05-21_20:00:00").unwrap())
}

#[test]
fn the_severe_options_diagnostics_are_read_from_the_diag_stream_word_for_word() {
    let dir = scratch("on");
    let history = dir.join("history.2024-05-21_20.00.00.nc");
    write_history(&history);
    write_diag(&dir.join("diag.2024-05-21_20.00.00.nc"), true);
    let frame = read_history_with_init(&history, None, time(), None).unwrap();
    for (_, wrf, values) in &SEVERE {
        assert_eq!(frame.fields[*wrf].1, 1, "{wrf} is a surface plane");
        assert_eq!(bits(&frame.fields[*wrf].0), bits(values), "{wrf}");
    }
    let diag = frame.companion_diag.as_ref().expect("the frame names the diag file it read");
    for (_, wrf, _) in &SEVERE {
        assert!(diag.fields.iter().any(|n| n == wrf), "{wrf} not recorded as a diag field: {:?}", diag.fields);
    }
    // The options this run did not turn on are not output-stream gaps.
    for m in FIELD_MAP.iter().filter(|m| is_optional(m)) {
        assert!(!frame.absent.iter().any(|n| n == m.wrf_name), "{} named absent", m.wrf_name);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn a_run_without_the_options_reads_as_before() {
    let dir = scratch("off");
    let history = dir.join("history.2024-05-21_20.00.00.nc");
    write_history(&history);
    write_diag(&dir.join("diag.2024-05-21_20.00.00.nc"), false);
    let frame = read_history_with_init(&history, None, time(), None).unwrap();
    assert!(FIELD_MAP.iter().filter(|m| is_optional(m)).all(|m| !frame.fields.contains_key(m.wrf_name)));
    // Every name the frame calls absent is a non-optional row: the absent list is the one it always was.
    for name in &frame.absent {
        assert!(FIELD_MAP.iter().any(|m| m.wrf_name == name && !is_optional(m) && field_set_allows("full", m))
                || FIELD_MAP.iter().any(|m| m.wrf_name == name && m.source != rw_mpas::fieldmap::FieldSource::History),
                "{name} is named absent");
    }
    assert!(frame.absent.iter().any(|n| n == "Q2"), "a mapped field neither stream carries stays absent");
    assert!(frame.companion_diag.as_ref().is_some_and(|d| d.fields == vec!["T2".to_string()]));
    std::fs::remove_dir_all(dir).unwrap();
}

//! The streamed digest is the SHA-256 of the file on disk, or nothing.
//!
//! Each case re-reads the finished file and hashes it with the crate's
//! portable SHA-256 (itself checked against the FIPS vectors), so the claim
//! tested is the one the Python identity relies on: the streamed value IS
//! the digest of the published bytes.

use netcdf_writer::{FinishedDigest, NcFormat, NcType, NcWriter, Schema, Sha256, VarData};

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("netcdf-writer-digest-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}.nc"))
}

fn file_sha(path: &std::path::Path) -> ([u8; 32], u64) {
    let bytes = std::fs::read(path).unwrap();
    let mut hasher = Sha256::new_portable();
    hasher.update(&bytes);
    (hasher.finalize(), bytes.len() as u64)
}

/// A wrfout-shaped schema: Times plus record variables, optionally a fixed one.
fn schema(format: NcFormat, with_fixed: bool) -> (Schema, Vec<usize>) {
    let mut schema = Schema::new(format);
    let time = schema.def_dim("Time", 0, true).unwrap();
    let strlen = schema.def_dim("DateStrLen", 19, false).unwrap();
    let z = schema.def_dim("z", 3, false).unwrap();
    let x = schema.def_dim("x", 5, false).unwrap();
    let mut vars = vec![schema.def_var("Times", NcType::Char, &[time, strlen]).unwrap()];
    vars.push(schema.def_var("T", NcType::Float, &[time, z, x]).unwrap());
    vars.push(schema.def_var("ITIMESTEP", NcType::Int, &[time]).unwrap());
    vars.push(schema.def_var("Q", NcType::Short, &[time, x]).unwrap());
    if with_fixed {
        vars.push(schema.def_var("H", NcType::Double, &[z]).unwrap());
    }
    schema
        .put_global_attr("TITLE", netcdf_writer::AttrValue::Text("digest".into()))
        .unwrap();
    (schema, vars)
}

fn write_record(writer: &mut NcWriter, vars: &[usize], recno: u64) {
    let times = format!("2026-01-01_00:00:{:02}", recno);
    writer.write_record(recno, vars[0], VarData::Char(times.as_bytes())).unwrap();
    let t: Vec<f32> = (0..15).map(|i| i as f32 * 0.5 + recno as f32).collect();
    writer.write_record(recno, vars[1], VarData::F32(&t)).unwrap();
    writer.write_record(recno, vars[2], VarData::I32(&[recno as i32])).unwrap();
    let q: Vec<i16> = (0..5).map(|i| i as i16 - 2).collect();
    writer.write_record(recno, vars[3], VarData::I16(&q)).unwrap();
}

#[test]
fn streamed_digest_equals_the_file_for_every_container() {
    for (label, format) in [("cdf1", NcFormat::Classic), ("cdf2", NcFormat::Offset64), ("cdf5", NcFormat::Cdf5)] {
        for records in [1u64, 3] {
            let path = tmp(&format!("{label}-{records}"));
            let (schema, vars) = schema(format, false);
            let mut writer = NcWriter::create(&path, schema).unwrap();
            writer.begin_digest(records).unwrap();
            for recno in 0..records {
                write_record(&mut writer, &vars, recno);
            }
            match writer.finish_with_digest().unwrap() {
                FinishedDigest::Streamed { sha256, bytes } => {
                    assert_eq!((sha256, bytes), file_sha(&path), "{label} x{records}");
                }
                other => panic!("{label} x{records}: {other:?}"),
            }
        }
    }
}

#[test]
fn fixed_variable_written_first_keeps_the_digest() {
    let path = tmp("fixed");
    let (schema, vars) = schema(NcFormat::Offset64, true);
    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer.begin_digest(1).unwrap();
    writer.write_var(vars[4], VarData::F64(&[1.0, 2.0, 3.0])).unwrap();
    write_record(&mut writer, &vars, 0);
    match writer.finish_with_digest().unwrap() {
        FinishedDigest::Streamed { sha256, bytes } => assert_eq!((sha256, bytes), file_sha(&path)),
        other => panic!("{other:?}"),
    }
}

#[test]
fn out_of_order_writes_withdraw_the_digest_but_not_the_file() {
    let path = tmp("out-of-order");
    let (schema, vars) = schema(NcFormat::Offset64, false);
    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer.begin_digest(1).unwrap();
    // T before Times: the first data write is not at the hashed end.
    let t: Vec<f32> = (0..15).map(|i| i as f32).collect();
    writer.write_record(0, vars[1], VarData::F32(&t)).unwrap();
    writer.write_record(0, vars[0], VarData::Char(b"2026-01-01_00:00:00")).unwrap();
    writer.write_record(0, vars[2], VarData::I32(&[0])).unwrap();
    writer.write_record(0, vars[3], VarData::I16(&[0, 0, 0, 0, 0])).unwrap();
    match writer.finish_with_digest().unwrap() {
        FinishedDigest::Unavailable(reason) => assert!(reason.contains("did not continue"), "{reason}"),
        other => panic!("{other:?}"),
    }
    assert!(std::fs::metadata(&path).unwrap().len() > 0);
}

#[test]
fn a_record_count_other_than_predicted_withdraws_the_digest() {
    let path = tmp("numrecs");
    let (schema, vars) = schema(NcFormat::Offset64, false);
    let mut writer = NcWriter::create(&path, schema).unwrap();
    writer.begin_digest(2).unwrap();
    write_record(&mut writer, &vars, 0);
    match writer.finish_with_digest().unwrap() {
        FinishedDigest::Unavailable(reason) => assert!(reason.contains("predicted"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn begin_digest_after_data_is_refused_and_without_it_finish_says_so() {
    let path = tmp("late");
    let (schema, vars) = schema(NcFormat::Offset64, false);
    let mut writer = NcWriter::create(&path, schema).unwrap();
    write_record(&mut writer, &vars, 0);
    assert!(writer.begin_digest(1).is_err());
    match writer.finish_with_digest().unwrap() {
        FinishedDigest::Unavailable(reason) => assert!(reason.contains("no streamed digest"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

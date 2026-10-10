//! The C ABI seam `woof/energy/sample_bridge.py` loads through ctypes.
//!
//! Same discipline as `rw-isobaric/src/capi.rs` and `obs-score`: cdylib +
//! ctypes, positional C signatures guarded by an ABI version probe, a
//! thread-local last-error string, a source-revision stamp and a
//! contract-marker symbol (`gpuwm_sitesample_wind_profile`, listed in
//! `woof/bridges.py` `BRIDGE_ABI_MARKERS`).
//!
//! Stateless: every call takes one time's field windows and writes into
//! caller buffers.  Fields cross as C-contiguous `f32` exactly as the
//! history stores them (level outermost); site indices, heights and every
//! output cross as `f64`.  `nz`, `ny`, `nx` are the window's MASS
//! dimensions.  Site outputs are site-major: `out[site * nh + h]`.
//! A site outside the window's mass grid, or with a non-finite index,
//! receives NaN everywhere.

use std::cell::RefCell;

use crate::{Dims, Field, Stagger, heights_agl, interpolate_heights, stencil, wind_columns};

/// The seam's ABI version.  Bump when a signature changes shape, never for a
/// rebuild.
pub const SITESAMPLE_ABI_VERSION: u32 = 1;

const OK: i32 = 0;
const ERR: i32 = -1;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn run(body: impl FnOnce() -> Result<(), String>) -> i32 {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(Ok(())) => {
            LAST_ERROR.with(|slot| slot.borrow_mut().clear());
            OK
        }
        Ok(Err(message)) => {
            LAST_ERROR.with(|slot| *slot.borrow_mut() = message);
            ERR
        }
        Err(payload) => {
            let detail = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic payload".to_string());
            LAST_ERROR.with(|slot| {
                if let Ok(mut message) = slot.try_borrow_mut() {
                    *message = format!("panic in the site sampler: {detail}");
                }
            });
            ERR
        }
    }
}

/// # Safety
/// `ptr` must point to `len` readable `T`, or be null when `len` is 0.
unsafe fn input<'a, T>(ptr: *const T, len: usize, what: &str) -> Result<&'a [T], String> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(format!("{what}: null pointer for {len} values"));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// # Safety
/// `ptr` must point to `len` writable `T`, or be null when `len` is 0.
unsafe fn output<'a, T>(ptr: *mut T, len: usize, what: &str) -> Result<&'a mut [T], String> {
    if len == 0 {
        return Ok(&mut []);
    }
    if ptr.is_null() {
        return Err(format!("{what}: null output pointer for {len} values"));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
}

fn checked_dims(nz: usize, ny: usize, nx: usize) -> Result<Dims, String> {
    let dims = Dims { nz, ny, nx };
    if dims.is_empty() {
        return Err(format!("empty window: nz={nz} ny={ny} nx={nx}"));
    }
    (nz + 1)
        .checked_mul(ny + 1)
        .and_then(|n| n.checked_mul(nx + 1))
        .ok_or_else(|| format!("window nz={nz} ny={ny} nx={nx} overflows"))?;
    Ok(dims)
}

#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_sitesample_abi_version() -> u32 {
    SITESAMPLE_ABI_VERSION
}

/// The `GPUWM_BRIDGE_SOURCE_REV=<commit>` stamp the release cut reads out of
/// the built library as bytes.
#[unsafe(no_mangle)]
pub extern "C" fn gpuwm_sitesample_source_rev() -> *const std::os::raw::c_char {
    static SOURCE_REV_STAMP: &str = concat!(
        "GPUWM_BRIDGE_SOURCE_REV=",
        env!("GPUWM_BRIDGE_SOURCE_REV"),
        "\0"
    );
    SOURCE_REV_STAMP.as_ptr().cast()
}

/// Copy the last error (UTF-8, no terminator) into `buf` and return its full
/// length; call with `cap == 0` to size the buffer.
///
/// # Safety
/// `buf` must point to `cap` writable bytes, or be null when `cap` is 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_sitesample_last_error(buf: *mut u8, cap: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        let bytes = message.as_bytes();
        if !buf.is_null() && cap > 0 {
            let n = bytes.len().min(cap);
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n) };
        }
        bytes.len()
    })
}

/// `out[s] = 1` when site `s` has a bilinear stencil on the `ny` by `nx`
/// mass grid, else 0.
///
/// # Safety
/// `fi`, `fj` hold `ns` values; `out` holds `ns` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_sitesample_inside(
    fi: *const f64,
    fj: *const f64,
    ns: usize,
    ny: usize,
    nx: usize,
    out: *mut u8,
) -> i32 {
    run(|| {
        let fi = unsafe { input(fi, ns, "fi") }?;
        let fj = unsafe { input(fj, ns, "fj") }?;
        let out = unsafe { output(out, ns, "inside") }?;
        for s in 0..ns {
            out[s] = u8::from(stencil(fi[s], fj[s], ny, nx).is_some());
        }
        Ok(())
    })
}

/// Height above model terrain of each mass level at each site,
/// `out[s * nz + k]`.  `ph`, `phb` are `bottom_top_stag` windows
/// (`(nz + 1) * ny * nx`), `hgt` a mass plane.
///
/// # Safety
/// Every pointer holds the length its shape states.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_sitesample_heights_agl(
    ph: *const f32,
    phb: *const f32,
    hgt: *const f32,
    nz: usize,
    ny: usize,
    nx: usize,
    gravity: f64,
    fi: *const f64,
    fj: *const f64,
    ns: usize,
    out: *mut f64,
) -> i32 {
    run(|| {
        let dims = checked_dims(nz, ny, nx)?;
        if !(gravity.is_finite() && gravity > 0.0) {
            return Err(format!("gravity {gravity} is not a positive number"));
        }
        let n = dims.len(Stagger::Z);
        let ph = unsafe { input(ph, n, "PH") }?;
        let phb = unsafe { input(phb, n, "PHB") }?;
        let hgt = unsafe { input(hgt, ny * nx, "HGT") }?;
        let fi = unsafe { input(fi, ns, "fi") }?;
        let fj = unsafe { input(fj, ns, "fj") }?;
        let out = unsafe { output(out, ns * nz, "heights") }?;
        for s in 0..ns {
            let row = &mut out[s * nz..(s + 1) * nz];
            match stencil(fi[s], fj[s], ny, nx) {
                Some(st) => row.copy_from_slice(&heights_agl(ph, phb, hgt, dims, gravity, &st)),
                None => row.fill(f64::NAN),
            }
        }
        Ok(())
    })
}

/// One profile variable, `scale * (field + plus) + offset` (plus nullable),
/// destaggered per `stagger` (0 mass, 1 x, 2 y, 3 z), interpolated
/// bilinearly to each site and linearly in height from the site's mass-level
/// heights `zagl` (`ns * nz`, from `gpuwm_sitesample_heights_agl`) to the
/// `nh` requested heights.  `out[s * nh + h]`.
///
/// # Safety
/// Every pointer holds the length its shape states; `plus` may be null.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_sitesample_profile(
    field: *const f32,
    plus: *const f32,
    scale: f64,
    offset: f64,
    stagger: u32,
    nz: usize,
    ny: usize,
    nx: usize,
    fi: *const f64,
    fj: *const f64,
    ns: usize,
    zagl: *const f64,
    heights: *const f64,
    nh: usize,
    out: *mut f64,
) -> i32 {
    run(|| {
        let dims = checked_dims(nz, ny, nx)?;
        let stagger =
            Stagger::from_code(stagger).ok_or_else(|| format!("unknown stagger code {stagger}"))?;
        let n = dims.len(stagger);
        let data = unsafe { input(field, n, "field") }?;
        let plus = if plus.is_null() {
            None
        } else {
            Some(unsafe { input(plus, n, "plus") }?)
        };
        let field = Field {
            data,
            plus,
            scale,
            offset,
            stagger,
            dims,
        };
        let fi = unsafe { input(fi, ns, "fi") }?;
        let fj = unsafe { input(fj, ns, "fj") }?;
        let zagl = unsafe { input(zagl, ns * nz, "zagl") }?;
        let heights = unsafe { input(heights, nh, "heights") }?;
        let out = unsafe { output(out, ns * nh, "profile") }?;
        for s in 0..ns {
            let row = &mut out[s * nh..(s + 1) * nh];
            match stencil(fi[s], fj[s], ny, nx) {
                Some(st) => {
                    let column = field.column(&st);
                    interpolate_heights(&column, &zagl[s * nz..(s + 1) * nz], heights, row);
                }
                None => row.fill(f64::NAN),
            }
        }
        Ok(())
    })
}

/// Earth-relative wind profile: `u` on `west_east_stag`, `v` on
/// `south_north_stag`, `sinalpha`/`cosalpha` mass planes.  Rotated at each
/// mass point of the stencil, then interpolated as `gpuwm_sitesample_profile`.
///
/// # Safety
/// Every pointer holds the length its shape states.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_sitesample_wind_profile(
    u: *const f32,
    v: *const f32,
    sinalpha: *const f32,
    cosalpha: *const f32,
    nz: usize,
    ny: usize,
    nx: usize,
    fi: *const f64,
    fj: *const f64,
    ns: usize,
    zagl: *const f64,
    heights: *const f64,
    nh: usize,
    out_u: *mut f64,
    out_v: *mut f64,
) -> i32 {
    run(|| {
        let dims = checked_dims(nz, ny, nx)?;
        let u = Field::plain(
            unsafe { input(u, dims.len(Stagger::X), "U") }?,
            Stagger::X,
            dims,
        );
        let v = Field::plain(
            unsafe { input(v, dims.len(Stagger::Y), "V") }?,
            Stagger::Y,
            dims,
        );
        let sin = unsafe { input(sinalpha, ny * nx, "SINALPHA") }?;
        let cos = unsafe { input(cosalpha, ny * nx, "COSALPHA") }?;
        let fi = unsafe { input(fi, ns, "fi") }?;
        let fj = unsafe { input(fj, ns, "fj") }?;
        let zagl = unsafe { input(zagl, ns * nz, "zagl") }?;
        let heights = unsafe { input(heights, nh, "heights") }?;
        let out_u = unsafe { output(out_u, ns * nh, "U") }?;
        let out_v = unsafe { output(out_v, ns * nh, "V") }?;
        for s in 0..ns {
            let (ru, rv) = (s * nh..(s + 1) * nh, s * nh..(s + 1) * nh);
            match stencil(fi[s], fj[s], ny, nx) {
                Some(st) => {
                    let (ue, ve) = wind_columns(&u, &v, sin, cos, &st);
                    let z = &zagl[s * nz..(s + 1) * nz];
                    interpolate_heights(&ue, z, heights, &mut out_u[ru]);
                    interpolate_heights(&ve, z, heights, &mut out_v[rv]);
                }
                None => {
                    out_u[ru].fill(f64::NAN);
                    out_v[rv].fill(f64::NAN);
                }
            }
        }
        Ok(())
    })
}

/// A surface (mass-plane) field interpolated bilinearly to each site.
///
/// # Safety
/// `field` holds `ny * nx` values, `fi`/`fj`/`out` hold `ns`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpuwm_sitesample_surface(
    field: *const f32,
    ny: usize,
    nx: usize,
    fi: *const f64,
    fj: *const f64,
    ns: usize,
    out: *mut f64,
) -> i32 {
    run(|| {
        let dims = checked_dims(1, ny, nx)?;
        let field = Field::plain(
            unsafe { input(field, ny * nx, "field") }?,
            Stagger::Mass,
            dims,
        );
        let fi = unsafe { input(fi, ns, "fi") }?;
        let fj = unsafe { input(fj, ns, "fj") }?;
        let out = unsafe { output(out, ns, "surface") }?;
        for s in 0..ns {
            out[s] = match stencil(fi[s], fj[s], ny, nx) {
                Some(st) => field.bilinear(0, &st),
                None => f64::NAN,
            };
        }
        Ok(())
    })
}

/// Earth-relative surface wind (U10/V10, mass planes): rotated at each mass
/// point of the stencil, then interpolated bilinearly.
///
/// # Safety
/// The four planes hold `ny * nx` values, `fi`/`fj`/outputs hold `ns`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn gpuwm_sitesample_wind_surface(
    u: *const f32,
    v: *const f32,
    sinalpha: *const f32,
    cosalpha: *const f32,
    ny: usize,
    nx: usize,
    fi: *const f64,
    fj: *const f64,
    ns: usize,
    out_u: *mut f64,
    out_v: *mut f64,
) -> i32 {
    run(|| {
        let dims = checked_dims(1, ny, nx)?;
        let plane = ny * nx;
        let u = Field::plain(unsafe { input(u, plane, "U10") }?, Stagger::Mass, dims);
        let v = Field::plain(unsafe { input(v, plane, "V10") }?, Stagger::Mass, dims);
        let sin = unsafe { input(sinalpha, plane, "SINALPHA") }?;
        let cos = unsafe { input(cosalpha, plane, "COSALPHA") }?;
        let fi = unsafe { input(fi, ns, "fi") }?;
        let fj = unsafe { input(fj, ns, "fj") }?;
        let out_u = unsafe { output(out_u, ns, "U10") }?;
        let out_v = unsafe { output(out_v, ns, "V10") }?;
        for s in 0..ns {
            match stencil(fi[s], fj[s], ny, nx) {
                Some(st) => {
                    let (ue, ve) = wind_columns(&u, &v, sin, cos, &st);
                    out_u[s] = ue[0];
                    out_v[s] = ve[0];
                }
                None => {
                    out_u[s] = f64::NAN;
                    out_v[s] = f64::NAN;
                }
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn last_error() -> String {
        let n = unsafe { gpuwm_sitesample_last_error(std::ptr::null_mut(), 0) };
        let mut buf = vec![0u8; n];
        unsafe { gpuwm_sitesample_last_error(buf.as_mut_ptr(), n) };
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn profile_refuses_an_unknown_stagger_and_reports_why() {
        let data = [0.0f32; 8];
        let (fi, fj, z, h) = ([0.5f64], [0.5f64], [5.0f64, 15.0], [10.0f64]);
        let mut out = [0.0f64; 1];
        let status = unsafe {
            gpuwm_sitesample_profile(
                data.as_ptr(),
                std::ptr::null(),
                1.0,
                0.0,
                7,
                2,
                2,
                2,
                fi.as_ptr(),
                fj.as_ptr(),
                1,
                z.as_ptr(),
                h.as_ptr(),
                1,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, ERR);
        assert!(last_error().contains("unknown stagger code 7"));
    }

    #[test]
    fn profile_samples_inside_and_fills_outside_with_nan() {
        // T = 10 k + i, mass (2, 2, 2); heights 5 and 15 m
        let data: Vec<f32> = (0..8).map(|n| (10 * (n / 4) + n % 2) as f32).collect();
        let fi = [0.25f64, 5.0];
        let fj = [0.5f64, 0.5];
        let z = [5.0f64, 15.0, 5.0, 15.0];
        let h = [10.0f64];
        let mut out = [0.0f64; 2];
        let status = unsafe {
            gpuwm_sitesample_profile(
                data.as_ptr(),
                std::ptr::null(),
                1.0,
                300.0,
                0,
                2,
                2,
                2,
                fi.as_ptr(),
                fj.as_ptr(),
                2,
                z.as_ptr(),
                h.as_ptr(),
                1,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, OK, "{}", last_error());
        assert!((out[0] - (300.0 + 5.0 + 0.25)).abs() < 1e-12, "{out:?}");
        assert!(out[1].is_nan());
        let mut inside = [9u8; 2];
        let status = unsafe {
            gpuwm_sitesample_inside(fi.as_ptr(), fj.as_ptr(), 2, 2, 2, inside.as_mut_ptr())
        };
        assert_eq!(status, OK);
        assert_eq!(inside, [1, 0]);
    }

    #[test]
    fn heights_refuse_a_non_positive_gravity() {
        let ph = [0.0f32; 12];
        let hgt = [0.0f32; 4];
        let (fi, fj) = ([0.5f64], [0.5f64]);
        let mut out = [0.0f64; 2];
        let status = unsafe {
            gpuwm_sitesample_heights_agl(
                ph.as_ptr(),
                ph.as_ptr(),
                hgt.as_ptr(),
                2,
                2,
                2,
                0.0,
                fi.as_ptr(),
                fj.as_ptr(),
                1,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, ERR);
        assert!(last_error().contains("gravity"));
    }

    #[test]
    fn abi_version_and_stamp_are_exported() {
        assert_eq!(gpuwm_sitesample_abi_version(), SITESAMPLE_ABI_VERSION);
        let stamp = unsafe { std::ffi::CStr::from_ptr(gpuwm_sitesample_source_rev()) };
        assert!(
            stamp
                .to_str()
                .unwrap()
                .starts_with("GPUWM_BRIDGE_SOURCE_REV=")
        );
    }
}

//! glibc239-libm verification harness (verification.log beside it).
//!
//! Build: a binary crate with this file as src/main.rs and the dependencies
//! `glibc239-libm = { path = ".../tools/rustwx/crates/glibc239-libm" }` and
//! `rw-libm = { path = ".../tools/rustwx/crates/rw-libm" }`, `cargo build
//! --release`.  The reference is the process libm (`exp`, `sin`, ... by
//! their C names).  To grade against glibc 2.39 itself on any Linux x86-64
//! machine with FMA and AVX2 (the CPU decides glibc's ifunc variant), run
//! the binary inside the stock Ubuntu 24.04 image, whose libc6 is glibc
//! 2.39: `docker run --rm -v $PWD:/w ubuntu:24.04 /w/harness random sin
//! 1000000000 2026`.  Run outside, it grades against the host's glibc,
//! which shows where that glibc's words are not 2.39's.
//!
//! Modes:
//!   random <f> <n> <seed>  n random arguments from mixed distributions
//!                          (any bit pattern, bit-uniform over each
//!                          function's ranges, uniform in typical ranges)
//!   edges <f>              every binade swept by high word, ±2048 ULP
//!                          around each branch threshold of the glibc
//!                          source, k*pi/2 neighbourhoods for sin, cos and
//!                          tan, and a special-value grid for pow and atan2
//!   f32 <f>                expf and logf on all 2^32 arguments, powf on
//!                          1e9 random pairs (the scalar binary32 entries,
//!                          rw-libm's transcriptions); asinf, glibc239-libm's
//!                          binary32 asinf, on all 2^32 arguments
//! A mismatch is a different word; two NaNs with different payloads are
//! counted separately.  Up to 20 mismatches are printed.
use std::sync::atomic::{AtomicU64, Ordering};

unsafe extern "C" {
    fn exp(x: f64) -> f64; fn log(x: f64) -> f64; fn log10(x: f64) -> f64;
    fn sin(x: f64) -> f64; fn cos(x: f64) -> f64; fn tan(x: f64) -> f64;
    fn atan(x: f64) -> f64; fn asin(x: f64) -> f64; fn acos(x: f64) -> f64;
    fn pow(x: f64, y: f64) -> f64; fn atan2(y: f64, x: f64) -> f64;
    fn expf(x: f32) -> f32; fn logf(x: f32) -> f32; fn powf(x: f32, y: f32) -> f32;
    fn dlsym(handle: *mut std::ffi::c_void, name: *const std::ffi::c_char) -> *mut std::ffi::c_void;
}

const THREADS: u64 = 12;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn unif(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * ((self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)) }
    fn bits(&mut self, lo: f64, hi: f64) -> f64 { let a = lo.to_bits(); let b = hi.to_bits(); f64::from_bits(a + self.next() % (b - a + 1)) }
    fn sgn(&mut self, x: f64) -> f64 { if self.next() & 1 == 1 { -x } else { x } }
}

fn unary(name: &str) -> (fn(f64) -> f64, unsafe extern "C" fn(f64) -> f64) {
    use glibc239_libm as g;
    match name {
        "exp" => (g::exp, exp), "log" => (g::log, log), "log10" => (g::log10, log10),
        "sin" => (g::sin, sin), "cos" => (g::cos, cos), "tan" => (g::tan, tan),
        "atan" => (g::atan, atan), "asin" => (g::asin, asin), "acos" => (g::acos, acos),
        _ => panic!("unknown function {name}"),
    }
}

fn binary(name: &str) -> (fn(f64, f64) -> f64, unsafe extern "C" fn(f64, f64) -> f64) {
    match name {
        "pow" => (glibc239_libm::pow, pow), "atan2" => (glibc239_libm::atan2, atan2),
        _ => panic!("unknown function {name}"),
    }
}

fn arg1(name: &str, r: &mut Rng) -> f64 {
    let k = r.next() % 8;
    if k == 0 { return f64::from_bits(r.next()); }
    match name {
        "exp" => match k { 1 | 2 => r.unif(-746.0, 710.0), 3 => r.unif(-5.0, 5.0), 4 => { let x = r.bits(1e-300, 1e3); r.sgn(x) }, _ => r.unif(-1.0, 1.0) },
        "log" | "log10" => match k { 1 | 2 => r.bits(5e-324, f64::MAX), 3 => r.unif(0.5, 2.0), 4 => r.unif(0.0, 1e6), 5 => 1.0 + r.unif(-1e-3, 1e-3), _ => r.unif(0.0, 1e3) },
        "sin" | "cos" | "tan" => match k { 1 => { let x = r.bits(1e-300, 1e300); r.sgn(x) }, 2 => { let x = r.bits(1e-10, 1e22); r.sgn(x) }, 3 | 4 => r.unif(-7.0, 7.0), 5 => r.unif(-1.6, 1.6), 6 => r.unif(-1e6, 1e6), _ => r.unif(-0.3, 0.3) },
        "atan" => match k { 1 | 2 => { let x = r.bits(1e-300, 1e300); r.sgn(x) }, 3 => r.unif(-3.0, 3.0), 4 => r.unif(-1.0, 1.0), 5 => r.unif(-30.0, 30.0), _ => r.unif(-0.2, 0.2) },
        _ => match k { 1 | 2 => { let x = r.bits(1e-300, 1.0); r.sgn(x) }, 3 | 4 => r.unif(-1.0, 1.0), 5 => { let x = 1.0 - r.bits(1e-17, 0.3); r.sgn(x) }, _ => r.unif(-0.6, 0.6) },
    }
}

fn arg2(name: &str, r: &mut Rng) -> (f64, f64) {
    let k = r.next() % 8;
    if k == 0 { return (f64::from_bits(r.next()), f64::from_bits(r.next())); }
    if name == "pow" {
        match k {
            1 => (r.bits(1e-300, 1e300), { let e = r.bits(1e-10, 1e3); r.sgn(e) }),
            2 => (r.unif(0.0, 4.0), r.unif(-10.0, 10.0)),
            3 => (1.0 + r.unif(-1e-2, 1e-2), r.unif(-1e3, 1e3)),
            4 => (r.unif(0.0, 1e4), r.unif(-3.0, 3.0)),
            5 => (r.unif(-100.0, 0.0), (r.next() % 41) as f64 - 20.0),
            6 => (r.unif(0.0, 2.0), r.unif(0.0, 1.0)),
            _ => (r.unif(1e-3, 1e5), r.unif(-0.5, 2.5)),
        }
    } else {
        match k {
            1 => { let a = r.bits(1e-300, 1e300); let b = r.bits(1e-300, 1e300); (r.sgn(a), r.sgn(b)) }
            2 | 3 => (r.unif(-3.0, 3.0), r.unif(-3.0, 3.0)),
            4 => (r.unif(-1.0, 1.0), r.unif(-1.0, 1.0)),
            5 => (r.unif(-1e6, 1e6), r.unif(-1e6, 1e6)),
            6 => (r.unif(-1e-3, 1e-3), r.unif(0.5, 2.0)),
            _ => (r.unif(-50.0, 50.0), r.unif(-1.0, 50.0)),
        }
    }
}

struct Tally { bad: AtomicU64, nan: AtomicU64, shown: AtomicU64 }
impl Tally {
    fn new() -> Self { Tally { bad: AtomicU64::new(0), nan: AtomicU64::new(0), shown: AtomicU64::new(0) } }
    fn check(&self, label: &str, x: f64, y: Option<f64>, got: f64, want: f64) {
        if got.to_bits() == want.to_bits() { return; }
        if got.is_nan() && want.is_nan() { self.nan.fetch_add(1, Ordering::Relaxed); return; }
        self.bad.fetch_add(1, Ordering::Relaxed);
        if self.shown.fetch_add(1, Ordering::Relaxed) < 20 {
            let y = y.map(|y| format!(", {y:e} [{:#018x}]", y.to_bits())).unwrap_or_default();
            println!("  {label}({x:e} [{:#018x}]{y}) port {:#018x} libm {:#018x}", x.to_bits(), got.to_bits(), want.to_bits());
        }
    }
    fn report(&self, what: &str, n: u64) {
        println!("{what:<40} n={n:>12} mismatches={} nan_payload_diffs={}",
                 self.bad.load(Ordering::Relaxed), self.nan.load(Ordering::Relaxed));
    }
}

fn random(name: &str, n: u64, seed: u64) {
    let tally = Tally::new();
    let per = n / THREADS;
    std::thread::scope(|s| for t in 0..THREADS {
        let tally = &tally;
        s.spawn(move || {
            let mut r = Rng(seed.wrapping_mul(1_000_003).wrapping_add(t));
            for _ in 0..per {
                if name == "pow" || name == "atan2" {
                    let (port, libm) = binary(name);
                    let (x, y) = arg2(name, &mut r);
                    tally.check(name, x, Some(y), port(x, y), unsafe { libm(x, y) });
                } else {
                    let (port, libm) = unary(name);
                    let x = arg1(name, &mut r);
                    tally.check(name, x, None, port(x), unsafe { libm(x) });
                }
            }
        });
    });
    tally.report(&format!("{name} random seed {seed}"), per * THREADS);
}

/// Branch thresholds of each glibc 2.39 source, as values.
fn thresholds(name: &str) -> Vec<f64> {
    let hw = |h: u64| f64::from_bits(h << 32);
    let mut v = vec![0.0, f64::MIN_POSITIVE, 5e-324, f64::MAX, 1.0, 0.5, 2.0];
    match name {
        "exp" => v.extend([hw(0x3c900000), 512.0, 1024.0, 709.782712893384, -708.3964185322641, -745.1332191019411, -744.4400719213812]),
        "log" | "log10" => v.extend([0.9375, 1.064697265625, 0.6875, 1.375, 10.0, 1e22]),
        "sin" | "cos" => v.extend([hw(0x3e400000), hw(0x3e500000), hw(0x3feb6000), hw(0x400368fd), hw(0x419921fb), 0.126, 0.855469, 2.426265, 105414350.0, 1e22, 1e300]),
        "tan" => v.extend([0.0608, 0.787, 25.0, 1e8, 1e22, 1e300]),
        "atan" => v.extend([0.0625, 0.125, 0.25, 1.0 / 0.0625, 8.0, 16.0, 1e8, 1e300, hw(0x3e400000)]),
        _ => v.extend([hw(0x3c880000), hw(0x3e500000), hw(0x3fc00000), hw(0x3fd00000), hw(0x3fe00000), hw(0x3fe80000), hw(0x3fed8000), hw(0x3fee8000), hw(0x3fef0000), hw(0x3ff00000)]),
    }
    v
}

fn edges(name: &str) {
    if name == "pow" || name == "atan2" {
        let (port, libm) = binary(name);
        let mut s: Vec<f64> = vec![0.0, 5e-324, f64::MIN_POSITIVE, 0.5, 1.0, 2.0, 3.0, 0.75, 10.0, 1e-300, 1e300,
            f64::MAX, f64::INFINITY, f64::NAN, f64::from_bits(0x7ff0000000000001), 1.0 + f64::EPSILON,
            1.0 - f64::EPSILON / 2.0, 1.5, 0.1, 9007199254740993.0, 4503599627370497.0, 2f64.powi(-65),
            2f64.powi(-63), 2f64.powi(63), 2f64.powi(64), 709.0, 1075.0, 2f64.powi(500), 2f64.powi(-500)];
        let neg: Vec<f64> = s.iter().map(|v| -v).collect();
        s.extend(neg);
        let tally = Tally::new();
        let mut n = 0u64;
        for &a in &s { for &b in &s {
            for (da, db) in [(0i64, 0i64), (1, 0), (-1, 0), (0, 1), (0, -1)] {
                let x = f64::from_bits((a.to_bits() as i64 + da) as u64);
                let y = f64::from_bits((b.to_bits() as i64 + db) as u64);
                n += 1;
                tally.check(name, x, Some(y), port(x, y), unsafe { libm(x, y) });
            }
        } }
        tally.report(&format!("{name} special grid"), n);
        // Near-ties of the reduction: x close to 1 and y large; y/x near the
        // atan2 table boundaries and exponent gaps.
        let tally = Tally::new();
        let mut r = Rng(7);
        let count = 50_000_000u64;
        for _ in 0..count {
            let (x, y) = if name == "pow" {
                let (d, y) = (r.bits(1e-17, 1e-3), r.bits(1.0, 1e18));
                (1.0 + r.sgn(d), r.sgn(y))
            } else {
                let x = r.bits(1e-6, 1e6);
                let x = r.sgn(x);
                let e = (r.next() % 130) as i32 - 65;
                (x * 2f64.powi(e) * r.unif(0.5, 2.0), x)
            };
            tally.check(name, x, Some(y), port(x, y), unsafe { libm(x, y) });
        }
        tally.report(&format!("{name} near-boundary pairs"), count);
        return;
    }
    let (port, libm) = unary(name);
    let tally = Tally::new();
    let n = AtomicU64::new(0);
    // Every high word that is a multiple of 0x100, and its neighbours, with
    // low words at both ends: each binade sampled 4096 times, uniformly.
    std::thread::scope(|s| for t in 0..THREADS {
        let (tally, n) = (&tally, &n);
        s.spawn(move || {
            let lows = [0u64, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff];
            let mut step = t;
            while step < 0x7ff00 {
                for offset in [0u64, 1, 0xff] {
                    let high = step * 0x100 + offset;
                    for sign in [0u64, 1 << 63] {
                        for &lo in &lows {
                            let x = f64::from_bits(sign | high << 32 | lo);
                            n.fetch_add(1, Ordering::Relaxed);
                            tally.check(name, x, None, port(x), unsafe { libm(x) });
                        }
                    }
                }
                step += THREADS;
            }
        });
    });
    // ±2048 ULP around each threshold, both signs.
    for t in thresholds(name) {
        for sign in [1.0, -1.0] {
            let b = (t * sign).to_bits() as i64;
            for d in -2048..=2048i64 {
                let x = f64::from_bits((b + d) as u64);
                n.fetch_add(1, Ordering::Relaxed);
                tally.check(name, x, None, port(x), unsafe { libm(x) });
            }
        }
    }
    // k*pi/2 neighbourhoods: the reductions' hardest cancellations.
    if matches!(name, "sin" | "cos" | "tan") {
        let mut r = Rng(11);
        let ks: Vec<f64> = (1..200_000).map(|k| k as f64)
            .chain((0..200_000).map(|_| (r.next() % (1u64 << 50)) as f64)).collect();
        for k in ks {
            let b = (k * std::f64::consts::FRAC_PI_2).to_bits() as i64;
            for d in -2..=2i64 {
                for sign in [0u64, 1 << 63] {
                    let x = f64::from_bits(((b + d) as u64) | sign);
                    n.fetch_add(1, Ordering::Relaxed);
                    tally.check(name, x, None, port(x), unsafe { libm(x) });
                }
            }
        }
    }
    for x in [f64::INFINITY, -f64::INFINITY, f64::NAN, -f64::NAN, f64::from_bits(0x7ff0000000000001), -0.0] {
        n.fetch_add(1, Ordering::Relaxed);
        tally.check(name, x, None, port(x), unsafe { libm(x) });
    }
    tally.report(&format!("{name} edges"), n.load(Ordering::Relaxed));
}

/// The process's default `asinf`, looked up at run time: a direct link
/// would bind glibc 2.43's versioned symbol, which glibc 2.39 lacks.
fn libm_asinf() -> unsafe extern "C" fn(f32) -> f32 {
    let f = unsafe { dlsym(std::ptr::null_mut(), c"asinf".as_ptr()) };
    assert!(!f.is_null(), "asinf is not in the process");
    unsafe { std::mem::transmute(f) }
}

fn f32_mode(name: &str) {
    let asinf = libm_asinf();
    let tally = Tally::new();
    let n = if name == "powf" { 1_000_000_000u64 } else { 1u64 << 32 };
    let per = n / THREADS;
    std::thread::scope(|s| for t in 0..THREADS {
        let tally = &tally;
        s.spawn(move || {
            let mut r = Rng(0x5eed ^ t);
            for i in t * per..(t + 1) * per {
                let (x, y, got, want) = match name {
                    "expf" => { let x = f32::from_bits(i as u32); (x, 0.0, rw_libm::expf(x), unsafe { expf(x) }) }
                    "logf" => { let x = f32::from_bits(i as u32); (x, 0.0, rw_libm::logf(x), unsafe { logf(x) }) }
                    "asinf" => { let x = f32::from_bits(i as u32); (x, 0.0, glibc239_libm::asinf(x), unsafe { asinf(x) }) }
                    _ => {
                        let w = r.next();
                        let x = if w & 3 == 0 { f32::from_bits(w as u32) } else { f32::from_bits(w as u32 & 0x7fffffff) % 4096.0 };
                        let y = if w & 12 == 0 { f32::from_bits((w >> 32) as u32) } else { ((w >> 32) as u32 as f32 / 4294967296.0 - 0.5) * 20.0 };
                        (x, y, rw_libm::powf(x, y), unsafe { powf(x, y) })
                    }
                };
                tally.check(name, x as f64, Some(y as f64), got as f64, want as f64);
            }
        });
    });
    tally.report(&format!("{name} (binary32)"), per * THREADS);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("random") => random(&a[2], a[3].parse().unwrap(), a[4].parse().unwrap()),
        Some("edges") => edges(&a[2]),
        Some("f32") => f32_mode(&a[2]),
        _ => eprintln!("usage: harness random <f> <n> <seed> | edges <f> | f32 <expf|logf|powf|asinf>"),
    }
}

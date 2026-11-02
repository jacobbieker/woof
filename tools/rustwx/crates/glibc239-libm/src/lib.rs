//! glibc 2.39's binary64 libm, transcribed, so every host computes its words.
//!
//! WRF's reference fixtures were computed with gfortran against glibc 2.39
//! on x86-64, where glibc's ifunc selects the variants compiled with
//! `-mfma -mavx2`; GCC contracts multiply-adds in those builds.  macOS's
//! libm, the Windows UCRT and other glibc releases round some of these
//! functions differently (glibc 2.44 even reduces huge `sin`, `cos` and
//! `tan` arguments differently), so the CPU preparation bridge's host math,
//! the static-fields projections and the host builds of the reference
//! mirrors call these instead of the platform C library.
//!
//! Each function follows glibc 2.39's own source statement for statement,
//! with `f64::mul_add` exactly where the installed FMA variant fuses (read
//! from its disassembly) and no other contraction: Rust never contracts,
//! and `mul_add` is the correctly rounded fused operation on every target.
//! Without hardware FMA it lowers to the C library's `fma`, which is
//! correctly rounded too (the tests below hold it to that), so a target's
//! FMA support changes speed, never a word.  The verification record is
//! tools/glibc239_libm_proof/.
//!
//! Licences, by file: exp.rs, log.rs and pow.rs (and their data) are Arm
//! optimized-routines under MIT; log10.rs is FDLIBM under Sun's notice;
//! dla.rs, branred.rs, sincos.rs, tan.rs, atan.rs, atan2.rs, asin.rs and
//! their tables are the IBM Accurate Mathematical Library as glibc
//! distributes it, under the GNU Lesser General Public License 2.1 or
//! later; asinf.rs (binary32) is FDLIBM with Stephen L. Moshier's
//! single-precision modifications, which glibc carries under the same
//! licence.  Each file opens with its own notice; see NOTICE.

mod exp;
mod log;
mod pow;
mod log10;
mod dla;
mod branred;
mod sincos;
mod tan;
mod atan;
mod atan2;
mod asin;
mod asinf;

pub use self::exp::exp;
pub use self::log::log;
pub use self::pow::pow;
pub use self::log10::log10;
pub use self::sincos::{sin, cos};
pub use self::tan::tan;
pub use self::atan::atan;
pub use self::atan2::atan2;
pub use self::asin::{asin, acos};
pub use self::asinf::asinf;

#[cfg(test)]
mod tests {
    /// `mul_add` must be the single-rounding fused operation wherever these
    /// run: a multiply then add (or a fused operation computed through
    /// binary80 or with double rounding) gives a different word on each.
    #[test]
    fn mul_add_is_correctly_rounded_fma() {
        let cases: [(u64, u64, u64, u64); 3] = [
            // (1 + 2^-52) * (1 - 2^-53) - 1 = 2^-53 - 2^-105: 0 unfused.
            (0x3ff0000000000001, 0x3fefffffffffffff, 0xbff0000000000000, 0x3c9ffffffffffffe),
            // (1 + 2^-52)^2 - (1 + 2^-51) = 2^-104: 0 unfused.
            (0x3ff0000000000001, 0x3ff0000000000001, 0xbff0000000000002, 0x3970000000000000),
            // (1 + 2^-52)^2 + 2^-53 lies 2^-104 above a binary64 tie: a
            // binary80 intermediate rounds it onto the tie, then to even.
            (0x3ff0000000000001, 0x3ff0000000000001, 0x3ca0000000000000, 0x3ff0000000000003),
        ];
        for (a, b, c, want) in cases {
            let (a, b, c) = (f64::from_bits(a), f64::from_bits(b), f64::from_bits(c));
            let got = std::hint::black_box(a).mul_add(std::hint::black_box(b), c);
            assert_eq!(got.to_bits(), want, "fma({a:e}, {b:e}, {c:e})");
        }
    }
}

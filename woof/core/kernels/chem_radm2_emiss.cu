// RADM2/SORGAM anthropogenic emission add (chem_opt = 1/2, emiss_opt = 3),
// WRF-Chem v4.7.1.
//
// SOURCE.  emissions_driver (chem/emissions_driver.F) calls, for RADM2 and
// RADM2SORG, add_anthropogenics (:1089) and then, for RADM2SORG only,
// sorgam_addemiss (:1647):
//
// * chem/module_emissions_anthropogenics.F:72-232, the non-MOZART, non-CRI
//   branch ("else is_mozart"), on levels kts .. min(kemit, kte):
//     conv_rho = 4.828e-4/rho_phy*dtstep/(dz8w*60.)
//     chem(p_x) = chem(p_x) + emis_ant(p_e_x)*conv_rho      ! mol km-2 hr-1
//   for every gas the package and the emission set both carry (the host
//   table, woof/core/chem_radm2_emiss.py GAS_TERMS, lists them in WRF's
//   order; each touches a different row, so the order is WRF's anyway).
//
// * chem/module_aerosols_sorgam.F:7769-8070, the emissopt /= 5 branch with
//   biom = 0 and no sea salt or dust, on levels kts .. kemit:
//     factor = alt*dtstep/dz8w
//     nu0  = nu0  + factor*factnumn*(anthfac*(e_pm25i+e_eci+e_naai)
//                     + orgfac*e_orgi + so4fac*e_so4i + no3fac*e_no3i)
//     ac0  = ac0  + factor*factnuma*(anthfac*(e_pm25j+e_ecj+e_naaj)
//                     + orgfac*e_orgj + so4fac*e_so4j + no3fac*e_no3j)
//     corn = corn + factor*factnumc*anthfac*e_pm_10
//     antha = antha + e_pm_10*factor;  p25j/p25i, ecj/eci, naaj/naai,
//     orgpaj/orgpai, so4aj/so4ai, no3aj/no3ai likewise from their e_* row.
//   (ug m-2 s-1 in, ug kg-1 and kg-1 out.)
//
// Evaluation order is gfortran -O0's: left to right inside each pair of
// parentheses, every product and sum rounded to float32 by an explicit
// round-to-nearest intrinsic so NVRTC cannot contract a*b + c into an FMA.
// An emission frame with fewer levels than kemit reads 0 above its top,
// which is WRF's emis_ant zero-filled up to kemit, computed rather than
// skipped.
//
// One thread per column.

#define EM(t) (k < nlev[t] ? reinterpret_cast<const float*>(emis[t])[idx] : 0.0f)
#define AER(r) reinterpret_cast<float*>(aer_field[r])[idx]

extern "C" __global__ void chem_radm2_emiss(
    const float* __restrict__ rho_phy,   // (nz, ny, nx) moist density, chem_prep
    const float* __restrict__ alt,       // (nz, ny, nx) inverse dry density
    const float* __restrict__ dz8w,      // (nz, ny, nx)
    const unsigned long long* __restrict__ gas_field,   // float* (nz, ny, nx), ngas
    const unsigned long long* __restrict__ emis,        // float* (nlev, ny, nx): ngas gas, then 13 aerosol
    const int* __restrict__ nlev,                       // levels of each frame
    const int ngas,
    const unsigned long long* __restrict__ aer_field,   // float* (nz, ny, nx), 16 rows, or unused
    const int with_aerosol,
    const float factnumn, const float factnuma, const float factnumc,
    const float anthfac, const float orgfac, const float so4fac, const float no3fac,
    const float dtstep,
    const int kemit,
    const int nz,
    const long long ncol)
{
    const long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ncol) return;
    const int kgas = kemit < nz ? kemit : nz;
    for (int k = 0; k < kgas; ++k) {
        const long long idx = (long long)k * ncol + c;
        // conv_rho = 4.828e-4/rho_phy*dtstep/(dz8w*60.)
        const float conv_rho = __fdiv_rn(__fmul_rn(__fdiv_rn(4.828e-4f, rho_phy[idx]), dtstep),
                                         __fmul_rn(dz8w[idx], 60.0f));
        for (int t = 0; t < ngas; ++t) {
            float* q = reinterpret_cast<float*>(gas_field[t]);
            q[idx] = __fadd_rn(q[idx], __fmul_rn(EM(t), conv_rho));
        }
    }
    if (!with_aerosol) return;
    // The aerosol frames follow the gas ones: pm25i, pm25j, pm_10, eci, ecj,
    // orgi, orgj, so4i, so4j, no3i, no3j, naaj, naai.
    const int a = ngas;
    for (int k = 0; k < kemit; ++k) {
        const long long idx = (long long)k * ncol + c;
        const float e_pm25i = EM(a + 0), e_pm25j = EM(a + 1), e_pm_10 = EM(a + 2);
        const float e_eci = EM(a + 3), e_ecj = EM(a + 4);
        const float e_orgi = EM(a + 5), e_orgj = EM(a + 6);
        const float e_so4i = EM(a + 7), e_so4j = EM(a + 8);
        const float e_no3i = EM(a + 9), e_no3j = EM(a + 10);
        const float e_naaj = EM(a + 11), e_naai = EM(a + 12);
        // factor = alt*dtstep/dz8w
        const float factor = __fdiv_rn(__fmul_rn(alt[idx], dtstep), dz8w[idx]);
        // nu0
        float s = __fmul_rn(anthfac, __fadd_rn(__fadd_rn(e_pm25i, e_eci), e_naai));
        s = __fadd_rn(s, __fmul_rn(orgfac, e_orgi));
        s = __fadd_rn(s, __fmul_rn(so4fac, e_so4i));
        s = __fadd_rn(s, __fmul_rn(no3fac, e_no3i));
        AER(0) = __fadd_rn(AER(0), __fmul_rn(__fmul_rn(factor, factnumn), s));
        // ac0
        s = __fmul_rn(anthfac, __fadd_rn(__fadd_rn(e_pm25j, e_ecj), e_naaj));
        s = __fadd_rn(s, __fmul_rn(orgfac, e_orgj));
        s = __fadd_rn(s, __fmul_rn(so4fac, e_so4j));
        s = __fadd_rn(s, __fmul_rn(no3fac, e_no3j));
        AER(1) = __fadd_rn(AER(1), __fmul_rn(__fmul_rn(factor, factnuma), s));
        // corn
        AER(2) = __fadd_rn(AER(2),
                              __fmul_rn(__fmul_rn(__fmul_rn(factor, factnumc), anthfac), e_pm_10));
        // antha, p25j, p25i, ecj, eci, naaj, naai, orgpaj, orgpai, so4aj,
        // so4ai, no3aj, no3ai: chem + e*factor
        const float e[13] = {e_pm_10, e_pm25j, e_pm25i, e_ecj, e_eci, e_naaj, e_naai,
                             e_orgj, e_orgi, e_so4j, e_so4i, e_no3j, e_no3i};
        for (int r = 0; r < 13; ++r)
            AER(3 + r) = __fadd_rn(AER(3 + r), __fmul_rn(e[r], factor));
    }
}

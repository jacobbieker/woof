// RRTMG-K longwave (WRF 4.7.1 ra_lw_physics = 14,
// phys/module_ra_rrtmg_lwk.F): the one gas-optics statement that differs
// from the legacy RRTMG LW.  Compiled after kernels/rrtmg_lw.cu (RLW_*
// arithmetic, FS/IS/A2/TAUG, TAUGB_PROLOGUE, SL_/SI_ slots) and
// kernels/rrtmg_lw_taugb06_09.cu (rlw_specfac_0609, rlw_ninemajor_0609)
// by woof/core/rrtmgk.py; the file name stays outside the
// rrtmg_lw_*.cu glob that assembles the legacy LW unit.
//
// taugb7, lower atmosphere (module_ra_rrtmg_lwk.F:6220-6494): the
// fac*01/fac*11 block that feeds tau_major1 opens with
//     if (specparm .lt. 0.125_rb) then        ! legacy: specparm1
// and its ``else if`` / ``else`` arms still test specparm1.  Every other
// statement is the legacy band (rlw_taugb7).  rlwk_taugb7_lower runs
// after the legacy rlw_taumol_batched and rewrites band 7's twelve
// g-points (77..88) for the layers at or below laytrop; the upper
// atmosphere and the Planck fractions are the legacy values unchanged.
//
// Fortran leaves fac201/fac211 unassigned in the ``else`` arm, so a layer
// with specparm >= 0.125 and specparm1 < 0.125 would read the previous
// layer's values in tau_major1.  That needs rat_h2oo3_1 > rat_h2oo3,
// i.e. chi_mls(1,jp+1)/chi_mls(3,jp+1) > chi_mls(1,jp)/chi_mls(3,jp) for
// a lower-atmosphere jp; woof/core/rrtmgk.py checks on the host that the
// ratio never increases over those jp (fails closed otherwise), so the
// arm's f2 terms are never read here.

// rrtmg_lw_taugb06_09.cu #undefs its table accessor at its end; the same
// (9,19,ng) Fortran-order element (j,i,g), under K's own name.
#define KA3_K7(tab, j, i, g) \
    tab[((j) - 1) + 9 * ((i) - 1) + 171 * ((g) - 1)]

extern "C" __global__ void rlwk_taugb7_lower(
    int ncol, int nl, const int* __restrict__ laytrop_v,
    const float* __restrict__ fs, const int* __restrict__ isv,
    const float* __restrict__ wx,
    const float* __restrict__ chi_mls, float oneminus,
    const float* const* __restrict__ tabs,
    float* __restrict__ taug, float* __restrict__ fracs)
{
    TAUGB_PROLOGUE
    (void)fracs;
    if (lay > laytrop) return;
    // GPU_BAND_TABS[7] = absa(585,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mco2(9,19,ng), kb_mco2(19,ng),
    //                    fracrefa(ng,9), fracrefb(ng)
    const float* absa = tabs[0];
    const float* selfref = tabs[2];
    const float* forref = tabs[3];
    const float* ka_mco2 = tabs[4];
    const int nspa7 = 9, ng7 = 12, gs = 76;

    float refrat_m_a = RLW_DV(chi_mls[0 + 7 * 2], chi_mls[2 + 7 * 2]);

    int jpv = IS(SI_JP);
    float colh2o = FS(SL_COLH2O);
    float colo3 = FS(SL_COLO3);

    float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OO3), colo3));
    float specparm = RLW_DV(colh2o, speccomb);
    if (specparm >= oneminus) specparm = oneminus;
    float specmult = RLW_MU(8.0f, specparm);
    int js = 1 + (int)specmult;
    float fsv = fmodf(specmult, 1.0f);

    float speccomb1 = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OO3_1), colo3));
    float specparm1 = RLW_DV(colh2o, speccomb1);
    if (specparm1 >= oneminus) specparm1 = oneminus;
    float specmult1 = RLW_MU(8.0f, specparm1);
    int js1 = 1 + (int)specmult1;
    float fs1v = fmodf(specmult1, 1.0f);

    float speccomb_mco2 = RLW_AD(colh2o, RLW_MU(refrat_m_a, colo3));
    float specparm_mco2 = RLW_DV(colh2o, speccomb_mco2);
    if (specparm_mco2 >= oneminus) specparm_mco2 = oneminus;
    float specmult_mco2 = RLW_MU(8.0f, specparm_mco2);
    int jmco2 = 1 + (int)specmult_mco2;
    float fmco2 = fmodf(specmult_mco2, 1.0f);

    float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
    float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2), chi_mls[1 + 7 * jpv]);
    float adjcolco2;
    if (ratco2 > 3.0f) {
        float adjfac = RLW_AD(3.0f, rlw_pow(RLW_SU(ratco2, 3.0f), 0.79f));
        adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                  FS(SL_COLDRY)), 1.e-20f);
    } else {
        adjcolco2 = FS(SL_COLCO2);
    }

    int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa7 + js;
    int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa7 + js1;
    int inds = IS(SI_INDSELF);
    int indf = IS(SI_INDFOR);
    int indm = IS(SI_INDMINOR);

    RlwSpecFac0609 f0 = rlw_specfac_0609(specparm, fsv,
                                         FS(SL_FAC00), FS(SL_FAC10));
    // K: the first arm tests specparm, the other two specparm1.  The
    // selector argument only picks rlw_specfac_0609's arm (0 < 0.125,
    // 1 > 0.875, 0.5 neither); every arm computes from fs1v.
    float arm1 = (specparm < 0.125f) ? 0.0f
               : ((specparm1 > 0.875f) ? 1.0f : 0.5f);
    RlwSpecFac0609 f1 = rlw_specfac_0609(arm1, fs1v,
                                         FS(SL_FAC01), FS(SL_FAC11));

    for (int ig = 1; ig <= ng7; ++ig) {
        float tauself = RLW_MU(FS(SL_SELFFAC),
            RLW_AD(A2(selfref, 10, inds, ig),
                   RLW_MU(FS(SL_SELFFRAC),
                          RLW_SU(A2(selfref, 10, inds + 1, ig),
                                 A2(selfref, 10, inds, ig)))));
        float taufor = RLW_MU(FS(SL_FORFAC),
            RLW_AD(A2(forref, 4, indf, ig),
                   RLW_MU(FS(SL_FORFRAC),
                          RLW_SU(A2(forref, 4, indf + 1, ig),
                                 A2(forref, 4, indf, ig)))));
        float co2m1 = RLW_AD(KA3_K7(ka_mco2, jmco2, indm, ig),
            RLW_MU(fmco2,
                   RLW_SU(KA3_K7(ka_mco2, jmco2 + 1, indm, ig),
                          KA3_K7(ka_mco2, jmco2, indm, ig))));
        float co2m2 = RLW_AD(KA3_K7(ka_mco2, jmco2, indm + 1, ig),
            RLW_MU(fmco2,
                   RLW_SU(KA3_K7(ka_mco2, jmco2 + 1, indm + 1, ig),
                          KA3_K7(ka_mco2, jmco2, indm + 1, ig))));
        float absco2 = RLW_AD(co2m1, RLW_MU(FS(SL_MINORFRAC),
                                            RLW_SU(co2m2, co2m1)));

        float tau_major = rlw_ninemajor_0609(specparm, speccomb, f0,
                                             absa, ind0, ig);
        float tau_major1 = rlw_ninemajor_0609(specparm1, speccomb1, f1,
                                              absa, ind1, ig);

        float t = RLW_AD(tau_major, tau_major1);
        t = RLW_AD(t, tauself);
        t = RLW_AD(t, taufor);
        t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
        TAUG(gs + ig) = t;
    }
}

#undef KA3_K7

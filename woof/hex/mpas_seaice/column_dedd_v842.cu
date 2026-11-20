// MPAS-Seaice v8.4.2: the column package's delta-Eddington shortwave with a
// computed surface temperature (config_shortwave_type = 'dEdd',
// config_calc_surface_temperature = .true.): colpkg_init_orbit's orbit,
// compute_coszen and run_dEdd (shortwave_dEdd, compute_dEdd, solution_dEdd,
// shortwave_dEdd_set_snow and shortwave_dEdd_set_pond), called from
// column_radiation.
//
// A modified derivative of MPAS-Model v8.4.2 (MPAS-Dev/MPAS-Model,
// f34984b2c89353cf31c235b08d3b4acd96b37bb3), transcribed from
//   src/core_seaice/shared/mpas_seaice_column.F:
//     505-533    seaice_init_column_shortwave (colpkg_init_orbit)
//     2118-2484  column_radiation
//   src/core_seaice/column/ice_colpkg.F90:
//     387-412    colpkg_init_orbit
//     2649-2972  colpkg_step_radiation
//   src/core_seaice/column/ice_orbital.F90:
//     35-96      compute_coszen
//     104-581    shr_orb_params
//     587-680    shr_orb_decl
//   src/core_seaice/column/ice_shortwave.F90:
//     673-1057   run_dEdd
//     1087-1489  shortwave_dEdd
//     1499-2961  compute_dEdd
//     2970-3440  solution_dEdd
//     3450-3523  shortwave_dEdd_set_snow
//     3533-3563  shortwave_dEdd_set_pond
// (Bruce P. Briegleb, Elizabeth C. Hunke, Adrian K. Turner, Erik Kluzek and
// the CICE Consortium). Copyright (c) 2013-2019, Los Alamos National
// Security, LLC (LANS) and the University Corporation for Atmospheric
// Research (UCAR); BSD-3-Clause, the terms in
// licenses/LICENSE-MPAS-BSD-3-Clause.txt. This is not the version available
// from LANS and UCAR.
//
// Appended to column_radiation_v842.cu. exp is glibc_exp (the MIT
// transcription of glibc's binary64 exp). The orbit and the solar zenith
// angle call glibc's binary64 sin, cos, atan and asin, which are not
// correctly rounded and are LGPL: they compile only with the opt-in
// transcription prepended (WOOF_GLIBC239_TRIG64, g239::g239_sin/cos/atan/
// asin); column_radiation_v842.py refuses the delta-Eddington shortwave
// otherwise. Fortran's MIN and MAX are gfortran's (sw_min, sw_max).
//
// v8.4.2's behaviour kept: shortwave_dEdd_set_snow's intent(out) snow
// fraction is assigned only where hs >= hs_min, so bare ice keeps the 0
// run_dEdd stored before the call (gfortran passes it by reference);
// shortwave_dEdd raises the cell's solar zenith cosine to puny and writes it
// back; compute_dEdd's zero-heat-capacity fold of the first layers'
// absorption into the surface runs once per surface type, on the sums so
// far; the bare-ice, snow and pond albedos keep their words in categories
// without ice; the chlorophyll absorption is kalg*(0.5/dzk(klev)) in the
// bottom layer's visible band only.
//
// One thread is one owned cell; nslyr is 1 (column_v842 refuses others), so
// klev = nilyr + 2 and the level arrays hold DEDD_NL >= klevp + 1 words.

#define MAX_DEDD_ICE_LAYERS 64 // column_v842.MAX_ICE_LAYERS
#define DEDD_NL (MAX_DEDD_ICE_LAYERS + 4)
#define P15 0.15
#define P25 0.25
#define DEDD_NSPINT 3
#define DEDD_NMBRAD 32
#define DEDD_HS_MIN 1.0e-4      // hs_min
#define DEDD_HPMIN 0.005        // hpmin
#define DEDD_HP0 0.200          // hp0
#define DEDD_HI_SSL 0.050       // hi_ssl
#define DEDD_HS_SSL 0.040       // hs_ssl
#define DEDD_RHOFRESH 1000.0    // rhofresh
#define DEDD_SECDAY 86400.0     // secday
#define DEDD_PII 3.141592653589793  // seaice_constants' pii

// the scalar parameters (column_radiation_v842._dedd_parameters)
#define DP_R_ICE 0
#define DP_R_PND 1
#define DP_R_SNW 2
#define DP_DT_MLT 3
#define DP_RSNW_MLT 4
#define DP_KALG 5
#define DP_HS0 6
#define DP_HS1 7
#define DP_HP1 8
#define DP_PNDASPECT 9
#define DP_DT 10
#define DP_YDAY 11
#define DP_COUNT 12

// pond flags
#define DEDD_POND_CESM 1
#define DEDD_POND_LVL 2
#define DEDD_POND_TOPO 4
#define DEDD_INITONLY 8
#define DEDD_HEAT_CAPACITY 16

__device__ const double DEDD_RSNW_TAB[DEDD_NMBRAD] = {
    5., 7., 10., 15., 20., 30., 40., 50., 65., 80., 100., 120., 140., 170., 200., 240., 290., 350., 420., 500.,
    570., 660., 760., 870., 1000., 1100., 1250., 1400., 1600., 1800., 2000., 2500.};
// Qs_tab, ws_tab, gs_tab (nspint, nmbrad): [nr][ns]
__device__ const double DEDD_QS_TAB[DEDD_NMBRAD][DEDD_NSPINT] = {
    {2.131798, 2.187756, 2.267358}, {2.104499, 2.148345, 2.236078}, {2.081580, 2.116885, 2.175067},
    {2.062595, 2.088937, 2.130242}, {2.051403, 2.072422, 2.106610}, {2.039223, 2.055389, 2.080586},
    {2.032383, 2.045751, 2.066394}, {2.027920, 2.039388, 2.057224}, {2.023444, 2.033137, 2.048055},
    {2.020412, 2.028840, 2.041874}, {2.017608, 2.024863, 2.036046}, {2.015592, 2.022021, 2.031954},
    {2.014083, 2.019887, 2.028853}, {2.012368, 2.017471, 2.025353}, {2.011092, 2.015675, 2.022759},
    {2.009837, 2.013897, 2.020168}, {2.008668, 2.012252, 2.017781}, {2.007627, 2.010813, 2.015678},
    {2.006764, 2.009577, 2.013880}, {2.006037, 2.008520, 2.012382}, {2.005528, 2.007807, 2.011307},
    {2.005025, 2.007079, 2.010280}, {2.004562, 2.006440, 2.009333}, {2.004155, 2.005898, 2.008523},
    {2.003794, 2.005379, 2.007795}, {2.003555, 2.005041, 2.007329}, {2.003264, 2.004624, 2.006729},
    {2.003037, 2.004291, 2.006230}, {2.002776, 2.003929, 2.005700}, {2.002590, 2.003627, 2.005276},
    {2.002395, 2.003391, 2.004904}, {2.002071, 2.002922, 2.004241}};
__device__ const double DEDD_WS_TAB[DEDD_NMBRAD][DEDD_NSPINT] = {
    {0.9999994, 0.9999673, 0.9954589}, {0.9999992, 0.9999547, 0.9938576}, {0.9999990, 0.9999382, 0.9917989},
    {0.9999985, 0.9999123, 0.9889724}, {0.9999979, 0.9998844, 0.9866190}, {0.9999970, 0.9998317, 0.9823021},
    {0.9999960, 0.9997800, 0.9785269}, {0.9999951, 0.9997288, 0.9751601}, {0.9999936, 0.9996531, 0.9706974},
    {0.9999922, 0.9995783, 0.9667577}, {0.9999903, 0.9994798, 0.9621007}, {0.9999885, 0.9993825, 0.9579541},
    {0.9999866, 0.9992862, 0.9541924}, {0.9999838, 0.9991434, 0.9490959}, {0.9999810, 0.9990025, 0.9444940},
    {0.9999772, 0.9988171, 0.9389141}, {0.9999726, 0.9985890, 0.9325819}, {0.9999670, 0.9983199, 0.9256405},
    {0.9999605, 0.9980117, 0.9181533}, {0.9999530, 0.9976663, 0.9101540}, {0.9999465, 0.9973693, 0.9035031},
    {0.9999382, 0.9969939, 0.8953134}, {0.9999289, 0.9965848, 0.8865789}, {0.9999188, 0.9961434, 0.8773350},
    {0.9999068, 0.9956323, 0.8668233}, {0.9998975, 0.9952464, 0.8589990}, {0.9998837, 0.9946782, 0.8476493},
    {0.9998699, 0.9941218, 0.8367318}, {0.9998515, 0.9933966, 0.8227881}, {0.9998332, 0.9926888, 0.8095131},
    {0.9998148, 0.9919968, 0.7968620}, {0.9997691, 0.9903277, 0.7677887}};
__device__ const double DEDD_GS_TAB[DEDD_NMBRAD][DEDD_NSPINT] = {
    {0.859913, 0.848003, 0.824415}, {0.867130, 0.858150, 0.848445}, {0.873381, 0.867221, 0.861714},
    {0.878368, 0.874879, 0.874036}, {0.881462, 0.879661, 0.881299}, {0.884361, 0.883903, 0.890184},
    {0.885937, 0.886256, 0.895393}, {0.886931, 0.887769, 0.899072}, {0.887894, 0.889255, 0.903285},
    {0.888515, 0.890236, 0.906588}, {0.889073, 0.891127, 0.910152}, {0.889452, 0.891750, 0.913100},
    {0.889730, 0.892213, 0.915621}, {0.890026, 0.892723, 0.918831}, {0.890238, 0.893099, 0.921540},
    {0.890441, 0.893474, 0.924581}, {0.890618, 0.893816, 0.927701}, {0.890762, 0.894123, 0.930737},
    {0.890881, 0.894397, 0.933568}, {0.890975, 0.894645, 0.936148}, {0.891035, 0.894822, 0.937989},
    {0.891097, 0.895020, 0.939949}, {0.891147, 0.895212, 0.941727}, {0.891189, 0.895399, 0.943339},
    {0.891225, 0.895601, 0.944915}, {0.891248, 0.895745, 0.945950}, {0.891277, 0.895951, 0.947288},
    {0.891299, 0.896142, 0.948438}, {0.891323, 0.896388, 0.949762}, {0.891340, 0.896623, 0.950916},
    {0.891356, 0.896851, 0.951945}, {0.891386, 0.897399, 0.954156}};
// the inherent optical properties' means, per band
__device__ const double DEDD_KI_SSL_MN[3] = {1000.1, 1003.7, 7042.};
__device__ const double DEDD_WI_SSL_MN[3] = {.9999, .9963, .9088};
__device__ const double DEDD_KI_DL_MN[3] = {100.2, 107.7, 1309.};
__device__ const double DEDD_WI_DL_MN[3] = {.9980, .9287, .0305};
__device__ const double DEDD_KI_INT_MN[3] = {20.2, 27.7, 1445.};
__device__ const double DEDD_WI_INT_MN[3] = {.9901, .7223, .0277};
__device__ const double DEDD_KI_P_SSL_MN[3] = {70.2, 77.7, 1309.};
__device__ const double DEDD_WI_P_SSL_MN[3] = {.9972, .9009, .0305};
__device__ const double DEDD_KI_P_INT_MN[3] = {20.2, 27.7, 1445.};
__device__ const double DEDD_WI_P_INT_MN[3] = {.9901, .7223, .0277};
__device__ const double DEDD_KW[3] = {0.20, 12.0, 729.};
#define DEDD_GI 0.94            // every gi_*_mn
#define DEDD_FP_ICE 0.15
#define DEDD_FM_ICE 0.15
#define DEDD_FP_PND 2.00
#define DEDD_FM_PND 0.50
#define DEDD_FR_MAX 1.00
#define DEDD_FR_MIN 0.80
#define DEDD_CP67 0.67
#define DEDD_CP78 0.78
#define DEDD_CP01 0.01
__device__ const double DEDD_GAUSPT[8] = {.9894009, .9445750, .8656312, .7554044, .6178762, .4580168, .2816036,
                                          .0950125};
__device__ const double DEDD_GAUSWT[8] = {.0271525, .0622535, .0951585, .1246290, .1495960, .1691565, .1826034,
                                          .1894506};
#define DEDD_REFINDX 1.310
#define DEDD_CP063 0.063
#define DEDD_CP455 0.455
#define DEDD_TRMIN 0.001

#ifdef WOOF_GLIBC239_TRIG64
// ice_orbital.F90 shr_orb_params' series (obliquity, eccentricity/fvelp, mvelp)
__device__ const double ORB_OBAMP[47] = {
    -2462.2214466, -857.3232075, -629.3231835, -414.2804924, -311.7632587, 308.9408604, -162.5533601,
    -116.1077911, 101.1189923, -67.6856209, 24.9079067, 22.5811241, -21.1648355, -15.6549876, 15.3936813,
    14.6660938, -11.7273029, 10.2742696, 6.4914588, 5.8539148, -5.4872205, -5.4290191, 5.1609570, 5.0786314,
    -4.0735782, 3.7227167, 3.3971932, -2.8347004, -2.6550721, -2.5717867, -2.4712188, 2.4625410, 2.2464112,
    -2.0755511, -1.9713669, -1.8813061, -1.8468785, 1.8186742, 1.7601888, -1.5428851, 1.4738838, -1.4593669,
    1.4192259, -1.1818980, 1.1756474, -1.1316126, 1.0896928};
__device__ const double ORB_OBRATE[47] = {
    31.609974, 32.620504, 24.172203, 31.983787, 44.828336, 30.973257, 43.668246, 32.246691, 30.599444,
    42.681324, 43.836462, 47.439436, 63.219948, 64.230478, 1.010530, 7.437771, 55.782177, 0.373813, 13.218362,
    62.583231, 63.593761, 76.438310, 45.815258, 8.448301, 56.792707, 49.747842, 12.058272, 75.278220, 65.241008,
    64.604291, 1.647247, 7.811584, 12.207832, 63.856665, 56.155990, 77.448840, 6.801054, 62.209418, 20.656133,
    48.344406, 55.145460, 69.000539, 11.071350, 74.291298, 11.047742, 0.636717, 12.844549};
__device__ const double ORB_OBPHAS[47] = {
    251.9025, 280.8325, 128.3057, 292.7252, 15.3747, 263.7951, 308.4258, 240.0099, 222.9725, 268.7809,
    316.7998, 319.6024, 143.8050, 172.7351, 28.9300, 123.5968, 20.2082, 40.8226, 123.4722, 155.6977, 184.6277,
    267.2772, 55.0196, 152.5268, 49.1382, 204.6609, 56.5233, 200.3284, 201.6651, 213.5577, 17.0374, 164.4194,
    94.5422, 131.9124, 61.0309, 296.2073, 135.4894, 114.8750, 247.0691, 256.6114, 32.1008, 143.6804, 16.8784,
    160.6835, 27.5932, 348.1074, 82.6496};
__device__ const double ORB_ECAMP[19] = {
    0.01860798, 0.01627522, -0.01300660, 0.00988829, -0.00336700, 0.00333077, -0.00235400, 0.00140015,
    0.00100700, 0.00085700, 0.00064990, 0.00059900, 0.00037800, -0.00033700, 0.00027600, 0.00018200,
    -0.00017400, -0.00012400, 0.00001250};
__device__ const double ORB_ECRATE[19] = {
    4.2072050, 7.3460910, 17.8572630, 17.2205460, 16.8467330, 5.1990790, 18.2310760, 26.2167580, 6.3591690,
    16.2100160, 3.0651810, 16.5838290, 18.4939800, 6.1909530, 18.8677930, 17.4255670, 6.1860010, 18.4174410,
    0.6678630};
__device__ const double ORB_ECPHAS[19] = {
    28.620089, 193.788772, 308.307024, 320.199637, 279.376984, 87.195000, 349.129677, 128.443387, 154.143880,
    291.269597, 114.860583, 332.092251, 296.414411, 145.769910, 337.237063, 152.092288, 126.839891, 210.667199,
    72.108838};
__device__ const double ORB_MVAMP[78] = {
    7391.0225890, 2555.1526947, 2022.7629188, -1973.6517951, 1240.2321818, 953.8679112, -931.7537108,
    872.3795383, 606.3544732, -496.0274038, 456.9608039, 346.9462320, -305.8412902, 249.6173246, -199.1027200,
    191.0560889, -175.2936572, 165.9068833, 161.1285917, 139.7878093, -133.5228399, 117.0673811, 104.6907281,
    95.3227476, 86.7824524, 86.0857729, 70.5893698, -69.9719343, -62.5817473, 61.5450059, -57.9364011,
    57.1899832, -57.0236109, -54.2119253, 53.2834147, 52.1223575, -49.0059908, -48.3118757, -45.4191685,
    -42.2357920, -34.7971099, 34.4623613, -33.8356643, 33.6689362, -31.2521586, -30.8798701, 28.4640769,
    -27.1960802, 27.0860736, -26.3437456, 24.7253740, 24.6732126, 24.4272733, 24.0127327, 21.7150294,
    -21.5375347, 18.1148363, -16.9603104, -16.1765215, 15.5567653, 15.4846529, 15.2150632, 14.5047426,
    -14.3873316, 13.1351419, 12.8776311, 11.9867234, 11.9385578, 11.7030822, 11.6018181, -11.2617293,
    -10.4664199, 10.4333970, -10.2377466, 10.1934446, -10.1280191, 10.0289441, -10.0034259};
__device__ const double ORB_MVRATE[78] = {
    31.609974, 32.620504, 24.172203, 0.636717, 31.983787, 3.138886, 30.973257, 44.828336, 0.991874, 0.373813,
    43.668246, 32.246691, 30.599444, 2.147012, 10.511172, 42.681324, 13.650058, 0.986922, 9.874455, 13.013341,
    0.262904, 0.004952, 1.142024, 63.219948, 0.205021, 2.151964, 64.230478, 43.836462, 47.439436, 1.384343,
    7.437771, 18.829299, 9.500642, 0.431696, 1.160090, 55.782177, 12.639528, 1.155138, 0.168216, 1.647247,
    10.884985, 5.610937, 12.658184, 1.010530, 1.983748, 14.023871, 0.560178, 1.273434, 12.021467, 62.583231,
    63.593761, 76.438310, 4.280910, 13.218362, 17.818769, 8.359495, 56.792707, 8.448301, 1.978796, 8.863925,
    0.186365, 8.996212, 6.771027, 45.815258, 12.002811, 75.278220, 65.241008, 18.870667, 22.009553, 64.604291,
    11.498094, 0.578834, 9.237738, 49.747842, 2.147012, 1.196895, 2.133898, 0.173168};
__device__ const double ORB_MVPHAS[78] = {
    251.9025, 280.8325, 128.3057, 348.1074, 292.7252, 165.1686, 263.7951, 15.3747, 58.5749, 40.8226, 308.4258,
    240.0099, 222.9725, 106.5937, 114.5182, 268.7809, 279.6869, 39.6448, 126.4108, 291.5795, 307.2848, 18.9300,
    273.7596, 143.8050, 191.8927, 125.5237, 172.7351, 316.7998, 319.6024, 69.7526, 123.5968, 217.6432, 85.5882,
    156.2147, 66.9489, 20.2082, 250.7568, 48.0188, 8.3739, 17.0374, 155.3409, 94.1709, 221.1120, 28.9300,
    117.1498, 320.5095, 262.3602, 336.2148, 233.0046, 155.6977, 184.6277, 267.2772, 78.9281, 123.4722, 188.7132,
    180.1364, 49.1382, 152.5268, 98.2198, 97.4808, 221.5376, 168.2438, 161.1199, 55.0196, 262.6495, 200.3284,
    201.6651, 294.6547, 99.8233, 213.5577, 154.1631, 232.7153, 138.3034, 204.6609, 106.5938, 250.4676, 332.3345,
    27.3039};
#endif

// ice_colpkg.F90:387-412 colpkg_init_orbit -> ice_orbital.F90 shr_orb_params
// with iyear_AD = 1950 and log_print false (no warnings; l_stop stays false,
// |yb4_1950AD| = 0). One thread writes orbit = {eccen, obliqr, lambm0,
// mvelpp, obliq, mvelp}.
extern "C" __global__ void seaice_column_init_orbit_v842(double *__restrict__ orbit)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
#ifdef WOOF_GLIBC239_TRIG64
    const double degrad = D(PI, 180.0);
    const double psecdeg = D(1.0, 3600.0);
    const double yb4_1950AD = S(1950.0, 1950.0);
    const double years = -yb4_1950AD;
    double obsum = 0.0;
    for (int i = 0; i < 47; ++i)
        obsum = A(obsum, M(M(ORB_OBAMP[i], psecdeg),
                           g239::g239_cos(M(A(M(M(ORB_OBRATE[i], psecdeg), years), ORB_OBPHAS[i]), degrad))));
    const double obliq = A(23.320556, obsum);
    double cossum = 0.0;
    for (int i = 0; i < 19; ++i)
        cossum = A(cossum, M(ORB_ECAMP[i], g239::g239_cos(M(A(M(M(ORB_ECRATE[i], psecdeg), years), ORB_ECPHAS[i]),
                                                             degrad))));
    double sinsum = 0.0;
    for (int i = 0; i < 19; ++i)
        sinsum = A(sinsum, M(ORB_ECAMP[i], g239::g239_sin(M(A(M(M(ORB_ECRATE[i], psecdeg), years), ORB_ECPHAS[i]),
                                                             degrad))));
    const double eccen2 = A(M(cossum, cossum), M(sinsum, sinsum));
    const double eccen = SQRT(eccen2);
    const double eccen3 = M(eccen2, eccen);
    double fvelp = 0.0;
    if (fabs(cossum) <= 1.0E-8) {
        if (sinsum == 0.0) fvelp = 0.0;
        else if (sinsum < 0.0) fvelp = M(1.5, PI);
        else if (sinsum > 0.0) fvelp = M(.5, PI);
    } else if (cossum < 0.0) {
        fvelp = A(g239::g239_atan(D(sinsum, cossum)), PI);
    } else if (cossum > 0.0) {
        if (sinsum < 0.0) fvelp = A(g239::g239_atan(D(sinsum, cossum)), M(2.0, PI));
        else fvelp = g239::g239_atan(D(sinsum, cossum));
    }
    double mvsum = 0.0;
    for (int i = 0; i < 78; ++i)
        mvsum = A(mvsum, M(M(ORB_MVAMP[i], psecdeg),
                           g239::g239_sin(M(A(M(M(ORB_MVRATE[i], psecdeg), years), ORB_MVPHAS[i]), degrad))));
    double mvelp = A(A(A(D(fvelp, degrad), M(M(50.439273, psecdeg), years)), 3.392506), mvsum);
    while (mvelp < 0.0) mvelp = A(mvelp, 360.0);
    while (mvelp >= 360.0) mvelp = S(mvelp, 360.0);
    const double obliqr = M(obliq, degrad);
    const double mvelpp = M(A(mvelp, 180.0), degrad);
    const double beta = SQRT(S(1.0, eccen2));
    const double t1 = M(M(A(M(.5, eccen), M(.125, eccen3)), A(1.0, beta)), g239::g239_sin(mvelpp));
    const double t2 = M(M(M(.250, eccen2), A(.5, beta)), g239::g239_sin(M(2.0, mvelpp)));
    const double t3 = M(M(M(.125, eccen3), A(D(1.0, 3.0), beta)), g239::g239_sin(M(3.0, mvelpp)));
    const double lambm0 = M(2.0, A(S(t1, t2), t3));
    orbit[0] = eccen;
    orbit[1] = obliqr;
    orbit[2] = lambm0;
    orbit[3] = mvelpp;
    orbit[4] = obliq;
    orbit[5] = mvelp;
#else
    __trap();   // the host refuses the delta-Eddington shortwave without the opt-in trig
#endif
}

// ice_orbital.F90:35-96 compute_coszen (not CCSMCOUPLED) with shr_orb_decl
__device__ double dedd_coszen(double tlat, double tlon, double yday, int sec, const double *orbit)
{
#ifdef WOOF_GLIBC239_TRIG64
    const double eccen = orbit[0], obliqr = orbit[1], lambm0 = orbit[2], mvelpp = orbit[3];
    const double ydayp1 = A(yday, D((double)sec, DEDD_SECDAY));
    // shr_orb_decl(ydayp1, eccen, mvelpp, lambm0, obliqr, decln, eccf)
    const double lambm = A(lambm0, D(M(M(S(ydayp1, 80.5), 2.0), PI), 365.0));
    const double lmm = S(lambm, mvelpp);
    const double sinl = g239::g239_sin(lmm);
    const double lamb = A(lambm, M(eccen, A(M(2.0, sinl),
        M(eccen, A(M(1.25, g239::g239_sin(M(2.0, lmm))),
                   M(eccen, S(M(D(13.0, 12.0), g239::g239_sin(M(3.0, lmm))), M(0.25, sinl))))))));
    const double decln = g239::g239_asin(M(g239::g239_sin(obliqr), g239::g239_sin(lamb)));
    return A(M(g239::g239_sin(tlat), g239::g239_sin(decln)),
             M(M(g239::g239_cos(tlat), g239::g239_cos(decln)),
               g239::g239_cos(A(M(M(S(D((double)sec, DEDD_SECDAY), 0.5), 2.0), PI), tlon))));
#else
    __trap();
    return 0.0;
#endif
}

// ice_shortwave.F90:3450-3523 shortwave_dEdd_set_snow. fs is intent(out) but
// set only where hs >= hs_min: elsewhere it keeps the caller's word.
__device__ void dedd_set_snow(double R_snw, double dT_mlt, double rsnw_mlt, double aice, double vsno, double Tsfc,
                              double *fs, double hs0, double *hs, double *rhosnw, double *rsnw)
{
    *hs = D(vsno, aice);
    if (*hs >= DEDD_HS_MIN) {
        *fs = 1.0;
        if (hs0 > PUNY) *fs = sw_min(D(*hs, hs0), 1.0);
    }
    const double dTs = S(SW_TIMELT, Tsfc);
    const double fT = -sw_min(S(D(dTs, dT_mlt), 1.0), 0.0);
    double rsnw_nm = S(500.0, M(R_snw, 250.0));
    rsnw_nm = sw_max(rsnw_nm, 100.0);
    rsnw_nm = sw_min(rsnw_nm, rsnw_mlt);
    // nslyr = 1
    rhosnw[0] = RHOS;
    rsnw[0] = A(rsnw_nm, M(S(rsnw_mlt, rsnw_nm), fT));
    rsnw[0] = sw_max(rsnw[0], 100.0);
    rsnw[0] = sw_min(rsnw[0], rsnw_mlt);
}

// ice_shortwave.F90:2970-3440 solution_dEdd
__device__ void dedd_solution(double coszen, int srftyp, int klev, int klevp, int nslyr, double exp_min,
                              const double *tau, const double *w0, const double *g, double albodr, double albodf,
                              double *trndir, double *trntdr, double *trndif, double *rupdir, double *rupdif,
                              double *rdndif)
{
    double rdir[DEDD_NL], rdif_a[DEDD_NL], rdif_b[DEDD_NL], tdir[DEDD_NL], tdif_a[DEDD_NL], tdif_b[DEDD_NL],
        trnlay[DEDD_NL];
    for (int k = 0; k <= klevp; ++k) {
        trndir[k] = 0.0;
        trntdr[k] = 0.0;
        trndif[k] = 0.0;
        rupdir[k] = 0.0;
        rupdif[k] = 0.0;
        rdndif[k] = 0.0;
    }
    trndir[0] = 1.0;
    trntdr[0] = 1.0;
    trndif[0] = 1.0;
    rdndif[0] = 0.0;
    const double mu0 = sw_max(coszen, P01);
    const double mu0nij = SQRT(S(1.0, D(S(1.0, M(mu0, mu0)), M(DEDD_REFINDX, DEDD_REFINDX))));
    int kfrsnl = 0;
    if (srftyp < 2) kfrsnl = nslyr + 2;
    for (int k = 0; k <= klev; ++k) {
        rdir[k] = 0.0;
        rdif_a[k] = 0.0;
        rdif_b[k] = 0.0;
        tdir[k] = 0.0;
        tdif_a[k] = 0.0;
        tdif_b[k] = 0.0;
        trnlay[k] = 0.0;
        if (trntdr[k] > DEDD_TRMIN) {
            const double tautot = tau[k], wtot = w0[k], gtot = g[k];
            const double ftot = M(gtot, gtot);
            const double ts = M(S(1.0, M(wtot, ftot)), tautot);                       // taus
            const double ws = D(M(S(1.0, ftot), wtot), S(1.0, M(wtot, ftot)));      // omgs
            const double gs = D(S(gtot, ftot), S(1.0, ftot));                       // asys
            const double lm = SQRT(M(M(3.0, S(1.0, ws)), S(1.0, M(ws, gs))));       // el
            const double ue = D(M(1.5, S(1.0, M(ws, gs))), lm);                     // u
            double mu0n = mu0nij;
            if (srftyp < 2 && k < kfrsnl) mu0n = mu0;
            const double extins = sw_max(exp_min, glibc_exp(-M(lm, ts)));
            const double ne = S(D(M(A(ue, 1.0), A(ue, 1.0)), extins), M(M(S(ue, 1.0), S(ue, 1.0)), extins));
            rdif_a[k] = D(M(S(M(ue, ue), 1.0), S(D(1.0, extins), extins)), ne);
            tdif_a[k] = D(M(4.0, ue), ne);
            trnlay[k] = sw_max(exp_min, glibc_exp(D(-ts, mu0n)));
            const double den0 = S(1.0, M(M(M(lm, lm), mu0n), mu0n));
            double alp = M(M(M(0.75, ws), mu0n), D(A(1.0, M(gs, S(1.0, ws))), den0));
            double gam = M(M(0.5, ws), D(A(1.0, M(M(M(M(3.0, gs), S(1.0, ws)), mu0n), mu0n)), den0));
            double apg = A(alp, gam);
            double amg = S(alp, gam);
            rdir[k] = A(M(apg, rdif_a[k]), M(amg, S(M(tdif_a[k], trnlay[k]), 1.0)));
            tdir[k] = A(M(apg, tdif_a[k]), M(A(S(M(amg, rdif_a[k]), apg), 1.0), trnlay[k]));
            double R1 = rdif_a[k];
            double T1 = tdif_a[k];
            double swt = 0.0, smr = 0.0, smt = 0.0;
            for (int ng = 0; ng < 8; ++ng) {
                const double mu = DEDD_GAUSPT[ng];
                const double gwt = DEDD_GAUSWT[ng];
                swt = A(swt, M(mu, gwt));
                const double trn = sw_max(exp_min, glibc_exp(D(-ts, mu)));
                const double den = S(1.0, M(M(M(lm, lm), mu), mu));
                alp = M(M(M(0.75, ws), mu), D(A(1.0, M(gs, S(1.0, ws))), den));
                gam = M(M(0.5, ws), D(A(1.0, M(M(M(M(3.0, gs), S(1.0, ws)), mu), mu)), den));
                apg = A(alp, gam);
                amg = S(alp, gam);
                const double rdr = S(A(M(apg, R1), M(M(amg, T1), trn)), amg);
                const double tdr = A(S(A(M(apg, T1), M(M(amg, R1), trn)), M(apg, trn)), trn);
                smr = A(smr, M(M(mu, rdr), gwt));
                smt = A(smt, M(M(mu, tdr), gwt));
            }
            rdif_a[k] = D(smr, swt);
            tdif_a[k] = D(smt, swt);
            rdif_b[k] = rdif_a[k];
            tdif_b[k] = tdif_a[k];
            if (k == kfrsnl) {
                R1 = D(S(mu0, M(DEDD_REFINDX, mu0n)), A(mu0, M(DEDD_REFINDX, mu0n)));
                const double R2 = D(S(M(DEDD_REFINDX, mu0), mu0n), A(M(DEDD_REFINDX, mu0), mu0n));
                T1 = D(M(2.0, mu0), A(mu0, M(DEDD_REFINDX, mu0n)));
                const double T2 = D(M(2.0, mu0), A(M(DEDD_REFINDX, mu0), mu0n));
                const double Rf_dir_a = M(0.5, A(M(R1, R1), M(R2, R2)));
                const double Tf_dir_a = D(M(M(M(0.5, A(M(T1, T1), M(T2, T2))), DEDD_REFINDX), mu0n), mu0);
                const double Rf_dif_a = DEDD_CP063;
                const double Tf_dif_a = S(1.0, Rf_dif_a);
                const double Rf_dif_b = DEDD_CP455;
                const double Tf_dif_b = S(1.0, Rf_dif_b);
                const double rintfc = D(1.0, S(1.0, M(Rf_dif_b, rdif_a[k])));
                tdir[k] = A(M(Tf_dir_a, tdir[k]), M(M(M(M(Tf_dir_a, rdir[k]), Rf_dif_b), rintfc), tdif_a[k]));
                rdir[k] = A(Rf_dir_a, M(M(M(Tf_dir_a, rdir[k]), rintfc), Tf_dif_b));
                rdif_a[k] = A(Rf_dif_a, M(M(M(Tf_dif_a, rdif_a[k]), rintfc), Tf_dif_b));
                rdif_b[k] = A(rdif_b[k], M(M(M(tdif_b[k], Rf_dif_b), rintfc), tdif_a[k]));
                tdif_a[k] = M(M(tdif_a[k], rintfc), Tf_dif_a);
                tdif_b[k] = M(M(tdif_b[k], rintfc), Tf_dif_b);
                trnlay[k] = M(Tf_dir_a, trnlay[k]);
            }
        }
        trndir[k + 1] = M(trndir[k], trnlay[k]);
        const double refkm1 = D(1.0, S(1.0, M(rdndif[k], rdif_a[k])));
        const double tdrrdir = M(trndir[k], rdir[k]);
        const double tdndif = S(trntdr[k], trndir[k]);
        trntdr[k + 1] = A(M(trndir[k], tdir[k]), M(M(A(tdndif, M(tdrrdir, rdndif[k])), refkm1), tdif_a[k]));
        rdndif[k + 1] = A(rdif_b[k], M(M(M(tdif_b[k], rdndif[k]), refkm1), tdif_a[k]));
        trndif[k + 1] = M(M(trndif[k], refkm1), tdif_a[k]);
    }
    rupdir[klevp] = albodr;
    rupdif[klevp] = albodf;
    for (int k = klev; k >= 0; --k) {
        const double refkp1 = D(1.0, S(1.0, M(rdif_b[k], rupdif[k + 1])));
        rupdir[k] = A(rdir[k], M(M(A(M(trnlay[k], rupdir[k + 1]), M(S(tdir[k], trnlay[k]), rupdif[k + 1])), refkp1),
                                 tdif_b[k]));
        rupdif[k] = A(rdif_a[k], M(M(M(tdif_a[k], rupdif[k + 1]), refkp1), tdif_b[k]));
    }
}

// the tuned inherent optical properties (compute_dEdd's adjust ice and ponded ice iops)
struct DeddIops {
    double ki_ssl[3], wi_ssl[3], ki_dl[3], wi_dl[3], ki_int[3], wi_int[3], ki_p_ssl[3], wi_p_ssl[3], ki_p_int[3],
        wi_p_int[3];
};

__device__ void dedd_tune(double ki_mn, double wi_mn, double f, double R, bool clamp, double *ki, double *wi)
{
    double sigp = M(M(ki_mn, wi_mn), A(1.0, M(f, R)));
    if (clamp) sigp = sw_max(sigp, 0.0);
    *ki = A(sigp, M(ki_mn, S(1.0, wi_mn)));
    *wi = D(sigp, *ki);
}

// ice_shortwave.F90:1499-2961 compute_dEdd, without the aerosols and the
// biogeochemistry (refused): kabs_chl is 0 except kabs_chl(1,klev).
__device__ void dedd_compute(int nilyr, int nslyr, int klev, int klevp, bool heat_capacity, double fnidr,
                             double coszen, double R_ice, double R_pnd, double kalg, double exp_min, double swvdr,
                             double swvdf, double swidr, double swidf, int srftyp, double hs, const double *rhosnw,
                             const double *rsnw, double hi, double hp, double fi, double *alvdr, double *alvdf,
                             double *alidr, double *alidf, double *fswsfc, double *fswint, double *fswthru,
                             double *Sswabs, double *Iswabs, double *fswpenl)
{
    double tau[DEDD_NL], w0[DEDD_NL], g[DEDD_NL], dzk[DEDD_NL];
    double trndir[DEDD_NL], trntdr[DEDD_NL], trndif[DEDD_NL], rupdir[DEDD_NL], rupdif[DEDD_NL], rdndif[DEDD_NL];
    double dfdir[DEDD_NL], dfdif[DEDD_NL];
    double Sabs[1], Iabs[DEDD_NL], fthrul[DEDD_NL], frsnw[1], wghtns[3];
    DeddIops io;
    const double rnilyr = D(1.0, (double)nilyr);
    const double rnslyr = D(1.0, (double)nslyr);
    const int kii = nslyr + 1;
    for (int k = 0; k <= nilyr; ++k) fthrul[k] = 0.0;
    for (int k = 0; k < nilyr; ++k) Iabs[k] = 0.0;
    double avdr = 0.0, avdf = 0.0, aidr = 0.0, aidf = 0.0, fsfc = 0.0, fint = 0.0, fthru = 0.0;
    wghtns[0] = 1.0;
    wghtns[1] = A(DEDD_CP67, M(S(DEDD_CP78, DEDD_CP67), S(1.0, fnidr)));
    wghtns[2] = S(1.0, wghtns[1]);
    for (int k = 0; k < nslyr; ++k) {
        frsnw[k] = M(A(M(DEDD_FR_MAX, fnidr), M(DEDD_FR_MIN, S(1.0, fnidr))), rsnw[k]);
        Sabs[k] = 0.0;
    }
    double dz = M(hs, rnslyr);
    dzk[0] = sw_min(DEDD_HS_SSL, D(dz, 2.0));
    dzk[1] = S(dz, dzk[0]);
    for (int k = 2; k <= nslyr; ++k) dzk[k] = dz;
    dz = M(hi, rnilyr);
    double dz_ssl = DEDD_HI_SSL;
    dz_ssl = sw_min(DEDD_HI_SSL, D(hi, 30.0));
    dz_ssl = sw_min(dz_ssl, D(dz, 2.0));
    dzk[kii] = dz_ssl;
    dzk[kii + 1] = S(dz, dz_ssl);
    for (int k = kii + 2; k <= klev; ++k) dzk[k] = dz;
    const bool ice_clamp = !(R_ice >= 0.0);
    const double fice = ice_clamp ? DEDD_FM_ICE : DEDD_FP_ICE;
    for (int ns = 0; ns < 3; ++ns) {
        dedd_tune(DEDD_KI_SSL_MN[ns], DEDD_WI_SSL_MN[ns], fice, R_ice, ice_clamp, &io.ki_ssl[ns], &io.wi_ssl[ns]);
        dedd_tune(DEDD_KI_DL_MN[ns], DEDD_WI_DL_MN[ns], fice, R_ice, ice_clamp, &io.ki_dl[ns], &io.wi_dl[ns]);
        dedd_tune(DEDD_KI_INT_MN[ns], DEDD_WI_INT_MN[ns], fice, R_ice, ice_clamp, &io.ki_int[ns], &io.wi_int[ns]);
    }
    const bool pnd_clamp = !(R_pnd >= 0.0);
    const double fpnd = pnd_clamp ? DEDD_FM_PND : DEDD_FP_PND;
    for (int ns = 0; ns < 3; ++ns) {
        dedd_tune(DEDD_KI_P_SSL_MN[ns], DEDD_WI_P_SSL_MN[ns], fpnd, R_pnd, pnd_clamp, &io.ki_p_ssl[ns],
                  &io.wi_p_ssl[ns]);
        dedd_tune(DEDD_KI_P_INT_MN[ns], DEDD_WI_P_INT_MN[ns], fpnd, R_pnd, pnd_clamp, &io.ki_p_int[ns],
                  &io.wi_p_int[ns]);
    }
    const int ksrf = srftyp == 1 ? 1 : nslyr + 2;
    // not (tr_bgc_N .and. dEdd_algae): kabs_chl(1,klev) only
    const double kabs_chl_klev = M(kalg, D(0.50, dzk[klev]));
#define KCHL(ns, k) (((ns) == 0 && (k) == klev) ? kabs_chl_klev : 0.0)

    for (int ns = 0; ns < 3; ++ns) {
        if (srftyp == 0) {
            for (int k = 0; k <= nslyr; ++k) {
                tau[k] = 0.0;
                w0[k] = 0.0;
                g[k] = 0.0;
            }
        } else if (srftyp == 1) {
            for (int k = 0; k <= nslyr; ++k) {
                const int ksnow = k - min(k - 1, 0) - 1;     // Fortran's ksnow, 0-based
                double Qs = 0.0, ws = 0.0, gs = 0.0;
                if (frsnw[ksnow] < DEDD_RSNW_TAB[0]) {
                    Qs = DEDD_QS_TAB[0][ns];
                    ws = DEDD_WS_TAB[0][ns];
                    gs = DEDD_GS_TAB[0][ns];
                } else if (frsnw[ksnow] >= DEDD_RSNW_TAB[DEDD_NMBRAD - 1]) {
                    Qs = DEDD_QS_TAB[DEDD_NMBRAD - 1][ns];
                    ws = DEDD_WS_TAB[DEDD_NMBRAD - 1][ns];
                    gs = DEDD_GS_TAB[DEDD_NMBRAD - 1][ns];
                } else {
                    for (int nr = 1; nr < DEDD_NMBRAD; ++nr) {
                        if (DEDD_RSNW_TAB[nr - 1] <= frsnw[ksnow] && frsnw[ksnow] < DEDD_RSNW_TAB[nr]) {
                            const double delr = D(S(frsnw[ksnow], DEDD_RSNW_TAB[nr - 1]),
                                                  S(DEDD_RSNW_TAB[nr], DEDD_RSNW_TAB[nr - 1]));
                            Qs = A(M(DEDD_QS_TAB[nr - 1][ns], S(1.0, delr)), M(DEDD_QS_TAB[nr][ns], delr));
                            ws = A(M(DEDD_WS_TAB[nr - 1][ns], S(1.0, delr)), M(DEDD_WS_TAB[nr][ns], delr));
                            gs = A(M(DEDD_GS_TAB[nr - 1][ns], S(1.0, delr)), M(DEDD_GS_TAB[nr][ns], delr));
                        }
                    }
                }
                const double ks = M(Qs, D(M(D(rhosnw[ksnow], RHOI), 3.0), M(M(4.0, frsnw[ksnow]), 1.0e-6)));
                tau[k] = M(A(ks, KCHL(ns, k)), dzk[k]);
                w0[k] = M(D(ks, A(ks, KCHL(ns, k))), ws);
                g[k] = gs;
            }
        } else {
            dz = D(hp, A(D(1.0, rnslyr), 1.0));
            for (int k = 0; k <= nslyr; ++k) {
                tau[k] = M(DEDD_KW[ns], dz);
                w0[k] = 0.0;
                g[k] = 0.0;
            }
        }
        if (srftyp <= 1) {
            int k = kii;
            tau[k] = M(A(io.ki_ssl[ns], KCHL(ns, k)), dzk[k]);
            w0[k] = M(D(io.ki_ssl[ns], A(io.ki_ssl[ns], KCHL(ns, k))), io.wi_ssl[ns]);
            g[k] = DEDD_GI;
            k = kii + 1;
            const double fs = D(P25, rnilyr);
            tau[k] = M(M(A(io.ki_dl[ns], KCHL(ns, k)), dzk[k]), fs);
            w0[k] = M(D(io.ki_dl[ns], A(io.ki_dl[ns], KCHL(ns, k))), io.wi_dl[ns]);
            g[k] = DEDD_GI;
            for (k = kii + 2; k <= klev - 1; ++k) {
                tau[k] = M(A(io.ki_int[ns], KCHL(ns, k)), dzk[k]);
                w0[k] = M(D(io.ki_int[ns], A(io.ki_int[ns], KCHL(ns, k))), io.wi_int[ns]);
                g[k] = DEDD_GI;
            }
            k = klev;
            double kabs = M(io.ki_int[ns], S(1.0, io.wi_int[ns]));
            if (ns == 0) kabs = A(kabs, KCHL(ns, k));
            const double sig = M(io.ki_int[ns], io.wi_int[ns]);
            tau[k] = M(A(kabs, sig), dzk[k]);
            w0[k] = D(sig, A(sig, kabs));
            g[k] = DEDD_GI;
        } else {
            int k = kii;
            tau[k] = M(io.ki_p_ssl[ns], dzk[k]);
            w0[k] = io.wi_p_ssl[ns];
            g[k] = DEDD_GI;
            k = kii + 1;
            tau[k] = M(io.ki_p_int[ns], dzk[k]);
            w0[k] = io.wi_p_int[ns];
            g[k] = DEDD_GI;
            for (k = kii + 2; k <= klev; ++k) {
                tau[k] = M(io.ki_p_int[ns], dzk[k]);
                w0[k] = io.wi_p_int[ns];
                g[k] = DEDD_GI;
            }
            if (DEDD_HPMIN <= hp && hp <= DEDD_HP0) {
                k = kii;
                double sig_i = M(io.ki_ssl[ns], io.wi_ssl[ns]);
                double sig_p = M(io.ki_p_ssl[ns], io.wi_p_ssl[ns]);
                double sig = A(sig_i, M(S(sig_p, sig_i), D(hp, DEDD_HP0)));
                double kext = A(sig, M(io.ki_p_ssl[ns], S(1.0, io.wi_p_ssl[ns])));
                tau[k] = M(kext, dzk[k]);
                w0[k] = D(sig, kext);
                g[k] = DEDD_GI;
                k = kii + 1;
                const double fs = D(P25, rnilyr);
                sig_i = M(M(io.ki_dl[ns], io.wi_dl[ns]), fs);
                sig_p = M(io.ki_p_int[ns], io.wi_p_int[ns]);
                sig = A(sig_i, M(S(sig_p, sig_i), D(hp, DEDD_HP0)));
                kext = A(sig, M(io.ki_p_int[ns], S(1.0, io.wi_p_int[ns])));
                tau[k] = M(kext, dzk[k]);
                w0[k] = D(sig, kext);
                g[k] = DEDD_GI;
                for (k = kii + 2; k <= klev; ++k) {
                    sig_i = M(io.ki_int[ns], io.wi_int[ns]);
                    sig_p = M(io.ki_p_int[ns], io.wi_p_int[ns]);
                    sig = A(sig_i, M(S(sig_p, sig_i), D(hp, DEDD_HP0)));
                    kext = A(sig, M(io.ki_p_int[ns], S(1.0, io.wi_p_int[ns])));
                    tau[k] = M(kext, dzk[k]);
                    w0[k] = D(sig, kext);
                    g[k] = DEDD_GI;
                }
            }
        }
        const double rns = (double)ns;
        const double albodr = M(DEDD_CP01, S(1.0, sw_min(rns, 1.0)));
        const double albodf = M(DEDD_CP01, S(1.0, sw_min(rns, 1.0)));
        dedd_solution(coszen, srftyp, klev, klevp, nslyr, exp_min, tau, w0, g, albodr, albodf, trndir, trntdr,
                      trndif, rupdir, rupdif, rdndif);
        for (int k = 0; k <= klevp; ++k) {
            const double refk = D(1.0, S(1.0, M(rdndif[k], rupdif[k])));
            dfdir[k] = S(A(trndir[k], M(M(S(trntdr[k], trndir[k]), S(1.0, rupdif[k])), refk)),
                         M(M(M(trndir[k], rupdir[k]), S(1.0, rdndif[k])), refk));
            if (dfdir[k] < PUNY) dfdir[k] = 0.0;
            dfdif[k] = M(M(trndif[k], S(1.0, rupdif[k])), refk);
            if (dfdif[k] < PUNY) dfdif[k] = 0.0;
        }
        if (ns == 0) {
            const double swdr = swvdr, swdf = swvdf;
            avdr = rupdir[0];
            avdf = rupdif[0];
            const double tmp_0 = A(M(dfdir[0], swdr), M(dfdif[0], swdf));
            const double tmp_ks = A(M(dfdir[ksrf], swdr), M(dfdif[ksrf], swdf));
            const double tmp_kl = A(M(dfdir[klevp], swdr), M(dfdif[klevp], swdf));
            for (int k = nslyr + 2; k <= klevp; ++k) fthrul[k - nslyr - 2] = A(M(dfdir[k], swdr), M(dfdif[k], swdf));
            fsfc = S(A(fsfc, tmp_0), tmp_ks);
            fint = S(A(fint, tmp_ks), tmp_kl);
            fthru = A(fthru, tmp_kl);
            if (srftyp == 1) {
                int ki = 0;
                for (int k = 1; k <= nslyr; ++k) {
                    const int km = k, kp = km + 1;
                    Sabs[ki] = S(A(A(Sabs[ki], M(dfdir[km], swdr)), M(dfdif[km], swdf)),
                                 A(M(dfdir[kp], swdr), M(dfdif[kp], swdf)));
                    ++ki;
                }
            }
            int ki = 0;
            for (int k = nslyr + 2; k <= nslyr + 1 + nilyr; ++k) {
                int km = k, kp = km + 1;
                if (srftyp == 1 && k == nslyr + 2) {
                    km = k - 1;
                    kp = km + 2;
                }
                Iabs[ki] = S(A(A(Iabs[ki], M(dfdir[km], swdr)), M(dfdif[km], swdf)),
                             A(M(dfdir[kp], swdr), M(dfdif[kp], swdf)));
                ++ki;
            }
        } else {
            const double swdr = swidr, swdf = swidf;
            aidr = A(aidr, M(rupdir[0], wghtns[ns]));
            aidf = A(aidf, M(rupdif[0], wghtns[ns]));
            double tmp_0 = A(M(dfdir[0], swdr), M(dfdif[0], swdf));
            double tmp_ks = A(M(dfdir[ksrf], swdr), M(dfdif[ksrf], swdf));
            double tmp_kl = A(M(dfdir[klevp], swdr), M(dfdif[klevp], swdf));
            tmp_0 = M(tmp_0, wghtns[ns]);
            tmp_ks = M(tmp_ks, wghtns[ns]);
            tmp_kl = M(tmp_kl, wghtns[ns]);
            fsfc = S(A(fsfc, tmp_0), tmp_ks);
            fint = S(A(fint, tmp_ks), tmp_kl);
            fthru = A(fthru, tmp_kl);
            if (srftyp == 1) {
                int ki = 0;
                for (int k = 1; k <= nslyr; ++k) {
                    const int km = k, kp = km + 1;
                    Sabs[ki] = A(Sabs[ki], M(S(A(M(dfdir[km], swdr), M(dfdif[km], swdf)),
                                               A(M(dfdir[kp], swdr), M(dfdif[kp], swdf))), wghtns[ns]));
                    ++ki;
                }
            }
            int ki = 0;
            for (int k = nslyr + 2; k <= nslyr + 1 + nilyr; ++k) {
                int km = k, kp = km + 1;
                if (srftyp == 1 && k == nslyr + 2) {
                    km = k - 1;
                    kp = km + 2;
                }
                Iabs[ki] = A(Iabs[ki], M(S(A(M(dfdir[km], swdr), M(dfdif[km], swdf)),
                                           A(M(dfdir[kp], swdr), M(dfdif[kp], swdf))), wghtns[ns]));
                ++ki;
            }
        }
    }
#undef KCHL
    *alvdr = avdr;
    *alvdf = avdf;
    *alidr = aidr;
    *alidf = aidf;
    *fswsfc = A(*fswsfc, M(fsfc, fi));
    *fswint = A(*fswint, M(fint, fi));
    *fswthru = A(*fswthru, M(fthru, fi));
    for (int k = 0; k < nslyr; ++k) Sswabs[k] = A(Sswabs[k], M(Sabs[k], fi));
    for (int k = 0; k < nilyr; ++k) {
        Iswabs[k] = A(Iswabs[k], M(Iabs[k], fi));
        fswpenl[k] = A(fswpenl[k], M(fthrul[k], fi));
        if (k == nilyr - 1) fswpenl[k + 1] = A(fswpenl[k + 1], M(fthrul[k + 1], fi));
    }
    if (!heat_capacity) {
        *fswsfc = A(A(*fswsfc, Iswabs[0]), Sswabs[0]);
        *fswint = 0.0;
        Iswabs[0] = 0.0;
        Sswabs[0] = 0.0;
    }
}

// ice_shortwave.F90:1087-1489 shortwave_dEdd (no aerosols, l_print_point false)
__device__ void dedd_shortwave(int nslyr, int nilyr, double *coszen, bool heat_capacity, double aice, double vice,
                               double hs, double fs, const double *rhosnw, const double *rsnw, double fp, double hp,
                               double R_ice, double R_pnd, double kalg, double exp_min, double swvdr, double swvdf,
                               double swidr, double swidf, double *alvdr, double *alvdf, double *alidr,
                               double *alidf, double *fswsfc, double *fswint, double *fswthru, double *Sswabs,
                               double *Iswabs, double *albice, double *albsno, double *albpnd, double *fswpenl)
{
    const int klev = nslyr + nilyr + 1;
    const int klevp = klev + 1;
    const double hstmp = 0.0;
    double hi = 0.0, fi = 0.0;
    *alvdr = 0.0;
    *alvdf = 0.0;
    *alidr = 0.0;
    *alidf = 0.0;
    double avdrl = 0.0, avdfl = 0.0, aidrl = 0.0, aidfl = 0.0;
    *fswsfc = 0.0;
    *fswint = 0.0;
    *fswthru = 0.0;
    double fnidr = 0.0;
    if (A(swidr, swidf) > PUNY) fnidr = D(swidr, A(swidr, swidf));
    *albice = 0.0;
    *albsno = 0.0;
    *albpnd = 0.0;
    for (int k = 0; k <= nilyr; ++k) fswpenl[k] = 0.0;
    for (int k = 0; k < nslyr; ++k) Sswabs[k] = 0.0;
    for (int k = 0; k < nilyr; ++k) Iswabs[k] = 0.0;
    double netsw = A(A(A(swvdr, swidr), swvdf), swidf);
    if (netsw > PUNY) {
        *coszen = sw_max(PUNY, *coszen);
        hi = D(vice, aice);
        fi = S(S(1.0, fs), fp);
        if (fi > 0.0) {
            dedd_compute(nilyr, nslyr, klev, klevp, heat_capacity, fnidr, *coszen, R_ice, R_pnd, kalg, exp_min,
                         swvdr, swvdf, swidr, swidf, 0, hstmp, rhosnw, rsnw, hi, hp, fi, &avdrl, &avdfl, &aidrl,
                         &aidfl, fswsfc, fswint, fswthru, Sswabs, Iswabs, fswpenl);
            *alvdr = A(*alvdr, M(avdrl, fi));
            *alvdf = A(*alvdf, M(avdfl, fi));
            *alidr = A(*alidr, M(aidrl, fi));
            *alidf = A(*alidf, M(aidfl, fi));
            *albice = A(A(A(A(*albice, M(SW_AWTVDR, avdrl)), M(SW_AWTIDR, aidrl)), M(SW_AWTVDF, avdfl)),
                        M(SW_AWTIDF, aidfl));
        }
    }
    netsw = A(A(A(swvdr, swidr), swvdf), swidf);
    if (netsw > PUNY) {
        *coszen = sw_max(PUNY, *coszen);
        if (fs > 0.0) {
            dedd_compute(nilyr, nslyr, klev, klevp, heat_capacity, fnidr, *coszen, R_ice, R_pnd, kalg, exp_min,
                         swvdr, swvdf, swidr, swidf, 1, hs, rhosnw, rsnw, hi, hp, fs, &avdrl, &avdfl, &aidrl, &aidfl,
                         fswsfc, fswint, fswthru, Sswabs, Iswabs, fswpenl);
            *alvdr = A(*alvdr, M(avdrl, fs));
            *alvdf = A(*alvdf, M(avdfl, fs));
            *alidr = A(*alidr, M(aidrl, fs));
            *alidf = A(*alidf, M(aidfl, fs));
            *albsno = A(A(A(A(*albsno, M(SW_AWTVDR, avdrl)), M(SW_AWTIDR, aidrl)), M(SW_AWTVDF, avdfl)),
                        M(SW_AWTIDF, aidfl));
        }
    }
    hi = 0.0;
    netsw = A(A(A(swvdr, swidr), swvdf), swidf);
    if (netsw > PUNY) {
        *coszen = sw_max(PUNY, *coszen);
        hi = D(vice, aice);
        if (fp > PUNY) {
            dedd_compute(nilyr, nslyr, klev, klevp, heat_capacity, fnidr, *coszen, R_ice, R_pnd, kalg, exp_min,
                         swvdr, swvdf, swidr, swidf, 2, hs, rhosnw, rsnw, hi, hp, fp, &avdrl, &avdfl, &aidrl, &aidfl,
                         fswsfc, fswint, fswthru, Sswabs, Iswabs, fswpenl);
            *alvdr = A(*alvdr, M(avdrl, fp));
            *alvdf = A(*alvdf, M(avdfl, fp));
            *alidr = A(*alidr, M(aidrl, fp));
            *alidf = A(*alidf, M(aidfl, fp));
            *albpnd = A(A(A(A(*albpnd, M(SW_AWTVDR, avdrl)), M(SW_AWTIDR, aidrl)), M(SW_AWTVDF, avdfl)),
                        M(SW_AWTIDF, aidfl));
        }
    }
    netsw = A(A(A(swvdr, swidr), swvdf), swidf);
    if (netsw <= PUNY) {
        *alvdr = 1.0;
        *alvdf = 1.0;
        *alidr = 1.0;
        *alidf = 1.0;
    }
}

// mpas_seaice_column.F:2118-2484 column_radiation's loop over the owned
// cells, with colpkg_step_radiation's calc_Tsfc and 'dEdd' branch: run_dEdd.
// The orbit comes from seaice_column_init_orbit_v842.
extern "C" __global__ void seaice_column_radiation_dedd_v842(
    int ncat, int nilyr, int nslyr, int nbio, int flags, int sec, int nCellsSolve,
    const double *__restrict__ par, const double *__restrict__ orbit, const double *__restrict__ latCell,
    const double *__restrict__ lonCell, const double *__restrict__ aicenAll, const double *__restrict__ vicenAll,
    const double *__restrict__ vsnonAll, const double *__restrict__ TsfcnAll, const double *__restrict__ alvlnAll,
    const double *__restrict__ apndnAll, const double *__restrict__ hpndnAll, const double *__restrict__ ipndnAll,
    const double *__restrict__ ffracnAll, const double *__restrict__ fsnowAll, const double *__restrict__ swvdrAll,
    const double *__restrict__ swvdfAll, const double *__restrict__ swidrAll, const double *__restrict__ swidfAll,
    double *__restrict__ coszenAll, double *__restrict__ alvdrnAll, double *__restrict__ alvdfnAll,
    double *__restrict__ alidrnAll, double *__restrict__ alidfnAll, double *__restrict__ fswsfcnAll,
    double *__restrict__ fswintnAll, double *__restrict__ fswthrunAll, double *__restrict__ fswpenlnAll,
    double *__restrict__ IswabsnAll, double *__restrict__ SswabsnAll, double *__restrict__ albicenAll,
    double *__restrict__ albsnonAll, double *__restrict__ albpndnAll, double *__restrict__ apeffnAll,
    double *__restrict__ snowfracnAll, double *__restrict__ dhsnAll, double *__restrict__ zbionAll)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nCellsSolve) return;
    const size_t cn = (size_t)c * ncat;
    double *alvdrn = alvdrnAll + cn, *alidrn = alidrnAll + cn, *alvdfn = alvdfnAll + cn, *alidfn = alidfnAll + cn;
    double *fswsfcn = fswsfcnAll + cn, *fswintn = fswintnAll + cn, *fswthrun = fswthrunAll + cn;
    double *fswpenln = fswpenlnAll + cn * (nilyr + 1);
    double *Iswabsn = IswabsnAll + cn * nilyr;
    double *Sswabsn = SswabsnAll + cn * nslyr;
    const double swvdr = swvdrAll[c], swvdf = swvdfAll[c], swidr = swidrAll[c], swidf = swidfAll[c];
    // colpkg_step_radiation's zeroing
    for (int n = 0; n < ncat; ++n) {
        alvdrn[n] = 0.0;
        alidrn[n] = 0.0;
        alvdfn[n] = 0.0;
        alidfn[n] = 0.0;
        fswsfcn[n] = 0.0;
        fswintn[n] = 0.0;
        fswthrun[n] = 0.0;
    }
    for (int i = 0; i < ncat * (nilyr + 1); ++i) fswpenln[i] = 0.0;
    for (int i = 0; i < ncat * nilyr; ++i) Iswabsn[i] = 0.0;
    for (int i = 0; i < ncat * nslyr; ++i) Sswabsn[i] = 0.0;
    if (nbio > 0)
        for (int i = 0; i < ncat * nbio; ++i) zbionAll[cn * nbio + i] = 0.0;

    // column_radiation's longitude in (-pi, pi]
    double tlon = lonCell[c];
    if (tlon > DEDD_PII) tlon = S(tlon, M(2.0, DEDD_PII));
    const double R_ice = par[DP_R_ICE], R_pnd = par[DP_R_PND], R_snw = par[DP_R_SNW], dT_mlt = par[DP_DT_MLT];
    const double rsnw_mlt = par[DP_RSNW_MLT], kalg = par[DP_KALG], hs0 = par[DP_HS0], hs1 = par[DP_HS1];
    const double hp1 = par[DP_HP1], pndaspect = par[DP_PNDASPECT], dt = par[DP_DT];
    const bool initonly = (flags & DEDD_INITONLY) != 0;
    const bool heat_capacity = (flags & DEDD_HEAT_CAPACITY) != 0;
    const double fsnow = fsnowAll[c];

    // run_dEdd
    const double exp_min = glibc_exp(-10.0);
    double coszen = dedd_coszen(latCell[c], tlon, par[DP_YDAY], sec, orbit);
    const double *aicen = aicenAll + cn, *vicen = vicenAll + cn, *vsnon = vsnonAll + cn, *Tsfcn = TsfcnAll + cn;
    double *apeffn = apeffnAll + cn, *snowfracn = snowfracnAll + cn;
    for (int n = 0; n < ncat; ++n) {
        double fsn = 0.0, hsn = 0.0;
        double rhosnwn[1] = {0.0}, rsnwn[1] = {0.0};
        apeffn[n] = 0.0;
        snowfracn[n] = 0.0;
        if (aicen[n] > PUNY) {
            dedd_set_snow(R_snw, dT_mlt, rsnw_mlt, aicen[n], vsnon[n], Tsfcn[n], &fsn, hs0, &hsn, rhosnwn, rsnwn);
            double fpn, hpn;
            if (flags & DEDD_POND_CESM) {
                fpn = apndnAll[cn + n];
                hpn = hpndnAll[cn + n];
                if (hsn >= DEDD_HS_MIN && hs0 > PUNY) {
                    const double asnow = sw_min(D(hsn, hs0), 1.0);
                    fpn = M(S(1.0, asnow), fpn);
                    hpn = M(pndaspect, fpn);
                }
                if (hpn < DEDD_HPMIN) fpn = 0.0;
                fsn = sw_min(fsn, S(1.0, fpn));
                apeffn[n] = fpn;
            } else if (flags & DEDD_POND_LVL) {
                const double alvl = alvlnAll[cn + n], apnd = apndnAll[cn + n];
                fpn = 0.0;
                hpn = 0.0;
                const double ipn = M(M(alvl, apnd), ipndnAll[cn + n]);
                double dhs = dhsnAll[cn + n];
                if (!initonly && ipn > PUNY && dhs < PUNY && M(fsnow, dt) > DEDD_HS_MIN) dhs = S(hsn, M(fsnow, dt));
                const double spn = S(hsn, dhs);
                if (!initonly && M(ipn, spn) < PUNY) dhs = 0.0;
                dhsnAll[cn + n] = dhs;
                fpn = M(apnd, alvl);
                hpn = hpndnAll[cn + n];
                fpn = M(S(1.0, ffracnAll[cn + n]), fpn);
                if (dhs > PUNY && spn >= PUNY && hs1 > PUNY) {
                    const double asnow = sw_min(D(spn, hs1), 1.0);
                    fpn = M(S(1.0, asnow), fpn);
                }
                double hp = hpn;
                if (hp > PUNY) {
                    const double hs = hsn;
                    const double rp = D(M(DEDD_RHOFRESH, hp), A(M(DEDD_RHOFRESH, hp), M(RHOS, hs)));
                    if (rp < P15) {
                        fpn = 0.0;
                        hpn = 0.0;
                    } else {
                        const double hmx = D(M(hs, S(DEDD_RHOFRESH, RHOS)), DEDD_RHOFRESH);
                        const double tmp = sw_max(0.0, copysign(1.0, S(hp, hmx)));
                        hp = D(A(M(DEDD_RHOFRESH, hp), M(M(RHOS, hs), tmp)), S(DEDD_RHOFRESH, M(RHOS, S(1.0, tmp))));
                        hsn = S(hs, M(M(hp, fpn), S(1.0, tmp)));
                        hpn = M(hp, tmp);
                        fpn = M(fpn, tmp);
                    }
                }
                if (hpn < DEDD_HPMIN) fpn = 0.0;
                fsn = sw_min(fsn, S(1.0, fpn));
                apeffn[n] = fpn;
            } else if (flags & DEDD_POND_TOPO) {
                const double apnd = apndnAll[cn + n];
                if (M(apnd, aicen[n]) > PUNY && ipndnAll[cn + n] < hp1) fpn = apnd;
                else fpn = 0.0;
                if (apnd > PUNY) {
                    hpn = hpndnAll[cn + n];
                } else {
                    fpn = 0.0;
                    hpn = 0.0;
                }
                if (hpn < DEDD_HPMIN) fpn = 0.0;
                fsn = sw_min(fsn, S(1.0, fpn));
                apeffn[n] = fpn;
            } else {
                // shortwave_dEdd_set_pond
                const double dTs = S(SW_TIMELT, Tsfcn[n]);
                const double fT = -sw_min(S(D(dTs, 1.0), 1.0), 0.0);
                fpn = M(M(0.3, fT), S(1.0, fsn));
                hpn = M(M(0.3, fT), S(1.0, fsn));
                apeffn[n] = fpn;
                fpn = 0.0;
                hpn = 0.0;
            }
            snowfracn[n] = fsn;
            dedd_shortwave(nslyr, nilyr, &coszen, heat_capacity, aicen[n], vicen[n], hsn, fsn, rhosnwn, rsnwn, fpn,
                           hpn, R_ice, R_pnd, kalg, exp_min, swvdr, swvdf, swidr, swidf, &alvdrn[n], &alvdfn[n],
                           &alidrn[n], &alidfn[n], &fswsfcn[n], &fswintn[n], &fswthrun[n], &Sswabsn[n * nslyr],
                           &Iswabsn[n * nilyr], &albicenAll[cn + n], &albsnonAll[cn + n], &albpndnAll[cn + n],
                           &fswpenln[n * (nilyr + 1)]);
        }
    }
    coszenAll[c] = coszen;
}

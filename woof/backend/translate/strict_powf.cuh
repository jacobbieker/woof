// Correctly rounded binary32 pow, for the MLX facade's cupy.power under
// GPUWM_WRF_EXACT=1 (woof/backend/facade/_ops.py).  Written for WOOF from
// textbook methods; its constants come from tools/strict_powf/constants.py
// (Python's decimal module, no library tables).  Never compiled for CUDA,
// whose power stays CuPy's own; a host C++ build (no __CUDACC__) exists for
// tests/test_strict_powf.py's sweeps.
//
// Method.  |x| = m 2^e with m in [0.75, 1.5).  A 97-entry table gives
// invc ~ 1/m to 24 bits, so r = m invc - 1 is exact and |r| < 2^-7.5, and
// log2 x = e + log2(1/invc) + log2(1 + r) (Taylor series).  t = y log2 x is
// formed as a double-double; 2^t = 2^k 2^(j/64) 2^g with |g| <= 2^-7, from a
// 64-entry table and a Taylor series.
//
// * Fast phase, in binary64: the result has a relative error below 2^-60
//   (log series to r^10, remainder < 2^-71 relative; exp series to g^7,
//   remainder < 2^-64; rounding errors of a few 2^-60).  It is kept when
//   the interval z (1 +- 2^-54) rounds to one binary32 word.
// * Accurate phase, in double-double: log series to r^18, exp series to
//   g^14, relative error below 2^-95.  The value is rounded by its side of
//   the neighbouring binary32 midpoints.
// * A value within 2^-93 of a midpoint is tested for an exact midpoint:
//   for binary32 x and y > 0 that happens only when y = n or n/2^K (K <= 4)
//   and the odd part of x is an exact power whose result has 25 bits (for
//   example 4097^2 or 257^1.5); the test is exact integer arithmetic, and
//   an exact midpoint rounds to even.
//
// Special cases follow C99 Annex F (pow(x, +-0) = pow(1, y) = 1, NaN in
// gives NaN out, signed zeros and infinities, NaN for a negative base and
// a non-integer exponent).  Subnormal operands and results are exact
// binary32 behaviour: the result is rounded once, from binary64.
#ifndef WOOF_STRICT_POWF
#define WOOF_STRICT_POWF

#ifdef __CUDACC__
#define WSP_HD __device__ inline
#define WSP_TABLE static __device__ const
#define WSP_DBITS(x) __double_as_longlong(x)
#define WSP_FROM_DBITS(i) __longlong_as_double(i)
#define WSP_FBITS(x) __float_as_int(x)
#define WSP_FROM_FBITS(i) __int_as_float(i)
#define WSP_FMA(a, b, c) fma((a), (b), (c))
#define WSP_RINT(a) rint(a)
#else
#include <cmath>
#include <cstring>
#define WSP_HD inline
#define WSP_TABLE static const
inline long long wsp_dbits(double x) {long long i; std::memcpy(&i, &x, 8); return i;}
inline double wsp_from_dbits(long long i) {double x; std::memcpy(&x, &i, 8); return x;}
inline int wsp_fbits(float x) {int i; std::memcpy(&i, &x, 4); return i;}
inline float wsp_from_fbits(int i) {float x; std::memcpy(&x, &i, 4); return x;}
#define WSP_DBITS(x) wsp_dbits(x)
#define WSP_FROM_DBITS(i) wsp_from_dbits(i)
#define WSP_FBITS(x) wsp_fbits(x)
#define WSP_FROM_FBITS(i) wsp_from_fbits(i)
#define WSP_FMA(a, b, c) std::fma((a), (b), (c))
#define WSP_RINT(a) std::rint(a)
#endif

// CONSTANTS BEGIN (tools/strict_powf/constants.py)
#define WSP_LOG_LO (-32)
#define WSP_LOG_ENTRIES 97
#define WSP_FAST_LOG_TERMS 10
#define WSP_ACCURATE_LOG_TERMS 18
#define WSP_FAST_EXP_TERMS 7
#define WSP_ACCURATE_EXP_TERMS 14
WSP_TABLE double wsp_invc[97] = {
    0x1.5555560000000p+0, 0x1.51d07e0000000p+0, 0x1.4e5e0a0000000p+0,
    0x1.4afd6a0000000p+0, 0x1.47ae140000000p+0, 0x1.446f860000000p+0,
    0x1.4141420000000p+0, 0x1.3e22cc0000000p+0, 0x1.3b13b20000000p+0,
    0x1.3813820000000p+0, 0x1.3521d00000000p+0, 0x1.323e340000000p+0,
    0x1.2f684c0000000p+0, 0x1.2c9fb40000000p+0, 0x1.29e4120000000p+0,
    0x1.27350c0000000p+0, 0x1.24924a0000000p+0, 0x1.21fb780000000p+0,
    0x1.1f70480000000p+0, 0x1.1cf06a0000000p+0, 0x1.1a7b960000000p+0,
    0x1.1811820000000p+0, 0x1.15b1e60000000p+0, 0x1.135c820000000p+0,
    0x1.1111120000000p+0, 0x1.0ecf560000000p+0, 0x1.0c97140000000p+0,
    0x1.0a68100000000p+0, 0x1.0842100000000p+0, 0x1.0624de0000000p+0,
    0x1.0410420000000p+0, 0x1.0204080000000p+0, 0x1.0000000000000p+0,
    0x1.fc07f00000000p-1, 0x1.f81f820000000p-1, 0x1.f4465a0000000p-1,
    0x1.f07c200000000p-1, 0x1.ecc07c0000000p-1, 0x1.e9131a0000000p-1,
    0x1.e573ac0000000p-1, 0x1.e1e1e20000000p-1, 0x1.de5d6e0000000p-1,
    0x1.dae6080000000p-1, 0x1.d77b660000000p-1, 0x1.d41d420000000p-1,
    0x1.d0cb580000000p-1, 0x1.cd85680000000p-1, 0x1.ca4b300000000p-1,
    0x1.c71c720000000p-1, 0x1.c3f8f00000000p-1, 0x1.c0e0700000000p-1,
    0x1.bdd2b80000000p-1, 0x1.bacf920000000p-1, 0x1.b7d6c40000000p-1,
    0x1.b4e81c0000000p-1, 0x1.b203640000000p-1, 0x1.af286c0000000p-1,
    0x1.ac57020000000p-1, 0x1.a98ef60000000p-1, 0x1.a6d01a0000000p-1,
    0x1.a41a420000000p-1, 0x1.a16d400000000p-1, 0x1.9ec8ea0000000p-1,
    0x1.9c2d140000000p-1, 0x1.99999a0000000p-1, 0x1.970e500000000p-1,
    0x1.948b100000000p-1, 0x1.920fb40000000p-1, 0x1.8f9c180000000p-1,
    0x1.8d30180000000p-1, 0x1.8acb900000000p-1, 0x1.886e600000000p-1,
    0x1.8618620000000p-1, 0x1.83c9780000000p-1, 0x1.8181820000000p-1,
    0x1.7f40600000000p-1, 0x1.7d05f40000000p-1, 0x1.7ad2200000000p-1,
    0x1.78a4c80000000p-1, 0x1.767dce0000000p-1, 0x1.745d180000000p-1,
    0x1.7242880000000p-1, 0x1.702e060000000p-1, 0x1.6e1f760000000p-1,
    0x1.6c16c20000000p-1, 0x1.6a13ce0000000p-1, 0x1.6816820000000p-1,
    0x1.661ec60000000p-1, 0x1.642c860000000p-1, 0x1.623fa80000000p-1,
    0x1.6058160000000p-1, 0x1.5e75bc0000000p-1, 0x1.5c98820000000p-1,
    0x1.5ac0560000000p-1, 0x1.58ed240000000p-1, 0x1.571ed40000000p-1,
    0x1.5555560000000p-1
};
WSP_TABLE double wsp_logc_hi[97] = {
    (-0x1.a8ff99fab991dp-2), (-0x1.99b06fafae365p-2), (-0x1.8a897eb027b02p-2),
    (-0x1.7b89f015dd637p-2), (-0x1.6cb0f45c5ddccp-2), (-0x1.5dfdcd9681230p-2),
    (-0x1.4f6fbe9a14f18p-2), (-0x1.4106026313941p-2), (-0x1.32bff1d2620d3p-2),
    (-0x1.249cd663a544dp-2), (-0x1.169c06a7938bbp-2), (-0x1.08bcddc88c274p-2),
    (-0x1.f5fd8c01b8598p-3), (-0x1.dac224f29e71ep-3), (-0x1.bfc6745e58544p-3),
    (-0x1.a509500525022p-3), (-0x1.8a898953f695dp-3), (-0x1.70460263cfbd2p-3),
    (-0x1.563dc4114f416p-3), (-0x1.3c6fad7aa88cfp-3), (-0x1.22dadb72090e4p-3),
    (-0x1.097e425d2ff08p-3), (-0x1.e0b1af47da109p-4), (-0x1.aed3a581afc75p-4),
    (-0x1.7d605d9f9a247p-4), (-0x1.4c55ffab94bfbp-4), (-0x1.1bb314bc1250dp-4),
    (-0x1.d6ebb51765786p-5), (-0x1.773935884e226p-5), (-0x1.184bb316406a1p-5),
    (-0x1.743f41d467d22p-6), (-0x1.72c7ae96537f8p-7), 0x0p+0,
    0x1.6fe516f994381p-7, 0x1.6e7966ead8ac5p-6, 0x1.11cd1acadf723p-5,
    0x1.6bad2043a8791p-5, 0x1.c4df9816b67cbp-5, 0x1.0eb392fe79defp-4,
    0x1.3aa304acd04f4p-4, 0x1.663f6e3b3cbb2p-4, 0x1.918a19f536b15p-4,
    0x1.bc841cd4346d3p-4, 0x1.e72eb841d5082p-4, 0x1.08c587b8a8459p-3,
    0x1.1dcd1f96f9b03p-3, 0x1.32aea1c2de0a0p-3, 0x1.476aa1c23e268p-3,
    0x1.5c01a22e68f24p-3, 0x1.70742e079a632p-3, 0x1.84c2be7444b1ap-3,
    0x1.98edd46f8f540p-3, 0x1.acf5de2afc49ap-3, 0x1.c0db6bf6c0150p-3,
    0x1.d49ee012d3176p-3, 0x1.e840bea311339p-3, 0x1.fbc16a1ed20a6p-3,
    0x1.0790ac9a79044p-2, 0x1.11307dc445fecp-2, 0x1.1ac05ca5fe214p-2,
    0x1.2440796db68c3p-2, 0x1.2db10e538534dp-2, 0x1.37124a7b0e57ap-2,
    0x1.40646707c3973p-2, 0x1.49a7834b7d429p-2, 0x1.52dbddf71fd7ap-2,
    0x1.5c01a2e7132d6p-2, 0x1.651900878bb3cp-2, 0x1.6e22207523f6dp-2,
    0x1.771d2eb8c3300p-2, 0x1.800a59ccb4ee3p-2, 0x1.88e9c392b7fbbp-2,
    0x1.91bba6c447dcfp-2, 0x1.9a80224eb84b9p-2, 0x1.a3375ec3372a1p-2,
    0x1.abe186df47b97p-2, 0x1.b47ebfcfdd47ap-2, 0x1.bd0f30c877b4fp-2,
    0x1.c592fb2eead30p-2, 0x1.ce0a4a2d1a3cfp-2, 0x1.d6753b2085b50p-2,
    0x1.ded3fd15f8d60p-2, 0x1.e726a9208b3bep-2, 0x1.ef6d6a09ac6bbp-2,
    0x1.f7a85434872d2p-2, 0x1.ffd795ea4ce80p-2, 0x1.03fda781da546p-1,
    0x1.0809d07c70e13p-1, 0x1.0c104f268ee09p-1, 0x1.101139e927eecp-1,
    0x1.140c9fb5a8f7fp-1, 0x1.18028c04ca092p-1, 0x1.1bf31371c6bf0p-1,
    0x1.1fde3ea801dafp-1, 0x1.23c41b2f89133p-1, 0x1.27a4bfd967b6ep-1,
    0x1.2b803302a3372p-1
};
WSP_TABLE double wsp_logc_lo[97] = {
    0x1.b85b552732499p-58, (-0x1.fce853ff027fap-56), 0x1.4f8714574779dp-57,
    0x1.1d4ce385cfbe7p-57, 0x1.de975927718f4p-57, 0x1.efb9d089cf90ap-56,
    (-0x1.e6eccdbac6240p-56), 0x1.5bb94c1280f7bp-56, (-0x1.272814b48d397p-56),
    0x1.974d9ecc7b282p-56, 0x1.13c37a49fc23fp-56, 0x1.7ce9c2884efd7p-58,
    (-0x1.2af9dd21de84ap-57), 0x1.bb917c4158b96p-57, 0x1.022e75a2b3bd3p-63,
    0x1.7a9c418d84860p-58, (-0x1.ee701984198e3p-57), 0x1.9fe6887659803p-58,
    0x1.ca52b9f12ee9fp-58, 0x1.461660fcb5729p-58, 0x1.f4461b11bb8d0p-59,
    (-0x1.9c964e9e682b6p-59), 0x1.9dab517f95157p-59, (-0x1.468188dd06880p-62),
    (-0x1.9840c4a58049ap-58), 0x1.9c82970a55bd0p-58, (-0x1.93d70ff724b5ep-59),
    (-0x1.646c2cfed751bp-59), (-0x1.4260ba51ac614p-59), (-0x1.ddf9277cf26dep-59),
    0x1.dbe125fc21ee6p-60, (-0x1.cf9a702ff830ep-62), 0x0p+0,
    (-0x1.7c1ff82f44adcp-61), (-0x1.cba91dae29988p-60), (-0x1.efa68b88a0ae7p-64),
    (-0x1.8ee324ff21847p-60), (-0x1.ad73136c614cbp-61), 0x1.89deb91c31ac3p-59,
    (-0x1.4b3077c27b8b4p-58), 0x1.3e721192791a3p-61, (-0x1.336be543a9993p-60),
    (-0x1.b4f1b618fe611p-58), (-0x1.f7c9478921bd7p-58), (-0x1.eaebfe80f652ap-57),
    0x1.9a5aba13ee9c5p-57, 0x1.069536106578ep-59, 0x1.8bdb8834b12ffp-58,
    (-0x1.a8e7cd17ca46cp-59), 0x1.bd6b99f1a542fp-59, (-0x1.95599ba086466p-57),
    0x1.1d8e54f6b70ddp-58, 0x1.7e0e0aa2ae35fp-57, (-0x1.a829cc1512df9p-57),
    0x1.c4b8ca0f6d7a5p-58, 0x1.cbfe95c59cb8ap-57, 0x1.a5e946d48367ep-57,
    0x1.3ee2f0e46be64p-57, 0x1.05224944efcc0p-57, (-0x1.1c2140fbe02dbp-56),
    0x1.8de37e9a6110bp-57, 0x1.31cb428346b53p-57, 0x1.5e5cb8e201f79p-56,
    (-0x1.3ca60d9385e27p-57), (-0x1.2a4397d8912bap-56), (-0x1.d9b1af94a58afp-56),
    (-0x1.ebb9c6fdca778p-57), 0x1.1f6461fef16c6p-60, 0x1.33d26f4e1b0f4p-56,
    (-0x1.796e600ac247ep-58), 0x1.40affe91b5623p-57, (-0x1.fb8ab87be1f10p-57),
    (-0x1.fc48be2eed2eap-57), (-0x1.bcddc6a14952cp-58), (-0x1.995570a48531ep-57),
    0x1.234b98e55c106p-60, 0x1.d1fbd4b1ef502p-58, (-0x1.57b3fe5f0580cp-58),
    (-0x1.0eb98c5df43eap-59), (-0x1.c50ac20cc5c35p-62), 0x1.8ee3853c07c26p-57,
    (-0x1.2caa8abe0356cp-56), (-0x1.23b4c757e83b3p-57), (-0x1.3186b98f3f398p-57),
    0x1.7a98db9e82e5fp-60, (-0x1.16c968c39b3b9p-56), (-0x1.ad19c6c0abab0p-58),
    0x1.f7e57c5ce4173p-55, 0x1.7991c3b3db30ap-55, 0x1.70576a5aa8a36p-57,
    (-0x1.b8e77375e280ep-56), 0x1.9995c81c00c2dp-55, (-0x1.a4e55c102570cp-57),
    (-0x1.369b2e485c22cp-58), (-0x1.e6fd924c782c9p-55), (-0x1.2ebcc9e150d9dp-57),
    (-0x1.c8f4955b19b6dp-55)
};
WSP_TABLE double wsp_exp2_hi[64] = {
    0x1.0000000000000p+0, 0x1.02c9a3e778061p+0, 0x1.059b0d3158574p+0,
    0x1.0874518759bc8p+0, 0x1.0b5586cf9890fp+0, 0x1.0e3ec32d3d1a2p+0,
    0x1.11301d0125b51p+0, 0x1.1429aaea92de0p+0, 0x1.172b83c7d517bp+0,
    0x1.1a35beb6fcb75p+0, 0x1.1d4873168b9aap+0, 0x1.2063b88628cd6p+0,
    0x1.2387a6e756238p+0, 0x1.26b4565e27cddp+0, 0x1.29e9df51fdee1p+0,
    0x1.2d285a6e4030bp+0, 0x1.306fe0a31b715p+0, 0x1.33c08b26416ffp+0,
    0x1.371a7373aa9cbp+0, 0x1.3a7db34e59ff7p+0, 0x1.3dea64c123422p+0,
    0x1.4160a21f72e2ap+0, 0x1.44e086061892dp+0, 0x1.486a2b5c13cd0p+0,
    0x1.4bfdad5362a27p+0, 0x1.4f9b2769d2ca7p+0, 0x1.5342b569d4f82p+0,
    0x1.56f4736b527dap+0, 0x1.5ab07dd485429p+0, 0x1.5e76f15ad2148p+0,
    0x1.6247eb03a5585p+0, 0x1.6623882552225p+0, 0x1.6a09e667f3bcdp+0,
    0x1.6dfb23c651a2fp+0, 0x1.71f75e8ec5f74p+0, 0x1.75feb564267c9p+0,
    0x1.7a11473eb0187p+0, 0x1.7e2f336cf4e62p+0, 0x1.82589994cce13p+0,
    0x1.868d99b4492edp+0, 0x1.8ace5422aa0dbp+0, 0x1.8f1ae99157736p+0,
    0x1.93737b0cdc5e5p+0, 0x1.97d829fde4e50p+0, 0x1.9c49182a3f090p+0,
    0x1.a0c667b5de565p+0, 0x1.a5503b23e255dp+0, 0x1.a9e6b5579fdbfp+0,
    0x1.ae89f995ad3adp+0, 0x1.b33a2b84f15fbp+0, 0x1.b7f76f2fb5e47p+0,
    0x1.bcc1e904bc1d2p+0, 0x1.c199bdd85529cp+0, 0x1.c67f12e57d14bp+0,
    0x1.cb720dcef9069p+0, 0x1.d072d4a07897cp+0, 0x1.d5818dcfba487p+0,
    0x1.da9e603db3285p+0, 0x1.dfc97337b9b5fp+0, 0x1.e502ee78b3ff6p+0,
    0x1.ea4afa2a490dap+0, 0x1.efa1bee615a27p+0, 0x1.f50765b6e4540p+0,
    0x1.fa7c1819e90d8p+0
};
WSP_TABLE double wsp_exp2_lo[64] = {
    0x0p+0, (-0x1.19083535b085dp-56), 0x1.d73e2a475b465p-55,
    0x1.186be4bb284ffp-57, 0x1.8a62e4adc610bp-54, 0x1.03a1727c57b53p-59,
    (-0x1.6c51039449b3ap-54), (-0x1.32fbf9af1369ep-54), (-0x1.19041b9d78a76p-55),
    0x1.e5b4c7b4968e4p-55, 0x1.e016e00a2643cp-54, 0x1.dc775814a8495p-55,
    0x1.9b07eb6c70573p-54, 0x1.2bd339940e9d9p-55, 0x1.612e8afad1255p-55,
    0x1.0024754db41d5p-54, 0x1.6f46ad23182e4p-55, 0x1.32721843659a6p-54,
    (-0x1.63aeabf42eae2p-54), (-0x1.5e436d661f5e3p-56), 0x1.ada0911f09ebcp-55,
    (-0x1.ef3691c309278p-58), 0x1.89b7a04ef80d0p-59, 0x1.3c1a3b69062f0p-56,
    0x1.d4397afec42e2p-56, (-0x1.4b309d25957e3p-54), (-0x1.07abe1db13cadp-55),
    0x1.9bb2c011d93adp-54, 0x1.6324c054647adp-54, 0x1.ba6f93080e65ep-54,
    (-0x1.383c17e40b497p-54), (-0x1.bb60987591c34p-54), (-0x1.bdd3413b26456p-54),
    (-0x1.bbe3a683c88abp-57), (-0x1.16e4786887a99p-55), (-0x1.0245957316dd3p-54),
    (-0x1.41577ee04992fp-55), 0x1.05d02ba15797ep-56, (-0x1.d4c1dd41532d8p-54),
    (-0x1.fc6f89bd4f6bap-54), 0x1.6e9f156864b27p-54, 0x1.5cc13a2e3976cp-55,
    (-0x1.75fc781b57ebcp-57), (-0x1.d185b7c1b85d1p-54), 0x1.c7c46b071f2bep-56,
    (-0x1.359495d1cd533p-54), (-0x1.d2f6edb8d41e1p-54), 0x1.0fac90ef7fd31p-54,
    0x1.7a1cd345dcc81p-54, (-0x1.2805e3084d708p-57), (-0x1.5584f7e54ac3bp-56),
    0x1.23dd07a2d9e84p-55, 0x1.11065895048ddp-55, 0x1.2884dff483cadp-54,
    0x1.503cbd1e949dbp-56, (-0x1.cbc3743797a9cp-54), 0x1.2ed02d75b3707p-55,
    0x1.c2300696db532p-54, (-0x1.1a5cd4f184b5cp-54), 0x1.39e8980a9cc8fp-55,
    (-0x1.e9c23179c2893p-54), 0x1.dc7f486a4b6b0p-54, 0x1.9d3e12dd8a18bp-54,
    0x1.74853f3a5931ep-55
};
// log2(1 + r) = sum_k wsp_log_hi[k-1] r^k (+ wsp_log_lo for the accurate phase)
WSP_TABLE double wsp_log_hi[18] = {
    0x1.71547652b82fep+0, (-0x1.71547652b82fep-1), 0x1.ec709dc3a03fdp-2,
    (-0x1.71547652b82fep-2), 0x1.2776c50ef9bfep-2, (-0x1.ec709dc3a03fdp-3),
    0x1.a61762a7aded9p-3, (-0x1.71547652b82fep-3), 0x1.484b13d7c02a9p-3,
    (-0x1.2776c50ef9bfep-3), 0x1.0c9a84994022dp-3, (-0x1.ec709dc3a03fdp-4),
    0x1.c68f568d31760p-4, (-0x1.a61762a7aded9p-4), 0x1.89f3b1694cffep-4,
    (-0x1.71547652b82fep-4), 0x1.5b9ac9b743f0dp-4, (-0x1.484b13d7c02a9p-4)
};
WSP_TABLE double wsp_log_lo[18] = {
    0x1.777d0ffda0d24p-56, (-0x1.777d0ffda0d24p-57), 0x1.d27f05548af0cp-56,
    (-0x1.777d0ffda0d24p-58), 0x1.e4b29ccc535d4p-56, (-0x1.d27f05548af0cp-57),
    0x1.fb22e490ee2f0p-58, (-0x1.777d0ffda0d24p-59), (-0x1.e55fc724e0a2cp-61),
    (-0x1.e4b29ccc535d4p-57), 0x1.42b91d166906ap-58, (-0x1.d27f05548af0cp-58),
    0x1.ce23c4e96378ep-60, (-0x1.fb22e490ee2f0p-59), (-0x1.2339777c86c81p-61),
    (-0x1.777d0ffda0d24p-60), 0x1.0d0e5e1d8f408p-58, 0x1.e55fc724e0a2cp-62
};
// 2^g = 1 + sum_k wsp_exp_hi[k-1] g^k (+ wsp_exp_lo for the accurate phase)
WSP_TABLE double wsp_exp_hi[14] = {
    0x1.62e42fefa39efp-1, 0x1.ebfbdff82c58fp-3, 0x1.c6b08d704a0c0p-5,
    0x1.3b2ab6fba4e77p-7, 0x1.5d87fe78a6731p-10, 0x1.430912f86c787p-13,
    0x1.ffcbfc588b0c7p-17, 0x1.62c0223a5c824p-20, 0x1.b5253d395e7c4p-24,
    0x1.e4cf5158b8ecap-28, 0x1.e8cac7351bb25p-32, 0x1.c3bd650fc2986p-36,
    0x1.816193166d0f9p-40, 0x1.314964d5878a9p-44
};
WSP_TABLE double wsp_exp_lo[14] = {
    0x1.abc9e3b39803fp-56, (-0x1.5e43a53e44da3p-57), (-0x1.d331627513351p-59),
    0x1.4e65df05a9f75p-62, 0x1.0717f69a514bfp-66, 0x1.bd2c2a261ac8dp-67,
    (-0x1.e53ab8cde09c6p-71), (-0x1.3800cfc92c41ep-79), (-0x1.2dac78d2d8038p-79),
    (-0x1.204bc4d5a312dp-85), (-0x1.f8543350dc6f6p-87), (-0x1.d4a9781e85d12p-92),
    0x1.8a06b0c03dc6cp-94, 0x1.cfc50d89572bbp-98
};
// CONSTANTS END

struct wsp_dd {double h, l;};

WSP_HD wsp_dd wsp_make(double h, double l) {wsp_dd r; r.h = h; r.l = l; return r;}

WSP_HD wsp_dd wsp_two_sum(double a, double b)
{
    double s = a + b;
    double bb = s - a;
    return wsp_make(s, (a - (s - bb)) + (b - bb));
}

WSP_HD wsp_dd wsp_fast_two_sum(double a, double b)  // |a| >= |b| or a == 0
{
    double s = a + b;
    return wsp_make(s, b - (s - a));
}

WSP_HD wsp_dd wsp_add(wsp_dd a, wsp_dd b)
{
    wsp_dd s = wsp_two_sum(a.h, b.h);
    return wsp_fast_two_sum(s.h, s.l + (a.l + b.l));
}

WSP_HD wsp_dd wsp_mul(wsp_dd a, wsp_dd b)
{
    double p = a.h * b.h;
    double e = WSP_FMA(a.h, b.h, -p);
    e = WSP_FMA(a.h, b.l, e);
    e = WSP_FMA(a.l, b.h, e);
    return wsp_fast_two_sum(p, e);
}

WSP_HD wsp_dd wsp_mul_d(wsp_dd a, double b)
{
    double p = a.h * b;
    double e = WSP_FMA(a.h, b, -p);
    e = WSP_FMA(a.l, b, e);
    return wsp_fast_two_sum(p, e);
}

WSP_HD double wsp_pow2(int k) {return WSP_FROM_DBITS((long long)(k + 1023) << 52);}

// The odd integer and the exponent of a positive finite double: v = odd 2^exp.
struct wsp_odd {long long odd; int exp;};

WSP_HD wsp_odd wsp_odd_part(double v)
{
    long long b = WSP_DBITS(v);
    int eb = (int)((b >> 52) & 0x7ff);
    long long m = b & 0xfffffffffffffLL;
    int e;
    if (eb == 0) {
        e = -1074;
    } else {
        m |= 0x10000000000000LL;
        e = eb - 1075;
    }
    while ((m & 1) == 0) {
        m >>= 1;
        e += 1;
    }
    wsp_odd r;
    r.odd = m;
    r.exp = e;
    return r;
}

// base^n with n >= 1, or -1 when it exceeds 2^26.
WSP_HD long long wsp_small_power(long long base, long long n)
{
    long long v = 1;
    for (long long i = 0; i < n; ++i) {
        v *= base;
        if (v > (1LL << 26)) return -1;
    }
    return v;
}

// Is ax^y exactly the double m (a binary32 rounding midpoint)?  ax > 0
// finite, y finite.  A midpoint's odd part has 25 bits, so x's odd part is
// not 1, y > 0, and y = n (an integer) or n / 2^K with K <= 4.
WSP_HD bool wsp_exact_midpoint(float ax, float y, double m)
{
    if (!(y > 0.0f)) return false;
    wsp_odd M = wsp_odd_part(m);
    wsp_odd X = wsp_odd_part((double)ax);
    wsp_odd Y = wsp_odd_part((double)y);
    if (X.odd == 1) return false;
    if (Y.exp >= 0) {
        if (Y.exp > 6 || Y.odd > 64) return false;
        long long n = Y.odd << Y.exp;
        if (n > 64) return false;
        long long v = wsp_small_power(X.odd, n);
        return v == M.odd && (long long)X.exp * n == (long long)M.exp;
    }
    int K = -Y.exp;
    if (K > 4) return false;
    long long R = 1LL << K;
    // the R-th root of x's odd part, if it is a perfect R-th power
    long long s = 0;
    for (long long c = 3; c <= 4096; c += 2) {
        long long p = wsp_small_power(c, R);
        if (p < 0 || p > X.odd) break;
        if (p == X.odd) {
            s = c;
            break;
        }
    }
    if (s == 0 || X.exp % R != 0) return false;
    long long v = wsp_small_power(s, Y.odd);
    return v == M.odd && (long long)(X.exp / R) * Y.odd == (long long)M.exp;
}

// The binary32 neighbours of f0 (a non-negative word, possibly +inf) as
// binary64 midpoints.  2^128 stands for the word after the largest finite.
struct wsp_mid {double dn, up;};

WSP_HD wsp_mid wsp_midpoints(int u)
{
    const double big = 0x1p128;
    double f = u >= 0x7f800000 ? big : (double)WSP_FROM_FBITS(u);
    double up = u >= 0x7f7fffff ? big : (double)WSP_FROM_FBITS(u + 1);
    double dn = u == 0 ? -1.0 : (double)WSP_FROM_FBITS(u - 1);
    wsp_mid r;
    r.dn = u == 0 ? -1.0 : 0.5 * (f + dn);
    r.up = u >= 0x7f800000 ? 0x1p1023 * 2.0 : 0.5 * (f + up);
    return r;
}

// |x|^y for ax positive finite, ax != 1, y finite and nonzero.
WSP_HD float wsp_positive(float ax, float y)
{
    // |x| = m 2^e, m in [0.75, 1.5)
    long long bx = WSP_DBITS((double)ax);
    int e = (int)((bx >> 52) & 0x7ff) - 1023;
    double m = WSP_FROM_DBITS((bx & 0xfffffffffffffLL) | 0x3ff0000000000000LL);
    if (m >= 1.5) {
        m *= 0.5;
        e += 1;
    }
    int idx = (int)WSP_RINT((m - 1.0) * 128.0) - WSP_LOG_LO;
    double r = WSP_FMA(m, wsp_invc[idx], -1.0);  // exact
    double yd = (double)y;

    // fast phase: log2 x = e + logc + r c1 + r^2 (c2 + c3 r + ...)
    double q = wsp_log_hi[WSP_FAST_LOG_TERMS - 1];
    for (int k = WSP_FAST_LOG_TERMS - 2; k >= 1; --k) q = WSP_FMA(q, r, wsp_log_hi[k]);
    q = q * r * r;
    double p = r * wsp_log_hi[0];
    double pe = WSP_FMA(r, wsp_log_hi[0], -p);
    wsp_dd s = wsp_two_sum((double)e, wsp_logc_hi[idx]);
    wsp_dd t = wsp_two_sum(s.h, p);
    double lo = t.l + s.l + pe + wsp_logc_lo[idx] + r * wsp_log_lo[0] + q;
    wsp_dd L = wsp_fast_two_sum(t.h, lo);
    double th = yd * L.h;
    double tl = WSP_FMA(yd, L.h, -th) + yd * L.l;
    if (th > 129.0) return WSP_FROM_FBITS(0x7f800000);
    if (th < -151.0) return 0.0f;
    double nd = WSP_RINT(th * 64.0);
    int n = (int)nd;
    int j = n & 63;
    int k = (n - j) / 64;
    double g = (th - nd * 0.015625) + tl;
    double a = wsp_exp_hi[WSP_FAST_EXP_TERMS - 1];
    for (int i = WSP_FAST_EXP_TERMS - 2; i >= 0; --i) a = WSP_FMA(a, g, wsp_exp_hi[i]);
    a = a * g;
    double z = (wsp_exp2_hi[j] + (wsp_exp2_lo[j] + wsp_exp2_hi[j] * a)) * wsp_pow2(k);
    float f0 = (float)z;
    int u = WSP_FBITS(f0);
    wsp_mid mid = wsp_midpoints(u);
    if (z * (1.0 - 0x1p-54) > mid.dn && z * (1.0 + 0x1p-54) < mid.up) return f0;

    // accurate phase, in double-double
    wsp_dd acc = wsp_make(wsp_log_hi[WSP_ACCURATE_LOG_TERMS - 1], wsp_log_lo[WSP_ACCURATE_LOG_TERMS - 1]);
    for (int i = WSP_ACCURATE_LOG_TERMS - 2; i >= 0; --i)
        acc = wsp_add(wsp_mul_d(acc, r), wsp_make(wsp_log_hi[i], wsp_log_lo[i]));
    acc = wsp_mul_d(acc, r);
    wsp_dd Lx = wsp_add(wsp_add(wsp_make((double)e, 0.0),
                                wsp_make(wsp_logc_hi[idx], wsp_logc_lo[idx])), acc);
    wsp_dd T = wsp_mul_d(Lx, yd);
    nd = WSP_RINT(T.h * 64.0);
    n = (int)nd;
    j = n & 63;
    k = (n - j) / 64;
    wsp_dd G = wsp_fast_two_sum(T.h - nd * 0.015625, T.l);
    wsp_dd b = wsp_make(wsp_exp_hi[WSP_ACCURATE_EXP_TERMS - 1], wsp_exp_lo[WSP_ACCURATE_EXP_TERMS - 1]);
    for (int i = WSP_ACCURATE_EXP_TERMS - 2; i >= 0; --i)
        b = wsp_add(wsp_mul(b, G), wsp_make(wsp_exp_hi[i], wsp_exp_lo[i]));
    b = wsp_mul(b, G);
    wsp_dd E = wsp_make(wsp_exp2_hi[j], wsp_exp2_lo[j]);
    wsp_dd Z = wsp_add(E, wsp_mul(E, b));
    double scale = wsp_pow2(k);
    double vh = Z.h * scale, vl = Z.l * scale;
    f0 = (float)vh;
    u = WSP_FBITS(f0);
    mid = wsp_midpoints(u);
    double eps = vh * 0x1p-93;
    double d_up = (vh - mid.up) + vl;
    double d_dn = (vh - mid.dn) + vl;
    if (d_up > eps) return WSP_FROM_FBITS(u + 1);
    if (d_dn < -eps) return WSP_FROM_FBITS(u - 1);
    if (d_up >= -eps) {  // at the upper midpoint
        if (wsp_exact_midpoint(ax, y, mid.up)) return WSP_FROM_FBITS((u & 1) ? u + 1 : u);
        return WSP_FROM_FBITS(d_up > 0.0 ? u + 1 : u);
    }
    if (d_dn <= eps) {  // at the lower midpoint
        if (wsp_exact_midpoint(ax, y, mid.dn)) return WSP_FROM_FBITS((u & 1) ? u - 1 : u);
        return WSP_FROM_FBITS(d_dn < 0.0 ? u - 1 : u);
    }
    return f0;
}

WSP_HD bool wsp_is_nan(float v) {return (WSP_FBITS(v) & 0x7fffffff) > 0x7f800000;}

WSP_HD float woof_strict_powf(float x, float y)
{
    const float inf = WSP_FROM_FBITS(0x7f800000);
    if (y == 0.0f || x == 1.0f) return 1.0f;
    if (wsp_is_nan(x) || wsp_is_nan(y)) return x + y;
    float ax = x < 0.0f ? -x : x;
    float ay = y < 0.0f ? -y : y;
    bool y_int = ay >= 0x1p24f || (float)(long long)y == y;
    bool y_odd = y_int && ay < 0x1p24f && (((long long)y) & 1) != 0;
    if (ay == inf) {
        if (ax == 1.0f) return 1.0f;
        return (ax < 1.0f) == (y > 0.0f) ? 0.0f : inf;
    }
    if (ax == 0.0f) {  // the sign of zero is the sign bit
        bool neg = WSP_FBITS(x) < 0;
        if (y < 0.0f) return (y_odd && neg) ? -inf : inf;
        return y_odd ? x : 0.0f;
    }
    if (ax == inf) {
        float v = y < 0.0f ? 0.0f : inf;
        return (x < 0.0f && y_odd) ? -v : v;
    }
    if (x < 0.0f && !y_int) return WSP_FROM_FBITS(0x7fc00000);
    if (ax == 1.0f) return (x < 0.0f && y_odd) ? -1.0f : 1.0f;
    float v = wsp_positive(ax, y);
    return (x < 0.0f && y_odd) ? -v : v;
}

#endif

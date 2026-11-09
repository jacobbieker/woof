// The df64 libm (f64_df64.h) as CUDA's binary64 math names, for the fast
// numerics' prelude only.  The same signatures the vendored binary64 libm
// (woof/backend/translate/vendor_libm64.cuh) and CORE-MATH's cbrt give a
// strict unit, so overload resolution in a translated unit is unchanged.
static inline f64 exp(f64 a) { return f64::bits(df_exp_word(a.b)); }
static inline f64 expm1(f64 a) { return f64::bits(df_expm1_word(a.b)); }
static inline f64 log(f64 a) { return f64::bits(df_log_word(a.b)); }
static inline f64 log2(f64 a) { return f64::bits(df_log2_word(a.b)); }
static inline f64 log10(f64 a) { return f64::bits(df_log10_word(a.b)); }
static inline f64 log1p(f64 a) { return f64::bits(df_log1p_word(a.b)); }
static inline f64 pow(f64 a, f64 b) { return f64::bits(df_pow_word(a.b, b.b)); }
static inline f64 sin(f64 a) { return f64::bits(df_sin_word(a.b)); }
static inline f64 cos(f64 a) { return f64::bits(df_cos_word(a.b)); }
static inline f64 tan(f64 a) { return f64::bits(df_tan_word(a.b)); }
static inline f64 atan(f64 a) { return f64::bits(df_atan_word(a.b)); }
static inline f64 atan2(f64 y, f64 x) { return f64::bits(df_atan2_word(y.b, x.b)); }
static inline f64 tanh(f64 a) { return f64::bits(df_tanh_word(a.b)); }
static inline f64 hypot(f64 a, f64 b) { return f64::bits(df_hypot_word(a.b, b.b)); }
static inline f64 cbrt(f64 a) { return f64::bits(df_cbrt_word(a.b)); }

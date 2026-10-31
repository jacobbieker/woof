/* The C library names gfortran's REAL(4) intrinsics lower to, bound to
 * woof's ports (libm_ports.cpp).  Linked into the oracle executable, these
 * definitions take precedence over the platform libm for every call the
 * Fortran objects make; build.sh checks the executable defines each one.
 * No <math.h>: the definitions must not meet the platform's prototypes. */
float woof_expf(float), woof_logf(float), woof_log10f(float),
    woof_powf(float, float), woof_sinf(float), woof_cosf(float),
    woof_acosf(float), woof_sinhf(float);

float expf(float x) { return woof_expf(x); }
float logf(float x) { return woof_logf(x); }
float log10f(float x) { return woof_log10f(x); }
float powf(float x, float y) { return woof_powf(x, y); }
float sinf(float x) { return woof_sinf(x); }
float cosf(float x) { return woof_cosf(x); }
float acosf(float x) { return woof_acosf(x); }
float sinhf(float x) { return woof_sinhf(x); }

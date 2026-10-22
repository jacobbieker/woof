/* Batch entry over the verbatim upstream CORE-MATH cr_cbrt (upstream/cbrt.c),
 * for sweep_cbrt.py.  Build (from this directory):
 *
 *   gcc -O2 -ffp-contract=off -fopenmp -shared -fPIC \
 *       cbrt_batch.c upstream/cbrt.c -o cbrt_batch.so
 */
#include <stddef.h>

double cr_cbrt(double x);

void cbrt_batch(const double *x, double *y, long long n)
{
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < n; ++i)
        y[i] = cr_cbrt(x[i]);
}

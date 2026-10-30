// woof's spectral discrete-ordinate solver (one thread per column x wavelength).
//
// CLEAN-ROOM IMPLEMENTATION.  Written from the published method only; no
// DISORT, cDISORT, twostr or libRadtran source text was read.  libRadtran's
// uvspec is used as a black-box oracle in tests (tools/uvspec_oracle).
//   * Stamnes, Tsay, Wiscombe & Jayaweera (1988), Appl. Opt. 27, 2502:
//     Fourier decomposition in azimuth, double-Gauss quadrature, the beam
//     particular solution, the multi-layer boundary-value problem in scaled
//     exponentials, intensities at user angles by integrating the source
//     function along the line of sight.
//   * Stamnes & Swanson (1981), J. Atmos. Sci. 38, 387: the eigenvalue
//     problem reduced to order N, (alpha-beta)(alpha+beta) v = k^2 v.  Here it
//     is symmetrised (the quadrature weights and cosines are a diagonal
//     similarity) and solved by Cholesky + cyclic Jacobi rotations.
//   * Wiscombe (1977), J. Atmos. Sci. 34, 1408: delta-M scaling.
//   * Dahlback & Stamnes (1991), Planet. Space Sci. 39, 671: pseudo-
//     spherical beam (slant optical depth through spherical shells; inside
//     a layer the beam decays with the layer's average secant).
//   * Nakajima & Tanaka (1988), JQSRT 40, 51: TMS correction of radiances.
//   * Kylling, Stamnes & Tsay (1995), J. Atmos. Chem. 21, 115: the two-stream
//     solver as the order-1 discrete-ordinate problem with mu1 = 1/sqrt(3).
// Conventions: tau grows downward; mu > 0 is upward-travelling radiation;
// I(tau, mu, phi) = sum_m I^m cos(m (phi - phi0)); the beam carries
// irradiance fbeam normal to itself, so the direct horizontal irradiance at
// the top is umu0 * fbeam.
//
// Arithmetic: binary64 throughout; exp/log/cos through woof's word-exact
// ports (glibc_flt64.cuh), sqrt and division IEEE; compiled without FMA
// contraction so CUDA and the Metal translation produce the same words.

#define RT_MAXN 16
#define RT_MAXSTR 32
#define RT_MAXUMU 32
#define RT_PI 3.14159265358979323846

#define RT_DELTAM 1
#define RT_SPHERICAL 2
#define RT_TWOSTREAM 4
#define RT_TMS 8

#define RT_DITHER 1e-9

__device__ __forceinline__ double rt_exp(double x) { return glibc_exp(x); }
__device__ __forceinline__ double rt_abs(double x) { return x < 0.0 ? -x : x; }

// Gauss-Legendre nodes/weights on (0,1), ascending (Newton on P_n).
__device__ void rt_double_gauss(int n, double *mu, double *w)
{
    for (int i = 0; i < n; ++i) {
        double z = uw_cos(RT_PI * ((double)i + 0.75) / ((double)n + 0.5));
        double dp = 1.0;
        for (int it = 0; it < 100; ++it) {
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; ++k) {
                double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / (double)k;
                p0 = p1;
                p1 = p2;
            }
            dp = (double)n * (z * p1 - p0) / (z * z - 1.0);
            double dz = p1 / dp;
            z = z - dz;
            if (rt_abs(dz) <= 1e-16) break;
        }
        {
            double p0 = 1.0, p1 = z;
            for (int k = 2; k <= n; ++k) {
                double p2 = ((2.0 * k - 1.0) * z * p1 - (k - 1.0) * p0) / (double)k;
                p0 = p1;
                p1 = p2;
            }
            dp = (double)n * (z * p1 - p0) / (z * z - 1.0);
        }
        // roots come in descending z; store ascending mu
        int j = n - 1 - i;
        mu[j] = 0.5 * (1.0 + z);
        w[j] = 1.0 / ((1.0 - z * z) * dp * dp);   // (2/((1-z^2)dp^2)) / 2
    }
}

// Lambda_l^m(x) = sqrt((l-m)!/(l+m)!) P_l^m(x) for l = 0..lmax (0 for l < m).
__device__ void rt_lambda(int m, int lmax, double x, double *out)
{
    for (int l = 0; l <= lmax; ++l) out[l] = 0.0;
    if (m > lmax) return;
    double s = 1.0 - x * x;
    s = s > 0.0 ? sqrt(s) : 0.0;
    double pmm = 1.0;
    for (int k = 1; k <= m; ++k) pmm = pmm * sqrt((2.0 * k - 1.0) / (2.0 * k)) * s;
    out[m] = pmm;
    if (m + 1 <= lmax) out[m + 1] = sqrt(2.0 * m + 1.0) * x * pmm;
    for (int l = m + 2; l <= lmax; ++l) {
        double a = (2.0 * l - 1.0) * x * out[l - 1];
        double b = sqrt((l - 1.0) * (l - 1.0) - (double)m * m) * out[l - 2];
        out[l] = (a - b) / sqrt((double)l * l - (double)m * m);
    }
}

// Cyclic Jacobi eigen-decomposition of a symmetric n x n matrix a (row-major,
// overwritten).  Eigenvalues in d, eigenvectors in the columns of v.
__device__ void rt_jacobi(int n, double *a, double *d, double *v)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) v[i * n + j] = (i == j) ? 1.0 : 0.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0, dia = 0.0;
        for (int i = 0; i < n; ++i) {
            dia += a[i * n + i] * a[i * n + i];
            for (int j = i + 1; j < n; ++j) off += a[i * n + j] * a[i * n + j];
        }
        if (off <= 1e-34 * dia || off == 0.0) break;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double apq = a[p * n + q];
                if (apq == 0.0) continue;
                double app = a[p * n + p], aqq = a[q * n + q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = 1.0 / (rt_abs(theta) + sqrt(theta * theta + 1.0));
                if (theta < 0.0) t = -t;
                double c = 1.0 / sqrt(t * t + 1.0);
                double s = t * c;
                for (int k = 0; k < n; ++k) {           // columns p, q
                    double akp = a[k * n + p], akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {           // rows p, q
                    double apk = a[p * n + k], aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for (int k = 0; k < n; ++k) {
                    double vkp = v[k * n + p], vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) d[i] = a[i * n + i];
}

// Dense solve a x = b (n x n row-major, partial pivoting); b is overwritten.
__device__ void rt_dense_solve(int n, double *a, double *b)
{
    for (int k = 0; k < n; ++k) {
        int p = k;
        double big = rt_abs(a[k * n + k]);
        for (int r = k + 1; r < n; ++r) {
            double v = rt_abs(a[r * n + k]);
            if (v > big) { big = v; p = r; }
        }
        if (p != k) {
            for (int c = k; c < n; ++c) {
                double t = a[k * n + c]; a[k * n + c] = a[p * n + c]; a[p * n + c] = t;
            }
            double t = b[k]; b[k] = b[p]; b[p] = t;
        }
        double piv = a[k * n + k];
        for (int r = k + 1; r < n; ++r) {
            double l = a[r * n + k] / piv;
            if (l == 0.0) continue;
            for (int c = k + 1; c < n; ++c) a[r * n + c] = a[r * n + c] - l * a[k * n + c];
            b[r] = b[r] - l * b[k];
        }
    }
    for (int k = n - 1; k >= 0; --k) {
        double s = b[k];
        for (int c = k + 1; c < n; ++c) s = s - a[k * n + c] * b[c];
        b[k] = s / a[k * n + k];
    }
}

// Banded LU with partial pivoting (column-major band storage, element
// (r, c) at ab[(ml + mu + r - c) + c * ldab], ldab = 2 ml + mu + 1), then
// the solve of one right-hand side.
__device__ __forceinline__ long rt_bidx(int ml, int mu, int ldab, int r, int c)
{
    return (long)(ml + mu + r - c) + (long)c * ldab;
}

__device__ void rt_band_solve(int n, int ml, int mu, double *ab, double *piv, double *b)
{
    int ldab = 2 * ml + mu + 1;
    for (int k = 0; k < n; ++k) {
        int rmax = k + ml < n - 1 ? k + ml : n - 1;
        int cmax = k + ml + mu < n - 1 ? k + ml + mu : n - 1;
        int p = k;
        double big = rt_abs(ab[rt_bidx(ml, mu, ldab, k, k)]);
        for (int r = k + 1; r <= rmax; ++r) {
            double v = rt_abs(ab[rt_bidx(ml, mu, ldab, r, k)]);
            if (v > big) { big = v; p = r; }
        }
        piv[k] = (double)p;
        if (p != k) {
            for (int c = k; c <= cmax; ++c) {
                long i1 = rt_bidx(ml, mu, ldab, k, c), i2 = rt_bidx(ml, mu, ldab, p, c);
                double t = ab[i1]; ab[i1] = ab[i2]; ab[i2] = t;
            }
        }
        double pv = ab[rt_bidx(ml, mu, ldab, k, k)];
        for (int r = k + 1; r <= rmax; ++r) {
            long ir = rt_bidx(ml, mu, ldab, r, k);
            double l = ab[ir] / pv;
            ab[ir] = l;
            if (l == 0.0) continue;
            for (int c = k + 1; c <= cmax; ++c) {
                long irc = rt_bidx(ml, mu, ldab, r, c);
                ab[irc] = ab[irc] - l * ab[rt_bidx(ml, mu, ldab, k, c)];
            }
        }
    }
    // forward: apply interchanges and L in elimination order
    for (int k = 0; k < n; ++k) {
        int p = (int)piv[k];
        if (p != k) { double t = b[k]; b[k] = b[p]; b[p] = t; }
        int rmax = k + ml < n - 1 ? k + ml : n - 1;
        for (int r = k + 1; r <= rmax; ++r)
            b[r] = b[r] - ab[rt_bidx(ml, mu, ldab, r, k)] * b[k];
    }
    for (int k = n - 1; k >= 0; --k) {
        int cmax = k + ml + mu < n - 1 ? k + ml + mu : n - 1;
        double s = b[k];
        for (int c = k + 1; c <= cmax; ++c) s = s - ab[rt_bidx(ml, mu, ldab, k, c)] * b[c];
        b[k] = s / ab[rt_bidx(ml, mu, ldab, k, k)];
    }
}

// int_0^d e^{-k(d-t)} e^{-t/u} dt/u  (k, u > 0)
__device__ double rt_grow(double k, double u, double d)
{
    double x = k * u - 1.0;
    double ku = k * u;
    double tol = 1e-6 * (ku > 1.0 ? ku : 1.0);
    if (rt_abs(x) < tol) return d / u * rt_exp(-d / u);
    return (rt_exp(-d / u) - rt_exp(-k * d)) / x;
}

// int_0^d e^{-k t} e^{-t/u} dt/u
__device__ __forceinline__ double rt_decay(double k, double u, double d)
{
    return (1.0 - rt_exp(-(k + 1.0 / u) * d)) / (1.0 + k * u);
}

// Workspace per problem, in doubles.
__device__ __forceinline__ long rt_work_size(int nlyr, int N, int nstr)
{
    long n = 2L * N * nlyr;
    long ldab = 9L * N - 2;
    return 6L * nlyr + 2L * (nlyr + 1) + (long)nlyr * nstr
         + (long)nlyr * 2 * N * N + (long)nlyr * N + (long)nlyr * 2 * N
         + ldab * n + 2 * n + 2L * N;
}

extern "C" __global__ void radtran_work_size(int nlyr, int N, int nstr, long long *out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = (long long)rt_work_size(nlyr, N, nstr);
}

extern "C" __global__ void radtran_disort(
    int prob0, int nprob, int nlyr, int nmom, int nstr, int flags,
    const double *dtau_in, const double *ssa_in, const double *pmom_in,
    const double *umu0_in, const double *fbeam_in, const double *albedo_in,
    const double *zlev_in, double radius,
    int nout, const int *outlev,
    int numu, const double *umu_in, int nphi, const double *phi_in,
    const double *ptrue_in,
    double *work, long long wstride,
    double *flux_out, double *rad_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int prob = prob0 + tid;
    if (prob >= nprob) return;
    const int twostream = (flags & RT_TWOSTREAM) != 0;
    if (twostream) nstr = 2;
    const int N = nstr / 2;
    const int L = nlyr;
    const double umu0 = umu0_in[prob];
    const double fbeam = fbeam_in[prob];
    const double albedo = albedo_in[prob];
    const double *dtau = dtau_in + (long)prob * L;
    const double *ssa = ssa_in + (long)prob * L;
    const double *pmom = pmom_in + (long)prob * L * (nmom + 1);

    double *wk = work + (long)tid * wstride;
    double *dtaus = wk;              // L   scaled optical depth
    double *ssas = dtaus + L;        // L   scaled single-scattering albedo
    double *fz = ssas + L;           // L   delta-M truncation fraction
    double *lam = fz + L;            // L   in-layer beam secant
    double *dtu = lam + L;           // L   unscaled optical depth
    double *spare = dtu + L;         // L
    double *ch = spare + L;          // L+1 slant optical depth (scaled)
    double *chu = ch + (L + 1);      // L+1 slant optical depth (unscaled)
    double *gm = chu + (L + 1);      // L*nstr scaled moments * (2l+1) * ssas / 2
    double *Gv = gm + (long)L * nstr; // L*2N*N
    double *Kv = Gv + (long)L * 2 * N * N;
    double *Zv = Kv + (long)L * N;
    double *ab = Zv + (long)L * 2 * N;
    const int nun = 2 * N * L;
    const int ml = 3 * N - 1, mub = 3 * N - 1;
    const long ldab = 2 * ml + mub + 1;
    double *rhs = ab + ldab * nun;
    double *piv = rhs + nun;

    // ---- delta-M scaling (Wiscombe 1977) ----
    const int deltam = (flags & RT_DELTAM) != 0;
    for (int n = 0; n < L; ++n) {
        double w0 = ssa[n];
        if (w0 > 1.0 - RT_DITHER) w0 = 1.0 - RT_DITHER;
        double f = 0.0;
        if (deltam && twostream) {
            // two-stream: delta-Eddington truncation f = g^2 (Joseph, Wiscombe &
            // Weinman 1976), g the first moment
            double g1 = nmom >= 1 ? pmom[(long)n * (nmom + 1) + 1] : 0.0;
            f = g1 * g1;
        } else if (deltam && nmom >= nstr) {
            f = pmom[(long)n * (nmom + 1) + nstr];
        }
        dtu[n] = dtau[n];
        double sc = 1.0 - w0 * f;
        dtaus[n] = sc * dtau[n];
        ssas[n] = w0 * (1.0 - f) / sc;
        fz[n] = f;
        for (int l = 0; l < nstr; ++l) {
            double g = l <= nmom ? pmom[(long)n * (nmom + 1) + l] : 0.0;
            g = (g - f) / (1.0 - f);
            gm[(long)n * nstr + l] = (2.0 * l + 1.0) * g * 0.5 * ssas[n];
        }
    }

    // ---- direct beam: slant optical depth to each level ----
    const int spherical = (flags & RT_SPHERICAL) != 0;
    ch[0] = 0.0;
    chu[0] = 0.0;
    if (!spherical) {
        double s = 0.0, su = 0.0;
        for (int n = 0; n < L; ++n) {
            s = s + dtaus[n];
            su = su + dtu[n];
            ch[n + 1] = s / umu0;
            chu[n + 1] = su / umu0;
        }
    } else {
        const double *z = zlev_in + (long)prob * (L + 1);
        double sin2 = 1.0 - umu0 * umu0;
        for (int k = 1; k <= L; ++k) {
            double rk = radius + z[k];
            double p2 = rk * rk * sin2;
            double t = 0.0, tu = 0.0;
            for (int j = 0; j < k; ++j) {
                double r0 = radius + z[j], r1 = radius + z[j + 1];
                double ds = sqrt(r0 * r0 - p2) - sqrt(r1 * r1 - p2);
                double fac = ds / (r0 - r1);
                t = t + dtaus[j] * fac;
                tu = tu + dtu[j] * fac;
            }
            ch[k] = t;
            chu[k] = tu;
        }
    }
    for (int n = 0; n < L; ++n)
        lam[n] = dtaus[n] > 0.0 ? (ch[n + 1] - ch[n]) / dtaus[n] : 1.0 / umu0;

    // ---- quadrature ----
    double mu[RT_MAXN], wq[RT_MAXN];
    if (twostream) {
        mu[0] = 1.0 / sqrt(3.0);
        wq[0] = 1.0;
    } else {
        rt_double_gauss(N, mu, wq);
    }

    // outputs
    double *flux = flux_out + (long)prob * nout * 6;
    double *rad = rad_out + (long)prob * nout * numu * nphi;
    for (int o = 0; o < nout * 6; ++o) flux[o] = 0.0;
    for (int o = 0; o < nout * numu * nphi; ++o) rad[o] = 0.0;

    const int mmax = (numu > 0 && !twostream) ? nstr - 1 : 0;
    double lq[RT_MAXSTR * RT_MAXN];      // Lambda_l^m(mu_i): [l][i]
    double l0[RT_MAXSTR];
    double tmp[RT_MAXSTR];
    double sg[RT_MAXSTR];

    for (int m = 0; m <= mmax; ++m) {
        for (int i = 0; i < N; ++i) {
            rt_lambda(m, nstr - 1, mu[i], tmp);
            for (int l = 0; l < nstr; ++l) lq[l * N + i] = tmp[l];
        }
        rt_lambda(m, nstr - 1, umu0, l0);
        for (int l = 0; l < nstr; ++l) sg[l] = ((l + m) % 2 == 0) ? 1.0 : -1.0;
        const double beamfac = fbeam / (2.0 * RT_PI) * (m == 0 ? 1.0 : 2.0);

        // ---- homogeneous and particular solutions per layer ----
        for (int n = 0; n < L; ++n) {
            const double *gl = gm + (long)n * nstr;
            double dpm[RT_MAXN * RT_MAXN], dmm[RT_MAXN * RT_MAXN];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    double sp = 0.0, sm = 0.0;
                    for (int l = m; l < nstr; ++l) {
                        double t = gl[l] * lq[l * N + i] * lq[l * N + j];
                        sp = sp + t;
                        sm = sm + sg[l] * t;
                    }
                    dpm[i * N + j] = sp;   // D(mu_i, mu_j)
                    dmm[i * N + j] = sm;   // D(mu_i, -mu_j)
                }
            // P2 = -(T2) = I - M^-1/2 W^1/2 (D+ + D-) W^1/2 M^-1/2  (SPD)
            // P1 = -(T1) = I - M^-1/2 W^1/2 (D+ - D-) W^1/2 M^-1/2
            double p2[RT_MAXN * RT_MAXN], p1[RT_MAXN * RT_MAXN];
            double sc[RT_MAXN];
            for (int i = 0; i < N; ++i) sc[i] = sqrt(wq[i] / mu[i]);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    double s = sc[i] * sc[j];
                    double e = (i == j) ? 1.0 / mu[i] : 0.0;
                    p2[i * N + j] = e - s * (dpm[i * N + j] + dmm[i * N + j]);
                    p1[i * N + j] = e - s * (dpm[i * N + j] - dmm[i * N + j]);
                }
            // Cholesky p2 = Lc Lc^T (lower, in place)
            for (int j = 0; j < N; ++j) {
                double s = p2[j * N + j];
                for (int k = 0; k < j; ++k) s = s - p2[j * N + k] * p2[j * N + k];
                double djj = sqrt(s);
                p2[j * N + j] = djj;
                for (int i = j + 1; i < N; ++i) {
                    double t = p2[i * N + j];
                    for (int k = 0; k < j; ++k) t = t - p2[i * N + k] * p2[j * N + k];
                    p2[i * N + j] = t / djj;
                }
                for (int i = 0; i < j; ++i) p2[i * N + j] = 0.0;
            }
            // C = Lc^T P1 Lc
            double cm[RT_MAXN * RT_MAXN], t1[RT_MAXN * RT_MAXN];
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    double s = 0.0;
                    for (int k = j; k < N; ++k) s = s + p1[i * N + k] * p2[k * N + j];
                    t1[i * N + j] = s;                       // P1 Lc
                }
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    double s = 0.0;
                    for (int k = i; k < N; ++k) s = s + p2[k * N + i] * t1[k * N + j];
                    cm[i * N + j] = s;                       // Lc^T (P1 Lc)
                }
            for (int i = 0; i < N; ++i)                      // exact symmetry
                for (int j = i + 1; j < N; ++j) {
                    double a = 0.5 * (cm[i * N + j] + cm[j * N + i]);
                    cm[i * N + j] = a;
                    cm[j * N + i] = a;
                }
            double ev[RT_MAXN], vv[RT_MAXN * RT_MAXN];
            rt_jacobi(N, cm, ev, vv);
            double *Gn = Gv + (long)n * 2 * N * N;
            double *Kn = Kv + (long)n * N;
            for (int j = 0; j < N; ++j) {
                double y[RT_MAXN], v[RT_MAXN], ab2[RT_MAXN];
                // y = Lc^{-T} u
                for (int i = N - 1; i >= 0; --i) {
                    double s = vv[i * N + j];
                    for (int k = i + 1; k < N; ++k) s = s - p2[k * N + i] * y[k];
                    y[i] = s / p2[i * N + i];
                }
                for (int i = 0; i < N; ++i) v[i] = y[i] * sc[i] / wq[i];   // W^-1/2 M^-1/2 y
                double k2 = ev[j];
                double kk = k2 > 0.0 ? sqrt(k2) : 0.0;
                Kn[j] = kk;
                for (int i = 0; i < N; ++i) {
                    double s = 0.0;
                    for (int c = 0; c < N; ++c) s = s + (dpm[i * N + c] + dmm[i * N + c]) * wq[c] * v[c];
                    ab2[i] = (s - v[i]) / mu[i];
                }
                for (int i = 0; i < N; ++i) {
                    double q = ab2[i] / kk;
                    Gn[i * N + j] = 0.5 * (v[i] + q);          // +mu_i (upward)
                    Gn[(N + i) * N + j] = 0.5 * (v[i] - q);    // -mu_i (downward)
                }
            }
            // particular solution for the beam: (1 + lam mu_i) Z_i - sum_j w_j D(mu_i, mu_j) Z_j = X_i
            double am[4 * RT_MAXN * RT_MAXN], zz[2 * RT_MAXN];
            const int n2 = 2 * N;
            for (int i = 0; i < n2; ++i) {
                int ii = i < N ? i : i - N;
                double mui = i < N ? mu[ii] : -mu[ii];
                double x0 = 0.0;
                for (int l = m; l < nstr; ++l) {
                    double li = lq[l * N + ii] * (i < N ? 1.0 : sg[l]);
                    x0 = x0 + gl[l] * sg[l] * l0[l] * li;
                }
                zz[i] = beamfac * x0;
                for (int j = 0; j < n2; ++j) {
                    int jj = j < N ? j : j - N;
                    double d = 0.0;
                    for (int l = m; l < nstr; ++l) {
                        double li = lq[l * N + ii] * (i < N ? 1.0 : sg[l]);
                        double lj = lq[l * N + jj] * (j < N ? 1.0 : sg[l]);
                        d = d + gl[l] * li * lj;
                    }
                    am[i * n2 + j] = -wq[jj] * d + (i == j ? 1.0 + lam[n] * mui : 0.0);
                }
            }
            rt_dense_solve(n2, am, zz);
            double *Zn = Zv + (long)n * 2 * N;
            for (int i = 0; i < n2; ++i) Zn[i] = zz[i];
        }

        // ---- boundary-value problem (banded) ----
        for (long i = 0; i < ldab * nun; ++i) ab[i] = 0.0;
        int row = 0;
        for (int n = 0; n < L; ++n) {
            const double *Gn = Gv + (long)n * 2 * N * N;
            const double *Kn = Kv + (long)n * N;
            const double *Zn = Zv + (long)n * 2 * N;
            double e[RT_MAXN];
            for (int j = 0; j < N; ++j) e[j] = rt_exp(-Kn[j] * dtaus[n]);
            const double att = rt_exp(-ch[n]), attb = rt_exp(-ch[n + 1]);
            const int c0 = 2 * N * n;
            if (n == 0) {
                // I-(top) = 0
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        ab[rt_bidx(ml, mub, ldab, row + i, c0 + j)] = Gn[(N + i) * N + j];
                        ab[rt_bidx(ml, mub, ldab, row + i, c0 + N + j)] = Gn[i * N + j] * e[j];
                    }
                    rhs[row + i] = -Zn[N + i] * att;
                }
                row += N;
            }
            if (n < L - 1) {
                const double *G1 = Gv + (long)(n + 1) * 2 * N * N;
                const double *K1 = Kv + (long)(n + 1) * N;
                const double *Z1 = Zv + (long)(n + 1) * 2 * N;
                double e1[RT_MAXN];
                for (int j = 0; j < N; ++j) e1[j] = rt_exp(-K1[j] * dtaus[n + 1]);
                const int c1 = c0 + 2 * N;
                for (int half = 0; half < 2; ++half) {
                    // half 0: upward directions, half 1: downward
                    for (int i = 0; i < N; ++i) {
                        int ia = half == 0 ? i : N + i;      // row in Gv for the A term
                        int ib = half == 0 ? N + i : i;      // row in Gv for the B term
                        for (int j = 0; j < N; ++j) {
                            ab[rt_bidx(ml, mub, ldab, row + i, c0 + j)] = Gn[ia * N + j] * e[j];
                            ab[rt_bidx(ml, mub, ldab, row + i, c0 + N + j)] = Gn[ib * N + j];
                            ab[rt_bidx(ml, mub, ldab, row + i, c1 + j)] = -G1[ia * N + j];
                            ab[rt_bidx(ml, mub, ldab, row + i, c1 + N + j)] = -G1[ib * N + j] * e1[j];
                        }
                        rhs[row + i] = Z1[ia] * rt_exp(-ch[n + 1]) - Zn[ia] * attb;
                    }
                    row += N;
                }
            } else {
                // I+(bottom) - 2 A sum_j w_j mu_j I-_j(bottom) = A/pi umu0 F0 e^{-ch}  (m = 0)
                for (int i = 0; i < N; ++i) {
                    for (int j = 0; j < N; ++j) {
                        double ra = 0.0, rb = 0.0;
                        if (m == 0 && albedo != 0.0) {
                            for (int c = 0; c < N; ++c) {
                                double rw = 2.0 * albedo * wq[c] * mu[c];
                                ra = ra + rw * Gn[(N + c) * N + j];
                                rb = rb + rw * Gn[c * N + j];
                            }
                        }
                        ab[rt_bidx(ml, mub, ldab, row + i, c0 + j)] = (Gn[i * N + j] - ra) * e[j];
                        ab[rt_bidx(ml, mub, ldab, row + i, c0 + N + j)] = Gn[(N + i) * N + j] - rb;
                    }
                    double r = -Zn[i] * attb;
                    if (m == 0 && albedo != 0.0) {
                        double s = 0.0;
                        for (int c = 0; c < N; ++c) s = s + 2.0 * albedo * wq[c] * mu[c] * Zn[N + c];
                        r = r + s * attb + albedo / RT_PI * umu0 * fbeam * attb;
                    }
                    rhs[row + i] = r;
                }
                row += N;
            }
        }
        rt_band_solve(nun, ml, mub, ab, piv, rhs);

        // ---- fluxes at the output levels (m = 0) ----
        if (m == 0) {
            for (int o = 0; o < nout; ++o) {
                int k = outlev[o];
                int n = k < L ? k : L - 1;
                double t = k < L ? 0.0 : dtaus[n];
                const double *Gn = Gv + (long)n * 2 * N * N;
                const double *Kn = Kv + (long)n * N;
                const double *Zn = Zv + (long)n * 2 * N;
                const double *A = rhs + 2 * N * n;
                const double *B = A + N;
                double bfac = rt_exp(-ch[k]);
                double fup = 0.0, fdn = 0.0, uup = 0.0, udn = 0.0;
                for (int i = 0; i < N; ++i) {
                    double up = Zn[i] * bfac, dn = Zn[N + i] * bfac;
                    for (int j = 0; j < N; ++j) {
                        double ea = rt_exp(-Kn[j] * t);
                        double eb = rt_exp(-Kn[j] * (dtaus[n] - t));
                        up = up + Gn[i * N + j] * A[j] * ea + Gn[(N + i) * N + j] * B[j] * eb;
                        dn = dn + Gn[(N + i) * N + j] * A[j] * ea + Gn[i * N + j] * B[j] * eb;
                    }
                    fup = fup + wq[i] * mu[i] * up;
                    fdn = fdn + wq[i] * mu[i] * dn;
                    uup = uup + wq[i] * up;
                    udn = udn + wq[i] * dn;
                }
                double edir_s = umu0 * fbeam * rt_exp(-ch[k]);
                double edir_u = umu0 * fbeam * rt_exp(-chu[k]);
                double *fo = flux + o * 6;
                fo[0] = edir_u;
                fo[1] = 2.0 * RT_PI * fdn + edir_s - edir_u;
                fo[2] = 2.0 * RT_PI * fup;
                // mean intensities keep the delta-M split (scaled direct beam),
                // irradiances report the unscaled direct beam (uvspec's conventions)
                fo[3] = fbeam * rt_exp(-ch[k]) / (4.0 * RT_PI);
                fo[4] = 0.5 * udn;
                fo[5] = 0.5 * uup;
            }
        }

        // ---- intensities at user angles ----
        if (numu > 0 && !twostream) {
            for (int iu = 0; iu < numu; ++iu) {
                const double u = umu_in[iu];
                double lu[RT_MAXSTR];
                rt_lambda(m, nstr - 1, u > 0.0 ? u : -u, lu);
                double lus[RT_MAXSTR];
                for (int l = 0; l < nstr; ++l) lus[l] = u > 0.0 ? lu[l] : lu[l] * sg[l];
                double cosm[RT_MAXUMU];
                for (int ip = 0; ip < nphi; ++ip)
                    cosm[ip] = m == 0 ? 1.0 : uw_cos((double)m * phi_in[ip] * (RT_PI / 180.0));
                double val = 0.0;
                if (u > 0.0) {
                    if (m == 0) {
                        // Lambert surface: albedo/pi times the total downward irradiance
                        int n = L - 1;
                        const double *Gn = Gv + (long)n * 2 * N * N;
                        const double *Kn = Kv + (long)n * N;
                        const double *Zn = Zv + (long)n * 2 * N;
                        const double *A = rhs + 2 * N * n;
                        const double *B = A + N;
                        double fdn = 0.0;
                        for (int i = 0; i < N; ++i) {
                            double dn = Zn[N + i] * rt_exp(-ch[L]);
                            for (int j = 0; j < N; ++j)
                                dn = dn + Gn[(N + i) * N + j] * A[j] * rt_exp(-Kn[j] * dtaus[n])
                                        + Gn[i * N + j] * B[j];
                            fdn = fdn + wq[i] * mu[i] * dn;
                        }
                        val = albedo / RT_PI * (2.0 * RT_PI * fdn + umu0 * fbeam * rt_exp(-ch[L]));
                    }
                    for (int o = 0; o < nout; ++o)
                        if (outlev[o] == L)
                            for (int ip = 0; ip < nphi; ++ip)
                                rad[(o * numu + iu) * nphi + ip] += val * cosm[ip];
                    for (int n = L - 1; n >= 0; --n) {
                        const double *gl = gm + (long)n * nstr;
                        const double *Gn = Gv + (long)n * 2 * N * N;
                        const double *Kn = Kv + (long)n * N;
                        const double *Zn = Zv + (long)n * 2 * N;
                        const double *A = rhs + 2 * N * n;
                        const double *B = A + N;
                        double d = dtaus[n];
                        double s = 0.0, sz = 0.0;
                        double du[2 * RT_MAXN];
                        for (int i = 0; i < 2 * N; ++i) {
                            int ii = i < N ? i : i - N;
                            double t = 0.0;
                            for (int l = m; l < nstr; ++l)
                                t = t + gl[l] * lus[l] * lq[l * N + ii] * (i < N ? 1.0 : sg[l]);
                            du[i] = t * wq[ii];
                        }
                        double x0 = 0.0;
                        for (int l = m; l < nstr; ++l) x0 = x0 + gl[l] * sg[l] * l0[l] * lus[l];
                        sz = beamfac * x0;
                        for (int i = 0; i < 2 * N; ++i) sz = sz + du[i] * Zn[i];
                        for (int j = 0; j < N; ++j) {
                            double sa = 0.0, sb = 0.0;
                            for (int i = 0; i < N; ++i) {
                                sa = sa + du[i] * Gn[i * N + j] + du[N + i] * Gn[(N + i) * N + j];
                                sb = sb + du[i] * Gn[(N + i) * N + j] + du[N + i] * Gn[i * N + j];
                            }
                            s = s + sa * A[j] * rt_decay(Kn[j], u, d) + sb * B[j] * rt_grow(Kn[j], u, d);
                        }
                        s = s + sz * rt_exp(-ch[n]) * rt_decay(lam[n], u, d);
                        val = val * rt_exp(-d / u) + s;
                        for (int o = 0; o < nout; ++o)
                            if (outlev[o] == n)
                                for (int ip = 0; ip < nphi; ++ip)
                                    rad[(o * numu + iu) * nphi + ip] += val * cosm[ip];
                    }
                } else {
                    const double a = -u;
                    for (int o = 0; o < nout; ++o)
                        if (outlev[o] == 0)
                            for (int ip = 0; ip < nphi; ++ip)
                                rad[(o * numu + iu) * nphi + ip] += 0.0;
                    for (int n = 0; n < L; ++n) {
                        const double *gl = gm + (long)n * nstr;
                        const double *Gn = Gv + (long)n * 2 * N * N;
                        const double *Kn = Kv + (long)n * N;
                        const double *Zn = Zv + (long)n * 2 * N;
                        const double *A = rhs + 2 * N * n;
                        const double *B = A + N;
                        double d = dtaus[n];
                        double s = 0.0, sz = 0.0;
                        double du[2 * RT_MAXN];
                        for (int i = 0; i < 2 * N; ++i) {
                            int ii = i < N ? i : i - N;
                            double t = 0.0;
                            for (int l = m; l < nstr; ++l)
                                t = t + gl[l] * lus[l] * lq[l * N + ii] * (i < N ? 1.0 : sg[l]);
                            du[i] = t * wq[ii];
                        }
                        double x0 = 0.0;
                        for (int l = m; l < nstr; ++l) x0 = x0 + gl[l] * sg[l] * l0[l] * lus[l];
                        sz = beamfac * x0;
                        for (int i = 0; i < 2 * N; ++i) sz = sz + du[i] * Zn[i];
                        for (int j = 0; j < N; ++j) {
                            double sa = 0.0, sb = 0.0;
                            for (int i = 0; i < N; ++i) {
                                sa = sa + du[i] * Gn[i * N + j] + du[N + i] * Gn[(N + i) * N + j];
                                sb = sb + du[i] * Gn[(N + i) * N + j] + du[N + i] * Gn[i * N + j];
                            }
                            s = s + sa * A[j] * rt_grow(Kn[j], a, d) + sb * B[j] * rt_decay(Kn[j], a, d);
                        }
                        s = s + sz * rt_exp(-ch[n]) * rt_grow(lam[n], a, d);
                        val = val * rt_exp(-d / a) + s;
                        for (int o = 0; o < nout; ++o)
                            if (outlev[o] == n + 1)
                                for (int ip = 0; ip < nphi; ++ip)
                                    rad[(o * numu + iu) * nphi + ip] += val * cosm[ip];
                    }
                }
            }
        }
    }

    // ---- TMS (Nakajima & Tanaka 1988): exact single scattering after delta-M ----
    if ((flags & RT_TMS) && deltam && numu > 0 && !twostream) {
        const double *ptrue = ptrue_in + (long)prob * L * numu * nphi;
        for (int iu = 0; iu < numu; ++iu) {
            const double u = umu_in[iu];
            for (int ip = 0; ip < nphi; ++ip) {
                double cosang = -umu0 * u;
                double su = 1.0 - u * u, s0 = 1.0 - umu0 * umu0;
                su = su > 0.0 ? sqrt(su) : 0.0;
                s0 = s0 > 0.0 ? sqrt(s0) : 0.0;
                cosang = cosang + su * s0 * uw_cos(phi_in[ip] * (RT_PI / 180.0));
                double pl[RT_MAXSTR];
                rt_lambda(0, nstr - 1, cosang, pl);
                double val = 0.0;
                if (u > 0.0) {
                    for (int o = 0; o < nout; ++o)
                        if (outlev[o] == L) rad[(o * numu + iu) * nphi + ip] += 0.0;
                    for (int n = L - 1; n >= 0; --n) {
                        double ptr = 0.0;
                        for (int l = 0; l < nstr; ++l) ptr = ptr + gm[(long)n * nstr + l] * pl[l];
                        // gm holds (2l+1) g' ssas/2, so ptr = ssas P*/2
                        double pt = ptrue[((long)n * numu + iu) * nphi + ip];
                        double src = fbeam / (4.0 * RT_PI) * (ssas[n] * pt / (1.0 - fz[n]) - 2.0 * ptr);
                        double d = dtaus[n];
                        val = val * rt_exp(-d / u) + src * rt_exp(-ch[n]) * rt_decay(lam[n], u, d);
                        for (int o = 0; o < nout; ++o)
                            if (outlev[o] == n) rad[(o * numu + iu) * nphi + ip] += val;
                    }
                } else {
                    const double a = -u;
                    for (int n = 0; n < L; ++n) {
                        double ptr = 0.0;
                        for (int l = 0; l < nstr; ++l) ptr = ptr + gm[(long)n * nstr + l] * pl[l];
                        double pt = ptrue[((long)n * numu + iu) * nphi + ip];
                        double src = fbeam / (4.0 * RT_PI) * (ssas[n] * pt / (1.0 - fz[n]) - 2.0 * ptr);
                        double d = dtaus[n];
                        val = val * rt_exp(-d / a) + src * rt_exp(-ch[n]) * rt_grow(lam[n], a, d);
                        for (int o = 0; o < nout; ++o)
                            if (outlev[o] == n + 1) rad[(o * numu + iu) * nphi + ip] += val;
                    }
                }
            }
        }
    }
}

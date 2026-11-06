// cupy/complex.cuh for the Metal translation: thrust::complex as CuPy's
// vendored Thrust defines it (cupy/_core/include/cupy/complex/), for the
// operations woof's units use.  NVRTC finds CuPy's own header through the
// unit's -I options; a Mac has no CuPy, so this stand-in is found instead.
//
// Each operation is CuPy's text, operation for operation, so the words are
// CuPy's: arithmetic.h's division (scaled by |re| + |im|), complex_inl.h's
// comparisons, and csqrtf.h's square root (Algorithm 312, CACM vol 10,
// Oct 1967, as FreeBSD msun's s_csqrt.c adapted by Filipe Maia).
//
// Third-party notice.  Thrust: Copyright 2008-2013 NVIDIA Corporation,
// Copyright 2013 Filipe RNC Maia, Apache License 2.0
// (licenses/LICENSE-CCCL.txt).  csqrtf: Copyright (c) 2007 David Schultz
// <das@FreeBSD.ORG>, BSD 2-Clause: redistribution and use in source and
// binary forms, with or without modification, are permitted provided that
// redistributions retain this notice; THIS SOFTWARE IS PROVIDED BY THE
// AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES
// ARE DISCLAIMED.  CuPy: licenses/LICENSE-CuPy-MIT.txt.
#pragma once

namespace thrust {

template <typename T>
struct complex {
  T __woof_re;
  T __woof_im;
  // by value: Metal binds a const reference to one address space only
  __device__ complex(T re = T(), T im = T()) : __woof_re(re), __woof_im(im) {}
  template <typename U>
  __device__ complex(const complex<U>& z) : __woof_re(T(z.real())), __woof_im(T(z.imag())) {}
  __device__ T real() const { return __woof_re; }
  __device__ T imag() const { return __woof_im; }
  __device__ void real(T re) { __woof_re = re; }
  __device__ void imag(T im) { __woof_im = im; }
};

template <typename T>
__device__ inline complex<T> operator+(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs.real() + rhs.real(), lhs.imag() + rhs.imag());
}
template <typename T>
__device__ inline complex<T> operator+(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs.real() + rhs, lhs.imag());
}
template <typename T>
__device__ inline complex<T> operator+(const T& lhs, const complex<T>& rhs) {
  return complex<T>(rhs.real() + lhs, rhs.imag());
}
template <typename T>
__device__ inline complex<T> operator-(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs.real() - rhs.real(), lhs.imag() - rhs.imag());
}
template <typename T>
__device__ inline complex<T> operator-(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs.real() - rhs, lhs.imag());
}
template <typename T>
__device__ inline complex<T> operator-(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs - rhs.real(), -rhs.imag());
}
template <typename T>
__device__ inline complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs) {
  return complex<T>(lhs.real() * rhs.real() - lhs.imag() * rhs.imag(),
                    lhs.real() * rhs.imag() + lhs.imag() * rhs.real());
}
template <typename T>
__device__ inline complex<T> operator*(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs.real() * rhs, lhs.imag() * rhs);
}
template <typename T>
__device__ inline complex<T> operator*(const T& lhs, const complex<T>& rhs) {
  return complex<T>(rhs.real() * lhs, rhs.imag() * lhs);
}
template <typename T>
__device__ inline complex<T> operator/(const complex<T>& lhs, const complex<T>& rhs) {
  T s = abs(rhs.real()) + abs(rhs.imag());
  T oos = T(1.0) / s;
  T ars = lhs.real() * oos;
  T ais = lhs.imag() * oos;
  T brs = rhs.real() * oos;
  T bis = rhs.imag() * oos;
  s = (brs * brs) + (bis * bis);
  oos = T(1.0) / s;
  complex<T> quot(((ars * brs) + (ais * bis)) * oos, ((ais * brs) - (ars * bis)) * oos);
  return quot;
}
template <typename T>
__device__ inline complex<T> operator/(const complex<T>& lhs, const T& rhs) {
  return complex<T>(lhs.real() / rhs, lhs.imag() / rhs);
}
template <typename T>
__device__ inline complex<T> operator/(const T& lhs, const complex<T>& rhs) {
  return complex<T>(lhs) / rhs;
}
template <typename T>
__device__ inline complex<T> operator-(const complex<T>& rhs) {
  return rhs * -T(1);
}
template <typename T>
__device__ inline bool operator==(const complex<T>& lhs, const complex<T>& rhs) {
  return lhs.real() == rhs.real() && lhs.imag() == rhs.imag();
}
template <typename T>
__device__ inline bool operator==(const complex<T>& lhs, const T& rhs) {
  return lhs.real() == rhs && lhs.imag() == 0;
}
template <typename T>
__device__ inline bool operator!=(const complex<T>& lhs, const complex<T>& rhs) {
  return !(lhs == rhs);
}
template <typename T>
__device__ inline T real(const complex<T>& z) { return z.real(); }
template <typename T>
__device__ inline T imag(const complex<T>& z) { return z.imag(); }
template <typename T>
__device__ inline complex<T> conj(const complex<T>& z) { return complex<T>(z.real(), -z.imag()); }

__device__ inline complex<float> sqrt(const complex<float>& z) {
  float a = z.real(), b = z.imag();
  float t;
  int scale;
  complex<float> result;

  /* We risk spurious overflow for components >= FLT_MAX / (1 + sqrt(2)). */
  const float THRESH = 1.40949553037932e+38f;

  /* Handle special cases. */
  // z == 0.0f and result * 2.0f, spelled out (complex_inl.h's operator==,
  // arithmetic.h's scalar product): in a unit that keeps subnormals float
  // is f32s, and a float literal does not deduce the template's T.
  if (z.real() == 0.0f && z.imag() == 0.0f) return (complex<float>(0, b));
  if (isinf(b)) return (complex<float>(__int_as_float(0x7f800000), b));
  if (isnan(a)) {
    t = (b - b) / (b - b);         /* raise invalid if b is not a NaN */
    return (complex<float>(a, t)); /* return NaN + NaN i */
  }
  if (isinf(a)) {
    if (signbit(a))
      return (complex<float>(fabsf(b - b), copysignf(a, b)));
    else
      return (complex<float>(a, copysignf(b - b, b)));
  }

  // FLT_MIN*2
  const float low_thresh = 2.35098870164458e-38f;
  scale = 0;

  if (fabsf(a) >= THRESH || fabsf(b) >= THRESH) {
    /* Scale to avoid overflow. */
    a *= 0.25f;
    b *= 0.25f;
    scale = 1;
  } else if (fabsf(a) <= low_thresh && fabsf(b) <= low_thresh) {
    /* Scale to avoid underflow. */
    a *= 4.f;
    b *= 4.f;
    scale = 2;
  }

  /* Algorithm 312, CACM vol 10, Oct 1967. */
  if (a >= 0.0f) {
    t = sqrtf((a + hypotf(a, b)) * 0.5f);
    result = complex<float>(t, b / (2.0f * t));
  } else {
    t = sqrtf((-a + hypotf(a, b)) * 0.5f);
    result = complex<float>(fabsf(b) / (2.0f * t), copysignf(t, b));
  }

  /* Rescale. */
  if (scale == 1)
    return (complex<float>(result.real() * 2.0f, result.imag() * 2.0f));
  else if (scale == 2)
    return (complex<float>(result.real() * 0.5f, result.imag() * 0.5f));
  else
    return (result);
}

// binary64: arithmetic.h's abs, arg and norm; csqrt.h; clog.h (FreeBSD msun's
// clog by Stephen Montgomery-Smith, Copyright (c) 2012, BSD 2-Clause, under
// the notice above).
__device__ inline double abs(const complex<double>& z) { return ::hypot(z.real(), z.imag()); }
__device__ inline double arg(const complex<double>& z) { return ::atan2(z.imag(), z.real()); }
__device__ inline double norm(const complex<double>& z) {
  if (::fabs(z.real()) < ::sqrt(2.2250738585072014e-308) && ::fabs(z.imag()) < ::sqrt(2.2250738585072014e-308)) {
    double a = z.real() * 4.0;
    double b = z.imag() * 4.0;
    return (a * a + b * b) / 16.0;
  }
  return z.real() * z.real() + z.imag() * z.imag();
}

__device__ inline complex<double> sqrt(const complex<double>& z) {
  complex<double> result;
  double a, b;
  double t;
  int scale;

  /* We risk spurious overflow for components >= DBL_MAX / (1 + ::sqrt(2)). */
  const double THRESH = 7.446288774449766337959726e+307;

  a = z.real();
  b = z.imag();

  /* Handle special cases. */
  if (a == 0.0 && b == 0.0) return (complex<double>(0.0, b));
  if (isinf(b)) return (complex<double>(__longlong_as_double(0x7ff0000000000000LL), b));
  if (isnan(a)) {
    t = (b - b) / (b - b);          /* raise invalid if b is not a NaN */
    return (complex<double>(a, t)); /* return NaN + NaN i */
  }
  if (isinf(a)) {
    if (signbit(a))
      return (complex<double>(::fabs(b - b), ::copysign(a, b)));
    else
      return (complex<double>(a, ::copysign(b - b, b)));
  }

  // DBL_MIN*2
  const double low_thresh = 4.450147717014402766180465e-308;
  scale = 0;

  if (::fabs(a) >= THRESH || ::fabs(b) >= THRESH) {
    /* Scale to avoid overflow. */
    a *= 0.25;
    b *= 0.25;
    scale = 1;
  } else if (::fabs(a) <= low_thresh && ::fabs(b) <= low_thresh) {
    /* Scale to avoid underflow. */
    a *= 4.0;
    b *= 4.0;
    scale = 2;
  }

  /* Algorithm 312, CACM vol 10, Oct 1967. */
  if (a >= 0.0) {
    t = ::sqrt((a + ::hypot(a, b)) * 0.5);
    result = complex<double>(t, b / (2 * t));
  } else {
    t = ::sqrt((-a + ::hypot(a, b)) * 0.5);
    result = complex<double>(::fabs(b) / (2 * t), ::copysign(t, b));
  }

  /* Rescale. */
  if (scale == 1)
    return (complex<double>(result.real() * 2.0, result.imag() * 2.0));
  else if (scale == 2)
    return (complex<double>(result.real() * 0.5, result.imag() * 0.5));
  else
    return (result);
}

/* round down to 18 = 54/3 bits */
__device__ inline double __woof_clog_trim(double x) {
  unsigned long long w = (unsigned long long)__double_as_longlong(x);
  return __longlong_as_double((long long)(w & 0xfffffff800000000ULL));
}

__device__ inline complex<double> log(const complex<double>& z) {
  // Adapted from FreeBSDs msun
  double x, y;
  double ax, ay;
  double x0, y0, x1, y1, x2, y2, t, hm1;
  double val[12];
  int i, sorted;
  const double e = 2.7182818284590452354;

  x = z.real();
  y = z.imag();

  /* Handle NaNs using the general formula to mix them right. */
  if (x != x || y != y) {
    return (complex<double>(::log(norm(z)), ::atan2(y, x)));
  }

  ax = ::fabs(x);
  ay = ::fabs(y);
  if (ax < ay) {
    t = ax;
    ax = ay;
    ay = t;
  }

  if (ay > 5e307) {
    return (complex<double>(::log(::hypot(x / e, y / e)) + 1.0, ::atan2(y, x)));
  }
  if (ax == 1.) {
    if (ay < 1e-150) {
      return (complex<double>((ay * 0.5) * ay, ::atan2(y, x)));
    }
    return (complex<double>(::log1p(ay * ay) * 0.5, ::atan2(y, x)));
  }

  if (ax < 1e-50 || ay < 1e-50 || ax > 1e50 || ay > 1e50) {
    return (complex<double>(::log(::hypot(x, y)), ::atan2(y, x)));
  }

  /* Some easy cases. */

  if (ax >= 1.0) {
    return (complex<double>(::log1p((ax - 1) * (ax + 1) + ay * ay) * 0.5,
                            ::atan2(y, x)));
  }

  if (ax * ax + ay * ay <= 0.7) {
    return (complex<double>(::log(ax * ax + ay * ay) * 0.5, ::atan2(y, x)));
  }

  x0 = __woof_clog_trim(ax);
  ax = ax - x0;
  x1 = __woof_clog_trim(ax);
  x2 = ax - x1;
  y0 = __woof_clog_trim(ay);
  ay = ay - y0;
  y1 = __woof_clog_trim(ay);
  y2 = ay - y1;

  val[0] = x0 * x0;
  val[1] = y0 * y0;
  val[2] = 2 * x0 * x1;
  val[3] = 2 * y0 * y1;
  val[4] = x1 * x1;
  val[5] = y1 * y1;
  val[6] = 2 * x0 * x2;
  val[7] = 2 * y0 * y2;
  val[8] = 2 * x1 * x2;
  val[9] = 2 * y1 * y2;
  val[10] = x2 * x2;
  val[11] = y2 * y2;

  /* Bubble sort. */

  do {
    sorted = 1;
    for (i = 0; i < 11; i++) {
      if (val[i] < val[i + 1]) {
        sorted = 0;
        t = val[i];
        val[i] = val[i + 1];
        val[i + 1] = t;
      }
    }
  } while (!sorted);

  hm1 = -1;
  for (i = 0; i < 12; i++) {
    hm1 += val[i];
  }
  return (complex<double>(0.5 * ::log1p(hm1), ::atan2(y, x)));
}

}  // namespace thrust

using thrust::conj;
using thrust::real;
using thrust::imag;

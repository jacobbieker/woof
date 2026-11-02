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
  __device__ complex(const T& re = T(), const T& im = T()) : __woof_re(re), __woof_im(im) {}
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
  if (z == 0.0f) return (complex<float>(0, b));
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
    return (result * 2.0f);
  else if (scale == 2)
    return (result * 0.5f);
  else
    return (result);
}

}  // namespace thrust

using thrust::complex;
using thrust::conj;
using thrust::real;
using thrust::imag;

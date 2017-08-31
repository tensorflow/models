// @file normalize_cpu.cpp
// @brief Normalize block implementation (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "normalize.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory.h>

/* ---------------------------------------------------------------- */
/*                             Fast approximated numerical routines */
/* ---------------------------------------------------------------- */

#ifndef _MSC_VER
#include <x86intrin.h>
#pragma GCC optimize ("fast-math")
#pragma GCC optimize ("tree-vectorize")
//#pragma GCC target ("veclibabi=svml")
//#pragma GCC target "sse4"
#endif
#define restrict __restrict

#define VL_NNNORMALIZE_FAST
#define max(a,b) (((a)>=(b))?a:b)
#define xat(t) x[(t) * offset]
#define yat(t) y[(t) * offset]
#define zat(t) z[(t) * offset]

#ifndef VL_NNNORMALIZE_FAST
inline double fast_pow(double a, double b) { return pow(a,b) ; }
inline float fast_pow(float a, float b) { return powf(a,b) ; }
#else
//#define VERY_FAST
#ifndef VERY_FAST
inline double fast_pow(double x, double y)
{
  double z ;
  double const plog3 = 0.164042561333445 ;
  double const plog2 = -0.606737602222409 ;
  double const plog1 = 1.442695040888963 ;
  double const pexp3 = 0.079441541679836 ;
  double const pexp2 = 0.227411277760219 ;
  double const pexp1 = 0.693147180559945 ;
  typedef long long int int_t;
  const int_t offset = 1023LL << 52 ;

  int_t ix = *(int_t*)&x - offset ;
  int_t imx = (ix & ((1LL<<52)-1LL)) + offset;
  double fx = (double)(ix >> 52) ;
  double mx = *((double*)&imx) - 1 ;
  double mx2 = mx*mx ;
  double mx3 = mx2*mx ;
  double t = y * (fx + mx*plog1 + mx2*plog2 + mx3*plog3) ;
  //  double t = y * (fx + mx) ;

  double fz = floor(t) ;
  double rz = t - fz ;
  double rz2 = rz*rz ;
  double rz3 = rz2*rz ;
  double tz = fz + rz*pexp1 + rz2*pexp2 + rz3*pexp3 ;
  // double tz = fz + rz ;

  //  mexPrintf("%g %g -- ix %ld imx %ld fx %g mx %g t %g\n", x,y, ix,imx, fx, mx, t) ;
  *((int_t*)&z) = (int_t)(tz * (1LL<<52)) + offset ;
  //z = exp(t * log(2.0)) ;
  return z ;
}
#else
inline double fast_pow(double a, double b)
{
  double z ;
  typedef long long int int_t;
  const int_t offset = 1023L << 52 ;
  int_t ai = *((int_t*)&a) ;
  *((int_t*)&z) = (int_t)(b * (ai - offset)) + offset ;
  return z ;
}
#endif

#ifndef VERY_FAST
inline float fast_pow(float x, float y)
{
  float z ;
  float const plog3 = 0.164042561333445F ;
  float const plog2 = -0.606737602222409F ;
  float const plog1 = 1.442695040888963F ;
  float const pexp3 = 0.079441541679836F ;
  float const pexp2 = 0.227411277760219F ;
  float const pexp1 = 0.693147180559945F ;
  typedef int int_t;
  const int_t offset = 127 << 23 ;

  int_t ix = *(int_t*)&x - offset ;
  int_t imx = (ix & ((1<<23)-1)) + offset;
  float fx = (float)(ix >> 23) ;
  float mx = *((float*)&imx) - 1 ;
  float mx2 = mx*mx ;
  float mx3 = mx2*mx ;
  float t = y * (fx + mx*plog1 + mx2*plog2 + mx3*plog3) ;

  float fz = floor(t) ;
  float rz = t - fz ;
  float rz2 = rz*rz ;
  float rz3 = rz2*rz ;
  float tz = fz + rz*pexp1 + rz2*pexp2 + rz3*pexp3 ;

  *((int_t*)&z) = (int_t)(tz * (1<<23)) + offset ;
  return z ;
}
#else
inline float fast_pow(float a, float b)
{
  float z ;
  typedef int int_t;
  const int_t offset = 127 << 23 ;
  int_t ai = *((int_t*)&a) ;
  *((int_t*)&z) = (int_t)(b * (ai - offset)) + offset ;
  return z ;
}
#endif
#endif


namespace vl { namespace impl {

  template<typename type>
  struct lrn<vl::VLDT_CPU, type>
  {
    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    forward(type* output,
            type const* data,
            size_t width,
            size_t height,
            size_t depth,
            size_t num,
            size_t normDepth,
            type kappa, type alpha, type beta)
    {
      int t ;
      int m1 = ((signed)normDepth-1)/2 ;
      int m2 = (int)normDepth - m1 - 1 ;
      int offset = (int)width*(int)height ;
#ifndef VL_NNNORMALIZE_FAST
      for (int k = 0 ; k < num ; ++k) {
        for (int h = 0 ; h < height ; ++h) {
          for (int w = 0 ; w < width ; ++w) {
            type const* x = data + w + h * width ;
            T* y = output + w + h * width ;
            type acc = 0 ;
            for (t = -m2 ; t < (signed)depth ; ++t) {
              type ap = 0 ;
              type am = 0 ;
              if (t-m1-1 >= 0) { am = xat(t-m1-1) ; }
              if (t+m2 < depth) { ap = xat(t+m2) ; }
              acc += ap*ap - am*am ;
              if (0 <= t && t < depth) {
                yat(t) = xat(t) * fast_pow(kappa + alpha * acc, -beta) ;
              }
            }
          }
        }
        data += width*height*depth ;
        output += width*height*depth ;
      }
#else
      type * acc = (type*) calloc(sizeof(type), width*height) ;
      for (int k = 0 ; k < num ; ++k) {
        memset(acc, 0, sizeof(type) * width*height) ;
        for (t = -m2 ; t < (signed)depth ; ++t) {
          int tm = t - m1 - 1 ;
          int tp = t + m2 ;
          type const* xam = data + offset * (t-m1-1) ;
          type const* xap = data + offset * (t+m2) ;
          type *end = acc + width*height ;
          if (0 <= tm && tp < depth) {
            for(type *xacc = acc ; xacc != end ; ++xacc, ++xam, ++xap) {
              type am = *xam ;
              type ap = *xap ;
              *xacc += ap*ap - am*am ;
            }
          } else if (0 > tm && tp < depth) {
            for(type *xacc = acc ; xacc != end ; ++xacc, ++xap) {
              type ap = *xap ;
              *xacc += ap*ap ;
            }
          } else if (0 <= tm && tp >= depth) {
            for(type *xacc = acc ; xacc != end ; ++xacc, ++xam) {
              type am = *xam ;
              *xacc -= am*am ;
            }
          }
          if (0 <= t && t < depth) {
            type const* xx = data + offset * t ;
            type * xy = output + offset * t ;
            for(type *xacc = acc ; xacc != end ; ++xacc, ++xx, ++xy) {
              (*xy) = (*xx) * fast_pow(kappa + alpha * (*xacc), -beta) ;
            }
          }
        }
        data += width*height*depth ;
        output += width*height*depth ;
      }
      free(acc) ;
#endif
      return vl::VLE_Success ;
    }

    /* ------------------------------------------------------------ */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward(type * output,
             type const* data,
             type const* derOutput,
             size_t width,
             size_t height,
             size_t depth,
             size_t num,
             size_t normDepth,
             type kappa, type alpha, type beta)
    {
      int m1 = ((signed)normDepth-1)/2 ;
      int m2 = (int)normDepth - m1 - 1 ;
      int offset = (int)width*(int)height ;
      type ab2 = 2*alpha*beta ;
      int t, q ;

#ifndef VL_NNNORMALIZE_FAST
      for (int k = 0 ; k < num ; ++k) {
        for (int h = 0 ; h < height ; ++h) {
          for (int w = 0 ; w < width ; ++w) {
            type const* x = data + w + h * width ;
            T* y = output + w + h * width ;
            type const* z = derOutput + w + h * width ;
            type acc = 0 ;
            for (t = 0 ; t < (signed)depth ; ++t) {
              yat(t) = 0 ;
            }
            for (t = -m2 ; t < (signed)depth ; ++t) {
              int q1 = t-m1 ;
              int q2 = t+m2 ;
              type ap = 0 ;
              type am = 0 ;
              if (t-m1-1 >= 0) { am = xat(t-m1-1) ; } else { q1 = 0 ; }
              if (t+m2 < depth) { ap = xat(t+m2) ; } else { q2 = depth - 1 ; }
              acc += ap*ap - am*am ;
              type L = kappa + alpha * acc ;
              type Lbeta = fast_pow(L, -beta) ;
              type Lbeta1 = Lbeta / L ;

              if (0 <= t && t < depth) {
                yat(t) += zat(t) * Lbeta ;
                for (q = q1 ; q <= q2 ; ++ q) {
                  yat(q) -= zat(t) * xat(t) * xat(q) * ab2 * Lbeta1 ;
                }
              }
            }
          }
        }
        data += width*height*depth ;
        output += width*height*depth ;
        derOutput += width*height*depth ;
      }
#else
      type * restrict acc = (type*) malloc(sizeof(type) * width*height) ;
      type * restrict acc2 = (type*) malloc(sizeof(type) * width*height*depth) ;
      for (int k = 0 ; k < num ; ++k) {
        memset(acc, 0, sizeof(type) * width*height) ;
        for (t = -m2 ; t < (signed)depth ; ++t) {
          /*
           Compue the square of the input data x.^2 summed in the normalization window. This is done
           incrementally, by updating the previous normalization window sum.
           */
          {
            int const tm = t - m1 - 1 ;
            int const tp = t + m2 ;
            type const* restrict datam_ = data + offset * tm ;
            type const* restrict datap_ = data + offset * tp ;
            type *end = acc + width*height ;

            if (0 <= tm && tp < depth) {
              for(type * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datap_, ++datam_) {
                type am = *datam_ ;
                type ap = *datap_ ;
                *acc_ += ap*ap - am*am ;
              }
            } else if (0 > tm && tp < depth) {
              for(type * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datap_) {
                type ap = *datap_ ;
                *acc_ += ap*ap ;
              }
            } else if (0 <= tm && tp >= depth) {
              for(type * restrict acc_ = acc ; acc_ != end ; ++acc_, ++datam_) {
                type am = *datam_ ;
                *acc_ -= am*am ;
              }
            }
          }

          /*
           Compute the arguments of the summation in the derivative
           expression, storing them into acc2.
           */
          if (0 <= t && t < depth) {
            type const* restrict data_ = data + offset * t ;
            type const* restrict derOutput_ = derOutput + offset * t ;
            type * restrict output_ = output + offset * t ;
            type * restrict acc2_ = acc2 + offset * t ;
            type * end = acc + width*height ;
            for(type * restrict acc_ = acc ; acc_ != end ;
                ++acc_, ++acc2_, ++data_, ++derOutput_, ++output_) {
              type L = kappa + alpha * (*acc_) ;
              type Lbeta = fast_pow(L, -beta) ;
              type temp1 = (*derOutput_) * Lbeta ;
              type temp2 = (*data_) * ab2 * temp1 / L ;
              *output_ = temp1 ;
              *acc2_ = temp2 ;
            }
          }
        }

        /*
         Integrate along feature channels in acc2, summing plane t-1 to
         plane t.
         */
        for (t = 1 ; t < (signed)depth ; ++t) {
          type * restrict acc2_ = acc2 + t * offset ;
          type const* restrict src_ = acc2_ - offset ;
          type const* end = acc2_ + offset ;
          for( ; acc2_ != end ; ++acc2_, ++src_) {
            *acc2_ += *src_ ;
          }
        }

        /*
         Compute summation in the derivative expression from the integral
         just obtained.
         */
        for (t = 0 ; t < (signed)depth ; ++t) {
          int q1 = t - m2 - 1 ;
          int q2 = ((t + m1) <= (depth - 1)) ? t + m1 : depth - 1 ;
          type const* restrict acc22_ = acc2 + offset * q2 ;
          type const* restrict acc21_ = acc2 + offset * q1 ;
          type const* restrict data_  = data + offset * t ;
          type const* restrict end = data_  + width*height ;
          type * restrict output_ = output + offset * t ;
          if (q1 >= 0) {
            for( ; data_ != end ; ++data_, ++acc22_, ++acc21_, ++output_) {
              *output_ -= (*acc22_ - *acc21_) * (*data_) ;
            }
          } else {
            for( ; data_ != end ; ++data_, ++acc22_, ++output_) {
              *output_ -= (*acc22_) * (*data_) ;
            }
          }
        }
        data += width*height*depth ;
        output += width*height*depth ;
        derOutput += width*height*depth ;
      }
      free(acc) ;
      free(acc2) ;
#endif
      return vl::VLE_Success ;
    }

  } ;

} }

// Instantiations
template struct vl::impl::lrn<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::lrn<vl::VLDT_CPU, double> ;
#endif
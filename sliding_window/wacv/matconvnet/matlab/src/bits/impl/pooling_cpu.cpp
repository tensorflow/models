// @file pooling_cpu.cpp
// @brief Pooling block implementation (GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "pooling.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>

/* ---------------------------------------------------------------- */
/*                                               Max pooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_max
{
  inline acc_max(int poolHeight, int poolWidth, type derOutput = 0)
  :
  value(-std::numeric_limits<type>::infinity()),
  derOutput(derOutput),
  derDataActivePt(NULL)
  { }

  inline void accumulate_forward(type x) {
    value = std::max(value, x) ;
  }

  inline void accumulate_backward(type const* data, type* derDataPt) {
    type x = *data ;
    if (x > value) {
      value = x ;
      derDataActivePt = derDataPt ;
    }
  }

  inline type done_forward() const {
    return value ;
  }

  inline void done_backward() const {
    if (derDataActivePt) { *derDataActivePt += derOutput ; }
  }

  type value ;
  type derOutput ;
  type* derDataActivePt ;
} ;

/* ---------------------------------------------------------------- */
/*                                           Average pooling helper */
/* ---------------------------------------------------------------- */

template <typename type>
struct acc_sum
{
  inline acc_sum(int poolHeight, int poolWidth, type derOutput = 0)
  :
  value(0),
  scale(type(1)/type(poolHeight*poolWidth)),
  derOutput(derOutput)
  { }

  inline void accumulate_forward(type x) {
    value += x ;
  }

  /* note: data is unused */
  inline void accumulate_backward(type const* data, type* derDataPt) {
    *derDataPt += derOutput * scale ;
  }

  inline type done_forward() const {
    return value * scale ;
  }

  inline void done_backward() const { }

  type value ;
  type derOutput ;
  type scale ;
} ;

/* ---------------------------------------------------------------- */
/*                                                pooling_*_forward */
/* ---------------------------------------------------------------- */

/*
 Reverse accumulation style (better for writing).
 - pick an input coordiante xi; goal is to compute dz/dxi
 - look for all the pools Pj that cointain xi
 -  compute dfj/dxi (Pj)
 -  accumulate into dz/dxi += dz/dfj dfj/dxi (Pj)

 The advantage of this method is that dz/dxi can be processed in parallel
 without conflicts from other threads writing on different dz/dxi. The
 disadvantage is that for eac xi we need to know dfj/dxi (Pj) for all
 the pools Pj that contain xi. Barring special cases (e.g. linear) this
 usually requires additional information to be available. For instance,
 for max pooling knowing the output in addition to the input of the
 pooling operator.

 Direct accumulation style.
 - pick an output coordiante fj and its pool Pj
 - for all the input point xi in the pool Pj
 - compute dfj/dxi (Pj)
 - accumulate to dz/dxi += dz/dfj dfj/dxi (Pj)

 The difference with the last method is that different output pools Pj
 will share several input pixels xi; hence this will cause write conflicts if
 Pj are processed in parallel.
 */

template<typename type, typename Accumulator> static inline void
pooling_forward_cpu(type* pooled,
                    type const* data,
                    size_t width, size_t height, size_t depth,
                    size_t windowWidth, size_t windowHeight,
                    size_t strideX, size_t strideY,
                    size_t padLeft, size_t padRight, size_t padTop, size_t padBottom)
{
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < pooledHeight; ++y) {
      for (int x = 0; x < pooledWidth; ++x) {
        int x1 = x * (signed)strideX - (signed)padLeft ;
        int y1 = y * (signed)strideY - (signed)padTop ;
        int x2 = std::min(x1 + windowWidth, width) ;
        int y2 = std::min(y1 + windowHeight, height) ;
        x1 = std::max(x1, 0) ;
        y1 = std::max(y1, 0) ;
        Accumulator acc(y2 - y1, x2 - x1) ;
        for (int v = y1 ; v < y2 ; ++v) {
          for (int u = x1 ; u < x2 ; ++u) {
            acc.accumulate_forward(data[v * width + u]) ;
          }
        }
        pooled[y * pooledWidth + x] = acc.done_forward() ;
      }
    }
    data += width*height ;
    pooled += pooledWidth*pooledHeight ;
  }
}

/* ---------------------------------------------------------------- */
/*                                               pooling_*_backward */
/* ---------------------------------------------------------------- */

/*
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */

/* Todo: transpose */

template<typename type, typename Accumulator> static inline void
pooling_backward_cpu(type* derData,
                     type const* data,
                     type const* derPooled,
                     size_t width, size_t height, size_t depth,
                     size_t windowWidth, size_t windowHeight,
                     size_t strideX, size_t strideY,
                     size_t padLeft, size_t padRight, size_t padTop, size_t padBottom)
{
  int pooledWidth = (width + (padLeft + padRight) - windowWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop + padBottom) - windowHeight)/strideY + 1 ;
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < pooledHeight; ++y) {
      for (int x = 0; x < pooledWidth; ++x) {
        int x1 = x * (signed)strideX - (signed)padLeft ;
        int y1 = y * (signed)strideY - (signed)padTop ;
        int x2 = std::min(x1 + windowWidth, width) ;
        int y2 = std::min(y1 + windowHeight, height) ;
        x1 = std::max(x1, 0) ;
        y1 = std::max(y1, 0) ;
        Accumulator acc(y2 - y1, x2 - x1, derPooled[y * pooledWidth + x]) ;
        for (int v = y1 ; v < y2 ; ++v) {
          for (int u = x1 ; u < x2 ; ++u) {
            acc.accumulate_backward(&data[v * width + u],
                                    &derData[v * width + u]) ;
          }
        }
        acc.done_backward() ;
      }
    }
    data += width*height ;
    derData += width*height ;
    derPooled += pooledWidth*pooledHeight ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct pooling_max<vl::VLDT_CPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      pooling_forward_cpu<type, acc_max<type> > (pooled,
                                                 data,
                                                 height, width, depth,
                                                 poolHeight, poolWidth,
                                                 strideY, strideX,
                                                 padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* data,
             type const* derOutput,
             size_t height, size_t width, size_t depth,
             size_t poolHeight, size_t poolWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
      pooling_backward_cpu<type, acc_max<type> > (derData,
                                                  data, derOutput,
                                                  height, width, depth,
                                                  poolHeight, poolWidth,
                                                  strideY, strideX,
                                                  padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }
  } ; // pooling_max

  template <typename type>
  struct pooling_average<vl::VLDT_CPU, type>
  {

    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      pooling_forward_cpu<type, acc_sum<type> > (pooled,
                                                 data,
                                                 height, width, depth,
                                                 poolHeight, poolWidth,
                                                 strideY, strideX,
                                                 padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    backward(type* derData,
             type const* derPooled,
             size_t height, size_t width, size_t depth,
             size_t poolHeight, size_t poolWidth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom,
             size_t padLeft, size_t padRight)
    {
      pooling_backward_cpu<type, acc_sum<type> > (derData,
                                                  NULL, derPooled,
                                                  height, width, depth,
                                                  poolHeight, poolWidth,
                                                  strideY, strideX,
                                                  padTop, padBottom, padLeft, padRight) ;
      return VLE_Success ;
    }
  } ; // pooling_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::pooling_max<vl::VLDT_CPU, float> ;
template struct vl::impl::pooling_average<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::pooling_max<vl::VLDT_CPU, double> ;
template struct vl::impl::pooling_average<vl::VLDT_CPU, double> ;
#endif


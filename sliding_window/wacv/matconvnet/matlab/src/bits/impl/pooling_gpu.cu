// @file pooling_gpu.cu
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
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <sm_20_atomic_functions.h>

/* ---------------------------------------------------------------- */
/*                                              pooling_max_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
pooling_max_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    data += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;

    T bestValue = data[y1 * width + x1] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        bestValue = max(bestValue, data[y * width + x]) ;
      }
    }
    pooled[pooledIndex] = bestValue ;
  }
}

/* ---------------------------------------------------------------- */
/*                                          pooling_average_forward */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
pooling_average_kernel
(T* pooled,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  /* pooledIndex = x + y * pooledWidth + z * (pooledWidth * pooledHeight) */
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;
    data += pz * (width*height) ;
    T accum = 0;
    T poolSize = (y2 - y1)*(x2 - x1);
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        accum += data[y * width + x] ;
      }
    }
    pooled[pooledIndex] = accum / poolSize ;
  }
}

/* ---------------------------------------------------------------- */
/*                                             pooling_max_backward */
/* ---------------------------------------------------------------- */

#ifdef VLNN_CAFFELIKE_BPPOOL
// In order to be able to use this, BP would need to have access to both
// bottom data and pooled data (currently only passed bottom data...)
template <typename T> __global__ void
pooling_max_backward_with_pooled_data
(T* derData,
 const T* data,
 const T* pooled,
 const T* derPooled,
 const int nthreads,
 const int pooledWidth,
 const int pooledHeight,
 const int width,
 const int height,
 const int depth,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    // find out the local index
    // find out the local offset
    int x = index % width;
    int y = (index / width) % height;
    int z = (index / width / height) % depth;
    int py1 = (y < poolHeight) ? 0 : (y - poolHeight) / strideY + 1;
    int py2 = min(y / strideY + 1, pooledHeight);
    int px1 = (x < poolWidth) ? 0 : (x - poolWidth) / strideX + 1;
    int px2 = min(x / strideX + 1, pooledWidth);
    T gradient = 0;
    T datum = data[(z * height + y) * width + x];
    pooled += z * pooledHeight * pooledWidth;
    dzdy += z * pooledHeight * pooledWidth;
    for (int py = py1; py < py2; ++py) {
      for (int px = px1; px < px2; ++px) {
        gradient += dzdy[py * pooledWidth + px] *
        (datum == pooled[py * pooledWidth + px]);
      }
    }
    dzdx[index] = gradient;
  }
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// an implementation of atomicAdd() for double (really slow) for older CC
static __device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template<typename T> __global__ void
pooling_max_backward_kernel
(T* derData,
 const T* data,
 const T* derPooled,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    data += pz * (width*height) ;
    derData += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = min(x1 + poolWidth, width) ;
    int y2 = min(y1 + poolHeight, height) ;
    x1 = max(x1, 0) ;
    y1 = max(y1, 0) ;

    int bestIndex = y1 * width + x1 ;
    T bestValue = data[bestIndex] ;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        int index = y * width + x ;
        T value = data[index] ;
        if (value > bestValue) {
          bestValue = value ;
          bestIndex = index ;
        }
      }
    }
    /*
     This is bad, but required to eliminate a race condition when writing
     to bottom_diff.
     Caffe goes the other way around, but requrires remembering the layer
     output, or the maximal indexes.
     atomicAdd(add, val)
     */
    atomicAdd(derData + bestIndex, derPooled[pooledIndex]) ;
  }
}

/* ---------------------------------------------------------------- */
/*                                         pooling_average_backward */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
pooling_average_backward_kernel(T* derData,
                                const T* derPooled,
                                const int nthreads,
                                const int pooledWidth,
                                const int pooledHeight,
                                const int width,
                                const int height,
                                const int depth,
                                const int poolWidth,
                                const int poolHeight,
                                const int strideX,
                                const int strideY,
                                const int padLeft,
                                const int padTop)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    /* To understand the logic of this piece of code see the
     comments to of the row2im backward kernel */
    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - poolWidth ;
    int dy = y_data + padTop - poolHeight ;
    int px1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int py1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int px2 = min((x_data + padLeft) / strideX, pooledWidth - 1) ;
    int py2 = min((y_data + padTop) / strideY, pooledHeight - 1) ;
    T accumulator = 0 ;
    derPooled += z * pooledHeight * pooledWidth;
    for (int py = py1 ; py <= py2 ; ++py) {
      for (int px = px1 ; px <= px2 ; ++px) {
        int x1 = px * strideX - padLeft ;
        int y1 = py * strideY - padTop ;
        int x2 = min(x1 + poolWidth, width) ;
        int y2 = min(y1 + poolHeight, height) ;
        x1 = max(x1, 0) ;
        y1 = max(y1, 0) ;
        T poolSize = (y2 - y1) * (x2 - x1);
        accumulator += derPooled[py * pooledWidth + px] / poolSize ;
      }
    }
    derData[index] = accumulator ;
  }
}

/* ---------------------------------------------------------------- */
/*                                                        Interface */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct pooling_max<vl::VLDT_GPU, type>
  {
    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom,
            size_t padLeft, size_t padRight)
    {
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int pooledVolume = pooledWidth * pooledHeight * depth ;

      pooling_max_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data,
       pooledHeight, pooledWidth, pooledVolume,
       height, width,
       poolHeight, poolWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
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
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int pooledVolume = pooledWidth * pooledHeight * depth ;

      pooling_max_backward_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, data, derOutput,
       pooledHeight, pooledWidth, pooledVolume,
       height, width,
       poolHeight, poolWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // pooling_max

  template <typename type>
  struct pooling_average<vl::VLDT_GPU, type>
  {

    static vl::ErrorCode
    forward(type* pooled,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t poolHeight, size_t poolWidth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int pooledVolume = pooledWidth * pooledHeight * depth ;

      pooling_average_kernel<type>
      <<< divideAndRoundUp(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (pooled, data,
       pooledHeight, pooledWidth, pooledVolume,
       height, width,
       poolHeight, poolWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
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
      int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
      int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
      int dataVolume = width * height * depth ;

      pooling_average_backward_kernel<type>
      <<< divideAndRoundUp(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData, derPooled,
       dataVolume,
       pooledHeight, pooledWidth,
       height, width, dataVolume,
       poolHeight, poolWidth,
       strideY, strideX,
       padTop, padLeft);

      cudaError_t status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }
  } ; // pooling_average

} } ; // namespace vl::impl

// Instantiations
template struct vl::impl::pooling_max<vl::VLDT_GPU, float> ;
template struct vl::impl::pooling_average<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::pooling_max<vl::VLDT_GPU, double> ;
template struct vl::impl::pooling_average<vl::VLDT_GPU, double> ;
#endif


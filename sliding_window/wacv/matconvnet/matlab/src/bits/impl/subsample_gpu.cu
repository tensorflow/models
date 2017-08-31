// @file subsampling_gpu.cu
// @brief Subsampling block implementation (GPU)
// @author Andrea Vedaldi
// @author Karel Lenc

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "subsample.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <float.h>
#include <iostream>

#ifndef ENABLE_GPU
#error "subsample_gpu.cu cannot be compiled without GPU support"
#endif

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                         subsample forward kernel */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
subsample_forward_kernel
(T* output,
 const T* data,
 const int outputHeight,
 const int outputWidth,
 const int outputVolume,
 const int height,
 const int width,
 const int strideY,
 const int strideX,
 const int padTop,
 const int padLeft)
{
  int outputIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (outputIndex < outputVolume) {
    /* outputIndex = x
     + y * outputWidth
     + z * (outputWidth * outputHeight) ;
     */
    int py = outputIndex ;
    int px = py / outputHeight ;
    int channel = px / outputWidth ;
    px %= outputWidth ;
    py %= outputHeight ;
    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    data += channel * (width*height) ;
    T value = 0 ;
    if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
      value = data[x1 * height + y1] ;
    }
    output[outputIndex] =  value ;
  }
}

/* ---------------------------------------------------------------- */
/*                                        subsample backward kernel */
/* ---------------------------------------------------------------- */

template<typename T>
__global__ void subsample_backward_kernel
(T* derData,
 const T* derOutput,
 const int outputHeight,
 const int outputWidth,
 const int dataVolume,
 const int height,
 const int width,
 const int strideY,
 const int strideX,
 const int padTop,
 const int padLeft)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < dataVolume) {
    int y = index ;
    int x = y / height ;
    int channel = x / width ;
    x %= width ;
    y %= height ;
    derOutput += channel * outputHeight * outputWidth ;
    int px = (x + padLeft) / strideX ;
    int py = (y + padTop) / strideY ;
    if (x == strideX * px - padLeft &&
        y == strideY * py - padTop) {
      derData[index] = derOutput[px * outputHeight + py] ;
    } else {
      derData[index] = 0 ;
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                          drivers */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template <typename type>
  struct subsample<vl::VLDT_GPU, type>
  {

    static vl::ErrorCode
    forward(vl::Context& context,
            type* output,
            type const* data,
            size_t height, size_t width, size_t depth,
            size_t strideY, size_t strideX,
            size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      int outputWidth = (width + (padLeft+padRight) - 1)/strideX + 1 ;
      int outputHeight = (height + (padTop+padBottom) - 1)/strideY + 1 ;
      int outputVolume = outputWidth * outputHeight * depth ;

      subsample_forward_kernel<type>
      <<< divideAndRoundUp(outputVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (output, data,
       outputHeight, outputWidth, outputVolume,
       height, width,
       strideY, strideX,
       padTop, padLeft);
      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
    }

    static vl::ErrorCode
    backward(vl::Context& context,
             type* derData,
             type const* derOutput,
             size_t height, size_t width, size_t depth,
             size_t strideY, size_t strideX,
             size_t padTop, size_t padBottom, size_t padLeft, size_t padRight)
    {
      int outputWidth = (width + (padLeft+padRight) - 1)/strideX + 1 ;
      int outputHeight = (height + (padTop+padBottom) - 1)/strideY + 1 ;
      int dataVolume = width * height * depth ;

      subsample_backward_kernel<type>
      <<< divideAndRoundUp(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      (derData,
       derOutput,
       outputHeight, outputWidth, dataVolume,
       height, width,
       strideY, strideX,
       padTop, padLeft);
      return context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
    }
  } ;

} }

// Instantiations
template struct vl::impl::subsample<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::subsample<vl::VLDT_GPU, double> ;
#endif

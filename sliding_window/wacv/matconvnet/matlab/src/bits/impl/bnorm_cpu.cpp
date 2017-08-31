// @file   bnorm_cpu.cpp
// @brief  Batch normalization implementation (CPU)
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bnorm.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory.h>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cassert>

/* ---------------------------------------------------------------- */
/*          compute_moments, compute_ders, compute_ders_and_moments	*/
/* ---------------------------------------------------------------- */

// Compute moments (means and sigmas) from the batch data
// WH is the product of the data width and height
// moments is a 2 x depth array with means and sigmas

template<typename T> inline void
compute_moments(T * moments,
                T const * data,
                int WH,
                int depth,
                int num,
                T epsilon)
{
  int mass = WH * num ;
  for(int channel = 0; channel < depth; ++channel) {
    for(int element = 0; element < num; ++element) {
      for(int wh = 0; wh < WH; ++wh){
        T x = data[wh + channel*WH + element*(depth*WH)] ;
        moments[channel] += x ; // mean
        moments[channel + depth] += x * x; // sigma
      }
    }
  }
  for(int i = 0; i < depth; ++i) {
    T mean = moments[i] / mass ;
    T sigma2 = std::max((T).0, moments[i + depth]/mass - mean*mean) ;
    moments[i] = mean ;
    moments[i + depth] = sqrt(sigma2 + epsilon);
  }
}

// this version assumes that moments is precomputed
template<typename T> inline void
compute_ders(T * derMultipliers,
             T * derBiases,
             T const * moments,
             T const * data,
             T const * derOutput,
             int WH, int depth, int num,
             T epsilon)
{
  memset(derMultipliers, 0, sizeof(T) * depth) ;
  memset(derBiases, 0, sizeof(T) * depth) ;
  for(int channel = 0; channel < depth; ++channel){
    for(int element = 0; element < num; ++element ){
      for(int wh = 0; wh < WH; ++wh){
        int offset = wh + channel*WH + element * (WH*depth) ;
        derMultipliers[channel] += derOutput[offset] * data[offset];
        derBiases[channel] += derOutput[offset];
      }
    }
  }

  T mass = WH*num;
  for(int i = 0; i < depth; ++i) {
    T mean = moments[i] ;
    T sigma = moments[i + depth] ;
    derMultipliers[i] = (derMultipliers[i] - mean*derBiases[i]) / sigma;
  }
}

template<typename T> inline void
compute_ders_and_moments(T * derMultipliers,
                         T * derBiases,
                         T * moments,
                         T const * data,
                         T const * derOutput,
                         int WH, int depth, int num,
                         T epsilon)
{
  memset(derMultipliers, 0, sizeof(T) * depth) ;
  memset(derBiases, 0, sizeof(T) * depth) ;
  for(int channel = 0; channel < depth; ++channel){
    for(int element = 0; element < num; ++element ){
      for(int wh = 0; wh < WH; ++wh){
        int offset = wh + channel*WH + element * (WH*depth) ;
        moments[channel] += data[offset] ;
        moments[channel + depth] += data[offset] * data[offset];
        derMultipliers[channel] += derOutput[offset] * data[offset];
        derBiases[channel] += derOutput[offset];
      }
    }
  }

  T mass = WH*num;
  for(int i = 0; i < depth; ++i) {
    T mean = moments[i] / mass ;
    T sigma2 = std::max((T).0, moments[i + depth]/mass - mean*mean) ;
    T sigma = sqrt(sigma2 + epsilon);
    moments[i] = mean ;
    moments[i + depth] = sigma ;
    derMultipliers[i] = (derMultipliers[i] - mean*derBiases[i]) / sigma;
  }
}

/* ---------------------------------------------------------------- */
/*                                         batch_normalize_backward	*/
/* ---------------------------------------------------------------- */

template<typename T> inline void
batch_normalize_backward(T * derData,
                         T const * moments,
                         T const * data,
                         T const * multipliers,
                         T const * derMultipliers,
                         T const * derBiases,
                         T const * derOutput,
                         int WH,
                         int depth,
                         int num)
{
  T mass = WH*num;
  for(int channel = 0; channel < depth; ++channel ) {
    T mean = moments[channel] ;
    T sigma = moments[channel + depth] ;

    T muz = derBiases[channel]/mass;
    T G1 = multipliers[channel]/sigma ;
    T G2 = G1 * derMultipliers[channel]/(mass*sigma);

    for(int element = 0; element < num; ++element){
      for(int wh = 0; wh < WH; ++wh){
        int offset = wh + channel*WH + element * (WH*depth) ;
        derData[offset] = G1 * (derOutput[offset] - muz) - G2 * (data[offset]-mean) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                           driver */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template<typename T>
  struct bnorm<vl::VLDT_CPU,T>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    forward_given_moments(Context& context,
                          T* output,
                          T const* moments,
                          T const* data,
                          T const* multipliers,
                          T const* biases,
                          size_t height, size_t width, size_t depth, size_t num)
    {
      int WH = height * width ;
      for(int channel = 0; channel < depth; ++channel) {
        T mean = moments[channel] ;
        T sigma = moments[channel + depth] ;
        T bias = biases[channel];
        T coefficient = multipliers[channel] / sigma ;

        for(int element = 0; element < num; ++element) {
          for(int wh = 0; wh < WH; ++wh){
            int offset = wh + channel*WH + element * (depth*WH) ;
            output[offset] = coefficient * (data[offset] - mean) + bias ;
          }
        }
      }
      return VLE_Success;
    }

    static vl::ErrorCode
    forward(Context& context,
            T* output,
            T* moments,
            T const* data,
            T const* multipliers,
            T const* biases,
            size_t height, size_t width, size_t depth, size_t size,
            T epsilon)
    {
      vl::ErrorCode error = VLE_Success ;
      bool ownMoments = false ;
      if (moments == NULL) {
        moments = (T*)calloc(sizeof(T),2*depth);
        if (!moments) {
          error = VLE_OutOfMemory ;
          goto done ;
        }
        ownMoments = true ;
      } else {
        memset(moments, 0, sizeof(T) * 2*depth) ;
      }
      compute_moments<T>(moments,
                         data, width*height, depth, size,
                         epsilon) ;

      error = bnorm<vl::VLDT_CPU,T>::forward_given_moments
      (context,
       output,
       moments, data,
       multipliers, biases,
       height, width, depth, size) ;

      // Delete intermediate variable
    done:
      if (ownMoments)  { free(moments) ; }
      return error ;
    }

    /*------------------------------------------------------------- */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward_given_moments(Context& context,
                           T* derData,
                           T* derMultipliers,
                           T* derBiases,
                           T const* moments,
                           T const* data,
                           T const* multipliers,
                           T const* biases,
                           T const* derOutput,
                           size_t height, size_t width, size_t depth, size_t size,
                           T epsilon)
    {
      vl::ErrorCode error = VLE_Success ;
      int WH = width * height;

      // Compute derMultipliers, derBiases, muz, and moments
      compute_ders<T>(derMultipliers, derBiases,
                      moments, data, derOutput,
                      WH, depth, size,
                      epsilon);

      // Compute derData
      batch_normalize_backward<T>(derData,
                                  moments, data,
                                  multipliers,
                                  derMultipliers, derBiases, derOutput,
                                  WH, depth, size);
    done:;
      return error ;
    }

    static vl::ErrorCode
    backward(Context& context,
             T* derData,
             T* derMultipliers,
             T* derBiases,
             T* moments,
             T const* data,
             T const* multipliers,
             T const* biases,
             T const* derOutput,
             size_t height, size_t width, size_t depth, size_t size,
             T epsilon)
    {
      vl::ErrorCode error = VLE_Success ;
      int WH = width * height;

      // Get workspace if needed
      if (moments == NULL) {
        moments = (T*)context.getWorkspace(vl::VLDT_CPU, sizeof(T)*2*depth) ;
        if (!moments) {
          error = VLE_OutOfMemory ;
          goto done ;
        }
      }
      memset(moments, 0, sizeof(T) * 2*depth) ;

      // Compute derMultipliers, derBiases, and moments
      compute_ders_and_moments<T>(derMultipliers, derBiases, moments,
                                  data, derOutput,
                                  WH, depth, size,
                                  epsilon);

      // Compute derData
      batch_normalize_backward<T>(derData,
                                  moments, data,
                                  multipliers,
                                  derMultipliers, derBiases, derOutput,
                                  WH, depth, size);

    done:;
      return error ;
    }
  } ;

} } // namespace vl::impl

template struct vl::impl::bnorm<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bnorm<vl::VLDT_CPU, double> ;
#endif


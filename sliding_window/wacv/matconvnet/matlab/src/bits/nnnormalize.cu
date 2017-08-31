// @file nnnormalize.cu
// @brief Normalization block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnnormalize.hpp"
#include "impl/normalize.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
//#include "impl/normalize_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                    nnlrn_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, type) \
error = vl::impl::lrn<deviceType,type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
normDepth, kappa, alpha, beta) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnlrn_forward(vl::Context& context,
                  vl::Tensor output,
                  vl::Tensor data,
                  size_t normDepth,
                  double kappa, double alpha, double beta)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DataType dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         error = vl::impl::nnlrn_forward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (error == vl::VLE_Success) { return error ; }
         if (error != vl::UNSUPPORTED) { return error ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      if (error != vl::VLE_Success) { context.getCudaHelper().catchCudaError(__func__) ; }
      break ;
#endif
  }
  if (error != vl::VLE_Success) {
    context.setError(error, __func__) ;
  }
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                                   nnlrn_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH

#define DISPATCH(deviceType, type) \
error = vl::impl::lrn<deviceType,type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), (type const*)derOutput.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
normDepth, kappa, alpha, beta) ;

vl::ErrorCode
vl::nnlrn_backward(vl::Context& context,
                   vl::Tensor derData,
                   vl::Tensor data,
                   vl::Tensor derOutput,
                   size_t normDepth,
                   double kappa, double alpha, double beta)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        /*
         error = vl::impl::nnlrn_backward_cudnn<float>(context, output, data, filters, biases,
         strideY, strideX,
         padTop, padBottom,
         padLeft, padRight) ;
         if (error == vl::VLE_Success) { return error ; }
         if (error != vl::UNSUPPORTED) { return error ; }
         */
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      if (error != vl::VLE_Success) { context.getCudaHelper().catchCudaError(__func__) ; }
      break ;
#endif
  }
  if (error != vl::VLE_Success) {
    context.setError(error, __func__) ;
  }
  return error ;
}

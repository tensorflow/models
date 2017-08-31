// @file nnbnorm.cu
// @brief Batch normalization block
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbnorm.hpp"
#include "impl/bnorm.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#if ENABLE_CUDNN
#include "impl/nnbnorm_cudnn.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::forward \
(context, \
 (type*)output.getMemory(), \
 (type*)moments.getMemory(), \
 (type const*)data.getMemory(), \
 (type*)multipliers.getMemory(), \
 (type*)biases.getMemory(), \
 data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
 epsilon);

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnbnorm_cudnn<dataType>::forward \
(context, output, moments, \
data, multipliers, biases, epsilon) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case VLDT_Float : DISPATCHCUDNN(VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCHCUDNN(VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnbnorm_forward(vl::Context& context,
                    vl::Tensor output,
                    vl::Tensor moments,
                    vl::Tensor data,
                    vl::Tensor multipliers,
                    vl::Tensor biases,
                    double epsilon)
{
  vl::ErrorCode error = VLE_Success ;
  vl::DataType dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (error == vl::VLE_Success) { return error ; }
        if (error != vl::VLE_Unsupported) { return error ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      if (error == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break;
#endif
  }
  return context.passError(error, __func__) ;
}

#undef DISPATCH
#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::forward_given_moments \
(context, \
(type*)output.getMemory(), \
(type const*)moments.getMemory(), \
(type const*)data.getMemory(), \
(type*)multipliers.getMemory(), \
(type*)biases.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize()) ;

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnbnorm_cudnn<dataType>::forward_given_moments \
(context, output, moments, \
data, multipliers, biases) ;

vl::ErrorCode
vl::nnbnorm_forward_given_moments(vl::Context& context,
                                  vl::Tensor output,
                                  vl::Tensor moments,
                                  vl::Tensor data,
                                  vl::Tensor multipliers,
                                  vl::Tensor biases)
{
  vl::ErrorCode error = VLE_Success ;
  vl::DataType dataType = output.getDataType() ;

  switch (output.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (error == vl::VLE_Success) { return error ; }
        if (error != vl::VLE_Unsupported) { return error ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      if (error == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("nnbnorm_*_forward")) ;
      }
      break;
#endif
  }
  return context.passError(error, "nnbnorm_forward_given_moments") ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::backward \
(context, \
 (type*)derData.getMemory(), \
 (type*)derMultipliers.getMemory(), \
 (type*)derBiases.getMemory(), \
 (type*)moments.getMemory(), \
 (type const*)data.getMemory(), \
 (type const*)multipliers.getMemory(), \
 (type const*)biases.getMemory(), \
 (type const*)derOutput.getMemory(), \
 data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
 epsilon);

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnbnorm_cudnn<dataType>::backward \
(context, derData, derMultipliers, derBiases, \
moments, data, multipliers, \
biases, derOutput, epsilon) ;

vl::ErrorCode
vl::nnbnorm_backward(Context& context,
                     vl::Tensor derData,
                     vl::Tensor derMultipliers,
                     vl::Tensor derBiases,
                     vl::Tensor moments,
                     vl::Tensor data,
                     vl::Tensor multipliers,
                     vl::Tensor biases,
                     vl::Tensor derOutput,
                     double epsilon)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
#if ENABLE_CUDNN
      if (context.getCudaHelper().getCudnnEnabled()) {
        DISPATCHCUDNN2() ;
        if (error == vl::VLE_Success) { return error ; }
        if (error != vl::VLE_Unsupported) { return error ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      if (error == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break;
#endif
  }
  return context.passError(error, __func__) ;
}

#undef DISPATCH
#define DISPATCH(deviceType,type) \
error = vl::impl::bnorm<deviceType,type>::backward_given_moments \
(context, \
(type*)derData.getMemory(), \
(type*)derMultipliers.getMemory(), \
(type*)derBiases.getMemory(), \
(type*)moments.getMemory(), \
(type const*)data.getMemory(), \
(type const*)multipliers.getMemory(), \
(type const*)biases.getMemory(), \
(type const*)derOutput.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
epsilon);

vl::ErrorCode
vl::nnbnorm_backward_given_moments(Context& context,
                                   vl::Tensor derData,
                                   vl::Tensor derMultipliers,
                                   vl::Tensor derBiases,
                                   vl::Tensor moments,
                                   vl::Tensor data,
                                   vl::Tensor multipliers,
                                   vl::Tensor biases,
                                   vl::Tensor derOutput,
                                   double epsilon)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (derOutput.getDeviceType()) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      if (error == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break;
#endif
  }
  return context.passError(error, __func__) ;
}

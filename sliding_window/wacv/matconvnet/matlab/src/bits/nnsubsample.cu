// @file nnsubsample.cu
// @brief Subsampling block
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi and Karel Lenc.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnsubsample.hpp"
#include "impl/subsample.hpp"
#include "impl/blashelper.hpp"
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/* Implementations                                                  */
/* ---------------------------------------------------------------- */

template<vl::DeviceType deviceType, vl::DataType dataType> vl::ErrorCode
nnsubsample_forward_impl(Context& context,
                         Tensor output,
                         Tensor data,
                         Tensor biases,
                         int strideY, int strideX,
                         int padTop, int padBottom,
                         int padLeft, int padRight)
{
  assert(output) ;
  assert(data) ;

  vl::ErrorCode error ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;

  ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
  type const* allOnesMemory = (type*) context.getAllOnes(deviceType, dataType, numOutputPixels) ;

  if (allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < data.getSize() ; ++image) {
    ptrdiff_t dataOffset = (data.getHeight()*data.getWidth()*data.getDepth()) * image ;
    ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;
    error = vl::impl::subsample<deviceType,type>::forward
    (context,
     (type*)output.getMemory() + outputOffset,
     (type const*)data.getMemory() + dataOffset,
     data.getHeight(), data.getWidth(), data.getDepth(),
     strideY, strideX,
     padTop, padBottom, padLeft, padRight) ;
    if (error != vl::VLE_Success) { goto done ; }
    if (biases) {
      type alpha = 1 ;
      type beta = 1 ;
      error = vl::impl::blas<deviceType, dataType>::gemm
      (context,
       'n', 'n',
       numOutputPixels, biases.getNumElements(), 1,
       alpha,
       allOnesMemory, numOutputPixels,
       (type*)biases.getMemory(), 1,
       beta,
       (type*)output.getMemory() + outputOffset, numOutputPixels) ;
      if (error != vl::VLE_Success) { goto done ; }
    }
  }
done:
  return context.passError(error, __func__) ;
}

template<vl::DeviceType deviceType, vl::DataType dataType> vl::ErrorCode
nnsubsample_backward_impl(Context& context,
                          Tensor derData,
                          Tensor derBiases,
                          Tensor derOutput,
                          int strideY, int strideX,
                          int padTop, int padBottom,
                          int padLeft, int padRight)
{
  assert(derOutput) ;

  vl::ErrorCode error ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;

  ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;
  type const* allOnesMemory = (type*) context.getAllOnes(deviceType, dataType, numOutputPixels) ;

  if (allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < derData.getSize() ; ++image) {
    ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

    /* compute derBiases = dz/dbias */
    if (derBiases) {
      type alpha = 1 ;
      type beta = (image > 0) ; /* this saves init. the output array with 0 */
      error = vl::impl::blas<deviceType,dataType>::gemv
      (context,
       't',
       numOutputPixels, derOutput.getDepth(),
       alpha,
       (type const*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
       allOnesMemory, 1,
       beta,
       (type*)derBiases.getMemory(), 1) ;
      if (error != vl::VLE_Success) { goto done ; }
    }

    /* compute derData = dz/dx */
    if (derData) {
      ptrdiff_t derDataOffset = (derData.getHeight()*derData.getWidth()*derData.getDepth()) * image ;
      error = vl::impl::subsample<deviceType,type>::backward
      (context,
       (type*)derData.getMemory() + derDataOffset,
       (type const*)derOutput.getMemory() + derOutputOffset,
       derData.getHeight(), derData.getWidth(), derData.getDepth(),
       strideY, strideX,
       padTop, padBottom, padLeft, padRight) ;
      if (error != vl::VLE_Success) { goto done ; }
    }
  }
done:
  return context.passError(error, __func__) ;
}

/* ---------------------------------------------------------------- */
/* Dispatchers                                                      */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, dataType) \
error = nnsubsample_forward_impl<deviceType, dataType> \
(context, output, data, biases, \
 strideY, strideX, \
 padTop, padBottom, \
 padLeft, padRight) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnsubsample_forward(Context& context,
                        Tensor output,
                        Tensor data,
                        Tensor biases,
                        int strideY, int strideX,
                        int padTop, int padBottom,
                        int padLeft, int padRight)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType() ;
  vl::DataType dataType = output.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      if (error == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break ;
#endif
  }
  return context.passError(error, __func__) ;
}

#undef DISPATCH
#define DISPATCH(deviceType, dataType) \
error = nnsubsample_backward_impl<deviceType, dataType> \
(context, derData, derBiases, derOutput, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight) ;

vl::ErrorCode
vl::nnsubsample_backward(vl::Context& context,
                         vl::Tensor derData,
                         vl::Tensor derBiases,
                         vl::Tensor derOutput,
                         int strideY, int strideX,
                         int padTop, int padBottom,
                         int padLeft, int padRight)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::DeviceType deviceType = derOutput.getDeviceType() ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;

    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      if (error == VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError("GPU")) ;
      }
      break ;
#endif
  }
  return context.passError(error, __func__) ;
}

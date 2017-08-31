// @file nnconv.cu
// @brief Convolution block
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015-16 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnconv.hpp"
#include "nnbias.hpp"
#include "impl/nnconv_blas.hpp"
#if ENABLE_CUDNN
#include "impl/nnconv_cudnn.hpp"
#endif
#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                                   nnconv_forward */
/* ---------------------------------------------------------------- */

/*
 for output: must have data and optional filters or biases
 */


#define DISPATCH(deviceType, dataType) \
error = vl::impl::nnconv_forward_blas<deviceType, dataType> \
(context, \
output, outputMult, \
data, dataMult, \
filters, biases, \
strideY, strideX, \
padTop, padBottom, \
padLeft, padRight, \
dilateY, dilateX) ;

#define DISPATCH2(deviceType) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnconv_cudnn<dataType>::forward \
(context, \
 output, outputMult, \
 data, dataMult, \
 filters, biases, \
 strideY, strideX, \
 padTop, padBottom, \
 padLeft, padRight, \
 dilateY, dilateX) ;

#define DISPATCHCUDNN2() \
switch (dataType) { \
case VLDT_Float : DISPATCHCUDNN(VLDT_Float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCHCUDNN(VLDT_Double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnconv_forward(Context& context,
                   Tensor output, double outputMult,
                   Tensor data, double dataMult,
                   Tensor filters,
                   Tensor biases,
                   int strideY, int strideX,
                   int padTop, int padBottom,
                   int padLeft, int padRight,
                   int dilateY, int dilateX)
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
        if (error != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                                  nnconv_backward */
/* ---------------------------------------------------------------- */


/*
 for derBiases:  must have derOuptut
 for derData:    must have derData, derOutput and filters
 for derFilters: must have derFilters, derOutput and data
 */

#undef DISPATCH
#define DISPATCH(deviceType, dataType) \
error = vl::impl::nnconv_backward_blas<deviceType, dataType> \
(context, \
 derData, derFilters, derBiases, \
 data, filters, derOutput, \
 strideY, strideX, \
 padTop, padBottom, \
 padLeft, padRight, \
 dilateY, dilateX) ;

#undef DISPATCHCUDNN
#define DISPATCHCUDNN(dataType) \
error = vl::impl::nnconv_cudnn<dataType>::backward \
(context, \
 derData, derFilters, derBiases, \
 data, filters, derOutput, \
 strideY, strideX, \
 padTop, padBottom, \
 padLeft, padRight, \
 dilateY, dilateX) ;

vl::ErrorCode
vl::nnconv_backward(Context& context,
                    Tensor derData,
                    Tensor derFilters,
                    Tensor derBiases,
                    Tensor data,
                    Tensor filters,
                    Tensor derOutput,
                    int strideY, int strideX,
                    int padTop, int padBottom,
                    int padLeft, int padRight,
                    int dilateY, int dilateX)
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
        if (error != vl::VLE_Unsupported) { goto done ; }
        /* this case was not supported by CUDNN -- fallback */
      }
#endif
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
#if ENABLE_CUDNN
done:
#endif
  return error ;
}


/* ---------------------------------------------------------------- */
/*                                                  nnconvt_forward */
/* ---------------------------------------------------------------- */

vl::ErrorCode
vl::nnconvt_forward(Context& context,
                    Tensor output,
                    Tensor data,
                    Tensor filters,
                    Tensor biases,
                    int upsampleY, int upsampleX,
                    int cropTop, int cropBottom,
                    int cropLeft, int cropRight)
{
  vl::ErrorCode error = VLE_Success ;
  size_t dataOffset = data.getHeight()*data.getWidth()*data.getDepth() ;
  size_t outputOffset = output.getHeight()*output.getWidth()*output.getDepth() ;

  // we need to process this down per image as nnconv_backward would otherwise
  // accumulate everything into a single feature field in the output
  for (int image = 0 ; image < data.getSize() ; ++image) {
    Tensor dataSlice(data) ;
    Tensor outputSlice(output) ;

    switch (data.getDataType()) {
      case VLDT_Float:
        dataSlice.setMemory((float*)data.getMemory() + dataOffset * image) ;
        outputSlice.setMemory((float*)output.getMemory() + outputOffset * image) ;
        break ;
      case VLDT_Double:
        dataSlice.setMemory((double*)data.getMemory() + dataOffset * image) ;
        outputSlice.setMemory((double*)output.getMemory() + outputOffset * image) ;
        break ;
      default:
        assert(false) ;
    }
    dataSlice.setSize(1) ;
    outputSlice.setSize(1) ;

    error = vl::nnconv_backward(context,
                                outputSlice, Tensor(), Tensor(),
                                Tensor(), filters, dataSlice,
                                upsampleY, upsampleX,
                                cropTop, cropBottom,
                                cropLeft, cropRight,
                                1, 1) ;
    if (error != VLE_Success) { goto done ; }
  }
  if (biases) {
    error = vl::nnbias_forward(context,
                               output, 1,
                               Tensor(), 0,
                               biases, 1) ;
  }
done:
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                                 nnconvt_backward */
/* ---------------------------------------------------------------- */

vl::ErrorCode
vl::nnconvt_backward(Context& context,
                     Tensor derData,
                     Tensor derFilters,
                     Tensor derBiases,
                     Tensor data,
                     Tensor filters,
                     Tensor derOutput,
                     int upsampleY, int upsampleX,
                     int cropTop, int cropBottom,
                     int cropLeft, int cropRight)
{
  vl::ErrorCode error = vl::VLE_Success ;

  if (derData) {
    error = vl::nnconv_forward(context,
                               derData, 0,
                               derOutput, 1,
                               filters, Tensor(),
                               upsampleY, upsampleX,
                               cropTop, cropBottom,
                               cropLeft, cropRight,
                               1, 1) ;
    if (error != VLE_Success) { goto done ; }
  }

  if (derFilters) {
    error = vl::nnconv_backward(context,
                                Tensor(), derFilters, Tensor(),
                                derOutput, Tensor(), data,
                                upsampleY, upsampleX,
                                cropTop, cropBottom,
                                cropLeft, cropRight,
                                1, 1) ;
    if (error != VLE_Success) { goto done ; }
  }

  if (derBiases) {
    error = vl::nnbias_backward(context,
                                Tensor(), 0,
                                derBiases, 0,
                                derOutput, 1) ;
  }

done:
  return error ;
}

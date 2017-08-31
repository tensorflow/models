// @file nnpooling_blas.cu
// @brief Pooling block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
 Copyright (C) 2015-16 Andrea Vedaldi.
 All rights reserved.

 This file is part of the VLFeat library and is made available under
 the terms of the BSD license (see the COPYING file).
 */

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnpooling_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnpooling_cudnn.hpp"
#include "cudnnhelper.hpp"
#include "../datacu.hpp"
#include <assert.h>

using namespace vl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
goto done ; \
} }

/* ---------------------------------------------------------------- */
/*                                         nnpooling_cudnn::forward */
/* ---------------------------------------------------------------- */


namespace vl { namespace impl {


  template<vl::DataType dataType>
  vl::ErrorCode
  nnpooling_cudnn<dataType>::forward(Context& context,
                                     Tensor output,
                                     Tensor data,
                                     PoolingMethod method,
                                     int poolHeight, int poolWidth,
                                     int strideY, int strideX,
                                     int padTop, int padBottom,
                                     int padLeft, int padRight)
  {
    assert(output) ;
    assert(data) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, dataDesc ;
    cudnnPoolingDescriptor_t poolingDesc ;
    bool outputDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool poolingDescInitialized = false ;

    if (padLeft != padRight) return vl::VLE_Unsupported ;
    if (padTop != padBottom) return vl::VLE_Unsupported ;

    if (method == vlPoolingAverage && (padLeft > 0 | padRight > 0)) {
      /* This seems like a bug in CUDNN? */
      return vl::VLE_Unsupported ;
    }

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
    vl::DataType dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     output.getSize(), // sizes
                                     output.getDepth(),
                                     output.getWidth(),
                                     output.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     data.getSize(),
                                     data.getDepth(),
                                     data.getWidth(),
                                     data.getHeight())) ;

    CHECK(cudnnCreatePoolingDescriptor(&poolingDesc)) ;
    poolingDescInitialized = true ;
    CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                      (method == vl::vlPoolingAverage) ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_MAX,
                                      IF_CUDNN_GE5(CUDNN_NOT_PROPAGATE_NAN COMMA)
                                      poolWidth, poolHeight,
                                      padLeft, padTop,
                                      strideX, strideY)) ;

    // Perform convolution for each filter group
    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      CHECK(cudnnPoolingForward(handle,
                                poolingDesc,
                                &alpha,
                                dataDesc, data.getMemory(),
                                &beta,
                                outputDesc, output.getMemory())) ;
    }

    /* cleanup */
  done:
    if (poolingDescInitialized) { cudnnDestroyPoolingDescriptor(poolingDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return context.passError(error, "nnpooling_cudnn::forward") ;
  }

  /* ---------------------------------------------------------------- */
  /*                                        nnpooling_cudnn::backward */
  /* ---------------------------------------------------------------- */

  template<vl::DataType dataType>
  vl::ErrorCode
  nnpooling_cudnn<dataType>::backward(Context& context,
                                      Tensor derData,
                                      Tensor data,
                                      Tensor output,
                                      Tensor derOutput,
                                      vl::PoolingMethod method,
                                      int poolHeight, int poolWidth,
                                      int strideY, int strideX,
                                      int padTop, int padBottom,
                                      int padLeft, int padRight)
  {
    assert(derData) ;
    assert(data) ;
    assert(output) ;
    assert(derOutput) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, dataDesc ;
    cudnnPoolingDescriptor_t poolingDesc ;
    bool outputDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool poolingDescInitialized = false ;

    if (padLeft != padRight) return vl::VLE_Unsupported ;
    if (padTop != padBottom) return vl::VLE_Unsupported ;

    if (method == vlPoolingAverage && (padLeft > 0 | padRight > 0)) {
      /* This seems like a bug in CuDNN? */
      return vl::VLE_Unsupported ;
    }

    cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
    vl::DataType dynDataType = output.getDataType() ;
    assert(dynDataType == dataType) ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     output.getSize(), // sizes
                                     output.getDepth(),
                                     output.getWidth(),
                                     output.getHeight())) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                     CUDNN_TENSOR_NCHW,
                                     cudnnDataType,
                                     data.getSize(),
                                     data.getDepth(),
                                     data.getWidth(),
                                     data.getHeight())) ;

    CHECK(cudnnCreatePoolingDescriptor(&poolingDesc)) ;
    poolingDescInitialized = true ;
    CHECK(cudnnSetPooling2dDescriptor(poolingDesc,
                                      (method == vl::vlPoolingAverage) ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_MAX,
                                      IF_CUDNN_GE5(CUDNN_NOT_PROPAGATE_NAN COMMA)
                                      poolWidth, poolHeight,
                                      padLeft, padTop,
                                      strideX, strideY)) ;

    // Perform convolution for each filter group
    {
      type alpha = 1.0f ;
      type beta = 0.0f ;
      CHECK(cudnnPoolingBackward(handle,
                                 poolingDesc,
                                 &alpha,
                                 outputDesc, (type const*)output.getMemory(),
                                 outputDesc, (type const*)derOutput.getMemory(),
                                 dataDesc, (type const*)data.getMemory(),
                                 &beta,
                                 dataDesc, (type*)derData.getMemory())) ;
    }

    /* cleanup */
  done:
    if (poolingDescInitialized) { cudnnDestroyPoolingDescriptor(poolingDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return context.passError(error, __func__) ;
  }
  
} }

// Instantiations
template struct vl::impl::nnpooling_cudnn<vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::nnpooling_cudnn<vl::VLDT_Double> ;
#endif




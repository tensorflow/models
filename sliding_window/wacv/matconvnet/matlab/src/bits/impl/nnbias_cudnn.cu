// @file nnbias_cudnn.cu
// @brief biasolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnbias_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnbias_cudnn.hpp"
#include "cudnnhelper.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <iostream>

using namespace vl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__LINE__) ":" STRINGIZE(__FILE__))) ; \
goto done ; \
} }

namespace vl { namespace impl {

  /* -------------------------------------------------------------- */
  /*                                           nnbias_forward_cudnn */
  /* -------------------------------------------------------------- */

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnbias_cudnn<dataType>::forward(vl::Context& context,
                                            vl::Tensor output, double outputMult,
                                            vl::Tensor data, double dataMult,
                                            vl::Tensor biases, double biasesMult)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, biasesDesc, dataDesc ;
    bool outputDescInitialized = false ;
    bool biasesDescInitialized = false ;
    bool dataDescInitialized = false ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get output tensor descripotr
    assert(output) ;
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     DataTypeToCudnn<dataType>::id,
                                     output.getSize(), // sizes
                                     output.getDepth(),
                                     output.getWidth(),
                                     output.getHeight())) ;

    if (biases) {
      CHECK(cudnnCreateTensorDescriptor(&biasesDesc)) ;
      biasesDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(biasesDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::id,
                                       1,
                                       biases.getNumElements(),
                                       1,
                                       1)) ;

      type alpha = biasesMult ;
      type beta = outputMult ;
#if (CUDNN_VERSION < 4000)
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_SAME_C,
                           &alpha,
                           biasesDesc, biases.getMemory(),
                           &beta,
                           outputDesc, output.getMemory())) ;
#else
      CHECK(cudnnAddTensor(handle,
                           &alpha,
                           biasesDesc, biases.getMemory(),
                           &beta,
                           outputDesc, output.getMemory())) ;
#endif
      outputMult = 1 ;
    }

    if (data) {
      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::id,
                                       data.getSize(),
                                       data.getDepth(),
                                       data.getWidth(),
                                       data.getHeight())) ;

      type alpha = dataMult ;
      type beta = outputMult ;
#if (CUDNN_VERSION < 4000)
      CHECK(cudnnAddTensor(handle,
                           CUDNN_ADD_FULL_TENSOR,
                           &alpha,
                           dataDesc, data.getMemory(),
                           &beta,
                           outputDesc, output.getMemory()));
#else
      CHECK(cudnnAddTensor(handle,
                           &alpha,
                           dataDesc, data.getMemory(),
                           &beta,
                           outputDesc, output.getMemory()));
#endif
    }

    /* cleanup */
  done:
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (biasesDescInitialized) { cudnnDestroyTensorDescriptor(biasesDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return context.passError(error, __func__) ;
  }

  /* ---------------------------------------------------------------- */
  /*                                            nnbias_backward_cudnn */
  /* ---------------------------------------------------------------- */

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnbias_cudnn<dataType>::backward(vl::Context& context,
                                             vl::Tensor derData, double derDataMult,
                                             vl::Tensor derBiases, double derBiasesMult,
                                             vl::Tensor derOutput, double derOutputMult)
  {
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derDataDesc needed as same as dataDesc */
    cudnnTensorDescriptor_t derDataDesc, derBiasesDesc, derOutputDesc ;
    bool derDataDescInitialized = false ;
    bool derBiasesDescInitialized = false ;
    bool derOutputDescInitialized = false ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                     CUDNN_TENSOR_NCHW,
                                     DataTypeToCudnn<dataType>::id,
                                     derOutput.getSize(), // sizes
                                     derOutput.getDepth(),
                                     derOutput.getWidth(),
                                     derOutput.getHeight())) ;

    // for derivatives w.r.t. bias
    if (derBiases) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasesDesc)) ;
      derBiasesDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derBiasesDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::id,
                                       1,
                                       derBiases.getNumElements(),
                                       1,
                                       1)) ;

      type alpha = derOutputMult ;
      type beta = derBiasesMult ;
      CHECK(cudnnConvolutionBackwardBias
            (handle,
             &alpha,
             derOutputDesc, (type const*)derOutput.getMemory(),
             &beta,
             derBiasesDesc, (type*)derBiases.getMemory())) ;
    }

    if (derData) {
      CHECK(cudnnCreateTensorDescriptor(&derDataDesc)) ;
      derDataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derDataDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::id,
                                       derData.getSize(),
                                       derData.getDepth(),
                                       derData.getWidth(),
                                       derData.getHeight())) ;
      // not implemented
      assert(false) ;
    }

  done:
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    if (derBiasesDescInitialized) { cudnnDestroyTensorDescriptor(derBiasesDesc) ; }
    if (derDataDescInitialized) { cudnnDestroyTensorDescriptor(derDataDesc) ; }
    return context.passError(error, __func__) ;
  }

} }

// Instantiations
template struct vl::impl::nnbias_cudnn<vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::nnbias_cudnn<vl::VLDT_Double> ;
#endif

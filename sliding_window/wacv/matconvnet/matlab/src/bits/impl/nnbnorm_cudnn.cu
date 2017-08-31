// @file nnbnorm_cudnn.hpp
// @brief bnorm CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2016 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "nnbnorm_cudnn.hpp"
#include "cudnnhelper.hpp"
#include "../datacu.hpp"
#include "copy.hpp"

#include <assert.h>

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__))) ; \
goto done ; \
} }

template<typename T>
__global__ void var_to_std(T * var, unsigned int num, T scale, T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    var[idx] = sqrt(scale * var[idx] + epsilon) ;
  }
}

template<typename T>
__global__ void inverse(T * ivar, unsigned int num)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    ivar[idx] = ((T)1) / ivar[idx] ;
  }
}

template<vl::DataType dataType>
vl::ErrorCode
vl::impl::nnbnorm_cudnn<dataType>::forward(vl::Context& context,
                                           vl::Tensor output,
                                           vl::Tensor moments, // can be null
                                           vl::Tensor data,
                                           vl::Tensor multipliers,
                                           vl::Tensor biases,
                                           double epsilon)
{
  assert(output) ;
  assert(data) ;
  assert(multipliers) ;
  assert(biases) ;

  typedef typename DataTypeTraits<dataType>::type type ;

  cudnnTensorDescriptor_t dataDesc, momentDesc ;
  bool dataDescInitialized = false ;
  bool momentDescInitialized = false ;

  cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
  vl::DataType dynDataType = output.getDataType() ;
  assert(dynDataType == dataType) ;

  cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
  vl::ErrorCode error = vl::VLE_Success ;
  cudnnHandle_t handle ;

  // Get CuDNN.
  CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

  // Get tensor descripotrs.
  CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
  dataDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   data.getSize(),
                                   data.getDepth(),
                                   data.getWidth(),
                                   data.getHeight())) ;

  CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
  dataDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   1, data.getDepth(), 1, 1)) ;


  // Run CuDNN batch normalization implementation.
  {
    type alpha = 1.0f ;
    type beta = 0.0f ;
    type * meanMemory = NULL ;
    type * varMemory = NULL ;
    if (moments) {
      meanMemory = (type*)moments.getMemory()  ;
      varMemory = meanMemory + data.getDepth() ;
      vl::impl::operations<vl::VLDT_GPU,type>::fill
      (meanMemory, 2 * data.getDepth() * sizeof(type), 0) ;
    }

    CHECK(cudnnBatchNormalizationForwardTraining
          (handle,
           CUDNN_BATCHNORM_SPATIAL,
           &alpha, &beta,
           dataDesc, data.getMemory(),
           dataDesc, output.getMemory(),
           momentDesc, multipliers.getMemory(), biases.getMemory(),
           0, NULL, NULL,
           epsilon,
           meanMemory, varMemory)) ;

    if (varMemory) {
      // CuDNN computes the variance without epsilon, whereas MCN
      // returns the standard deviation after adding epsilon.
      // Also, CuDNN returns the unbiased variance estimate, but it is
      // debatable that this is appropriate.
      //
      // We pick instead the caches, which are closer to the values we compute.
      // Also they do not need to be pre-initialized with zeros.

      size_t const blockSize = VL_CUDA_NUM_THREADS ;
      inverse<type> <<<divideAndRoundUp(data.getDepth(),blockSize),blockSize>>>
        (varMemory, data.getDepth()) ;
    }
  }

  // Cleanup.
done:
  if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
  if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
  return context.passError(error, "nnbnorm_cudnn::forward") ;
}


template<typename T>
__global__ void std_to_var(T * var, T const * std, unsigned int num, T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    var[idx] = std[idx]*std[idx] - epsilon ;
  }
}

template<vl::DataType dataType>
vl::ErrorCode
vl::impl::nnbnorm_cudnn<dataType>::forward_given_moments(vl::Context& context,
                                                         vl::Tensor output,
                                                         vl::Tensor moments,
                                                         vl::Tensor data,
                                                         vl::Tensor multipliers,
                                                         vl::Tensor biases)
{
  assert(output) ;
  assert(data) ;
  assert(moments) ;
  assert(multipliers) ;
  assert(biases) ;

  typedef typename DataTypeTraits<dataType>::type type ;
  size_t workspaceSize ;
  type * workspace ;

  cudnnTensorDescriptor_t dataDesc, momentDesc ;
  bool dataDescInitialized = false ;
  bool momentDescInitialized = false ;

  cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
  vl::DataType dynDataType = output.getDataType() ;
  assert(dynDataType == dataType) ;

  cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
  vl::ErrorCode error = vl::VLE_Success ;
  cudnnHandle_t handle ;

  // Get CuDNN.
  CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

  // Get tensor descripotrs.
  CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
  dataDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   data.getSize(),
                                   data.getDepth(),
                                   data.getWidth(),
                                   data.getHeight())) ;

  CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
  dataDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   1, data.getDepth(), 1, 1)) ;

  // Allocate workspace.
  workspaceSize = data.getDepth() ;
  workspace = (type*)context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;

  // Run CuDNN batch normalization implementation.
  {
    type alpha = 1.0f ;
    type beta = 0.0f ;
    type * meanMemory = moments ? (type*)moments.getMemory() : workspace ;
    type * stdMemory = meanMemory + data.getDepth() ;
    type * varMemory = workspace ;

    size_t const blockSize = VL_CUDA_NUM_THREADS ;
    std_to_var<type> <<<divideAndRoundUp(data.getDepth(),blockSize),blockSize>>>
    (varMemory, stdMemory, data.getDepth(), CUDNN_BN_MIN_EPSILON) ;

    CHECK(cudnnBatchNormalizationForwardInference
          (handle,
           CUDNN_BATCHNORM_SPATIAL,
           &alpha,
           &beta,
           dataDesc, data.getMemory(),
           dataDesc, output.getMemory(),
           momentDesc, multipliers.getMemory(), biases.getMemory(),
           meanMemory, varMemory, CUDNN_BN_MIN_EPSILON)) ;
  }

  // Cleanup.
done:
  if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
  if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
  return context.passError(error, "nnbnorm_cudnn::forward") ;
}

template<vl::DataType dataType>
vl::ErrorCode
vl::impl::nnbnorm_cudnn<dataType>::backward(Context& context,
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
  assert(derData) ;
  assert(derMultipliers) ;
  assert(derBiases) ;
  assert(moments) ;
  assert(data) ;
  assert(multipliers) ;
  assert(biases) ;
  assert(derOutput) ;

  typedef typename DataTypeTraits<dataType>::type type ;
  size_t workspaceSize ;
  type * workspace ;
  size_t volume ;

  cudnnTensorDescriptor_t derOutputDesc, momentDesc ;
  bool derOutputDescInitialized = false ;
  bool momentDescInitialized = false ;

  cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
  vl::DataType dynDataType = derOutput.getDataType() ;
  assert(dynDataType == dataType) ;

  cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
  vl::ErrorCode error = vl::VLE_Success ;
  cudnnHandle_t handle ;

  // Get CuDNN.
  CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

  // Get tensor descripotrs.
  CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
  derOutputDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   derOutput.getSize(), // sizes
                                   derOutput.getDepth(),
                                   derOutput.getWidth(),
                                   derOutput.getHeight())) ;

  CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
  momentDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   1, data.getDepth(), 1, 1)) ;

  // Compute moments using CuDNN. Unfortunately CuDNN does not expose
  // the values of the moments in the backward pass, so we need to run
  // the forward code to get them.

  volume = derData.getNumElements() ;
  workspaceSize = (moments ? 0 : 2 * derData.getDepth()) + volume ;
  workspace = (type*)context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;

  {
    type alpha = 1.0f ;
    type beta = 0.0f ;
    type * outMemory = workspace ;
    type * meanMemory = moments ? (type*)moments.getMemory() : workspace + volume ;
    type * varMemory = meanMemory + data.getDepth() ;

    CHECK(cudnnBatchNormalizationForwardTraining
          (handle,
           CUDNN_BATCHNORM_SPATIAL,
           &alpha, &beta,
           derOutputDesc, data.getMemory(),
           derOutputDesc, outMemory, // will be discarded
           momentDesc, multipliers.getMemory(), biases.getMemory(),
           1.0, // cumulative factor for moments
           NULL, NULL,
           epsilon,
           meanMemory, varMemory)) ;

    CHECK(cudnnBatchNormalizationBackward
          (handle,
           CUDNN_BATCHNORM_SPATIAL,
           &alpha, &beta, // data
           &alpha, &beta, // params
           derOutputDesc, data.getMemory(), // input
           derOutputDesc, derOutput.getMemory(), // input
           derOutputDesc, derData.getMemory(), // output
           momentDesc, multipliers.getMemory(), // input
           derMultipliers.getMemory(), // output
           derBiases.getMemory(), // output
           epsilon,
           meanMemory, varMemory)) ;

    // Note: the CuDNN manual describes the varMemory output above
    // as inverse variance, but it is the inverse standard deviation instead.
    size_t const blockSize = VL_CUDA_NUM_THREADS ;
    inverse<type> <<<divideAndRoundUp(data.getDepth(),blockSize),blockSize>>>
    (varMemory, data.getDepth()) ;
  }

  // Cleanup.
done:
  if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
  if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
  return context.passError(error, "nnbnorm_cudnn::backward") ;
}

template<typename T>
__global__ void inverse(T * out, T * in, unsigned int num)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < num) {
    out[idx] = ((T)1) / in[idx] ;
  }
}

template<vl::DataType dataType>
vl::ErrorCode
vl::impl::nnbnorm_cudnn<dataType>::backward_given_moments(Context& context,
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
  assert(derData) ;
  assert(derMultipliers) ;
  assert(derBiases) ;
  assert(moments) ;
  assert(data) ;
  assert(multipliers) ;
  assert(biases) ;
  assert(derOutput) ;

  typedef typename DataTypeTraits<dataType>::type type ;
  size_t workspaceSize ;
  type * workspace ;

  cudnnTensorDescriptor_t derOutputDesc, dataDesc, momentDesc ;
  bool derOutputDescInitialized = false ;
  bool dataDescInitialized = false ;
  bool momentDescInitialized = false ;

  cudnnDataType_t cudnnDataType = DataTypeToCudnn<dataType>::id ;
  vl::DataType dynDataType = derOutput.getDataType() ;
  assert(dynDataType == dataType) ;

  cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
  vl::ErrorCode error = vl::VLE_Success ;
  cudnnHandle_t handle ;

  // Get CuDNN.
  CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

  // Get tensor descripotrs.
  CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
  derOutputDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(derOutputDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   derOutput.getSize(), // sizes
                                   derOutput.getDepth(),
                                   derOutput.getWidth(),
                                   derOutput.getHeight())) ;

  CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
  dataDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(dataDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   data.getSize(),
                                   data.getDepth(),
                                   data.getWidth(),
                                   data.getHeight())) ;

  CHECK(cudnnCreateTensorDescriptor(&momentDesc)) ;
  dataDescInitialized = true ;
  CHECK(cudnnSetTensor4dDescriptor(momentDesc,
                                   CUDNN_TENSOR_NCHW,
                                   cudnnDataType,
                                   1, data.getDepth(), 1, 1)) ;


  // Compute moments using CuDNN.
  workspaceSize = derData.getDepth() ;
  workspace = (type*)context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(type)) ;

  {
    type alpha = 1.0f ;
    type beta = 0.0f ;
    type * meanMemory = (type*)moments.getMemory() ;
    type * stdMemory = meanMemory + data.getDepth() ;
    type * istdMemory = workspace ;

    // Note: the CuDNN manual describes the varMemory output above
    // as inverse variance, but it is the inverse standard deviation instead.
    size_t const blockSize = VL_CUDA_NUM_THREADS ;
    inverse<type> <<<divideAndRoundUp(data.getDepth(),blockSize),blockSize>>>
    (istdMemory, stdMemory, data.getDepth()) ;

    CHECK(cudnnBatchNormalizationBackward
          (handle,
           CUDNN_BATCHNORM_SPATIAL,
           &alpha, &beta, // data
           &alpha, &beta, // params
           dataDesc, data.getMemory(), // input
           derOutputDesc, derOutput.getMemory(), // input
           dataDesc, derData.getMemory(), // output
           momentDesc, multipliers.getMemory(), // input
           derMultipliers.getMemory(), // output
           derBiases.getMemory(), // output
           epsilon,
           meanMemory, istdMemory)) ;
  }

  // Cleanup.
done:
  if (momentDescInitialized) { cudnnDestroyTensorDescriptor(momentDesc) ; }
  if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
  if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
  return context.passError(error, "nnbnorm_cudnn::backward_given_moments") ;
}

template struct vl::impl::nnbnorm_cudnn<vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::nnbnorm_cudnn<vl::VLDT_Double> ;
#endif

// @file data.hpp
// @brief Basic data structures (CUDA support)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__datacu__
#define __vl__datacu__

#ifndef ENABLE_GPU
#error "datacu.hpp cannot be compiled without GPU support"
#endif

#include "data.hpp"
#include <string>
#include <cuda.h>
#include <cublas_v2.h>
#if __CUDA_ARCH__ >= 200
#define VL_CUDA_NUM_THREADS 1024
#else
#define VL_CUDA_NUM_THREADS 512
#endif

#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif

namespace vl {

#if ENABLE_CUDNN
  namespace impl { template<vl::DataType type> struct nnconv_cudnn ; }
#endif

  class CudaHelper {
  public:
    // CUDA errors
    cudaError_t getLastCudaError() const ;
    std::string const& getLastCudaErrorMessage() const ;
    vl::ErrorCode catchCudaError(char const* description = NULL) ;

    // CUDA control
    vl::ErrorCode setStream(cudaStream_t streamId) ;
    cudaStream_t getStream() const ;

    // CuBLAS support
    cublasStatus_t getCublasHandle(cublasHandle_t* handle) ;
    void clearCublas() ;
    cublasStatus_t getLastCublasError() const ;
    std::string const& getLastCublasErrorMessage() const ;
    vl::ErrorCode catchCublasError(cublasStatus_t status,
                                   char const* description = NULL) ;

#if ENABLE_CUDNN
    // CuDNN support
    cudnnStatus_t getCudnnHandle(cudnnHandle_t* handle) ;
    void clearCudnn() ;
    bool getCudnnEnabled() const ;
    void setCudnnEnabled(bool active) ;

    // Convolution parameters
    void resetCudnnConvolutionSettings() ;
    void setCudnnConvolutionFwdAlgo(cudnnConvolutionFwdAlgo_t x) ;
    void setCudnnConvolutionFwdPreference(cudnnConvolutionFwdPreference_t x,
                                          size_t workSpaceLimit = 0) ;
    size_t getCudnnConvolutionFwdWorkSpaceUsed() const ;

    void setCudnnConvolutionBwdFilterAlgo(cudnnConvolutionBwdFilterAlgo_t x) ;
    void setCudnnConvolutionBwdFilterPreference(cudnnConvolutionBwdFilterPreference_t x,
                                                size_t workSpaceLimit = 0) ;
    size_t getCudnnConvolutionBwdFilterWorkSpaceUsed() const ;

    void setCudnnConvolutionBwdDataAlgo(cudnnConvolutionBwdDataAlgo_t x) ;
    void setCudnnConvolutionBwdDataPreference(cudnnConvolutionBwdDataPreference_t x,
                                              size_t workSpaceLimit = 0) ;
    size_t getCudnnConvolutionBwdDataWorkSpaceUsed() const ;

    cudnnStatus_t getLastCudnnError() const ;
    std::string const& getLastCudnnErrorMessage() const ;
    vl::ErrorCode catchCudnnError(cudnnStatus_t status,
                              char const* description = NULL) ;

    template<vl::DataType type> friend struct vl::impl::nnconv_cudnn ;
#endif

  protected:
    CudaHelper() ;
    ~CudaHelper() ;
    void clear() ;
    void invalidateGpu() ;
    friend class Context ;

  private:
    cudaError_t lastCudaError ;
    std::string lastCudaErrorMessage ;

    // Streams support
    cudaStream_t cudaStream ;

    // CuBLAS
    cublasHandle_t cublasHandle ;
    bool isCublasInitialized ;
    cublasStatus_t lastCublasError ;
    std::string lastCublasErrorMessage ;

#if ENABLE_CUDNN
    // CuDNN
    cudnnStatus_t lastCudnnError ;
    std::string lastCudnnErrorMessage ;
    cudnnHandle_t cudnnHandle ;
    bool isCudnnInitialized ;
    bool cudnnEnabled ;

    bool cudnnConvolutionFwdSpecificAlgo ;
    cudnnConvolutionFwdPreference_t cudnnConvolutionFwdPreference ;
    cudnnConvolutionFwdAlgo_t cudnnConvolutionFwdAlgo ;
    size_t cudnnConvolutionFwdWorkSpaceLimit ;
    size_t cudnnConvolutionFwdWorkSpaceUsed  ;

    bool cudnnConvolutionBwdFilterSpecificAlgo ;
    cudnnConvolutionBwdFilterPreference_t  cudnnConvolutionBwdFilterPreference;
    cudnnConvolutionBwdFilterAlgo_t cudnnConvolutionBwdFilterAlgo ;
    size_t cudnnConvolutionBwdFilterWorkSpaceLimit ;
    size_t cudnnConvolutionBwdFilterWorkSpaceUsed  ;

    bool cudnnConvolutionBwdDataSpecificAlgo ;
    cudnnConvolutionBwdDataPreference_t cudnnConvolutionBwdDataPreference ;
    cudnnConvolutionBwdDataAlgo_t cudnnConvolutionBwdDataAlgo ;
    size_t cudnnConvolutionBwdDataWorkSpaceLimit ;
    size_t cudnnConvolutionBwdDataWorkSpaceUsed  ;
#endif
  } ;
}
#endif /* defined(__vl__datacu__) */

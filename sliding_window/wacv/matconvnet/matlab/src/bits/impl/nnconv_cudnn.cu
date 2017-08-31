// @file nnconv_cudnn.cu
// @brief Convolution block CuDNN-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#if !defined(ENABLE_GPU) | !defined(ENABLE_CUDNN)
#error "nnconv_cudnn.hpp cannot be compiled without GPU and CUDNN support."
#endif

#include "nnconv_cudnn.hpp"
#include "cudnnhelper.hpp"
#include "../datacu.hpp"
#include <assert.h>
#include <algorithm>

using namespace vl ;

#define CHECK(x) \
{ \
cudnnError = x ; \
if (cudnnError != CUDNN_STATUS_SUCCESS) { \
error = context.setError(context.getCudaHelper().catchCudnnError(cudnnError, \
STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__))) ; \
goto done ; \
} }

/* ---------------------------------------------------------------- */
/*                                             nnconv_forward_cudnn */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnconv_cudnn<dataType>::forward(Context& context,
                                            Tensor output, double outputMult,
                                            Tensor data, double dataMult,
                                            Tensor filters,
                                            Tensor biases,
                                            int strideY, int strideX,
                                            int padTop, int padBottom,
                                            int padLeft, int padRight,
                                            int dilateY, int dilateX)
  {
    assert(output) ;
    assert(data) ;
    assert(filters) ;

    typedef typename DataTypeTraits<dataType>::type type ;

    cudnnTensorDescriptor_t outputDesc, biasesDesc, dataDesc ;
    cudnnFilterDescriptor_t filtersDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool outputDescInitialized = false ;
    bool biasesDescInitialized = false ;
    bool dataDescInitialized = false ;
    bool filtersDescInitialized = false ;
    bool convDescInitialized = false ;

    void* workSpace = NULL ;

    int numGroups = data.getDepth() / filters.getDepth() ;
    int numFiltersPerGroup = filters.getSize() / numGroups ;

    if (dilateX != 1 || dilateY != 1) return vl::VLE_Unsupported ;
    if (padLeft != padRight) return vl::VLE_Unsupported ;
    if (padTop != padBottom) return vl::VLE_Unsupported ;
    if (filters.getHeight() > data.getHeight()) return vl::VLE_Unsupported ;
    if (filters.getWidth() > data.getWidth()) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get tensor descripotrs
    CHECK(cudnnCreateTensorDescriptor(&outputDesc)) ;
    outputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(outputDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       output.getSize(), // sizes
                                       numFiltersPerGroup,
                                       output.getWidth(),
                                       output.getHeight(),
                                       output.getHeight()*output.getWidth()*output.getDepth(), //strides
                                       output.getHeight()*output.getWidth(),
                                       output.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
    dataDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                       DataTypeToCudnn<dataType>::id,
                                       data.getSize(),
                                       data.getDepth() / numGroups,
                                       data.getWidth(),
                                       data.getHeight(),
                                       data.getHeight()*data.getWidth()*data.getDepth(), //strides
                                       data.getHeight()*data.getWidth(),
                                       data.getHeight(),
                                       1)) ;

    CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
    filtersDescInitialized = true ;
    CHECK(cudnnSetFilter4dDescriptor(filtersDesc,
                                     DataTypeToCudnn<dataType>::id,
                                     IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                     numFiltersPerGroup,
                                     filters.getDepth(),
                                     filters.getWidth(),
                                     filters.getHeight())) ;

    if (biases) {
      CHECK(cudnnCreateTensorDescriptor(&biasesDesc)) ;
      biasesDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(biasesDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::id ,
                                       1,
                                       biases.getNumElements() / numGroups,
                                       1,
                                       1)) ;
    }

    // Get convolution descriptor
    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                          padLeft, padTop,
                                          strideX, strideY,
                                          1,1, // upscale
                                          CUDNN_CROSS_CORRELATION)) ;
    // Sanity check
#if 1
    {
      int n, c, h, w ;
      cudnnGetConvolution2dForwardOutputDim(convDesc,
                                            dataDesc,
                                            filtersDesc,
                                            &n, &c, &w, &h) ;
      bool sane =
      output.getSize() == n &&
      numFiltersPerGroup == c &&
      output.getWidth() == w &&
      output.getHeight() == h ;
      assert(sane) ;
    }
#endif

    context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

    if (!context.getCudaHelper().cudnnConvolutionFwdSpecificAlgo) {
      // Determine algorithm automatically
      CHECK(cudnnGetConvolutionForwardAlgorithm(handle,
                                                dataDesc,
                                                filtersDesc,
                                                convDesc,
                                                outputDesc,
                                                context.getCudaHelper().cudnnConvolutionFwdPreference,
                                                context.getCudaHelper().cudnnConvolutionFwdWorkSpaceLimit,
                                                &context.getCudaHelper().cudnnConvolutionFwdAlgo)) ;
    }

    // Get workspace size
    CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  dataDesc,
                                                  filtersDesc,
                                                  convDesc,
                                                  outputDesc,
                                                  context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                                  &context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed)) ;

    // Get workspace
    if (context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed > 0) {
      workSpace = context.getWorkspace(vl::VLDT_GPU, context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed) ;
      if (workSpace == NULL) {
        error = context.getLastError() ;
        goto done ;
      }
    }

    // Perform convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * filters.getDepth()) *  g ;
      ptrdiff_t filtersGrpOffset = (filters.getHeight() * filters.getWidth() * filters.getDepth()) * numFiltersPerGroup * g ;
      ptrdiff_t outputGrpOffset = (output.getHeight() * output.getWidth() * numFiltersPerGroup) * g ;
      ptrdiff_t biasesGrpOffset = numFiltersPerGroup * g ;

      type alpha = dataMult ;
      type beta = outputMult ;
      CHECK(cudnnConvolutionForward(handle,
                                    &alpha,
                                    dataDesc, (type const*)data.getMemory() + dataGrpOffset,
                                    filtersDesc, (type const*)filters.getMemory() + filtersGrpOffset,
                                    convDesc,
                                    context.getCudaHelper().cudnnConvolutionFwdAlgo,
                                    workSpace, context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed,
                                    &beta,
                                    outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;

      if (biases) {
        type alpha = 1.0f ;
        type beta = 1.0f ;
#if (CUDNN_VERSION < 4000)
        CHECK(cudnnAddTensor(handle,
                             CUDNN_ADD_SAME_C,
                             &alpha,
                             biasesDesc, (type const*)biases.getMemory() + biasesGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#else
        CHECK(cudnnAddTensor(handle,
                             &alpha,
                             biasesDesc, (type const*)biases.getMemory() + biasesGrpOffset,
                             &beta,
                             outputDesc, (type*)output.getMemory() + outputGrpOffset)) ;
#endif
      }
    }

    /* cleanup */
  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    if (biasesDescInitialized) { cudnnDestroyTensorDescriptor(biasesDesc) ; }
    if (outputDescInitialized) { cudnnDestroyTensorDescriptor(outputDesc) ; }
    return context.passError(error, __func__) ;
  }

  /* ---------------------------------------------------------------- */
  /*                                            nnconv_backward_cudnn */
  /* ---------------------------------------------------------------- */

  template<vl::DataType dataType>
  vl::ErrorCode
  vl::impl::nnconv_cudnn<dataType>::backward(Context& context,
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
    typedef typename DataTypeTraits<dataType>::type type ;

    /* no derDataDesc needed as same as dataDesc */
    cudnnTensorDescriptor_t dataDesc, derBiasesDesc, derOutputDesc ;
    cudnnFilterDescriptor_t filtersDesc ;
    cudnnConvolutionDescriptor_t convDesc ;
    bool dataDescInitialized = false ;
    bool derBiasesDescInitialized = false ;
    bool derOutputDescInitialized = false ;
    bool filtersDescInitialized = false ;
    bool convDescInitialized = false ;

#if (CUDNN_VERSION >= 3000)
    void* workSpace = NULL ;
    size_t workSpaceSize = 0 ;
#endif

    ptrdiff_t numGroups = 1 ;
    ptrdiff_t numFiltersPerGroup = 0 ;
    ptrdiff_t filtersVolume = 0 ;

    if (dilateX != 1 || dilateY != 1) return vl::VLE_Unsupported ;
    if (padLeft != padRight) return vl::VLE_Unsupported ;
    if (padTop != padBottom) return vl::VLE_Unsupported ;

    cudnnStatus_t cudnnError = CUDNN_STATUS_SUCCESS ;
    vl::ErrorCode error = vl::VLE_Success ;
    cudnnHandle_t handle ;

    // Get CuDNN
    CHECK(context.getCudaHelper().getCudnnHandle(&handle)) ;

    // Get the dimensions of the tensrors involved
    // If derData is specified (hence comptued as output), use this
    // tensor as a basis to compute such dimensions, otherwise use derFilters.

    if (derData) {
      assert(filters) ;
      numGroups = derData.getDepth() / filters.getDepth() ;
      numFiltersPerGroup = filters.getSize() / numGroups ;
      filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                         DataTypeToCudnn<dataType>::id ,
                                         derData.getSize(),
                                         derData.getDepth() / numGroups,
                                         derData.getWidth(),
                                         derData.getHeight(),
                                         derData.getHeight()*derData.getWidth()*derData.getDepth(), //strides
                                         derData.getHeight()*derData.getWidth(),
                                         derData.getHeight(),
                                         1)) ;

      CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
      filtersDescInitialized = true ;
      CHECK(cudnnSetFilter4dDescriptor(filtersDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       numFiltersPerGroup,
                                       filters.getDepth(),
                                       filters.getWidth(),
                                       filters.getHeight())) ;
    } else if (derFilters) {
      assert(data) ;
      numGroups = data.getDepth() / derFilters.getDepth() ;
      numFiltersPerGroup = derFilters.getSize() / numGroups ;
      filtersVolume = derFilters.getHeight() * derFilters.getWidth() * derFilters.getDepth() ;

      CHECK(cudnnCreateTensorDescriptor(&dataDesc)) ;
      dataDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptorEx(dataDesc,
                                         DataTypeToCudnn<dataType>::id ,
                                         data.getSize(),
                                         data.getDepth() / numGroups,
                                         data.getWidth(),
                                         data.getHeight(),
                                         data.getHeight()*data.getWidth()*data.getDepth(), //strides
                                         data.getHeight()*data.getWidth(),
                                         data.getHeight(),
                                         1)) ;

      CHECK(cudnnCreateFilterDescriptor(&filtersDesc)) ;
      filtersDescInitialized = true ;
      CHECK(cudnnSetFilter4dDescriptor(filtersDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       IF_CUDNN_GE5(CUDNN_TENSOR_NCHW COMMA)
                                       numFiltersPerGroup,
                                       derFilters.getDepth(),
                                       derFilters.getWidth(),
                                       derFilters.getHeight())) ;
    }

    CHECK(cudnnCreateConvolutionDescriptor(&convDesc)) ;
    convDescInitialized = true ;
    CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                          padLeft, padTop,
                                          strideX, strideY,
                                          1,1, // upscale
                                          CUDNN_CROSS_CORRELATION)) ;

    // Must have derOutput for all derivatives
    assert(derOutput) ;
    CHECK(cudnnCreateTensorDescriptor(&derOutputDesc)) ;
    derOutputDescInitialized = true ;
    CHECK(cudnnSetTensor4dDescriptorEx(derOutputDesc,
                                       DataTypeToCudnn<dataType>::id ,
                                       derOutput.getSize(), // sizes
                                       numFiltersPerGroup,
                                       derOutput.getWidth(),
                                       derOutput.getHeight(),
                                       derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth(), //strides
                                       derOutput.getHeight()*derOutput.getWidth(),
                                       derOutput.getHeight(),
                                       1)) ;

    // for derivatives w.r.t. bias
    if (derBiases) {
      CHECK(cudnnCreateTensorDescriptor(&derBiasesDesc)) ;
      derBiasesDescInitialized = true ;
      CHECK(cudnnSetTensor4dDescriptor(derBiasesDesc,
                                       CUDNN_TENSOR_NCHW,
                                       DataTypeToCudnn<dataType>::id ,
                                       1,
                                       derBiases.getNumElements() / numGroups,
                                       1,
                                       1)) ;
    }


    context.getCudaHelper().cudnnConvolutionFwdWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed = 0 ;
    context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed = 0 ;

#if (CUDNN_VERSION >= 3000)

    if (derFilters) {
      // Get filter derivatives algorithm
      CHECK(cudnnGetConvolutionBackwardFilterAlgorithm
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filtersDesc,
             context.getCudaHelper().cudnnConvolutionBwdFilterPreference,
             context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceLimit,
             &context.getCudaHelper().cudnnConvolutionBwdFilterAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize
            (handle,
             dataDesc,
             derOutputDesc,
             convDesc,
             filtersDesc,
             context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
             &context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, context.getCudaHelper().cudnnConvolutionBwdFilterWorkSpaceUsed) ;
    }

    if (derData) {
      // Get data derivatives
      CHECK(cudnnGetConvolutionBackwardDataAlgorithm
            (handle,
             filtersDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             context.getCudaHelper().cudnnConvolutionBwdDataPreference,
             context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceLimit,
             &context.getCudaHelper().cudnnConvolutionBwdDataAlgo)) ;

      // Get workspace size
      CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize
            (handle,
             filtersDesc,
             derOutputDesc,
             convDesc,
             dataDesc,
             context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
             &context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed)) ;
      workSpaceSize = std::max(workSpaceSize, context.getCudaHelper().cudnnConvolutionBwdDataWorkSpaceUsed) ;
    }

    // Get workspace
    if (workSpaceSize > 0) {
      workSpace = context.getWorkspace(vl::VLDT_GPU, workSpaceSize) ;
      if (workSpace == NULL) {
        error = context.getLastError() ;
        goto done ;
      }
    }
#endif

    // Perform backward convolution for each filter group
    for (int g = 0  ; g < numGroups ; ++g) {
      ptrdiff_t filtersGrpOffset = filtersVolume * numFiltersPerGroup  * g ;
      ptrdiff_t derOutputGrpOffset = (derOutput.getHeight() * derOutput.getWidth() * numFiltersPerGroup) * g ;

      if (derBiases) {
        ptrdiff_t derBiasesGrpOffset = numFiltersPerGroup * g ;
        type alpha = 1 ;
        type beta = 0 ;
        CHECK(cudnnConvolutionBackwardBias
              (handle,
               &alpha,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               &beta,
               derBiasesDesc, (type*)derBiases.getMemory() + derBiasesGrpOffset)) ;
      }

      if (derFilters) {
        ptrdiff_t dataGrpOffset = (data.getHeight() * data.getWidth() * derFilters.getDepth()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;
#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardFilter)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardFilter_v3)
              (handle,
               &alpha,
               dataDesc, (type const*)data.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               context.getCudaHelper().cudnnConvolutionBwdFilterAlgo,
               workSpace, workSpaceSize,
               &beta,
               filtersDesc, (type*)derFilters.getMemory() + filtersGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardFilter
              (handle,
               &alpha,
               dataDesc, (type const*)data.getMemory() + dataGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               filtersDesc, (type*)derFilters.getMemory() + filtersGrpOffset)) ;
#endif
      }

      if (derData) {
        ptrdiff_t dataGrpOffset = (derData.getHeight() * derData.getWidth() * filters.getDepth()) *  g ;
        type alpha = 1 ;
        type beta = 0 ;

#if (CUDNN_VERSION >= 3000)
        CHECK(
              IF_CUDNN_GE4(cudnnConvolutionBackwardData)
              IF_CUDNN_GE3_LT4(cudnnConvolutionBackwardData_v3)
              (handle,
               &alpha,
               filtersDesc, (type const*)filters.getMemory() + filtersGrpOffset,
               derOutputDesc, (type const*)derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               context.getCudaHelper().cudnnConvolutionBwdDataAlgo,
               workSpace, workSpaceSize,
               &beta,
               dataDesc, (type*)derData.getMemory() + dataGrpOffset)) ;
#else
        CHECK(cudnnConvolutionBackwardData
              (handle,
               &alpha,
               filtersDesc, filters.getMemory() + filtersGrpOffset,
               derOutputDesc, derOutput.getMemory() + derOutputGrpOffset,
               convDesc,
               &beta,
               dataDesc, derData.getMemory() + dataGrpOffset)) ;
#endif
      }
    }

  done:
    if (convDescInitialized) { cudnnDestroyConvolutionDescriptor(convDesc) ; }
    if (filtersDescInitialized) { cudnnDestroyFilterDescriptor(filtersDesc) ; }
    if (derOutputDescInitialized) { cudnnDestroyTensorDescriptor(derOutputDesc) ; }
    if (derBiasesDescInitialized) { cudnnDestroyTensorDescriptor(derBiasesDesc) ; }
    if (dataDescInitialized) { cudnnDestroyTensorDescriptor(dataDesc) ; }
    return context.passError(error, __func__) ;
  }

} }

// Instantiations
template struct vl::impl::nnconv_cudnn<vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::nnconv_cudnn<vl::VLDT_Double> ;
#endif




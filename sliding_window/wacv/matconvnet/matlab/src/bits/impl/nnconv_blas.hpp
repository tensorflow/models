// @file nnconv_blas.hpp
// @brief Convolution block BLAS-based implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnconv_blas__
#define __vl__nnconv_blas__

#include "im2row.hpp"
#include "blashelper.hpp"
#include <assert.h>

namespace vl { namespace impl {

  template<vl::DeviceType deviceType, vl::DataType dataType> inline vl::ErrorCode
  nnconv_forward_blas(Context& context,
                      Tensor output, double outputMult,
                      Tensor data, double dataMult,
                      Tensor filters,
                      Tensor biases,
                      int strideY, int strideX,
                      int padTop, int padBottom,
                      int padLeft, int padRight,
                      int dilateY, int dilateX) ;

  template<vl::DeviceType deviceType, vl::DataType dataType> inline vl::ErrorCode
  nnconv_backward_blas(Context& context,
                       Tensor derData,
                       Tensor derFilters,
                       Tensor derBiases,
                       Tensor data,
                       Tensor filters,
                       Tensor derOutput,
                       int strideY, int strideX,
                       int padTop, int padBottom,
                       int padLeft, int padRight,
                       int dilateY, int dilateX) ;

} }

/*

 One image at a time is processed.

 Filters are (optionally) divided in to groups, one for each group of dimensions.


                 patchVolume                  numFilters
                 +-------------------------+   +-----------------------+

                 filtersVolume              numFiltersPerGroup
                 +------------+------------+   +-----------+-----------+      +--------+--------+
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |  filter   |           |      |        |        |
                 |            |            |   |  group 1  |     0     |  =   |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------------------+      |        |        |
 numOutputPixels |   grp. 1   |   grp. 2   |   |           |           |      |        |        |
                 |            |            |   |           |  filter   |      |        |        |
                 |            |            |   |     0     |  group 2  |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   |           |           |      |        |        |
                 |            |            |   +-----------+-----------+      |        |        |
                 |            |            |                                  |        |        |
                 |            |            |            filters               |        |        |
                 |            |            |                                  |        |        |
                 +------------+------------+                                  +--------+--------+

                 temp                                                     output

 */

template<vl::DeviceType deviceType, vl::DataType dataType> inline vl::ErrorCode
vl::impl::nnconv_forward_blas(Context& context,
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

  vl::ErrorCode error ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;

  ptrdiff_t numGroups = data.getDepth() / filters.getDepth() ;
  ptrdiff_t numFiltersPerGroup = filters.getSize() / numGroups ;
  ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
  ptrdiff_t filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;
  ptrdiff_t tempVolume = numOutputPixels * filtersVolume * numGroups ;

  type* tempMemory = (type*) context.getWorkspace(deviceType, tempVolume * sizeof(type)) ;
  type const* allOnesMemory = (type*) context.getAllOnes(deviceType,
                                                         dataType,
                                                         numOutputPixels) ;
  if (tempMemory == NULL || allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < data.getSize() ; ++image) {

    ptrdiff_t dataOffset = (data.getHeight()*data.getWidth()*data.getDepth()) * image ;
    ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;

    error = vl::impl::im2row<deviceType,type>::forward
    (context,
     tempMemory,
     (type*)data.getMemory() + dataOffset,
     data.getHeight(), data.getWidth(), data.getDepth(),
     filters.getHeight(), filters.getWidth(),
     strideY, strideX,
     padTop, padBottom, padLeft, padRight,
     dilateY, dilateX) ;
    if (error != vl::VLE_Success) { goto done ; }

    for (int g = 0 ; g < numGroups ; ++ g) {
      ptrdiff_t filterGrpOffset = filtersVolume * numFiltersPerGroup * g ;
      ptrdiff_t tempGrpOffset = numOutputPixels * filtersVolume * g ;
      ptrdiff_t outputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
      type alpha = dataMult ;
      type beta = outputMult ;
      error = vl::impl::blas<deviceType,dataType>::gemm
      (context,
       'n', 'n',
       numOutputPixels, numFiltersPerGroup, filtersVolume,
       alpha,
       tempMemory + tempGrpOffset, numOutputPixels,
       (type*)filters.getMemory() + filterGrpOffset, filtersVolume,
       beta,
       (type*)output.getMemory() + outputOffset + outputGrpOffset, numOutputPixels) ;
      if (error != vl::VLE_Success) { goto done ; }
    }

    if (biases) {
      type alpha = 1 ;
      type beta = 1 ;
      error = vl::impl::blas<deviceType,dataType>::gemm
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

template<vl::DeviceType deviceType, vl::DataType dataType>
inline vl::ErrorCode
vl::impl::nnconv_backward_blas(Context& context,
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
  vl::ErrorCode error ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;

  ptrdiff_t numGroups = 0 ;
  ptrdiff_t numFiltersPerGroup = 0 ;
  ptrdiff_t filtersVolume = 0 ;
  type const* allOnesMemory = NULL ;
  ptrdiff_t tempVolume = 0 ;
  type* tempMemory = NULL ;

  // for all derivatives
  assert(derOutput) ;
  ptrdiff_t numOutputPixels = derOutput.getHeight() * derOutput.getWidth() ;

  if (derBiases) {
    // for derivative w.r.t. bias
    allOnesMemory = (type*) context.getAllOnes(deviceType,
                                               dataType,
                                               numOutputPixels) ;
    if (allOnesMemory == NULL) {
      error = context.getLastError() ;
      goto done ;
    }
  }

  if (derData) {
    // for derivative w.r.t. data
    assert(filters) ;
    numGroups = derData.getDepth() / filters.getDepth() ;
    filtersVolume = filters.getHeight() * filters.getWidth() * filters.getDepth() ;
  }
  else if (derFilters) {
    // for derivative w.r.t. filters
    assert(data) ;
    numGroups = data.getDepth() / derFilters.getDepth() ;
    filtersVolume = derFilters.getHeight() * derFilters.getWidth() * derFilters.getDepth() ;
  }
  numFiltersPerGroup = derOutput.getDepth() / numGroups ;

  // get scratch space
  tempVolume = numOutputPixels * filtersVolume * numGroups ;
  if (tempVolume) {
    tempMemory = (type*) context.getWorkspace(deviceType, tempVolume * sizeof(type)) ;
    if (tempMemory == NULL) {
      error = context.getLastError() ;
      goto done ;
    }
  }

  for (int image = 0 ; image < derOutput.getSize() ; ++image) {

    ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

    /* compute derData dz/dbias */
    if (derBiases) {
      // has derBiases, derOutput
      type alpha = 1 ;
      type beta = (image > 0) ; /* this saves init. the output array with 0 */
      error = vl::impl::blas<deviceType,dataType>::gemv
      (context,
       't',
       numOutputPixels, derOutput.getDepth(),
       alpha, /* alpha */
       (type const*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
       allOnesMemory, 1,
       beta, /* beta */
       (type*)derBiases.getMemory(), 1) ;
      if (error != vl::VLE_Success) { return error ; }
    }

    /* compute derData dz/dx */
    if (derData) {
      // has derData, derOutput, filters
      ptrdiff_t derDataOffset = (derData.getHeight()*derData.getWidth()*derData.getDepth()) * image ;
      for (int g = 0 ; g < numGroups ; ++ g) {
        ptrdiff_t filterGrpOffset = filtersVolume * numFiltersPerGroup * g ;
        ptrdiff_t tempGrpOffset = numOutputPixels * filtersVolume * g ;
        ptrdiff_t derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        type alpha = 1 ;
        type beta = 0 ;
        error = vl::impl::blas<deviceType,dataType>::gemm
        (context,
         'n', 't',
         numOutputPixels, filtersVolume, numFiltersPerGroup,
         alpha,
         (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
         (type*)filters.getMemory() + filterGrpOffset, filtersVolume,
         beta,
         tempMemory + tempGrpOffset, numOutputPixels) ;
        if (error != vl::VLE_Success) { return error ; }
      }
      error = vl::impl::im2row<deviceType,type>::backward
      (context,
       (type*)derData.getMemory() + derDataOffset,
       tempMemory,
       derData.getHeight(), derData.getWidth(), derData.getDepth(),
       filters.getHeight(), filters.getWidth(),
       strideY, strideX,
       padTop, padBottom, padLeft, padRight,
       dilateY, dilateX) ;
      if (error != vl::VLE_Success) { return error ; }
    }

    /* compute derFilters dz/dF */
    if (derFilters) {
      // has derFilters, derOutput, data
      ptrdiff_t dataOffset = (data.getHeight()*data.getWidth()*data.getDepth()) * image ;
      error = vl::impl::im2row<deviceType,type>::forward
      (context,
       (type*)tempMemory,
       (type*)data.getMemory() + dataOffset,
       data.getHeight(), data.getWidth(), data.getDepth(),
       derFilters.getHeight(), derFilters.getWidth(),
       strideY, strideX,
       padTop, padBottom, padLeft, padRight,
       dilateY, dilateX) ;
      if (error != vl::VLE_Success) { return error ; }
      for (int g = 0 ; g < numGroups ; ++ g) {
        ptrdiff_t filterGrpOffset = filtersVolume * numFiltersPerGroup * g ;
        ptrdiff_t tempGrpOffset = numOutputPixels * filtersVolume * g ;
        ptrdiff_t derOutputGrpOffset = numOutputPixels * numFiltersPerGroup * g  ;
        /* dzdF = temp' * dzdY */
        type alpha = 1 ;
        type beta = (image > 0) ; /* this saves init. the output array with 0 */
        error = vl::impl::blas<deviceType,dataType>::gemm
        (context,
         't', 'n',
         filtersVolume, numFiltersPerGroup, numOutputPixels,
         alpha,
         tempMemory + tempGrpOffset, numOutputPixels,
         (type*)derOutput.getMemory() + derOutputOffset + derOutputGrpOffset, numOutputPixels,
         beta,
         (type*)derFilters.getMemory() + filterGrpOffset, filtersVolume) ;
        if (error != vl::VLE_Success) { return error ; }
      }
    }
  }

done:
  return context.passError(error, __func__) ;
}

#endif /* defined(__vl__nnconv_blas__) */

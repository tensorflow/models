// @file nnbias_blas.hpp
// @brief Bias block BLAS implementation.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__nnbias_blas__
#define __vl__nnbias_blas__

#include <assert.h>
#include "blashelper.hpp"

namespace vl { namespace impl {

  template<vl::DeviceType deviceType, vl::DataType dataType>
  inline vl::ErrorCode
  nnbias_forward_blas(vl::Context& context,
                      vl::Tensor output, double outputMult,
                      vl::Tensor data, double dataMult,
                      vl::Tensor biases, double biasesMult) ;

  template<vl::DeviceType deviceType, vl::DataType dataType>
  inline vl::ErrorCode
  nnbias_backward_blas(vl::Context& context,
                       vl::Tensor derData, double derDataMult,
                       vl::Tensor derBiases, double derBiasesMult,
                       vl::Tensor derOutput, double derOutputMult) ;

} }

template<vl::DeviceType deviceType, vl::DataType dataType>
inline vl::ErrorCode
vl::impl::nnbias_forward_blas(vl::Context& context,
                              vl::Tensor output, double outputMult,
                              vl::Tensor data, double dataMult,
                              vl::Tensor biases, double biasesMult)
{
  vl::ErrorCode error ;
  ptrdiff_t numOutputPixels = output.getHeight() * output.getWidth() ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;

  type const* allOnesMemory = (type*) context.getAllOnes(deviceType,
                                                         dataType,
                                                         numOutputPixels) ;
  if (allOnesMemory == NULL) {
    error = context.getLastError() ;
    goto done ;
  }

  for (int image = 0 ; image < output.getSize() ; ++image) {
    ptrdiff_t outputOffset = (output.getHeight()*output.getWidth()*output.getDepth()) * image ;
    double alpha = outputMult ;

    if (biases) {
      error = vl::impl::blas<deviceType,dataType>::gemm
      (context,
       'n', 'n',
       numOutputPixels, biases.getNumElements(), 1,
       biasesMult,
       allOnesMemory, numOutputPixels,
       (type*)biases.getMemory(), 1,
       alpha,
       (type*)output.getMemory() + outputOffset, numOutputPixels) ;
      if (error != vl::VLE_Success) { goto done ; }
      alpha = 1 ;
    }

    if (data) {
      assert(false) ; // todo: not implemented
      if (error != vl::VLE_Success) { goto done ; }
    }
  }
done:
  return context.passError(error, __func__) ;
}

template<vl::DeviceType deviceType, vl::DataType dataType> inline vl::ErrorCode
vl::impl::nnbias_backward_blas(vl::Context& context,
                               vl::Tensor derData, double derDataMult,
                               vl::Tensor derBiases, double derBiasesMult,
                               vl::Tensor derOutput, double derOutputMult)
{
  vl::ErrorCode error ;
  typedef typename vl::DataTypeTraits<dataType>::type type ;
  type const* allOnesMemory = NULL ;

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
    assert(false) ; // not implemented
  }

  for (int image = 0 ; image < derOutput.getSize() ; ++image) {

    ptrdiff_t derOutputOffset = (derOutput.getHeight()*derOutput.getWidth()*derOutput.getDepth()) * image ;

    /* compute derData dz/dbias */
    if (derBiases) {
      // has derBiases, derOutput
      error = vl::impl::blas<deviceType,dataType>::gemv
      (context,
       't',
       numOutputPixels, derOutput.getDepth(),
       derOutputMult, /* alpha */
       (type*)derOutput.getMemory() + derOutputOffset, numOutputPixels,
       allOnesMemory, 1,
       (image == 0) ? derBiasesMult : 1.0, /* beta */
       (type*)derBiases.getMemory(), 1) ;
      if (error != vl::VLE_Success) { return error ; }
    }

    /* compute derData dz/dx */
    if (derData) {
      assert(false) ; // todo: not implemented
      if (error != vl::VLE_Success) { return error ; }
    }
  }

done:
  return context.passError(error, __func__) ;
}

#endif /* defined(__vl__nnbias_blas__) */

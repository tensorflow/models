// @file nnroipooling.cu
// @brief roipooling block
// @author Hakan Bilen
// @author Abishek Dutta
// @author Andrea Vedaldi

/*
Copyright (C) 2016 Hakan Bilen, Abishek Dutta, and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/


#include "nnroipooling.hpp"
#include "impl/roipooling.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

#include <assert.h>

using namespace vl ;

/* ---------------------------------------------------------------- */
/*                                             nnroipooling_forward */
/* ---------------------------------------------------------------- */

#define DISPATCH(deviceType, op, type) \
status = vl::impl::op<deviceType, type>::forward \
((type*)output.getMemory(), (type const*)data.getMemory(), \
data.getHeight(), data.getWidth(), data.getDepth(), data.getSize(), \
(type const *)rois.getMemory(), rois.getNumElements() / 5, \
subdivisions, transform) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH(deviceType, op, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, op, double) ; break ;) \
default: assert(false) ; return VLE_Unknown ; \
}

#define DISPATCH3(deviceType) \
switch (method) { \
case vlROIPoolingAverage : DISPATCH2(deviceType, roipooling_average) ; break ; \
case vlROIPoolingMax : DISPATCH2(deviceType, roipooling_max) ; break ; \
default: assert(false) ; return VLE_Unknown ; \
}

vl::ErrorCode
vl::nnroipooling_forward(vl::Context& context,
                         vl::Tensor output,
                         vl::Tensor data,
                         vl::Tensor rois,
                         ROIPoolingMethod method,
                         int const subdivisions[2],
                         double const transform[6])
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = output.getDeviceType() ;
  vl::DataType dataType = output.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#ifdef ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH3(vl::VLDT_GPU) ;
      if (status == vl::VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }
  return context.passError(status, "nnroipooling_forward") ;
}

/* ---------------------------------------------------------------- */
/*                                            nnroipooling_backward */
/* ---------------------------------------------------------------- */

#undef DISPATCH
#undef DISPATCH2

// backward max and average want slightly differet argument lists

#define DISPATCH_roipooling_average(deviceType, type) \
status = vl::impl::roipooling_average<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(), \
(const type *)rois.getMemory(), rois.getNumElements() / 5, \
(type const*)derOutput.getMemory(), \
subdivisions, transform) ;

#define DISPATCH_roipooling_max(deviceType, type) \
status = vl::impl::roipooling_max<deviceType, type>::backward \
((type*)derData.getMemory(), (type const*)data.getMemory(), \
derData.getHeight(), derData.getWidth(), derData.getDepth(), derData.getSize(), \
(const type *)rois.getMemory(), rois.getNumElements() / 5, \
(type const*)derOutput.getMemory(), \
subdivisions, transform) ;

#define DISPATCH2(deviceType, op) \
switch (dataType) { \
case VLDT_Float : DISPATCH_ ## op (deviceType, float) ; break ; \
IF_DOUBLE(case VLDT_Double : DISPATCH_ ## op (deviceType, double) ; break ;) \
default: assert(false) ; return vl::VLE_Unknown ; \
}

vl::ErrorCode
vl::nnroipooling_backward(vl::Context& context,
                          vl::Tensor derData,
                          vl::Tensor data,
                          vl::Tensor rois,
                          vl::Tensor derOutput,
                          ROIPoolingMethod method,
                          int const subdivisions[2],
                          double const transform[6])
{
  vl::ErrorCode status = VLE_Success ;
  vl::DeviceType deviceType = derOutput.getDeviceType() ;
  vl::DataType dataType = derOutput.getDataType() ;

  switch (deviceType) {
    default:
      assert(false) ;
      return vl::VLE_Unknown ;

    case vl::VLDT_CPU:
      DISPATCH3(vl::VLDT_CPU) ;
      break ;

#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH3(vl::VLDT_GPU) ;
      if (status == vl::VLE_Cuda) {
        context.setError(context.getCudaHelper().catchCudaError(__func__)) ;
      }
      break ;
#endif
  }

  return context.passError(status, "nnroipooling_backward") ;
}

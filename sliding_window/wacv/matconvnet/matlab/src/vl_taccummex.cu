/** @file vl_taccummx.cu
 ** @brief VL_TACCUM MEX helper.
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2016 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/impl/blashelper.hpp"
#include "bits/impl/copy.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <memory>
#include <assert.h>

enum {
  IN_ALPHA, IN_A, IN_BETA, IN_B, IN_END
} ;

enum {
  OUT_RESULT, OUT_END
} ;

/* option codes */
enum {
  opt_inplace = 0,
  opt_verbose,
} ;

/* options */
VLMXOption  options [] = {
  {"InPlace",               0,   opt_inplace               },
  {"Verbose",               0,   opt_verbose               },
  {0,                       0,   0                         }
} ;

namespace vl { namespace impl {

template<vl::DeviceType deviceType, vl::DataType dataType>
vl::ErrorCode
accumulate(vl::Context & context,
             Tensor output,
             double alpha, Tensor a,
             double beta, Tensor b)
{
  vl::ErrorCode error ;

  typedef typename vl::DataTypeTraits<dataType>::type type ;
  ptrdiff_t n = a.getShape().getNumElements() ;

  bool inplace = (output.getMemory() == a.getMemory()) ;

  if (!inplace) {
    vl::impl::operations<deviceType,type>::
      copy((type*)output.getMemory(), (type const*)a.getMemory(), n) ;
  }

  error = vl::impl::blas<deviceType,dataType>::scal(context, n, alpha,
                                                    (type*)output.getMemory(), 1) ;
  if (error != vl::VLE_Success) { goto done ; }

  error = vl::impl::blas<deviceType,dataType>::axpy(context, n, beta,
                                                    (const type*)b.getMemory(), 1,
                                                    (type*)output.getMemory(), 1) ;
  if (error != vl::VLE_Success) { goto done ; }

 done:
  return context.passError(error, __func__) ;
}

} } // namespace vl::impl

#define DISPATCH(deviceType, dataType)          \
  error = vl::impl::accumulate<deviceType,dataType> \
    (context, output, alpha, a, beta, b)        \

#define DISPATCH2(deviceType)                   \
switch (dataType) {                             \
 case VLDT_Float : DISPATCH(deviceType, VLDT_Float) ; break ; \
 IF_DOUBLE(case VLDT_Double : DISPATCH(deviceType, VLDT_Double) ; break ;) \
 default: assert(false) ; return VLE_Unknown ;  \
}

namespace vl {
vl::ErrorCode
accumulate(vl::Context & context,
             Tensor output,
             double alpha, Tensor a,
             double beta, Tensor b)
{
  vl::ErrorCode error ;
  vl::DeviceType deviceType = a.getDeviceType() ;
  vl::DataType dataType = a.getDataType() ;
  switch (deviceType) {
    default:
      assert(false) ;
      error = vl::VLE_Unknown ;
      break ;
    case vl::VLDT_CPU:
      DISPATCH2(vl::VLDT_CPU) ;
      break ;
#if ENABLE_GPU
    case vl::VLDT_GPU:
      DISPATCH2(vl::VLDT_GPU) ;
      break ;
#endif
  }
  return error ;
}

} // namespace vl

/* -------------------------------------------------------------- */
/*                                                        Context */
/* -------------------------------------------------------------- */

vl::MexContext context ;

/*
  Resetting the context here resolves a crash when MATLAB quits and
  the ~Context function is implicitly called on unloading the MEX file.
*/
void atExit()
{
  context.clear() ;
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int verbosity = 0 ;
  bool inplace = false ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  mexAtExit(atExit) ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 4) {
    mexErrMsgTxt("There are less than four arguments.") ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
    case opt_verbose :
      ++ verbosity ;
      break ;

    case opt_inplace :
      inplace = true ;
      break ;
    }
  }

  if (!vlmxIsPlainScalar(in[IN_ALPHA])) {
    mexErrMsgTxt("ALPHA is not a plain scalar.") ;
  }
  if (!vlmxIsPlainScalar(in[IN_BETA])) {
    mexErrMsgTxt("BETA is not a plain scalar.") ;
  }

  vl::MexTensor a(context) ;
  vl::MexTensor b(context) ;
  vl::MexTensor output(context) ;

  a.init(in[IN_A]) ;
  b.init(in[IN_B]) ;

  if (! vl::areCompatible(a,b)) {
    mexErrMsgTxt("A and B do not have compatible formats.") ;
  }
  if (a.getShape().getNumElements() != b.getShape().getNumElements()) {
    mexErrMsgTxt("A and B do not have the same number of elements.") ;
  }

  double alpha = mxGetScalar(in[IN_ALPHA]) ;
  double beta = mxGetScalar(in[IN_BETA]) ;

  if (!inplace) {
    output.init(a.getDeviceType(), a.getDataType(), a.getShape()) ;
  } else {
    output.init(in[IN_A]) ;
  }

  if (verbosity) {
    mexPrintf("vl_taccum: %s %s\n",
              inplace?"inplace":"not inplace",
              (a.getDeviceType() == vl::VLDT_GPU)?"GPU":"CPU") ;
    vl::print("vl_taccum: A: ", a) ;
    vl::print("vl_taccum: B: ", b) ;
  }

  vl::ErrorCode error = vl::accumulate(context, output, alpha, a, beta, b);

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }

  if (!inplace){
    out[OUT_RESULT] = output.relinquish() ;
  }
 }


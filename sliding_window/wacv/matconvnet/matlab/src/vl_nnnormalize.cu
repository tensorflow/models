// @file vl_nnnormalize.cu
// @brief Normalization block MEX wrapper
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnnormalize.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose = 0
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
  {0,                  0,   0                     }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_PARAM, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  size_t normDepth ;
  double normAlpha ;
  double normKappa ;
  double normBeta ;
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;
      default: break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }
  if (backMode && (data.getShape() != derOutput.getShape())) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have the same size.") ;
  }

  if (!mxIsNumeric(in[IN_PARAM]) ||
       mxGetClassID(in[IN_PARAM]) != mxDOUBLE_CLASS ||
       mxIsComplex(in[IN_PARAM]) ||
       mxGetNumberOfElements(in[IN_PARAM]) != 4)
  {
    mexErrMsgTxt("PARAM is not a plain 4 vector.") ;
  }
  normDepth = (size_t) mxGetPr(in[IN_PARAM])[0]  ;
  normKappa = mxGetPr(in[IN_PARAM])[1]  ;
  normAlpha = mxGetPr(in[IN_PARAM])[2]  ;
  normBeta = mxGetPr(in[IN_PARAM])[3]  ;
  if (normDepth < 1) {
    mexErrMsgTxt("The normalization depth is smaller than 1.") ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;
  if (!backMode) {
    output.init(deviceType, dataType, data.getShape()) ;
  } else {
    derData.init(deviceType, dataType, data.getShape()) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnnormalize: mode %s; %s\n",  (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu", backMode?"backward":"forward") ;
    mexPrintf("vl_nnnormalize: (depth,kappa,alpha,beta): (%d,%g,%g,%g)\n",
              normDepth, normKappa, normAlpha, normBeta) ;
    vl::print("vl_nnnormalize: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnnormalize: derOutput: ", derOutput) ;
      vl::print("vl_nnnormalize: derData: ", derData) ;
    } else {
      vl::print("vl_nnnormalize: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;

  if (!backMode) {
    error = vl::nnlrn_forward(context,
                                    output, data,
                                    normDepth,
                                    normKappa, normAlpha, normBeta) ;
  } else {
    error = vl::nnlrn_backward(context,
                                     derData, data, derOutput,
                                     normDepth,
                                     normKappa, normAlpha, normBeta) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}

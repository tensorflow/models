// @file vl_nnbnorm.cu
// @brief Batch normalization MEX wrapper
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/nnbnorm.hpp"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose = 0,
  opt_epsilon,
  opt_moments,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
  {"Epsilon",	         1,   opt_epsilon           },
  {"Moments",          1,   opt_moments           },
  {"Cudnn",            0,   opt_cudnn             },
  {"NoCudnn",          0,   opt_no_cudnn          },
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
  IN_DATA = 0, IN_MULTIPLIERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0,
  OUT_DERMULTIPLIERS,
  OUT_DERBIASES,
  OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  double epsilon = 1e-4 ;

  // For the moment true need to be fixed
  bool computeDerData = true ;
  bool computeDerMultipliers = true ;
  bool computeDerBiases = true ;
  bool givenMomentsMode = false ;
  bool returnMomentsMode = false ;
  mxArray const* momentsArray ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 3) {
    mexErrMsgTxt("The arguments are less than three.") ;
  }
  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }
  returnMomentsMode = backMode ? (nout > 3) : (nout > 1) ;

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {

      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_epsilon :
        if (!vlmxIsPlainScalar(optarg)) {
          mexErrMsgTxt("EPSILON is not a plain scalar.") ;
        }
        epsilon = mxGetPr(optarg)[0] ;
        break ;

      case opt_moments:
        momentsArray = optarg ;
        givenMomentsMode = true ;
        break ;

      case opt_no_cudnn:
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false) ;
#endif
        break ;

      case opt_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(true) ;
#endif
        break ;
        
      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor multipliers(context);
  vl::MexTensor biases(context);
  vl::MexTensor derOutput(context) ;
  vl::MexTensor moments(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  multipliers.init(in[IN_MULTIPLIERS]) ;
  multipliers.reshape(1) ;

  biases.init(in[IN_BIASES]) ;
  biases.reshape(1) ;

  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;
  }

  if (givenMomentsMode) {
    moments.init(momentsArray) ;
    moments.reshape(2) ;
  }

  /* Check for GPU/data class consistency */
  if (! vl::areCompatible(data, multipliers)) {
    mexErrMsgTxt("DATA and MULTIPLIERS do not have compatible formats.") ;
  }
  if (! vl::areCompatible(data, biases)) {
    mexErrMsgTxt("DATA and BIASES do not have compatible formats.") ;
  }
  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }
  if (backMode && (data.getShape() != derOutput.getShape())) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have the same size.") ;
  }
  if (givenMomentsMode && ! vl::areCompatible(data, moments))
  {
    mexErrMsgTxt("DATA and MOMENTS do not have compatible formats.") ;
  }

  /* Get the filter geometry */
  vl::TensorShape multipliersGeom(multipliers) ;
  if (multipliersGeom.getHeight() != data.getDepth()) {
    mexErrMsgTxt("The MULTIPLIERS size does not match the DATA depth.") ;
  }
  vl::TensorShape biasesGeom(biases);
  if (biasesGeom.getHeight() != data.getDepth()) {
    mexErrMsgTxt("The BIASES size does not match the DATA depth.") ;
  }
  if (givenMomentsMode) {
    vl::TensorShape momentsGeom(moments) ;
    if (momentsGeom.getNumElements() != 2*data.getDepth()) {
      mexErrMsgTxt("The MOMENTS size does not match the DATA depth.") ;
    }
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;
  vl::MexTensor derMultipliers(context) ;
  vl::MexTensor derBiases(context) ;

  if (returnMomentsMode & !givenMomentsMode) {
    vl::TensorShape momentsGeom(data.getDepth(), 2, 1, 1) ;
    moments.init(deviceType, dataType, momentsGeom) ;
  }

  if (!backMode) {
    output.init(deviceType, dataType, data.getShape()) ;
  } else {
    if (computeDerData) {
      derData.init(deviceType, dataType, data.getShape()) ;
    }
    if (computeDerMultipliers) {
      derMultipliers.init(deviceType, dataType, multipliers.getShape()) ;
    }
    if (computeDerBiases) {
      derBiases.init(deviceType, dataType, biases.getShape()) ;
    }
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnbnorm: mode %s; %s; moments %s/%s\n",
              (data.getDeviceType()==vl::VLDT_GPU)?"gpu":"cpu",
              backMode?"backward":"forward",
              givenMomentsMode?"given":"computed",
              returnMomentsMode?"returned":"discared") ;
    vl::print("vl_nnbnorm: data: ", data) ;
    vl::print("vl_nnbnorm: multipliers: ", multipliers) ;
    vl::print("vl_nnbnorm: biases: ", biases) ;
    if (backMode) {
      vl::print("vl_nnbnorm: derOutput: ", derOutput) ;
      vl::print("vl_nnbnorm: derData: ", derData) ;
      vl::print("vl_nnbnorm: derMultipliers: ", derMultipliers) ;
      vl::print("vl_nnbnorm: derBiases: ", derBiases) ;
    } else {
      vl::print("vl_nnbnorm: output: ", output) ;
    }
    if (moments) { vl::print("vl_nnbnorm: moments: ", moments) ; }
    mexPrintf("vl_nnbnorm: epsilon: %f\n", epsilon) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;

  if (!backMode) {
    if (!givenMomentsMode) {
      error = vl::nnbnorm_forward(context,
                                  output,
                                  moments, // ok if null
                                  data,
                                  multipliers,
                                  biases,
                                  epsilon) ;
    } else {
      error = vl::nnbnorm_forward_given_moments(context,
                                                output,
                                                moments,
                                                data,
                                                multipliers,
                                                biases) ;
    }
  } else {
    if (!givenMomentsMode) {
      error = vl::nnbnorm_backward(context,
                                   derData,
                                   derMultipliers,
                                   derBiases,
                                   moments,
                                   data,
                                   multipliers,
                                   biases,
                                   derOutput,
                                   epsilon);
    } else {
      error = vl::nnbnorm_backward_given_moments(context,
                                                 derData,
                                                 derMultipliers,
                                                 derBiases,
                                                 moments,
                                                 data,
                                                 multipliers,
                                                 biases,
                                                 derOutput,
                                                 epsilon) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (!backMode) {
    out[OUT_RESULT] = output.relinquish() ;
  } else {
    out[OUT_RESULT] = (computeDerData) ? derData.relinquish() : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERMULTIPLIERS] = (computeDerMultipliers)? derMultipliers.relinquish() : mxCreateDoubleMatrix(0,0,mxREAL) ;
    out[OUT_DERBIASES] = (computeDerBiases) ? derBiases.relinquish() : mxCreateDoubleMatrix(0,0,mxREAL) ;
  }
  if (moments) {
    out[backMode ? 3 : 1] = moments.relinquish() ;
  }
}

// @file vl_nnroipooling.cpp
// @brief roipooling block implementation (GPU)
// @author Hakan Bilen
// @author Abishek Dutta
// @author Andrea Vedaldi

/*
Copyright (C) 2016 Hakan Bilen, Abishek Dutta, and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnroipooling.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>
#include <algorithm>

/* option codes */
enum {
  opt_method = 0,
  opt_subdivisions,
  opt_transform,
  opt_verbose,
} ;

/* options */
VLMXOption  options [] = {
  {"Method",           1,   opt_method       },
  {"Subdivisions",     1,   opt_subdivisions },
  {"Transform",        1,   opt_transform    },
  {"Verbose",          0,   opt_verbose      },
  {0,                  0,   0                }
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
  IN_DATA = 0, IN_ROIS, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int subdivisions [] = {1, 1} ;
  double transform [] = {1., 0., 0., 1., 0., 0.} ;
  vl::ROIPoolingMethod method = vl::vlROIPoolingMax ;
  bool backMode = false ;
  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 2) {
    vlmxError(VLMXE_IllegalArgument, "There are less than two arguments.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose : {
        ++ verbosity ;
        break ;
      }

      case opt_method : {
        if (!vlmxIsString(optarg,-1)) {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "max")) {
          method = vl::vlROIPoolingMax ;
        } else if (vlmxIsEqualToStringI(optarg, "avg")) {
          method = vl::vlROIPoolingAverage ;
        } else {
          vlmxError(VLMXE_IllegalArgument, "METHOD is not a supported method.") ;
        }
        break ;
      }

      case opt_subdivisions : {
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "SUBDIVISIONS is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            subdivisions[0] = mxGetPr(optarg)[0] ;
            subdivisions[1] = mxGetPr(optarg)[0] ;

          case 2:
            subdivisions[0] = mxGetPr(optarg)[0] ;
            subdivisions[1] = mxGetPr(optarg)[1] ;
            break ;

          default:
            vlmxError(VLMXE_IllegalArgument, "SUBDIVISIONS does not have one or two elements.") ;
            break ;
        }
        if (subdivisions[0] < 1 || subdivisions[1] < 1) {
          vlmxError(VLMXE_IllegalArgument, "SUBDIVISIONS has an element smaller than 1.") ;
        }
        break ;
      }

      case opt_transform : {
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "TRANSFORM is not a plain matrix.") ;
        }
        int n = (int) mxGetNumberOfElements(optarg) ;
        switch (n) {
          case 1: case 2:
            transform[0] = mxGetPr(optarg)[std::min(n - 1, 0)] ;
            transform[3] = mxGetPr(optarg)[std::min(n - 1, 1)] ;
            transform[4] = 1. - transform[0] ;
            transform[5] = 1. - transform[3] ;
            break ;

          case 6:
            memcpy(transform, mxGetPr(optarg), 6 * sizeof(transform[0])) ;
            break ;

          default:
            vlmxError(VLMXE_IllegalArgument, "TRANSFORM is neither a 1 x 1, 2 x 1, or 2 x 3 matrix.") ;
        }
        break ;
      }

      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;
  vl::MexTensor rois(context) ;

  /* Get data */
  rois.init(in[IN_ROIS]);
  data.init(in[IN_DATA]) ;
  if (backMode) { derOutput.init(in[IN_DEROUTPUT]) ; }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    vlmxError(VLMXE_IllegalArgument, "DATA and DEROUTPUT do not have compatible formats.") ;
  }

  size_t numROIs = rois.getNumElements() / 5 ;

  if (! vl::areCompatible(data, rois)) {
    vlmxError(VLMXE_IllegalArgument, "DATA and ROI do not have compatible formats.") ;
  }

  if (rois.getNumElements() != numROIs * 5 || numROIs == 0) {
    vlmxError(VLMXE_IllegalArgument, "ROI is not a 5 x K array with K >= 1.") ;
  }
  rois.reshape(vl::TensorShape(1, 1, 5, numROIs)) ;

  vl::TensorShape dataShape = data.getShape();
  dataShape.reshape(4);

  /* Get the output geometry */
  vl::TensorShape outputShape(subdivisions[0],
                              subdivisions[1],
                              dataShape.getDepth(),
                              numROIs) ;

  vl::TensorShape derOutputShape = derOutput.getShape();
  /* in case there is only one roi */ 
  derOutputShape.reshape(4);

  if (backMode) {
    if (derOutputShape != outputShape) {
      vlmxError(VLMXE_IllegalArgument, "The shape of DEROUTPUT is incorrect.") ;
    }
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, dataShape) ;
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnroipool: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "GPU" : "CPU") ;
    mexPrintf("\nvl_nnroipool: method: %d; num ROIs: %d\n", method, numROIs);
    mexPrintf("vl_nnroipool: subdivisions: [%d x %d]\n", subdivisions[0], subdivisions[1]) ;
    mexPrintf("vl_nnroipool: transform: [%g %g %g ; %g %g %g]\n",
              transform[0], transform[2], transform[4],
              transform[1], transform[3], transform[5]) ;

    vl::print("vl_nnroipool: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnroipool: derOutput: ", derOutput) ;
      vl::print("vl_nnroipool: derData: ", derData) ;
    } else {
      vl::print("vl_nnroipool: output: ", output) ;
      vl::print("vl_nnroipool: rois: ", rois) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
    error = vl::nnroipooling_forward(context,
                                     output, data, rois,
                                     method,
                                     subdivisions,
                                     transform) ;
  } else {
    error = vl::nnroipooling_backward(context,
                                      derData, data, rois, derOutput,
                                      method,
                                      subdivisions,
                                      transform) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    vlmxError(VLMXE_IllegalArgument, context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}

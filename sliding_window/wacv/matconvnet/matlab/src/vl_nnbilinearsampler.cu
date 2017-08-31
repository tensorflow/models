// @file vl_nnbilinearsampler.cu
// @brief Bilinear Sampler MEX wrapper
// @author Ankush Gupta
// @author Andrea Vedaldi
/*
Copyright (C) 2016- Ankush Gupta and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

/** this is the mex-wrapper -- entry-point from matlab to cuda */

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnbilinearsampler.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose = 0,
  opt_cudnn,
  opt_no_cudnn
};

/* options */
VLMXOption  options [] = {
  {"Verbose",          0,   opt_verbose           },
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
  IN_DATA = 0, IN_GRID, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERGRID, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  // whether we are back-propagating or not:
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  // need at least data and grid to operate (2 args minimum)
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

      case opt_no_cudnn :
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
  vl::MexTensor grid(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ; // -> 4 dimensions

  grid.init(in[IN_GRID]);
  grid.reshape(4); // ->  4 dimensions

  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
  }

  if (! vl::areCompatible(data, grid)) {
    mexErrMsgTxt("DATA and GRID do not have compatible formats.") ;
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  /* Basic compatibility of shape */
  const int inHeight = data.getHeight(); // spatial dimension 1
  const int inWidth = data.getWidth(); // spatial dimension 2
  const int inDepth = data.getDepth(); // number of channels
  const int inBatch = data.getSize(); // batch-size

  /* Grid dimensions: note that the grid uses the first dimension as channels */
  const int gridHeight = grid.getWidth(); // *OUTPUT* spatial dimension
  const int gridWidth = grid.getDepth(); // *OUTPUT* spatial dimension 2
  const int gridDepth = grid.getHeight(); // number of channels :: should be 2
  const int gridBatch = grid.getSize(); // should be DIVISIBLE by inBatch

  if (gridDepth != 2) {
    char msg[200];
    sprintf(msg, "GRID has %d channels; expected 2.\n", gridDepth);
    mexErrMsgTxt(msg) ;
  }

  if ((gridBatch % inBatch) != 0) {
    mexErrMsgTxt("GRID batch-size is not a multiple of DATA batch-size.") ;
  }

  /* Get the output Shape */
  vl::TensorShape outputShape(gridHeight, gridWidth, inDepth, gridBatch);
  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with DATA and GRID.") ;
  }

  /* Create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;
  vl::MexTensor derGrid(context) ;

  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
    derGrid.initWithZeros(deviceType, dataType, grid.getShape()) ;
  }

  // log:
  if (verbosity > 0) {
    mexPrintf("vl_nnbilinearsampler: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "GPU" : "CPU") ;
    mexPrintf("; MatConvNet\n") ;
    vl::print("vl_nnbilinearsampler: data: ", data) ;
    vl::print("vl_nnbilinearsampler: grid: ", grid) ;
    if (backMode) {
      vl::print("vl_nnbilinearsampler: derOutput: ", derOutput) ;
      vl::print("vl_nnbilinearsampler: derData: ", derData) ;
      vl::print("vl_nnbilinearsampler: derGrid: ", derGrid) ;
    } else {
      vl::print("vl_nnbilinearsampler: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;
  if (!backMode) {
    error = vl::nnbilinearsampler_forward(context,
                                  output,
                                  data, grid);
  } else {
    error = vl::nnbilinearsampler_backward(context,
                                   derData, derGrid,
                                   data, grid, derOutput);
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
    out[OUT_DERGRID] = derGrid.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
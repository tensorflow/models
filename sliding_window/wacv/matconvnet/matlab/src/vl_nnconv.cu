// @file nnconv.cu
// @brief Convolution block MEX wrapper
// @author Andrea Vedaldi
// @author Max Jaderberg

/*
Copyright (C) 2014 Andrea Vedaldi and Max Jaderberg
Copyright (C) 2015 Andrea Vedaldi.

All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnconv.hpp"
#include "bits/nnfullyconnected.hpp"
#include "bits/nnsubsample.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <memory>
#include <assert.h>
#include <math.h>

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_dilate,
  opt_verbose,
  opt_no_der_data,
  opt_no_der_filters,
  opt_no_der_biases,
  opt_cudnn,
  opt_no_cudnn,
  opt_cudnn_workspace_limit,
  opt_transpose
} ;

/* options */
VLMXOption  options [] = {
  {"Stride",                1,   opt_stride                },
  {"Pad",                   1,   opt_pad                   },
  {"Dilate",                1,   opt_dilate                },
  {"Verbose",               0,   opt_verbose               },
  {"NoDerData",             0,   opt_no_der_data           },
  {"NoDerFilters",          0,   opt_no_der_filters        },
  {"NoderBiases",           0,   opt_no_der_biases         },
  {"Cudnn",                 0,   opt_cudnn                 },
  {"NoCudnn",               0,   opt_no_cudnn              },
  {"CudnnWorkSpaceLimit",   1,   opt_cudnn_workspace_limit },
  {0,                       0,   0                         }
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
  IN_DATA = 0, IN_FILTERS, IN_BIASES, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERFILTERS, OUT_DERBIASES, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  int dilateY = 1 ;
  int dilateX = 1 ;
  int numFilterGroups = 1 ;

  bool backMode = false ;
  bool hasFilters = false ;
  bool hasBiases = false ;
  bool fullyConnectedMode = false ;
  bool computeDerData = true ;
  bool computeDerFilters = true ;
  bool computederBiases = true ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 3) {
    vlmxError(VLMXE_IllegalArgument, "There are less than three arguments.") ;
  }

  if (nin > 3 && vlmxIsString(in[3],-1)) {
    next = 3 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 4) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            break ;
          case 2:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            vlmxError(VLMXE_IllegalArgument, "STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            break ;
          case 4:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            vlmxError(VLMXE_IllegalArgument, "PAD has neither one nor four elements.") ;
        }
        break ;

      case opt_dilate :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          vlmxError(VLMXE_IllegalArgument, "DILATE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            dilateY = (int)mxGetPr(optarg)[0] ;
            dilateX = dilateY ;
            break ;
          case 2:
            dilateY = (int)mxGetPr(optarg)[0] ;
            dilateX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            vlmxError(VLMXE_IllegalArgument, "DILATE has neither one nor two elements.") ;
        }
        break ;

      case opt_no_der_data :
        computeDerData = false ;
        break ;

      case opt_no_der_filters :
        computeDerFilters = false ;
        break ;

      case opt_no_der_biases :
        computederBiases = false ;
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

      case opt_cudnn_workspace_limit :
      {
#if ENABLE_CUDNN
        double x ;
        if (!vlmxIsScalar(optarg) || (x = mxGetScalar(optarg)) < 0) {
          vlmxError(VLMXE_IllegalArgument, "CudnnWorkSpaceLimit is not a non-negative scalar.") ;
        }
        context.getCudaHelper().setCudnnConvolutionFwdPreference
        ((x==mxGetInf() ?
          CUDNN_CONVOLUTION_FWD_PREFER_FASTEST :
          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT),
         (size_t)x) ;
        context.getCudaHelper().setCudnnConvolutionBwdFilterPreference
        ((x==mxGetInf() ?
          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST :
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT),
         (size_t)x) ;
        context.getCudaHelper().setCudnnConvolutionBwdDataPreference
        ((x==mxGetInf() ?
          CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST :
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT),
         (size_t)x) ;
        break ;
#endif
      }

      default: break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor filters(context) ;
  vl::MexTensor biases(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ;

  filters.init(in[IN_FILTERS]) ;
  filters.reshape(4) ;

  biases.init(in[IN_BIASES]) ;

  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ;
  }

  hasFilters = !filters.isEmpty() ;
  hasBiases = !biases.isEmpty() ;

  /* check for GPU/data class consistency */
  if (hasFilters && ! vl::areCompatible(data, filters)) {
    vlmxError(VLMXE_IllegalArgument, "DATA and FILTERS do not have compatible formats.") ;
  }
  if (hasBiases && ! vl::areCompatible(data, biases)) {
    vlmxError(VLMXE_IllegalArgument, "DATA and BIASES do not have compatible formats.") ;
  }
  if (backMode && ! vl::areCompatible(data, derOutput)) {
    vlmxError(VLMXE_IllegalArgument, "DATA and DEROUTPUT do not have compatible formats.") ;
  }

  /* basic argument checks */
  if (strideX < 1 || strideY < 1) {
    vlmxError(VLMXE_IllegalArgument, "At least one element of STRIDE is smaller than one.") ;
  }
  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    vlmxError(VLMXE_IllegalArgument, "An element of PAD is negative.") ;
  }
  if (dilateY < 1 || dilateX < 1) {
    vlmxError(VLMXE_IllegalArgument, "An element of DILATE is less than one.") ;
  }
  if (!hasFilters && (dilateY != 1 || dilateX != 1)) {
    vlmxError(VLMXE_IllegalArgument, "There are no filters and DILATE is not one.") ;
  }

  /* Get the filter shape */
  vl::TensorShape filtersShape(filters) ;
  int equivalentNumFilters ;
  if (hasFilters) {
    if (filtersShape.getHeight() == 0 || filtersShape.getWidth() == 0 || filtersShape.getDepth() == 0) {
      vlmxError(VLMXE_IllegalArgument, "A dimension of FILTERS is void.") ;
    }
    if (data.getHeight() + (padTop+padBottom) < (filters.getHeight() - 1)*dilateY + 1 ||
        data.getWidth() + (padLeft+padRight) < (filters.getWidth() - 1)*dilateX + 1) {
      vlmxError(VLMXE_IllegalArgument, "FILTERS are larger than the DATA (including padding).") ;
    }
    /* grouped filters */
    numFilterGroups = data.getDepth() / filters.getDepth() ;
    if (numFilterGroups * filters.getDepth() != data.getDepth()) {
      vlmxError(VLMXE_IllegalArgument, "The FILTERS depth does not divide the DATA depth.") ;
    }
    if (filters.getSize() % numFilterGroups != 0) {
      vlmxError(VLMXE_IllegalArgument, "The number of filter groups does not divide the number of filters.") ;
    }
    equivalentNumFilters = filters.getSize() ;
  } else {
    /* empty filters -> pretend the identity filter bank */
    filtersShape = vl::TensorShape(1, 1, data.getDepth(), data.getDepth()) ;
    numFilterGroups = 1 ;
    equivalentNumFilters = data.getDepth() ;
  }

  /* Get the output shape */
  int kernelExtentX = (filtersShape.getWidth() - 1)*dilateX + 1 ;
  int kernelExtentY = (filtersShape.getHeight() - 1)*dilateY + 1 ;

  vl::TensorShape outputShape((data.getHeight() + (padTop+padBottom) - kernelExtentY)/strideY + 1,
                                (data.getWidth()  + (padLeft+padRight) - kernelExtentX)/strideX + 1,
                                equivalentNumFilters,
                                data.getSize()) ;

  if (backMode && (derOutput != outputShape)) {
    vlmxError(VLMXE_IllegalArgument, "DEROUTPUT dimensions are incompatible with X and FILTERS.") ;
  }

  /* Check the biases sizes */
  if (hasBiases) {
    if (biases.getNumElements() != filtersShape.getSize()) {
      vlmxError(VLMXE_IllegalArgument, "The number of elements of BIASES is not the same as the number of filters.") ;
    }
  }

  /*
   Detect fully connected mode (further optimisations):
   the output is 1 x 1 pixels,
   no padding,
   one filter group,
   stride of one pixel
   */
  fullyConnectedMode = (outputShape.getHeight() == 1 &&
                        outputShape.getWidth() == 1 &&
                        strideY == 1 &&
                        strideX == 1 &&
                        padTop == 0 &&
                        padBottom == 0 &&
                        padLeft == 0 &&
                        padRight == 0 &&
                        dilateY == 1 &&
                        dilateX == 1 &&
                        numFilterGroups == 1) ;

  /* create output buffers */
  vl::DeviceType deviceType = data.getDeviceType() ;
  vl::DataType dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;
  vl::MexTensor derFilters(context) ;
  vl::MexTensor derBiases(context) ;

  if (!backMode) {
    output.init(deviceType, dataType, outputShape) ;
  } else {
    if (computeDerData) {
      derData.init(deviceType, dataType, data.getShape()) ;
    }
    if (computeDerFilters && hasFilters) {
      derFilters.init(deviceType, dataType, filters.getShape()) ;
    }
    if (computederBiases && hasBiases) {
      derBiases.init(deviceType, dataType, biases.getShape()) ;
    }
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnconv: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "GPU" : "CPU") ;
    if (data.getDeviceType() == vl::VLDT_GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "cuBLAS") ;
#else
      mexPrintf("; cuBLAS\n") ;
#endif
    } else {
      mexPrintf("; BLAS\n") ;
    }
    mexPrintf("vl_nnconv: stride: [%d %d], pad: [%d %d %d %d], dilate: [%d %d]\n"
              "vl_nnconv: num filter groups: %d, has bias: %d, has filters: %d, is fully connected: %d\n",
              strideY, strideX,
              padTop, padBottom, padLeft, padRight,
              dilateY, dilateX,
              numFilterGroups, hasBiases, hasFilters, fullyConnectedMode) ;
    vl::print("vl_nnconv: data: ", data) ;
    if (hasFilters) { vl::print("vl_nnconv: filters: ", filters) ; }
    if (hasBiases) { vl::print("vl_nnconv: biases: ", biases) ; }
    if (backMode) {
      vl::print("vl_nnconv: derOutput: ", derOutput) ;
      vl::print("vl_nnconv: derData: ", derData) ;
      if (hasFilters) { vl::print("vl_nnconv: derFilters: ", derFilters) ; }
      if (hasBiases) { vl::print("vl_nnconv: derBiases: ", derBiases) ; }
    } else {
      vl::print("vl_nnconv: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;

  /*
   special case: fully connected
   (could be done as a regular case, but it is faster this way)
   */
  if (fullyConnectedMode) {
    if (!backMode) {
      error = vl::nnfullyconnected_forward(context,
                                           output,
                                           data,
                                           filters,
                                           biases) ;
    } else {
      error = vl::nnfullyconnected_backward(context,
                                            derData,
                                            derFilters,
                                            derBiases,
                                            data,
                                            filters,
                                            derOutput) ;
    }
    goto doneok ;
  }

  /* special case: no filters = identity filter bank (subsample + bias) */
  if (!hasFilters) {
    if (!backMode) {
      error = vl::nnsubsample_forward(context,
                                      output,
                                      data,
                                      biases,
                                      strideY, strideX,
                                      padTop, padBottom, padLeft, padRight) ;
    } else {
      error = vl::nnsubsample_backward(context,
                                       derData,
                                       derBiases,
                                       derOutput,
                                       strideY, strideX,
                                       padTop, padBottom, padLeft, padRight) ;
    }
    goto doneok ;
  }

  /* regular case */
  if (!backMode) {
    error = vl::nnconv_forward(context,
                               output, 0,
                               data, 1,
                               filters,
                               biases,
                               strideY, strideX,
                               padTop, padBottom, padLeft, padRight,
                               dilateY, dilateX) ;
  } else {
    error = vl::nnconv_backward(context,
                                derData,
                                derFilters,
                                derBiases,
                                data,
                                filters,
                                derOutput,
                                strideY, strideX,
                                padTop, padBottom, padLeft, padRight,
                                dilateY, dilateX) ;
  }

doneok:
  if (verbosity > 0) {
#if ENABLE_CUDNN
    if (context.getCudaHelper().getCudnnEnabled()) {
      mexPrintf("vl_nnconv: cuDNN workspace used: "
                "fwd %.6g MB"
                ", bwd filter %.6g MB"
                ", bwd data %.6g MB\n",
                (double)context.getCudaHelper().getCudnnConvolutionFwdWorkSpaceUsed() / (1024*1024),
                (double)context.getCudaHelper().getCudnnConvolutionBwdFilterWorkSpaceUsed() / (1024*1024),
                (double)context.getCudaHelper().getCudnnConvolutionBwdDataWorkSpaceUsed() / (1024*1024)) ;
    }
#endif
  }

  /* -------------------------------------------------------------- */
  /*                                                        Cleanup */
  /* -------------------------------------------------------------- */

  if (error != vl::VLE_Success) {
    vlmxError(VLMXE_IllegalArgument, context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    mxClassID classID ;
    switch (derOutput.getDataType()) {
      case vl::VLDT_Float: classID = mxSINGLE_CLASS ; break ;
      case vl::VLDT_Double: classID = mxDOUBLE_CLASS ; break ;
      default: abort() ;
    }
    out[OUT_RESULT] = (computeDerData) ? derData.relinquish() : mxCreateNumericMatrix(0,0,classID,mxREAL) ;
    out[OUT_DERFILTERS] = (computeDerFilters & hasFilters)? derFilters.relinquish() : mxCreateNumericMatrix(0,0,classID,mxREAL) ;
    out[OUT_DERBIASES] = (computederBiases & hasBiases) ? derBiases.relinquish() : mxCreateNumericMatrix(0,0,classID,mxREAL) ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}

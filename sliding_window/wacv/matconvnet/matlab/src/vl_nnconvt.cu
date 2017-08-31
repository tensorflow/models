// @file nnconvt.cu
// @brief Convolution transpose block MEX wrapper
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.

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

/* option codes */
enum {
  opt_upsample = 0,
  opt_crop,
  opt_verbose,
  opt_num_groups,
  opt_no_der_data,
  opt_no_der_filters,
  opt_no_der_biases,
  opt_cudnn,
  opt_no_cudnn,
  opt_cudnn_workspace_limit,
} ;

/* options */
VLMXOption  options [] = {
  {"Upsample",              1,   opt_upsample              },
  {"Crop",                  1,   opt_crop                  },
  {"Verbose",               0,   opt_verbose               },
  {"NumGroups",             1,   opt_num_groups            },
  {"NoDerData",             0,   opt_no_der_data           },
  {"NoDerFilters",          0,   opt_no_der_filters        },
  {"NoDerBiases",           0,   opt_no_der_biases         },
  {"CUDNN",                 0,   opt_cudnn                 },
  {"NoCUDNN",               0,   opt_no_cudnn              },
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
  int upsampleX = 1 ;
  int upsampleY = 1 ;
  int cropLeft = 0 ;
  int cropRight = 0 ;
  int cropTop = 0 ;
  int cropBottom = 0 ;
  int numFilterGroups = 1 ;

  bool backMode = false ;
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
    mexErrMsgTxt("There are less than three arguments.") ;
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

      case opt_upsample :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("upsample is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            upsampleY = (int)mxGetPr(optarg)[0] ;
            upsampleX = upsampleY ;
            break ;
          case 2:
            upsampleY = (int)mxGetPr(optarg)[0] ;
            upsampleX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("upsample has neither one nor two elements.") ;
        }
        break ;

      case opt_crop :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("crop is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            cropLeft = (int)mxGetPr(optarg)[0] ;
            cropRight = cropLeft ;
            cropTop = cropLeft ;
            cropBottom = cropLeft ;
            break ;
          case 4:
            cropTop = (int)mxGetPr(optarg)[0] ;
            cropBottom = (int)mxGetPr(optarg)[1] ;
            cropLeft = (int)mxGetPr(optarg)[2] ;
            cropRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            mexErrMsgTxt("crop has neither one nor two elements.") ;
        }
        break ;

      case opt_num_groups :
        if (!vlmxIsPlainMatrix(optarg,1,1)) {
          mexErrMsgTxt("NUMGROUPS is not a plain scalar.") ;
        }
        numFilterGroups = (int)mxGetPr(optarg)[0] ;
        break;

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
          mexErrMsgTxt("CudnnWorkSpaceLimit is not a non-negative scalar.") ;
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

  hasBiases = !biases.isEmpty() ;

  /* check for GPU/data class consistency */
  if (! vl::areCompatible(data, filters)) {
    mexErrMsgTxt("DATA and FILTERS do not have compatible formats.") ;
  }
  if (hasBiases && ! vl::areCompatible(data, biases)) {
    mexErrMsgTxt("DATA and BIASES do not have compatible formats.") ;
  }
  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  /* basic argument checks */
  if (upsampleX < 1 || upsampleY < 1) {
    mexErrMsgTxt("At least one element of UPSAMPLE is smaller than one.") ;
  }
  if (cropLeft < 0 ||
      cropRight < 0 ||
      cropTop < 0 ||
      cropBottom < 0) {
    mexErrMsgTxt("An element of CROP is negative.") ;
  }

  /* Get the filter shape */
  vl::TensorShape filtersShape(filters) ;

  if (filtersShape.getHeight() == 0 || filtersShape.getWidth() == 0 || filtersShape.getDepth() == 0) {
    mexErrMsgTxt("A dimension of FILTERS is void.") ;
  }

  /* grouped filters */
  if (numFilterGroups < 1) {
    mexErrMsgTxt("NUMGROUPS is less than 1.") ;
  }
  if (filters.getSize() % numFilterGroups != 0) {
    mexErrMsgTxt("The number of filter groups does not divide the filter bank depth (fourth dimension of FILTERS).") ;
  }
  if (filters.getSize() != data.getDepth()) {
    mexErrMsgTxt("The filter bank depth (fourth dimension of FILTERS) is not the same as the data depth (third dimension of X).") ;
  }

  /* Get the output Shapeetry */
  vl::TensorShape outputShape((data.getHeight()-1)*upsampleY - (cropTop+cropBottom) + filtersShape.getHeight(),
                                (data.getWidth()-1)*upsampleX  - (cropLeft+cropRight) + filtersShape.getWidth(),
                                filtersShape.getDepth() * numFilterGroups,
                                data.getSize()) ;

  if (outputShape.getHeight() < 1 || outputShape.getWidth() < 1) {
    mexErrMsgTxt("The output array is empty due to CROP being too large.") ;
  }

  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and FILTERS.") ;
  }

  /* Check the biases sizes */
  if (hasBiases) {
    if (biases.getNumElements() != outputShape.getDepth()) {
      mexErrMsgTxt("The number of elements of BIASES is not the same as the dimenison of the filters.") ;
    }
  }

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
    if (computeDerFilters) {
      derFilters.init(deviceType, dataType, filters.getShape()) ;
    }
    if (computederBiases && hasBiases) {
      derBiases.init(deviceType, dataType, biases.getShape()) ;
    }
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnconvt: %s; %s", backMode?"backward":"forward", (data.getDeviceType()==vl::VLDT_GPU) ? "GPU" : "CPU") ;
    if (data.getDeviceType() == vl::VLDT_GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "cuBLAS") ;
#else
      mexPrintf("; cuBLAS\n") ;
#endif
    } else {
      mexPrintf("; BLAS\n") ;
    }
    mexPrintf("vl_nnconvt: upsample: [%d %d], crop: [%d %d %d %d]\n"
              "vl_nnconvt: num filter groups: %d, has bias: %d, is fully connected: %d\n",
              upsampleY, upsampleX,
              cropTop, cropBottom, cropLeft, cropRight,
              numFilterGroups, hasBiases, fullyConnectedMode) ;
    vl::print("vl_nnconvt: data: ", data) ;
    vl::print("vl_nnconvt: filters: ", filters) ;
    if (hasBiases) { vl::print("vl_nnconvt: biases: ", biases) ; }
    if (backMode) {
      vl::print("vl_nnconvt: derOutput: ", derOutput) ;
      vl::print("vl_nnconvt: derData: ", derData) ;
      vl::print("vl_nnconvt: derFilters: ", derFilters) ;
      if (hasBiases) { vl::print("vl_nnconvt: derBiases: ", derBiases) ; }
    } else {
      vl::print("vl_nnconvt: output: ", output) ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::ErrorCode error ;

  /* regular case */
  if (!backMode) {
    error = vl::nnconvt_forward(context,
                                output,
                                data,
                                filters,
                                biases,
                                upsampleY, upsampleX,
                                cropTop, cropBottom, cropLeft, cropRight) ;
  } else {
    error = vl::nnconvt_backward(context,
                                 derData,
                                 derFilters,
                                 derBiases,
                                 data,
                                 filters,
                                 derOutput,
                                 upsampleY, upsampleX,
                                 cropTop, cropBottom, cropLeft, cropRight) ;
  }

  if (verbosity > 0) {
#if ENABLE_CUDNN
    if (context.getCudaHelper().getCudnnEnabled()) {
      mexPrintf("vl_nnconvt: cuDNN workspace used: "
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
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    mxClassID classID ;
    switch (derOutput.getDataType()) {
      case vl::VLDT_Float: classID = mxSINGLE_CLASS ; break ;
      case vl::VLDT_Double: classID = mxDOUBLE_CLASS ; break ;
      default: abort() ;
    }
    out[OUT_RESULT] = (computeDerData) ? derData.relinquish() : mxCreateNumericMatrix(0,0,classID,mxREAL) ;
    out[OUT_DERFILTERS] = (computeDerFilters)? derFilters.relinquish() : mxCreateNumericMatrix(0,0,classID,mxREAL) ;
    out[OUT_DERBIASES] = (computederBiases & hasBiases) ? derBiases.relinquish() : mxCreateNumericMatrix(0,0,classID,mxREAL) ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}

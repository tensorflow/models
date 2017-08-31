/** @file vl_cudatool.cu
 ** @brief Low-level CUDA tricks.
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2016 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"

#include "gpu/mxGPUArray.h"
#include "bits/mexutils.h"
#include "cuda_profiler_api.h"

enum {
  IN_COMMAND, IN_ARG1, IN_ARG2, IN_END
} ;

enum {
  OUT_END
} ;

typedef struct Memory_ {
  void * address ;
  size_t size ;
  mxClassID classID ;
} Memory ;

Memory
getMemoryFromArray(const mxArray * array)
{
  Memory mem ;
  if (mxIsGPUArray(array)) {
    mxGPUArray* garray = (mxGPUArray*) mxGPUCreateFromMxArray(array) ;
    mem.address = (void*) mxGPUGetDataReadOnly(garray) ;
    mem.size = mxGPUGetNumberOfElements(garray) ;
    mem.classID = mxGPUGetClassID(garray) ;
    mxGPUDestroyGPUArray(garray) ;
  } else {
    mem.address = mxGetData((mxArray*)array) ;
    mem.size = mxGetNumberOfElements(array) ;
    mem.classID = mxGetClassID(array) ;
  }

  switch (mem.classID) {
  case mxDOUBLE_CLASS: mem.size *= sizeof(double) ; break ;
  case mxSINGLE_CLASS: mem.size *= sizeof(float) ; break ;
  default:
    vlmxError(VLMXE_IllegalArgument, "Data type unsupported.") ;
  }
  return mem ;
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{

  if (nin < 1) {
    vlmxError(VLMXE_IllegalArgument, "Not enough arguments.") ;
  }

  if (!vlmxIsString(in[0], -1)) {
    vlmxError(VLMXE_IllegalArgument, "COMMAND is not a string.") ;
  }

  if (vlmxCompareToStringI(in[0], "getMemory") == 0) {
    if (nin != 2) {
      vlmxError(VLMXE_IllegalArgument, "Incorrect number of arguments for 'getMemory'.") ;
    }
    Memory mem = getMemoryFromArray(in[1]) ;
    out[0] = mxCreateNumericMatrix(1,2,mxUINT64_CLASS,mxREAL) ;
    size_t* x = (size_t*)mxGetPr(out[0]) ;
    x[0] = (size_t)mem.address ;
    x[1] = mem.size ;
  }
  else if (vlmxCompareToStringI(in[0], "cudaRegister") == 0) {
    if (nin != 2) {
      vlmxError(VLMXE_IllegalArgument, "Incorrect number of arguments for 'cudaRegister'.") ;
    }
    Memory mem = getMemoryFromArray(in[1]) ;
    cudaError_t err = cudaHostRegister(mem.address,
                                       mem.size,
                                       cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      vlmxWarning(VLMXE_Execution, "cudaHostRegister failied\n");
    }
  }
  else if (vlmxCompareToStringI(in[0], "cudaUnregister") == 0) {
    if (nin != 2) {
      vlmxError(VLMXE_IllegalArgument, "Incorrect number of arguments for 'cudaUnregister'.") ;
    }
    Memory mem = getMemoryFromArray(in[1]) ;
    cudaError_t err = cudaHostUnregister(mem.address) ;
    if (err != cudaSuccess) {
      mexWarnMsgTxt("cudaHostUnregister failied\n");
    }
  }
  else if (vlmxCompareToStringI(in[0], "cudaCopyDeviceToHost") == 0) {
    if (nin != 3) {
      vlmxError(VLMXE_IllegalArgument, "Incorrect number of arguments for 'cudaCopyDeviceToHost'") ;
    }
    Memory mem = getMemoryFromArray(in[1]) ;
    Memory gmem = getMemoryFromArray(in[2]) ;
    cudaMemcpy(mem.address,
               gmem.address,
               gmem.size,
               cudaMemcpyDeviceToHost);
  }
  else if (vlmxCompareToStringI(in[0], "cudaCopyDeviceToHostAsync") == 0) {
    if (nin != 3) {
      vlmxError(VLMXE_IllegalArgument, "Incorrect number of arguments for 'cudaCopyDeviceToHostAsync'.") ;
    }
    Memory mem = getMemoryFromArray(in[1]) ;
    Memory gmem = getMemoryFromArray(in[2]) ;
    cudaMemcpyAsync(mem.address,
                    gmem.address,
                    gmem.size,
                    cudaMemcpyDeviceToHost,
                    0);
  }
  else if (vlmxCompareToStringI(in[0], "cudaProfilerStart") == 0) {
    cudaProfilerStart() ;
    mexPrintf("Enabled CUDA profiler.\n") ;
  }
  else if (vlmxCompareToStringI(in[0], "cudaProfilerStop") == 0) {
    cudaProfilerStop() ;
    mexPrintf("Disabled CUDA profiler.\n") ;
  }
  else {
    vlmxError(VLMXE_IllegalArgument, "Unrecognized command COMMAND.") ;
  }
}

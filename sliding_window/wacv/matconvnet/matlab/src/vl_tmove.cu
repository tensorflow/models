/** @file vl_tmove.cu
 ** @brief MEX internals of vl_tmove.m.
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

#include "bits/data.hpp"
#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include "bits/impl/tinythread.h"
#include "bits/impl/blashelper.hpp"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/un.h>
#include <sys/socket.h>

#include <memory>
#include <vector>
#include <algorithm>
#include <sstream>

/**
 \file vl_tmove.cu
 
 The `vl_tmove` utility implements an efficient mechanism to exchange
 tensor data between different MATLAB processes. Presently, it is
 limited to processes running on the same host, but future extensions
 can integrate networked environments. Even limited to a single
 host, this functionality is important because MATLAB multiple GPU
 support uses different processess for different GPUs.
 
 The key idea is to implement a reduction tree, in which each MATLAB
 process is connected to a parent and a number of children. When a tensor
 needs to be accumulated, a node receives copies form the children,
 sums them with its local copy, and sends the result to the parent.
 Eventually, the data flow reaches the root of the tree and the accumulated
 tensor is sent back towards the leaves. This communication mechanism
 is designed to reduce the amount of data transfers from O(n^2)
 for the trivial n-to-n communication of tensor copies to O(n).
 
 A second strategy used to significantly improve the speed is to allow
 the transfer of tensor data to proceed in the background, while MATLAB is busy
 running the rest of the network. This is achieved by isolating
 all communications in a supervisory thread.

 # Notable facts
 
 * Communications between thread uses UNIX-domain sockets (extensible
   to INet sockets in the future). These are used to send lightweight
   cohordination messages.
 
 * Data passing on local machines uses a shared memory map between
   processes. The shared memory contains a copy of each tensor for each
   process. GPU tensors may either be allocated internally
   by `vl_tmove` (in which case MATLAB may forget them)
   or may remember pointers to MATLAB's memory (inplace). 
   The latter is slightly unsafe, but much faster as it saves several copies.
   In any case, `vl_tmove` allocates a GPU buffer as large as
   the largest tensor as scratch space (and for direct GPU communication).

 * The supervisory and main threads collaborate through lock-less
   synchronization for speed. This is possible because at any point in time
   each tensor is managed by only one thread depending on its state.
   Thus a tensor moves from one thread to the other simply by swapping
   its state. There is, however, a condition variable to allow the
   main thread to wait for the supervisory thread when needed.

 * The supervisory thread waits by calling `poll()` on a number of sockets.
   However, sometimes the main thread needs to signal the supervisor too.
   This is realized by having a dummy `pipe()` between the two
   threads.

 **/

/* ---------------------------------------------------------------- */
/*                                                          Globals */
/* ---------------------------------------------------------------- */

enum {
  IN_COMMAND, IN_END
} ;

enum {
  OUT_RESULT, OUT_END
} ;

/* option codes */
enum {
  opt_inplace = 0,
  opt_verbose,
  opt_prefix,
} ;

/* options */
VLMXOption  options [] = {
  {"prefix",                1,   opt_prefix                },
  {"InPlace",               0,   opt_inplace               },
  {"Verbose",               0,   opt_verbose               },
  {0,                       0,   0                         }
} ;

int verbosity = 0 ;
vl::MexContext context ;

class SharedTensorDescriptor ;
class SharedTensorSpace ;
class ProcessPool ;

/* ---------------------------------------------------------------- */
/*                                                          Utility */
/* ---------------------------------------------------------------- */

static VLMXErrorCode vlmxParseDataType(vl::DataType & dataType, mxArray const * arg)
{
  if (vlmxCompareToStringI(arg, "double") == 0) {
    dataType = vl::VLDT_Double ;
    return VLMXE_Success ;
  } else if (vlmxCompareToStringI(arg, "single") == 0) {
    dataType = vl::VLDT_Float ;
    return VLMXE_Success ;
  } else {
    return VLMXE_IllegalArgument ;
  }
}

static VLMXErrorCode vlmxParseDeviceType(vl::DeviceType & deviceType, mxArray const * arg)
{
  if (vlmxCompareToStringI(arg, "cpu") == 0) {
    deviceType = vl::VLDT_CPU ;
    return VLMXE_Success ;
  } else if (vlmxCompareToStringI(arg, "gpu") == 0) {
    deviceType = vl::VLDT_GPU ;
    return VLMXE_Success ;
  } else {
    return VLMXE_IllegalArgument ;
  }
}

static VLMXErrorCode vlmxParseString(std::string & name, mxArray const * arg)
{
  char str [256] ;
  if (!vlmxIsString(arg, -1)) {
    return VLMXE_IllegalArgument ;
  }
  mxGetString(arg, str, sizeof(str)) ;
  name = str ;
  return VLMXE_Success ;
}

static VLMXErrorCode vlmxParseTensorShape(vl::TensorShape & shape, mxArray const * arg)
{
  size_t dimensions [32] ;
  if (!vlmxIsVector(arg, -1) || !vlmxIsPlain(arg)) {
    return VLMXE_IllegalArgument ;
  }
  int nd = mxGetNumberOfElements(arg) ;
  for (int k = 0 ; k < nd ; ++k) { dimensions[k] = (size_t)mxGetPr(arg)[k] ; }
  shape.setDimensions(dimensions, nd) ;
  return VLMXE_Success ;
}

/* ---------------------------------------------------------------- */
/*                                                           Logger */
/* ---------------------------------------------------------------- */

namespace vl {
  class Logger
  {
  public:
    Logger() ;
    ~Logger() ;
    std::ostringstream & getStream() ;
  protected:
    std::ostringstream stringStream ;
  private:
    // Disable
    Logger(const Logger&) ;
    Logger& operator= (const Logger&) ;
  } ;
}

vl::Logger::Logger()
{ }

vl::Logger::~Logger()
{
  printf("%s\n", stringStream.str().c_str()) ;
  //fflush(stdout) ;
}

std::ostringstream &
vl::Logger::getStream()
{
  return stringStream ;
}

#define LOGERROR \
vl::Logger().getStream() \
<<"[error]"<<__func__<<"::lab "<<lab<<"::"

#define LOG(level) \
if (verbosity < level) { } \
else vl::Logger().getStream() \
<<"[info] "<<__func__<<"::lab "<<lab<<"::"

/* ---------------------------------------------------------------- */
/*                                           SharedTensorDescriptor */
/* ---------------------------------------------------------------- */

#pragma mark -

// Describe one of the shared tensors: shape, data type,
// and device type.
class SharedTensorDescriptor
{
public:
  SharedTensorDescriptor() ;
  ~SharedTensorDescriptor() ;

  void init(vl::DeviceType deviceType,
            vl::DataType dataType,
            vl::TensorShape const & shape) ;
  void finalize() ;
  size_t getSizeInBytes() const ;
  SharedTensorDescriptor & operator=(SharedTensorDescriptor const & tensor) ;

  // Data.
  vl::DeviceType deviceType ;
  vl::DataType dataType ;
  vl::TensorShape shape ;
} ;

SharedTensorDescriptor::SharedTensorDescriptor()
{ }

SharedTensorDescriptor::~SharedTensorDescriptor()
{
  finalize() ;
}

SharedTensorDescriptor &
SharedTensorDescriptor::operator=(SharedTensorDescriptor const & tensor)
{
  deviceType = tensor.deviceType ;
  dataType = tensor.dataType ;
  shape = tensor.shape ;
  return *this ;
}

void SharedTensorDescriptor::init(vl::DeviceType newDeviceType,
                                  vl::DataType newDataType,
                                  vl::TensorShape const & newShape)
{
  assert(newDeviceType == vl::VLDT_CPU || newDeviceType == vl::VLDT_GPU) ;
  assert(newDataType == vl::VLDT_Float || newDataType == vl::VLDT_Double) ;
  deviceType = newDeviceType ;
  dataType = newDataType ;
  shape = newShape ;
}

void SharedTensorDescriptor::finalize()
{ }

size_t SharedTensorDescriptor::getSizeInBytes() const
{
  return shape.getNumElements() * getDataTypeSizeInBytes(dataType) ;
}

/* ---------------------------------------------------------------- */
/*                                                SharedTensorSpace */
/* ---------------------------------------------------------------- */

#pragma mark -

// SharedTensorSpace holds a list of tensors that can be accumulated
// between different processes.
//
// It encapsualtes in particular: the shared memory map,
// the GPU dispatch buffer, and, possibly, for non-inplace operations
// and GPU arrays, a copy of the GPU data.
//
// This class is not thread safe, so the MATLAB and flow supervisor thread
// must properly syncrhonize in accessing it.

class SharedTensorSpace
{
public:
  SharedTensorSpace() ;
  ~SharedTensorSpace() ;

  vl::ErrorCode mexInit(mxArray const *mexDescriptor) ;
  void finalize() ;
  vl::ErrorCode attach(std::string const & prefix, int lab, int numLabs) ;
  vl::ErrorCode attachPeer(int lab) ;

  void mexPrint() const ;
  void dump() const ;

private:
  bool initialized ;
  int lab ;
  int numLabs ;

  enum SharedTensorState {
    ready,
    accumulateChildren,
    waitParent,
    waitChildren,
  } state ;

  // This class represents an instance of a shared tensor. It contain
  // its state@transaction pair and information on its memory location.
  struct SharedTensorInstance
  {
    std::string name ;
    SharedTensorDescriptor descriptor ;
    SharedTensorState state ;
    size_t transaction ;
    size_t finalTransaction ;
    int numChildrenToAccumulate ;
    size_t memoryMapOffset ;
    void * cpuMemory ;
    void * gpuMemory ;
    bool gpuMemoryIsOwned ;
#if ENABLE_GPU
    cudaEvent_t gpuEvent ;
    bool gpuEventIsInitialized ;
#endif
    bool operator==(std::string const & theName) { return name == theName ; }
    SharedTensorInstance()
      : state(ready), transaction(0), finalTransaction((size_t)-1),
        cpuMemory(NULL), gpuMemory(NULL), gpuMemoryIsOwned(false)
#if ENABLE_GPU
    , gpuEvent(0), gpuEventIsInitialized(false)
#endif
    { }
  } ;
  typedef std::vector<SharedTensorInstance> tensors_t ;
  tensors_t tensors ;

  struct SharedTensorPeerInstance
  {
    int lab ;
    SharedTensorState state ;
    size_t transaction ;
    size_t finalTransaction ;
    void *mappedCpuMemory ;
    void *mappedGpuMemory ;
    bool accumulated ;
    bool operator==(int theLab) { return lab == theLab ; }
    SharedTensorPeerInstance()
      : lab(-1), state(ready), transaction(0),
        mappedCpuMemory(NULL), mappedGpuMemory(NULL), accumulated(false), 
        finalTransaction((size_t)-1) { }
  } ;
  typedef std::vector<std::vector<SharedTensorPeerInstance> > peerTensors_t ;
  peerTensors_t peerTensors ;
  SharedTensorPeerInstance & getPeerTensor(int tensorIndex, int lab) ;

  // Shared CPU memory
  void * memoryMap ;
  size_t memoryMapSize ;
  size_t memoryMapLabStride ;
  std::string memoryMapName ;
  int memoryMapFD ;
  bool memoryMapIsCudaRegistered ;

  // Additional GPU memory
  void * gpuDispatchMemory ;
  int gpuDevice ;

#if ENABLE_GPU
  // Todo: one for each mapped peer dispatch memory
  cudaIpcMemHandle_t gpuMemoryHandle ;
  cudaStream_t gpuHelperStream ;
  cudaEvent_t gpuHelperEvent ;
  bool gpuHelperStreamInitialized ;
  bool gpuHelperEventInitialized ;
#endif

  friend class ProcessPool ;
} ;

SharedTensorSpace::SharedTensorSpace()
  : initialized(false),
    memoryMapFD(-1),
    memoryMap(NULL),
    memoryMapIsCudaRegistered(false),
    memoryMapSize(0),
    gpuDevice(-1),
    gpuDispatchMemory(NULL)
#if ENABLE_GPU
,   gpuHelperStream(0),
    gpuHelperStreamInitialized(false),
    gpuHelperEventInitialized(false)
#endif
{ }

SharedTensorSpace::~SharedTensorSpace()
{
  finalize() ;
}

// This function initializes the SharedTensorSpace using
// a MATLAB cell array as descriptor for the space content.
// It can throw a MEX error, so it must be called from
// the MATLAB thread.

vl::ErrorCode SharedTensorSpace::mexInit(mxArray const *descriptor)
{
  assert(descriptor) ;

  if (initialized) {
    mexErrMsgTxt("Already initialized. Use 'reset' to clear.") ;
  }

  lab = -1 ;
  numLabs = 0 ;
  memoryMapName = "" ;
  memoryMapSize = 0 ;
  memoryMapLabStride = 0 ;

  // Parse tensor list
  if (!mxIsCell(descriptor)) {
    mexErrMsgTxt("DESCRIPTOR is not a cell array.") ;
  }
  if (mxGetNumberOfDimensions(descriptor) != 2) {
    mexErrMsgTxt("DESCRIPTOR does not have two dimensions.") ;
  }
  if (mxGetN(descriptor) != 3 &&
      mxGetN(descriptor) != 4) {
    mexErrMsgTxt("DESCRIPTOR does not have three or four columns.") ;
  }

  size_t numTensors = mxGetM(descriptor) ;
  size_t offset = 0 ;
  size_t const alignFactor = 16 ;
  bool useGPU = false ;

  for (int i = 0 ; i < numTensors ; ++i) {
    VLMXErrorCode error ;
    vl::DeviceType deviceType = vl::VLDT_CPU ;
    vl::DataType dataType ;
    vl::TensorShape shape ;
    std::string name ;

    error = vlmxParseDataType(dataType, mxGetCell(descriptor, 0*numTensors + i)) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "DESCRIPTOR{%d,1} is not a valid data type.", i+1) ;
    }

    error = vlmxParseTensorShape(shape, mxGetCell(descriptor, 1*numTensors + i)) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "DESCRIPTOR{%d,2} is not a valid tensor shape.", i+1) ;
    }

    error = vlmxParseString(name, mxGetCell(descriptor, 2*numTensors + i)) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "DESCRIPTOR{%d,3} is not a valid tensor name.", i+1) ;
    }

    if (mxGetN(descriptor) == 4) {
      error = vlmxParseDeviceType(deviceType, mxGetCell(descriptor, 3*numTensors + i)) ;
      if (error != VLMXE_Success) {
        vlmxError(error, "DESCRIPTOR{%d,4} is not a valid device type name.", i+1) ;
      }
    }

    if (deviceType == vl::VLDT_GPU) {
#if not defined(ENABLE_GPU)
      vlmxError(VLMXE_IllegalArgument, "GPU support not compiled.") ;
#endif
      useGPU = true ;
    }

    // Add the new tensor to the table.
    {
      SharedTensorInstance tensor ;
      tensor.name = name ;
      tensor.descriptor.init(deviceType, dataType, shape) ;
      tensor.memoryMapOffset = offset ;
      tensors.push_back(tensor) ;

      offset +=
        vl::divideAndRoundUp(tensor.descriptor.getSizeInBytes(), alignFactor) * alignFactor ;

      if (verbosity >= 2) {
        mexPrintf("[info] %s: registered tensor %s\n", __func__, name.c_str()) ;
      }
    }
  }

  // Size of the memory allocated for one lab (with a copy of all tensors).
  memoryMapName = "/mcn" ;
  size_t const pageSize = getpagesize() ;
  memoryMapLabStride = vl::divideAndRoundUp(offset, pageSize) * pageSize ;
  memoryMapSize = 0 ;

#if ENABLE_GPU
  if (useGPU) {
    cudaGetDevice(&gpuDevice) ; // to inform thread
    LOG(2) << "current CUDA device: " << gpuDevice ;
  }
#endif

  initialized = true ;
  return vl::VLE_Success ;
}

// Get the peer tensor corresponding to a given
// tensor and process index.

SharedTensorSpace::SharedTensorPeerInstance &
SharedTensorSpace::getPeerTensor(int tensorIndex, int lab)
{
  std::vector<SharedTensorPeerInstance>::iterator PT
  = std::find(peerTensors[tensorIndex].begin(), peerTensors[tensorIndex].end(), lab) ;
  assert(PT != peerTensors[tensorIndex].end()) ;
  return *PT ;
}

/// Attach the shared space. This allocates the shared memory map
/// for inter-process data transfers containing all tensors,
/// and the GPU dispatch memory.

vl::ErrorCode SharedTensorSpace::attach(std::string const & prefix, int lab, int numLabs)
{
  int error ;
  this->lab = lab ;
  this->numLabs = numLabs ;

  // Create the memory map name from the prefix.
  memoryMapName = std::string("/") + prefix ;

  // The root lab deletes a pre-existing memory object, if any.
  if (lab == 0) {
    error = shm_unlink(memoryMapName.c_str()) ;
    if (error == -1) {
      switch (errno) {
        case ENOENT:
          // Fine, there wasn't such a memory map anyways.
          break ;

        default:
          LOGERROR
          << "could not delete the stale memory map '"
          << memoryMapName.c_str()
          << "' because '" << strerror(errno) << '\'' ;
          return vl::VLE_Unknown ;
      }
    }
  }

  // Open/create the shared memory file descriptor.
  memoryMapSize = memoryMapLabStride * numLabs ;
  memoryMapFD = shm_open(memoryMapName.c_str(),
                         (lab == 0 ? O_CREAT:0)| O_RDWR, S_IRUSR | S_IWUSR) ;
  if (memoryMapFD == -1) {
    LOGERROR << "shm_open() failed because " << strerror(errno) ;
    close(memoryMapFD) ;
    memoryMapFD = -1 ;
    return vl::VLE_Unknown ;
  }

  // The root process set the size of the shared memory.
  if (lab == 0) {
    if (ftruncate(memoryMapFD, memoryMapSize) == -1) {
      LOGERROR << "truncate failed because " << strerror(errno) ;
      return vl::VLE_OutOfMemory ;
    }
  }

  // Map the memory.
  memoryMap = mmap(0, memoryMapSize,
                   PROT_READ | PROT_WRITE, MAP_SHARED,
                   memoryMapFD, 0) ;
  if (memoryMap == MAP_FAILED) {
    LOGERROR << "mmap failed because " << strerror(errno) ;
    memoryMap = NULL ;
    close(memoryMapFD) ;
    memoryMapFD = -1 ;
    return vl::VLE_Unknown ;
  }
  memoryMapIsCudaRegistered = false ;

  // The FD is not needed after mmap.
  close(memoryMapFD) ;
  memoryMapFD = -1 ;

  // Associate memory to tensors.
#if ENABLE_GPU
  size_t maxGPUTensorSize = 0 ;
#endif
  for (int t = 0 ; t < tensors.size() ; ++t) {
    tensors[t].cpuMemory = (char*)memoryMap
    + tensors[t].memoryMapOffset
    + lab * memoryMapLabStride ;
#if ENABLE_GPU
    if (tensors[t].descriptor.deviceType == vl::VLDT_GPU) {
      // Lazy allocation (to allow inplace operations).
      tensors[t].gpuMemory = NULL ;
      tensors[t].gpuMemoryIsOwned = false ;
      maxGPUTensorSize = std::max(maxGPUTensorSize,
                                  tensors[t].descriptor.getSizeInBytes()) ;

      cudaError_t cerror = cudaEventCreate(&tensors[t].gpuEvent) ;
      if (cerror != cudaSuccess) {
        LOGERROR
          << "CUDA could not create an event because '"
          << cudaGetErrorString(cerror) << '\'' ;
        return vl::VLE_Cuda ;
      }
      tensors[t].gpuEventIsInitialized = true ;
    }
#endif
  }

#if ENABLE_GPU
  if (maxGPUTensorSize > 0) {
    cudaError_t cerror ;
    cerror = cudaMalloc(&gpuDispatchMemory, maxGPUTensorSize) ;
    if (cerror != cudaSuccess) {
      LOGERROR
      << "could not allocate GPU memory for dispatch because '"
      << cudaGetErrorString(cerror) << '\'' ;
      gpuDispatchMemory = NULL ;
      return vl::VLE_Cuda ;
    }

    // To parallelize memory transfers we use a separate CUDA stream.
    cerror = cudaStreamCreateWithFlags(&gpuHelperStream, cudaStreamNonBlocking) ;
    if (cerror != cudaSuccess) {
      LOGERROR
      << "could not create a CUDA stream because '"
      << cudaGetErrorString(cerror) << '\'' ;
      return vl::VLE_Cuda ;
    }
    gpuHelperStreamInitialized = true ;

    // Pin all shared host memory.
    cerror = cudaHostRegister(memoryMap,
                              memoryMapSize,
                              cudaHostRegisterDefault) ;
    if (cerror != cudaSuccess) {
      LOGERROR
        << "CUDA generated an error while pinning the shared host memory: '"
        << cudaGetErrorString(cerror) << '\'' ;
    } else {
      LOG(2) << "pinned shared memory" ;
      memoryMapIsCudaRegistered = true ;
    }
  }
#endif

  return vl::VLE_Success ;
}

// attachPeer
vl::ErrorCode
SharedTensorSpace::attachPeer(int lab)
{
  if (peerTensors.size() != tensors.size()) {
    peerTensors.resize(tensors.size()) ;
  }
  for (int t = 0 ; t < tensors.size() ; ++t) {
    SharedTensorPeerInstance peerTensor ;
    peerTensor.lab = lab ;
    peerTensor.state = SharedTensorSpace::ready ;
    peerTensor.mappedCpuMemory = (char*)memoryMap
    + tensors[t].memoryMapOffset
    + lab * memoryMapLabStride ;
    peerTensor.accumulated = false ;
    peerTensors[t].push_back(peerTensor) ;
  }
  return vl::VLE_Success ;
}

// Destroy all resources
// 1) unmap and unlink shared memory map
// 2) ...

void SharedTensorSpace::finalize()
{
  int error ;

  initialized = false ;

#if ENABLE_GPU
  if (memoryMap && memoryMapIsCudaRegistered) {
    cudaHostUnregister(memoryMap) ;
  }

  // if (gpuHelperEventInitialized) {
  //   cudaEventDestroy(gpuHelperEvent) ;
  //   gpuHelperEventInitialized = false ;
  // }

  if (gpuHelperStreamInitialized) {
    cudaStreamDestroy(gpuHelperStream) ;
    gpuHelperStream = 0 ;
    gpuHelperStreamInitialized = false ;
  }

  if (gpuDispatchMemory) {
    cudaFree(gpuDispatchMemory) ;
    gpuDispatchMemory = NULL ;
  }

  for (tensors_t::iterator T = tensors.begin() ;
       T != tensors.end() ;
       T++)
  {
    if (T->gpuMemory && T->gpuMemoryIsOwned) {
      cudaFree(T->gpuMemory) ;
      T->gpuMemory = NULL ;
      T->gpuMemoryIsOwned = false ;
    }

    if (T->gpuEventIsInitialized) {
      cudaEventDestroy(T->gpuEvent) ;
      T->gpuEvent = 0 ;
      T->gpuEventIsInitialized = false ;
    }
  }
  gpuDevice = -1 ;
#endif

  if (memoryMap) {
    munmap(memoryMap, memoryMapSize) ;
    memoryMap = NULL ;
  }

  if (memoryMapFD != -1) {
    // This should have beeen closed right after mmap().
    close(memoryMapFD) ;
    memoryMapFD = -1 ;
  }

  error = shm_unlink(memoryMapName.c_str()) ;
  if (error == -1 && errno == EACCES) {
    LOGERROR << "Cannot clear the shared memory map due to a permission error." ;
  }

  tensors.clear() ;
  numLabs = -1 ;
}

// For debugging
void SharedTensorSpace::dump() const
{
  for (int tensorIndex = 0 ; tensorIndex < tensors.size() ; ++tensorIndex) {
    SharedTensorInstance const & T = tensors[tensorIndex] ;
    char const * stateString ;

    switch (T.state) {
    case ready: stateString="ready" ; break ;
    case accumulateChildren: stateString="accumulateChildren" ; break ;
    case waitParent: stateString="waitParent" ; break ;
    case waitChildren: stateString="waitChildren" ; break ;
    }
    LOG(0)<<"Tensor " << T.name ;
    LOG(0)<<"\tState: " << stateString ;
    LOG(0)<<"\ttransaction: "<<T.transaction ;
    if (peerTensors.size() > tensorIndex) {
      for (int p = 0 ; p < peerTensors[tensorIndex].size() ; ++p) {
        SharedTensorPeerInstance const & PT = peerTensors[tensorIndex][p] ;
        switch (PT.state) {
        case ready: stateString="ready" ; break ;
        case accumulateChildren: stateString="accumulateChildren" ; break ;
        case waitParent: stateString="waitParent" ; break ;
        case waitChildren: stateString="waitChildren" ; break ;
        }
        LOG(0)<<"\tPeer on lab " << PT.lab << ": " << stateString;
        LOG(0)<<"\t\ttransaction:" << PT.transaction ;
      }
    }
  }
}

void SharedTensorSpace::mexPrint() const
{
  mexPrintf("\tlab %d of %d\n", lab, numLabs) ;
  mexPrintf("\tshared memory: '%s', %d bytes mapped at address: 0x%zx\n",
            memoryMapName.c_str(),memoryMapSize,memoryMap) ;
  for (int tensorIndex = 0 ; tensorIndex < tensors.size() ; ++tensorIndex) {
    SharedTensorInstance const & T = tensors[tensorIndex] ;
    mexPrintf("\tTensor '%s'\n", T.name.c_str()) ;
    mexPrintf("\t\t[") ;
    for (int k = 0 ; k < T.descriptor.shape.getNumDimensions() ; ++k) {
      mexPrintf(" %d", T.descriptor.shape.getDimensions()[k]) ;
    }
    mexPrintf("] %s %s\n",
              T.descriptor.dataType == vl::VLDT_Double?"double":"single",
              T.descriptor.deviceType == vl::VLDT_CPU?"CPU":"GPU") ;
    mexPrintf("\t\tCPU address: 0x%zx\n", T.cpuMemory) ;
    mexPrintf("\t\tGPU address: 0x%zx\n", T.gpuMemory) ;

    if (peerTensors.size() > tensorIndex) {
      for (int p = 0 ; p < peerTensors[tensorIndex].size() ; ++p) {
        SharedTensorPeerInstance const & PT = peerTensors[tensorIndex][p] ;
        mexPrintf("\t\tPeer instance %d\n", p) ;
        mexPrintf("\t\t\tlab: %0d\n", PT.lab) ;
        mexPrintf("\t\t\tmapped CPU address: 0x%zx\n",PT.mappedCpuMemory) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                                                      ProcessPool */
/* ---------------------------------------------------------------- */

#pragma mark -

/// Represents a pool of collaborating MATLAB processes. Usually each
/// process corresponds to a certain MATLAB instance in a MATLAB pool.
class ProcessPool
{
public:
  /// Create an un-intialized ProcessPool. Before it is used,
  /// the pool must be initialized using init(). This design allows
  /// to catch errors during initialization without resorting to exceptions.
  ProcessPool() ;

  /// Automatically calls ::finalize().
  ~ProcessPool() ;

  /// Initialize the instance \a lab of \a numLabs pools. The function
  /// timesout.
  vl::ErrorCode init(std::string const & prefix, int lab,
                     int numLabs, SharedTensorSpace * space) ;

  /// Gracefully shutdown the connection with the other processes,
  /// waiting for them to finish updating as needed. After this, the
  /// supervisory thread quits, but the object remains initialized
  /// to allow reading off the final value of the tensor.
  ///
  /// The function timesout.
  vl::ErrorCode shutdown() ;

  /// Immediately terminate the ProcessPool instance and release all
  /// resources.
  void finalize() ;

  /// Print information.
  ///
  /// This function must be called from the MATLAB thread.
  void mexPrint() const ;

  /// Push a tensor in the pool for accumulation.
  ///
  /// This function must be called from the MATLAB thread. It throws
  /// a MEX error on error and can time out.
  void mexPush(std::string const & name, mxArray const * x,
               bool inplace = false) ;

  /// Pull an accumulated tensor from the pool.
  ///
  /// This function must be called from the MATLAB thread. It throws
  /// a MEX error on error and an time out.
  mxArray * mexPull(std::string const & name, bool inplace = false) ;


  /// Check whether the instance is intialized or not.
  bool isInitialized() const { return initialized ; }

private:
  bool initialized ;
  std::string prefix ;
  int lab ;
  int numLabs ;
  size_t timeoutInterval ;
  SharedTensorSpace * sharedSpace ;

  // Messages between peer processes.
  struct Message
  {
    enum MessageType {
      /// Sent from root to leaves to request initialization during
      /// hanshake.
      init,

      /// Sent from leaves to root to acknowledge initialization.
      initDone,

      /// Sent from root to leaves to request attching the shared
      /// resources (shared memory).
      attach,

      /// Sent to advertise a state change for a tensor.
      tensorStateChange,

      /// Shutdown sequence
      requestShutdown,

      /// Communicate the final transaction index for quitting.
      tensorFinalTransaction
    }
    type ;

    /// The transaction number.
    size_t transaction ;

    /// The final transaction number.
    size_t finalTransaction ;

    // Sender and destination process indexes.
    int16_t from ;
    int16_t to ;

    // Session identifier, used for sanity checks.
    uint32_t session ;

    // Tensort ID and state for a tensor state change.
    uint32_t tensorId ;
    SharedTensorSpace::SharedTensorState tensorState ;
    Message() : transaction(0), finalTransaction((size_t)-1), tensorId(0) { }
  } ;

  class Supervisor {
  public:
    Supervisor(ProcessPool& pool)
      : pool(pool), thread(NULL), state(down),
        socketFD(-1) { pipeFD[0] = -1 ; pipeFD[1] = -1 ; }
    ~Supervisor() { finalize() ; }

    vl::ErrorCode init() ;
    void finalize() ;
    vl::ErrorCode shutdown() ;
    vl::ErrorCode beginTransaction(int tensorIndex) ;
    vl::ErrorCode waitTensor(int tensorIndex) ;

  private:
    ProcessPool & pool ;

    tthread::thread * thread ;
    enum State {
      connecting,
      running,
      shuttingDown,
      down} state ;

    // Peer processes.
    struct Peer
    {
      int lab ;
      int socketFD ;
      bool cudaCanAccessPeer ; //cudaDeviceCanAccessPeer
      bool shutdownRequested ;
      Peer(int lab)
        : lab(lab), socketFD(-1),
          cudaCanAccessPeer(false),
          shutdownRequested(false)
      { }
      bool operator== (int lab) { return this->lab == lab ; }
    } ;
    typedef std::vector<Peer> peers_t ;
    peers_t peers ;

    // Comms.
    uint32_t session ;
    int pipeFD [2] ;
    int socketFD ;
    tthread::mutex mutex ;
    tthread::condition_variable waitingList ;
    bool shutdownRequested ; // local
    bool forceQuit ;

    static void threadEntryPoint(void * thing) ;
    void entryPoint() ;

    vl::ErrorCode connect() ;
    void disconnect() ;
    vl::ErrorCode handshake() ;
    vl::ErrorCode loop() ;
    vl::ErrorCode send(Message &msg, int to) ;
    vl::ErrorCode receive(Message &msg, int from, int timeout = -1) ;
    vl::ErrorCode handleAccumulateChildren(int tensorIndex) ;
    vl::ErrorCode handleWaitParent(int tensorIndex) ;
    vl::ErrorCode handleWaitChildren(int tensorIndex) ;
  } supervisor ;
} ;


ProcessPool::ProcessPool()
: supervisor(*this),
  initialized(false),
  lab(-1), numLabs(0)
{ }

ProcessPool::~ProcessPool()
{
  finalize() ;
}

vl::ErrorCode ProcessPool::init(std::string const & newPrefix, int newLab, int newNumLabs, SharedTensorSpace * newSharedSpace)
{
  vl::ErrorCode error ;

  assert(newLab >= 0) ;
  assert(newNumLabs > newLab) ;
  assert(newSharedSpace) ;

  // finalize process pool if previously initialized
  finalize() ;

  // set members
  prefix = newPrefix ;
  lab = newLab ;
  numLabs = newNumLabs ;
  sharedSpace = newSharedSpace ;
  timeoutInterval = 30UL * 1000UL * 1000UL ; // 30s in us

  error = supervisor.init() ;
  if (error == vl::VLE_Success) {
    initialized = true ;
  }
  return error ;
}

vl::ErrorCode ProcessPool::shutdown()
{
  return supervisor.shutdown() ;
}

void ProcessPool::finalize()
{
  supervisor.finalize() ;
  if (sharedSpace) {
    sharedSpace->finalize() ;
    delete sharedSpace ;
    sharedSpace = NULL ;
  }
  lab = -1 ;
  numLabs = 0 ;
  initialized = false ;
}

void ProcessPool::mexPrint() const
{
  tthread::lock_guard<tthread::mutex> (mutex) ;
  if (sharedSpace) {
    sharedSpace->mexPrint() ;
  } else {
    mexPrintf("Uninitialized.") ;
  }
}

void ProcessPool::mexPush(std::string const & name,
                          mxArray const * x,
                          bool inplace)
{
  // Search tensor by name.
  SharedTensorSpace::tensors_t::iterator T
  = std::find(sharedSpace->tensors.begin(), sharedSpace->tensors.end(), name) ;
  if (T == sharedSpace->tensors.end()) {
    vlmxError(VLMXE_IllegalArgument, "There is no tensor '%s'.", name.c_str()) ;
  }

  // Encapsulate MATLAB argument and check tensor compatibility.
  vl::MexTensor mtens(context) ;
  mtens.init(x) ;

  if (mtens.getDeviceType() != T->descriptor.deviceType) {
    vlmxError(VLMXE_IllegalArgument, "The tensor device type is incorrect.") ;
  }

  if (mtens.getDataType() != T->descriptor.dataType) {
    vlmxError(VLMXE_IllegalArgument, "The tensor data type is incorrect.") ;
  }

  if (mtens.getNumElements() != T->descriptor.shape.getNumElements()) {
    vlmxError(VLMXE_IllegalArgument, "The tensor shape is incorrect.") ;
  }

  if (inplace && T->descriptor.deviceType != vl::VLDT_GPU) {
    vlmxError(VLMXE_IllegalArgument, "Inplace operations are supported only for GPU arrays.") ;
  }

  // Wait until the tensor is in ready state
  vl::ErrorCode error = supervisor.waitTensor(T - sharedSpace->tensors.begin()) ;
  if (error != vl::VLE_Success) {
    vlmxError(VLMXE_Execution, "Timeout or disconnected while waiting for tensor '%s' to become ready.", T->name.c_str()) ;
  }

  // Copy memory to SharedSpace
  if (T->descriptor.deviceType == vl::VLDT_CPU) {
    memcpy(T->cpuMemory, mtens.getMemory(), T->descriptor.getSizeInBytes()) ;
  } else {
#if ENABLE_GPU
    cudaError_t cerror ;

    // sync main thread (do not start until the parameters have been computed!)
    cudaEventRecord(T->gpuEvent, 0) ;
    cudaStreamWaitEvent(sharedSpace->gpuHelperStream, T->gpuEvent, 0) ;

    if (inplace) {
      if (T->gpuMemoryIsOwned && T->gpuMemory) {
        // Free the previously allocated memory as we are going to use
        // an inplace operation on this tensor.
        cudaFree(T->gpuMemory) ;
        T->gpuMemory = NULL ;
      }
      T->gpuMemoryIsOwned = false ;
      T->gpuMemory = mtens.getMemory() ;
    } else {
      if (T->gpuMemoryIsOwned == false || T->gpuMemory == NULL) {
        cerror = cudaMalloc(&T->gpuMemory,
                            T->descriptor.getSizeInBytes()) ;
        if (cerror != cudaSuccess) {
          T->gpuMemory = NULL ;
          T->gpuMemoryIsOwned = false ;
          vlmxError(VLMXE_Alloc, "CUDA error while allocating GPU memory (%s).",
                    cudaGetErrorString(cerror)) ;
        }
        T->gpuMemoryIsOwned = true ;
        cerror = cudaMemcpyAsync (T->gpuMemory,
                                  mtens.getMemory(),
                                  T->descriptor.getSizeInBytes(),
                                  cudaMemcpyDeviceToDevice,
                                  sharedSpace->gpuHelperStream) ;
        if (cerror != cudaSuccess) {
          vlmxError(VLMXE_Execution, "CUDA error while copying GPU data (%s).",
                    cudaGetErrorString(cerror)) ;
        }
      }
    }
#endif
  }
  supervisor.beginTransaction(T - sharedSpace->tensors.begin()) ;
}

mxArray * ProcessPool::mexPull(std::string const & name, bool inplace)
{
  // Search the tensor with the specified name.
  SharedTensorSpace::tensors_t::const_iterator T
  = std::find(sharedSpace->tensors.begin(), sharedSpace->tensors.end(), name) ;

  if (T == sharedSpace->tensors.end()) {
    vlmxError(VLMXE_IllegalArgument, "There is no tensor with the specified name.") ;
  }

  if (inplace && T->descriptor.deviceType != vl::VLDT_GPU) {
    vlmxError(VLMXE_IllegalArgument, "Inplace operations are supported only for GPU arrays.") ;
  }

  // Wait until the tensor is in ready state
  vl::ErrorCode error = supervisor.waitTensor(T - sharedSpace->tensors.begin()) ;
  if (error != vl::VLE_Success) {
    vlmxError(VLMXE_Execution, "Timeout or disconnected while waiting for tensor '%s' to become ready.", T->name.c_str()) ;
  }

  if (inplace) {
    // With in-place operations, the only purpose of pull() is to wait until
    // the tensor is ready and can be accessed.
    return NULL ;
  } else {
    vl::MexTensor result(context) ;
    result.init(T->descriptor.deviceType, T->descriptor.dataType, T->descriptor.shape) ;

    if (T->descriptor.deviceType == vl::VLDT_CPU) {
      memcpy(result.getMemory(),
             T->cpuMemory,
             T->descriptor.getSizeInBytes()) ;
    } else {
#if ENABLE_GPU
      // Synchronous with main thread.
      cudaError_t cerror = cudaMemcpyAsync (result.getMemory(),
                                           T->gpuMemory,
                                           T->descriptor.getSizeInBytes(),
                                           cudaMemcpyDeviceToDevice,
                                           sharedSpace->gpuHelperStream) ;
      if (cerror != cudaSuccess) {
        vlmxError(VLMXE_Execution, "CUDA generated an error while copying GPU data: '%s'.",
                  cudaGetErrorString(cerror)) ;
      }

      cerror = cudaStreamSynchronize(sharedSpace->gpuHelperStream) ;
      if (cerror != cudaSuccess) {
        vlmxError(VLMXE_Execution, "CUDA generated an error while synchronizing a stream: '%s'.",
                  cudaGetErrorString(cerror)) ;
      }
#endif
    }
    return result.relinquish() ;
  }
}

/* ---------------------------------------------------------------- */
/*                                          ProcessPool::Supervisor */
/* ---------------------------------------------------------------- */

#pragma mark -

#undef LOGERROR
#define LOGERROR \
vl::Logger().getStream() \
<<"[error]"<<__func__<<"::lab "<<pool.lab<<"::"

#undef LOG
#define LOG(level) \
if (verbosity < level) { } \
else vl::Logger().getStream() \
<<"[info] "<<__func__<<"::lab "<<pool.lab<<"::"

void ProcessPool::Supervisor::threadEntryPoint(void * thing)
{
  ((ProcessPool::Supervisor*)thing)->entryPoint() ;
}

vl::ErrorCode ProcessPool::Supervisor::init()
{
  vl::ErrorCode error = vl::VLE_Success ;
  finalize() ;

  // Infer parent and children labs.
  int bit = ffs(pool.lab) - 1 ;
  if (bit == -1) { bit = 31 ; }

  int parent = pool.lab & (~(1 << bit)) ;
  if (parent != pool.lab) {
    // peers[0] always contain the parent (except for root)
    peers.push_back(Peer(parent)) ;
  }

  for (int k = 0 ; k < bit ; ++k) {
    int child = pool.lab | (1 << k) ;
    if (child < pool.numLabs) {
      // Which peers[] gets which children is determined later
      // during hadshake based on the random connection order.
      // Here we assign a provisional lab index using negative indexes
      // as these are needed to use send().
      peers.push_back(Peer(-child)) ;
    }
  }

  state = connecting ;
  shutdownRequested = false ;
  forceQuit = false ;
  thread = new tthread::thread(threadEntryPoint, this) ;

  // Wait for initialization to be complete.
  {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    while (state == connecting) {
      waitingList.wait(mutex) ;
    }
    if (state == running) {
      error = vl::VLE_Success ;
    } else {
      error = vl::VLE_Unknown ;
    }
  }

  return error ;
}

void ProcessPool::Supervisor::finalize()
{
  if (thread) {
    {
      tthread::lock_guard<tthread::mutex> lock(mutex) ;
      forceQuit = true ;
      if (pipeFD[1] >= 0) {
        char dummy = 1 ;
        write(pipeFD[1], &dummy, 1) ;
      }
    }
    if (thread->joinable()) {
      thread->join() ;
    }
    delete thread ;
    thread = NULL ;
  }
  peers.clear() ;
}

vl::ErrorCode ProcessPool::Supervisor::shutdown()
{
  // Signal the supervisory thread
  shutdownRequested = true ;
  char dummy = 1 ;
  write(pipeFD[1], &dummy, 1) ;

  // Wait for shutdown to complete
  {
    size_t start = vl::getTime() ;
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    while (state != down) {
      if (vl::getTime() > start + pool.timeoutInterval) {
        LOGERROR << "timeout while shutting down" ;
        return vl::VLE_Timeout ;
      }
      waitingList.wait(mutex) ;
    }
  }
  return vl::VLE_Success ;
}

vl::ErrorCode ProcessPool::Supervisor::beginTransaction(int tensorIndex)
{
  vl::ErrorCode error = vl::VLE_Success ;
  SharedTensorSpace::SharedTensorInstance & T = pool.sharedSpace->tensors[tensorIndex] ;

  T.transaction ++ ;
  T.numChildrenToAccumulate = 0 ;
  for (int p = (pool.lab > 0) ; p < peers.size() ; ++p) {
    SharedTensorSpace::SharedTensorPeerInstance & PT = pool.sharedSpace->peerTensors[tensorIndex][p] ;
    PT.accumulated = false ;
    T.numChildrenToAccumulate += 1;
  }
  asm volatile("": : :"memory") ; // Memory barrier: prevents compiler from reordering
  T.state = SharedTensorSpace::accumulateChildren ; // Must be last to close transaction

  // Signal the supervisory thread
  {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    char dummy = 1 ;
    write(pipeFD[1], &dummy, 1) ;
  }
  return error ;
}

vl::ErrorCode ProcessPool::Supervisor::waitTensor(int tensorIndex)
{
  SharedTensorSpace::SharedTensorInstance & T = pool.sharedSpace->tensors[tensorIndex] ;
  size_t start = vl::getTime() ;
  tthread::lock_guard<tthread::mutex> lock(mutex) ;
  while (T.state != SharedTensorSpace::ready) {
    if ((vl::getTime() - start) > pool.timeoutInterval) {
      return vl::VLE_Timeout ;
    }
    if (state != running) {
      return vl::VLE_Unknown ;
    }
    waitingList.wait(mutex) ;
  }
  return vl::VLE_Success ;
}

vl::ErrorCode ProcessPool::Supervisor::send(Message & msg, int to)
{
  // Find connection to peer.
  peers_t::const_iterator rel = std::find(peers.begin(), peers.end(), to) ;
  assert(rel != peers.end()) ;

  // Add complementery information to the message.
  msg.session = session ;
  msg.from = pool.lab ;
  msg.to = to ;

  // Send all bytes.
  int bytesWritten = 0 ;
  int status ;
  char * nextByte = (char*)&msg ;
  while (bytesWritten < sizeof(msg)) {
    status = write(rel->socketFD, nextByte, sizeof(msg) - bytesWritten) ;
    if (status == -1) {
      LOGERROR
      << "could not send message to " << to
      << " because '" << strerror(errno) << '\'' ;
      return vl::VLE_Unknown ;
    }
    bytesWritten += status ;
  }

  LOG(3)
  << "sent message to " << to
  << " (type "  << msg.type
  << ", state " << msg.tensorState
  << " tensor " << msg.tensorId
  << ')' ;
  return vl::VLE_Success ;
}

vl::ErrorCode ProcessPool::Supervisor::receive(Message & msg, int from, int timeout)
{
  size_t waited = 0 ; // us
  size_t const pollInterval = 1000 ; // us
  if (timeout < 0) { timeout = pool.timeoutInterval ; } // us

  // find connection to peer
  peers_t::const_iterator rel = std::find(peers.begin(), peers.end(), from) ;
  assert(rel != peers.end()) ;

  // receive all bytes
  {
    int bytesRead = 0 ;
    int status ;
    char * nextByte = (char*)&msg ;
    while (bytesRead < sizeof(msg)) {
      status = read(rel->socketFD, nextByte, sizeof(msg) - bytesRead) ;
      if (status == 0 || status == -1) {
        if (status == 0 || errno == EAGAIN) {
          if (timeout == 0 && bytesRead == 0) {
            // non blocking operation, no message, just return no data
            return vl::VLE_NoData ;
          }
          if (timeout > 0 && waited >= timeout) {
            if (verbosity >= 1) {
              LOGERROR
              << "timed out while receiving a message from lab " << from
              << " because '" << strerror(errno) << '\'' ;
            }
            return vl::VLE_Timeout ;
          }
          usleep(pollInterval) ;
          waited += pollInterval ;
          continue ;
        }
        if (verbosity >= 1) {
          LOGERROR
          << "error while receiving a message from lab " << from
          << ": '" << strerror(errno) << '\'' ;
        }
        return vl::VLE_Unknown ;
      }
      bytesRead += status ;
    }
  }

  // check message integrity
  if ((msg.type != Message::init &&
       msg.type != Message::initDone)
      && (msg.session != session &&
          msg.from != from &&
          msg.to != pool.lab)) {
        LOGERROR
        << "received an unexpected message from lab " << from
        << "\n\tmsg: session:" << msg.session
        << " from:" << msg.from
        << " to:"  << msg.to
        << " type:" << msg.type
        << "\n\tthis session:" << this->session ;
        return vl::VLE_Unknown ;
      }

  LOG(3)
  << "received message from "<<from
  << " (type "   << msg.type
  << ", state "  << msg.tensorState
  << ", tensor " << msg.tensorId
  << ')' ;

  return vl::VLE_Success ;
}

/// Establish connections with the peers.
vl::ErrorCode ProcessPool::Supervisor::connect()
{
  vl::ErrorCode error = vl::VLE_Success ;
  int result ;
  char socketName [256] ;
  struct sockaddr_un socketAddress ;
  size_t start = vl::getTime() ;
  pipeFD[0] = -1 ;
  pipeFD[1] = -1 ;
  socketFD = -1 ;

  // Lock for entire duration of connect()
  tthread::lock_guard<tthread::mutex> lock(mutex) ;

  // Advertise
  state = connecting ;
  waitingList.notify_all() ;

  // Cerate a pipe FD for notification between MATLAB's thread
  // and the supervisory thread. This is needed to allow awaking
  // the supervisory thread.
  result = pipe(pipeFD) ;
  if (result == -1) {
    pipeFD[0] = -1 ;
    pipeFD[1] = -1 ;
    LOGERROR
    << "cannot create inter-threads pipe because: '"
    << strerror(errno) << '\'' ;
    return vl::VLE_Unknown ;
  }

  // Create a socket and connect children.
  size_t numChildren = peers.size() - (pool.lab > 0) ;
  if (numChildren > 0) {

    // Get a UNID comain socket.
    snprintf(socketName, sizeof(socketName)/sizeof(socketName[0]),
             "/%s/%s-socket-%02d", P_tmpdir, pool.prefix.c_str(), pool.lab) ;
    socketFD = socket(AF_UNIX, SOCK_STREAM, 0) ;
    if (socketFD == -1) {
      LOGERROR
      << "cannot create socket " << socketName
      << "because: " << strerror(errno) ;
      return vl::VLE_Unknown ;
    }

    // Copy socket path into socketAddress.
    memset(&socketAddress, 0, sizeof(socketAddress)) ;
    socketAddress.sun_family = AF_UNIX;
    strncpy(socketAddress.sun_path, socketName,
            sizeof(socketAddress.sun_path) - 1) ;

    // Delete socket path if it exists before binding.
    if (access(socketAddress.sun_path, F_OK) == 0) {
      unlink(socketAddress.sun_path) ;
    }

    // Bind socket to address.
    result = bind(socketFD,
                  (struct sockaddr *)&socketAddress,
                  sizeof(socketAddress)) ;
    if (result == -1) {
      LOGERROR
      << "cannot bind socket " << socketName
      << "because: " << strerror(errno) ;
      return vl::VLE_Unknown ;
    }

    // Start listening for children connections
    result = listen(socketFD, numChildren) ;
    if (result == -1) {
      LOGERROR
      << "cannot listen to socket " << socketName
      << "because: " << strerror(errno) ;
      return vl::VLE_Unknown ;
    }

    // Do not block on accept().
    fcntl(socketFD, F_SETFL, fcntl(socketFD, F_GETFL, 0) | O_NONBLOCK);

    // Accept one connection per child.
    for (int p = (pool.lab > 0) ; p < peers.size() ; ++p) {
      peers[p].socketFD = -1 ;
      for (;;) {
        peers[p].socketFD = accept(socketFD, NULL, NULL) ;
        if (peers[p].socketFD == -1) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            if (vl::getTime() < start + pool.timeoutInterval) continue ; // retry
            LOGERROR
            << "timed out while accepting connection from peer " << peers[p].lab ;
            error = vl::VLE_Timeout ;
            goto done ;
          }
          LOGERROR
          << " cannot accept connection from peer " << peers[p].lab
          << " because: " << strerror(errno) ;
          error = vl::VLE_Unknown ;
          goto done ;
        }
        break ;
      }
      fcntl(peers[p].socketFD, F_SETFL,
            fcntl(peers[p].socketFD ,F_GETFL, 0) | O_NONBLOCK) ;
    }
  }

  // Connect parent.
  if (pool.lab > 0) {
    snprintf(socketName, sizeof(socketName)/sizeof(socketName[0]),
             "/%s/%s-socket-%02d", P_tmpdir, pool.prefix.c_str(), peers[0].lab) ;

    for (;;) {
      peers[0].socketFD = socket(AF_UNIX, SOCK_STREAM, 0) ;
      if (peers[0].socketFD == -1) {
        if (vl::getTime() < start + pool.timeoutInterval) {
          // Wait for parent to create socket file.
          usleep(100UL * 1000UL) ; // 100 ms (10 times a second)
          continue ;
        }
        LOGERROR
        << "cannot create socket '" << socketName
        << "' because '" << strerror(errno) << '"' ;
        error = vl::VLE_Unknown ;
        goto done ;
      }
      break ;
    }
    fcntl(peers[0].socketFD, F_SETFL,
          fcntl(peers[0].socketFD ,F_GETFL, 0) | O_NONBLOCK) ;

    // Copy socket path into socketAddress.
    memset(&socketAddress, 0, sizeof(socketAddress)) ;
    socketAddress.sun_family = AF_UNIX;
    strncpy(socketAddress.sun_path, socketName,
            sizeof(socketAddress.sun_path) - 1) ;

    // Establish connection with parent.
    for (int trials = 0 ; ; ++trials) {
      int result = ::connect(peers[0].socketFD,
                             (struct sockaddr *)&socketAddress,
                             sizeof(socketAddress)) ;
      if (result == 0) break ;
      if (vl::getTime() < start + pool.timeoutInterval) {
        // Wait for parent to start accepting connections.
        usleep(100UL * 1000UL) ; // 100 ms (10 times a second)
        continue ;
      }
      LOGERROR
      << "cannot connect socket " << socketName
      << " after trying " << trials
      << " times because '" << strerror(errno) << '"' ;
      error = vl::VLE_Unknown ;
      goto done ;
    }
  }

done:
  return error ;
}

void ProcessPool::Supervisor::disconnect()
{
  // Lock for entire duration of disconnect()
  tthread::lock_guard<tthread::mutex> lock(mutex) ;

  for (int p = 0 ; p < peers.size() ; ++p) {
    if (peers[p].socketFD != -1) {
      close(peers[p].socketFD) ;
      peers[p].socketFD = -1 ;
    }
  }

  if (socketFD != -1) {
    close(socketFD) ;
    socketFD = -1 ;
  }

  char socketName [256] ;
  snprintf(socketName, sizeof(socketName)/sizeof(socketName[0]),
           "/%s/%s-socket-%02d", P_tmpdir, pool.prefix.c_str(), pool.lab) ;
  unlink(socketName) ;

  for (int t = 1 ; t >= 0 ; --t) {
    if (pipeFD[t] != -1) {
      close(pipeFD[t]) ;
      pipeFD[t] = -1 ;
    }
  }

  state = down ;
  waitingList.notify_all() ;
}

// The purpose of the handshake sequence is to make sure that
// all processes are properly communicating and ready to go.
// It is also required to synchornize the root (which creates several
// shared resources) and the other nodes (which attach them).

vl::ErrorCode ProcessPool::Supervisor::handshake()
{
  Message msg ;
  vl::ErrorCode error = vl::VLE_Success ;

  // Lock for entire duration of handshake()
  tthread::lock_guard<tthread::mutex> lock(mutex) ;

  LOG(2) << "handshake begins" ;

  // receive message from parent (except for root)
  if (pool.lab == 0) {
    session = (uint32_t)vl::getTime() ;
    // root atteches first
    error = pool.sharedSpace->attach(pool.prefix, 0, pool.numLabs) ;
    if (error != vl::VLE_Success) {
      LOGERROR << "root could not attach the shared space" ;
      error = vl::VLE_Unknown ;
      goto done ;
    }
    LOG(2) << "root attached the shared tensor space" ;
  } else {
    error = receive(msg, peers[0].lab) ;
    if (error != vl::VLE_Success || msg.type != Message::init) {
      LOGERROR << "did not receive a message from parent" ;
      error = vl::VLE_Unknown ;
      goto done ;
    }
    session = msg.session ;
    // children attach now
    error = pool.sharedSpace->attach(pool.prefix, pool.lab, pool.numLabs) ;
    if (error != vl::VLE_Success || msg.type != Message::init) {
      LOGERROR << "could not attach shared space" ;
      error = vl::VLE_Unknown ;
      goto done ;
    }
    LOG(2) << "child attached the shared tensor space" ;
  }

  // send message to all children
  for (int p = (pool.lab > 0) ; p < peers.size() ; ++p) {
    msg.type = Message::init ;
    error = send(msg,peers[p].lab) ;
    if (error != vl::VLE_Success) {
      LOGERROR << "could not send a message to a child" ;
      goto done ;
    }
  }

  // receive message from all children
  for (int p = (pool.lab > 0) ; p < peers.size() ; ++p) {
    error = receive(msg,peers[p].lab) ;
    if (error != vl::VLE_Success || msg.type != Message::initDone) {
      error = vl::VLE_Unknown ;
      goto done ;
    }
    // now we can identify the child lab index
    peers[p].lab = msg.from ;
    LOG(2) << "connected lab " << msg.from ;
  }

  // register peer tensors in the same order as peer[]
  for (int p = 0 ; p < peers.size() ; ++p) {
    pool.sharedSpace->attachPeer(peers[p].lab) ;
  }

  // send message to parent (excep for root)
  if (pool.lab > 0) {
    msg.type = Message::initDone ;
    error = send(msg, peers[0].lab) ;
    if (error != vl::VLE_Success) {
      error = vl::VLE_Unknown ;
      goto done ;
    }
    session = msg.session ;
  }

done:
  if (error != vl::VLE_Success) {
    LOGERROR << "handshake failed" ;
  } else {
    LOG(2) << "handshake terminated successfully" ;
  }
  return error ;
}

void ProcessPool::Supervisor::entryPoint()
{
  vl::ErrorCode error = vl::VLE_Success ;

  // Make sure the supervisory thread operates on the same CUDA device
  // as the main thread.
#if ENABLE_GPU
  if (pool.sharedSpace->gpuDevice >= 0) {
    LOG(2) << "setting CUDA device" ;
    cudaError_t cerror = cudaSetDevice(pool.sharedSpace->gpuDevice) ;
    if (cerror != cudaSuccess) {
      LOGERROR
      << "could not switch supervisory thread to CUDA device "
      << pool.sharedSpace->gpuDevice ;
      error = vl::VLE_Cuda ;
    } else {
      LOG(2) << "supervisory thread switched to CUDA device " << pool.sharedSpace->gpuDevice ;
    }
  }
#endif

  if (error == vl::VLE_Success) {
    error = connect() ;
  }

  if (error == vl::VLE_Success) {
    error = handshake() ;
  }

  if (error == vl::VLE_Success) {
    error = loop() ;
  }

  disconnect() ;
}

vl::ErrorCode ProcessPool::Supervisor::handleAccumulateChildren(int tensorIndex)
{
  vl::ErrorCode error = vl::VLE_Success ;
  SharedTensorSpace::SharedTensorInstance & T = pool.sharedSpace->tensors[tensorIndex] ;

  // Search for children ready to be be accumulated.
  for (int p = (pool.lab > 0) ; p < peers.size() && error == vl::VLE_Success ; ++p)
  {
    int peerLab = peers[p].lab ;
    SharedTensorSpace::SharedTensorPeerInstance & PT
    = pool.sharedSpace->getPeerTensor(tensorIndex, peerLab) ;

    bool thisChildReadyForAccumulation =
      PT.transaction == T.transaction &&
      PT.state == SharedTensorSpace::waitParent &&
      PT.accumulated == false ;

    if (thisChildReadyForAccumulation) {

      switch (T.descriptor.deviceType) {

        case vl::VLDT_CPU: {
          switch (T.descriptor.dataType) {
            case vl::VLDT_Float:
              vl::impl::blas<vl::VLDT_CPU,vl::VLDT_Float>::axpy
              (context,
               T.descriptor.shape.getNumElements(),
               1.0f,
               (float*)PT.mappedCpuMemory, 1,
               (float*)T.cpuMemory, 1) ;
              break ;

            case vl::VLDT_Double:
              vl::impl::blas<vl::VLDT_CPU,vl::VLDT_Double>::axpy
              (context,
               T.descriptor.shape.getNumElements(),
               1.0,
               (double*)PT.mappedCpuMemory, 1,
               (double*)T.cpuMemory, 1) ;
              break ;

            default:
              assert(false) ;
              break ;
          }
          break ;
        }

        case vl::VLDT_GPU: {
#if ENABLE_GPU
          cudaError_t cerror ;

          if (T.gpuMemory == NULL) {
            LOGERROR << "internal error: GPU memory not allocated for tensor " << T.name ;
            error = vl::VLE_Unknown ;
            break ;
          }

          // Copy the copy of the tensor update in the host shared memory map
          // to a buffer in the GPU.

          cerror = cudaMemcpyAsync(pool.sharedSpace->gpuDispatchMemory,
                                   PT.mappedCpuMemory,
                                   T.descriptor.getSizeInBytes(),
                                   cudaMemcpyHostToDevice,
                                   pool.sharedSpace->gpuHelperStream) ;
          if (cerror != cudaSuccess) {
            LOGERROR
            << "CUDA generated an error while copying data from host to device: "
            << cudaGetErrorString(cerror) ;
            error = vl::VLE_Cuda ;
            break ;
          }

          // Sum the update to the current tensor vale.

          cudaStream_t previousStream = context.getCudaHelper().getStream() ;
          error = context.getCudaHelper().setStream(pool.sharedSpace->gpuHelperStream) ;
          if (error != vl::VLE_Success) {
            LOGERROR
            << "CUDA generated an error while switching to a different stream:"
            << context.getLastErrorMessage() ;
            break ;
          }

          switch (T.descriptor.dataType) {
            case vl::VLDT_Float:
              error = vl::impl::blas<vl::VLDT_GPU,vl::VLDT_Float>::axpy
              (context,
               T.descriptor.shape.getNumElements(),
               1.0f,
               (float*)pool.sharedSpace->gpuDispatchMemory, 1,
               (float*)T.gpuMemory, 1) ;
              break ;

            case vl::VLDT_Double:
              error = vl::impl::blas<vl::VLDT_GPU,vl::VLDT_Double>::axpy
              (context,
               T.descriptor.shape.getNumElements(),
               1.0,
               (double*)pool.sharedSpace->gpuDispatchMemory, 1,
               (double*)T.gpuMemory, 1) ;
              break ;

            default:
              assert(false) ;
              break ;
          }

          context.getCudaHelper().setStream(previousStream) ;

          if (error != vl::VLE_Success) {
            LOGERROR << "summing tensors:" << context.getLastErrorMessage() ;
          }
#endif
          break ;
        }

        default:
          assert(false) ;
          break ;
      }

      PT.accumulated = true ;
      -- T.numChildrenToAccumulate ;
      LOG(3)
      << "accumulated child " << PT.lab
      << "; " << T.numChildrenToAccumulate << " remaining" ;
    } // next peer
  }

  if (error != vl::VLE_Success) { return error ; }

  // If all children have been accumulated, then
  // notify the parent and switch to waitParent state.
  // Note that we change the PT state too as the peer
  // will switch to that upon receiving the notification.
  //
  // The root is a special case because it
  // does not have a parent, so it can switch
  // directly to the waitChildren state. However, in order
  // to reuse the generic code above, we also set it
  // to waitParent and let the next iteration pick this up.

  if (T.numChildrenToAccumulate == 0) {
    if (T.descriptor.deviceType == vl::VLDT_GPU) {
#if ENABLE_GPU
      cudaError_t cerror ;

      // Copy the GPU tensor to the shared host memory map for other
      // processes to use.
      cerror = cudaMemcpyAsync(T.cpuMemory,
                               T.gpuMemory,
                               T.descriptor.getSizeInBytes(),
                               cudaMemcpyDeviceToHost,
                               pool.sharedSpace->gpuHelperStream) ;
      if (cerror != cudaSuccess) {
        LOGERROR
        << "CUDA error while copying from device to host ("
        << cudaGetErrorString(cerror) << ")" ;
        return vl::VLE_Cuda ;
      }

      // Make this operation synchronous in order
      // to make sure that other processes will properly read the
      // update only when the copy is complete
      cerror = cudaStreamSynchronize(pool.sharedSpace->gpuHelperStream) ;
      if (cerror != cudaSuccess) {
        LOGERROR
        << "CUDA error while synchronizing a stream: '"
        << cudaGetErrorString(cerror) << '\'' ;
        return vl::VLE_Cuda ;
      }
#endif
    }

    T.state = SharedTensorSpace::waitParent ;
    if (pool.lab > 0) {
      int parentLab = peers[0].lab ;
      pool.sharedSpace->getPeerTensor(tensorIndex, parentLab).state = SharedTensorSpace::waitParent ;
      Message msg ;
      msg.type = Message::tensorStateChange ;
      msg.tensorId = tensorIndex ;
      msg.tensorState = T.state ;
      msg.transaction = T.transaction ;
      error = send(msg, parentLab) ;
    }
  }

  return error ;
}

vl::ErrorCode ProcessPool::Supervisor::handleWaitParent(int tensorIndex)
{
  vl::ErrorCode error = vl::VLE_Success ;
  SharedTensorSpace::SharedTensorInstance & T = pool.sharedSpace->tensors[tensorIndex] ;

  // Check if parent finished updating. If so, we can copy its value here
  // and notify the children to copy us by switching to waitParent state and
  // notifying the children. Note that we change the children peer state too
  // as these peers will switch to that upon being notified.

  if (pool.lab > 0) {
    int parentLab = peers[0].lab ;
    SharedTensorSpace::SharedTensorPeerInstance & PT
    = pool.sharedSpace->getPeerTensor(tensorIndex, parentLab) ;
    bool parentDone = (PT.transaction == T.transaction &&
                       PT.state == SharedTensorSpace::waitChildren) ;
    if (!parentDone) {
      return vl::VLE_Success ;
    }

    switch (T.descriptor.deviceType) {
      case vl::VLDT_CPU:
        memcpy(T.cpuMemory, PT.mappedCpuMemory, T.descriptor.getSizeInBytes()) ;
        break ;

      case vl::VLDT_GPU: {
#if ENABLE_GPU
        cudaError_t cerror = cudaMemcpyAsync(T.gpuMemory,
                                             PT.mappedCpuMemory,
                                             T.descriptor.getSizeInBytes(),
                                             cudaMemcpyHostToDevice,
                                             pool.sharedSpace->gpuHelperStream) ;
        if (cerror != cudaSuccess) {
          LOGERROR
          << "propagating parent to children: CUDA generated an error while copying from host to device: '"
          << cudaGetErrorString(cerror) << '\'' ;
          error = vl::VLE_Cuda ;
        }
#endif
        break ;
      }
    }
    if (error != vl::VLE_Success) { return error ; }
  }

  // We have copied data from parent (or there is no parent at all)
  // so we are ready to pass our data to the children and to release
  // the parent from waiting on us.
#if ENABLE_GPU
  if (T.descriptor.deviceType == vl::VLDT_GPU) {
    cudaError_t cerror ;
    if (peers.size() > (pool.lab > 0)) {
      // There are children (i.e. peers other than parent), so copy data to host
      // to deliver it to them.
      cerror = cudaMemcpyAsync(T.cpuMemory,
                               T.gpuMemory,
                               T.descriptor.getSizeInBytes(),
                               cudaMemcpyDeviceToHost,
                               pool.sharedSpace->gpuHelperStream) ;
      if (cerror != cudaSuccess) {
        LOGERROR
          << "CUDA generated an error while copying from device to host: '"
          << cudaGetErrorString(cerror) << '\'' ;
        error = vl::VLE_Cuda ;
      }
    }

    // Synchronize, so it is safe for children on other processes to read
    // the memory. Synchronize even if there are no children, so that inplace
    // reads from this process are safe.
    cerror = cudaStreamSynchronize(pool.sharedSpace->gpuHelperStream) ;
    if (cerror != cudaSuccess) {
      LOGERROR
        << "CUDA gnereated an error while synchronizing a stream: '"
        << cudaGetErrorString(cerror) << '\'' ;
      return vl::VLE_Cuda ;
    }
  }
#endif

  // Notify the parent that we are done copying its data and the children than we are waiting
  // on them to copy our data.
  T.state = SharedTensorSpace::waitChildren ;
  for (int p = 0 ; p < peers.size() ; ++p) {
    int peerLab = peers[p].lab ;
    SharedTensorSpace::SharedTensorPeerInstance & PT
    = pool.sharedSpace->getPeerTensor(tensorIndex, peerLab) ;
    PT.state = (pool.lab > 0 && p == 0) ? SharedTensorSpace::ready : SharedTensorSpace::waitChildren ;
    Message msg ;
    msg.type = Message::tensorStateChange ;
    msg.transaction = T.transaction ;
    msg.tensorId = tensorIndex ;
    msg.tensorState = (pool.lab > 0 && p == 0) ? SharedTensorSpace::ready : SharedTensorSpace::waitChildren ;
    error = send(msg, peerLab) ;
  }

  return error ;
}

vl::ErrorCode ProcessPool::Supervisor::handleWaitChildren(int tensorIndex)
{
  vl::ErrorCode error = vl::VLE_Success ;
  SharedTensorSpace::SharedTensorInstance & T = pool.sharedSpace->tensors[tensorIndex] ;

  // Check if all children finished updating. If so, we can switch
  // to ready state and notify the parent.
  // Note that we change the peer children state too
  // as these peers will switch to that upon being notified.

  bool allChildrenDone = true ;
  for (int p = (pool.lab > 0) ; p < peers.size() ; ++p) {
    int peerLab = peers[p].lab ;
    SharedTensorSpace::SharedTensorPeerInstance & PT
    = pool.sharedSpace->getPeerTensor(tensorIndex, peerLab) ;
    bool thisChildDone =((PT.transaction == T.transaction &&
                          PT.state == SharedTensorSpace::ready) ||
                          PT.transaction > T.transaction) ;
    allChildrenDone &= thisChildDone ;
  }
  if (allChildrenDone) {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    T.state = SharedTensorSpace::ready ;
    waitingList.notify_all() ;
  }

  return error ;
}

vl::ErrorCode ProcessPool::Supervisor::loop()
{
  vl::ErrorCode error = vl::VLE_Success ;

  LOG(2) << "loop begins" ;

  // Advertise. Note that we do not lock extensively in the main
  // loop. Syncrhonization with the main thread is kept efficient
  // using lock-free mechanisms.
  {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    state = running ;
    waitingList.notify_all() ;
  }

  int pollStatus = 0 ;
  size_t const pollInterval = 499UL ; // allow heartbeats (ms)
  size_t const heartbeatInterval = 500UL * 1000UL * 1000UL ; // (ns)
  size_t lastHeartbeat = vl::getTime() ;

  struct pollfd * polls = new struct pollfd [peers.size() + 1] ;
  for (int p = 0 ; p < peers.size() ; ++p) {
    polls[p].fd = peers[p].socketFD ;
    polls[p].events = POLLIN | POLLHUP | POLLERR | POLLNVAL ;
  }
  polls[peers.size()].fd = pipeFD[0] ;
  polls[peers.size()].events = POLLIN ;

  while (error == vl::VLE_Success && forceQuit == false)
  {
    // Generate regular heartbeats to wake up the main thread at
    // regular interval and allow it to time out on
    // user commands usch as pull() and push().
    size_t now = vl::getTime() ;
    if (now > lastHeartbeat + heartbeatInterval) {
      waitingList.notify_all() ; // no need to lock
      lastHeartbeat = now ;
    }

    // Wait for incoming messages or a timeout.
    pollStatus = poll(polls, peers.size() + 1, pollInterval) ;
    if (pollStatus < 0) {
      error = vl::VLE_Unknown ;
      continue ;
    }

    // Timeout!
    if (pollStatus == 0) {
      LOG(1) << "Polling timed out on lab " << pool.sharedSpace->lab ;
      // pool.sharedSpace->dump() ;
    }

    // Check for messages piped from the main thread.
    if (polls[peers.size()].revents & POLLIN) {
      LOG(3) << "supervisory thread notified by the main thread" ;
      char dummy ;
      read(pipeFD[0], &dummy, 1) ;
    }

    // Check for messages from other processes.
    for (int p = 0 ; p < peers.size() && error == vl::VLE_Success ; ++ p)
    {
      // Check for communication errors.
      if (polls[p].revents & (POLLHUP | POLLERR | POLLNVAL)) {
        LOG(3) << "one of the sockets generated an error, quitting" ;
        error = vl::VLE_Unknown ;
        break ;
      }

      // Skip this peer if there is no incoming data.
      if ((polls[p].revents & POLLIN) == 0) continue ;

      // Receive the message.
      Message msg ;
      error = receive(msg, peers[p].lab) ;
      if (error != vl::VLE_Success) {
        LOGERROR << "error while receiving a message from lab " << peers[p].lab ;
        break ;
      }

      // Process the message.
      switch (msg.type) {
        case Message::tensorStateChange: {
          // Record the new state for later.
          LOG(3)
          << "received tensor state change from lab " << msg.from
          << " for tensor " << pool.sharedSpace->tensors[msg.tensorId].name.c_str()
          << " to state " << msg.tensorState
          << " for transaction " << msg.transaction ;
          SharedTensorSpace::SharedTensorPeerInstance & T
          = pool.sharedSpace->getPeerTensor(msg.tensorId, msg.from) ;
          T.state = msg.tensorState ;
          T.transaction = msg.transaction ;
          break ;
        }

        case Message::requestShutdown: {
          peers_t::iterator P = std::find(peers.begin(), peers.end(), msg.from) ;
          P->shutdownRequested = true ;
          break ;
        }

        case Message::tensorFinalTransaction: {
          peers_t::iterator P = std::find(peers.begin(), peers.end(), msg.from) ;
          SharedTensorSpace::SharedTensorInstance & T = pool.sharedSpace->tensors[msg.tensorId];
          LOG(3)
          << "received final transaction from lab " << msg.from
          << " for tensor " << T.name.c_str()
          << " to transaction " << msg.finalTransaction ;
          int sourcePeer = msg.from ;
          if (msg.finalTransaction <  T.finalTransaction) {
            T.finalTransaction = msg.finalTransaction ;
            for (int q = 0 ; q < peers.size() ; ++q) {
              if (sourcePeer == peers[q].lab) continue ;
              error = send(msg, peers[q].lab) ;
              if (error != vl::VLE_Success) {
                LOGERROR
                  << "error while sending a message to lab "
                  << peers[p].lab ;
                break ;
              }
            }
          }
          break ;
        }

        default:
          // Unexpected message.
          LOGERROR << "received an unexpected message" ;
          error = vl::VLE_Unknown ;
          break ;
      }
    }

    // Check all tensors for actions. Keep updating each tensor until its
    // state does not change anymore.
    for (int tensorIndex = 0 ; tensorIndex < pool.sharedSpace->tensors.size() && error == vl::VLE_Success ; ++tensorIndex)
    {
      SharedTensorSpace::SharedTensorState currentState ;
      SharedTensorSpace::SharedTensorInstance & T = pool.sharedSpace->tensors[tensorIndex] ;
      do {

        currentState = T.state ;
        LOG(3) << "visiting tensor " << T.name << " in state " << T.state ;

        // Detect interruptions
        if (T.transaction > T.finalTransaction) {
          LOG(1) << "detected interrupded transaction for tensor " << T.name <<
            " (transaction:"<<T.transaction<<" > final_transaction:"<<T.finalTransaction<<")";
          error = vl::VLE_Interrupted ;
          continue ;
        }

        switch (T.state) {

          case SharedTensorSpace::ready:
            break ;

          case SharedTensorSpace::accumulateChildren:
            error = handleAccumulateChildren(tensorIndex) ;
            break ;

          case SharedTensorSpace::waitParent :
            error = handleWaitParent(tensorIndex) ;
            break ;

          case SharedTensorSpace::waitChildren :
            error = handleWaitChildren(tensorIndex) ;
            break ;
        }
      } while (T.state != currentState && error == vl::VLE_Success) ;
    }

    // Upon shutting down, propagate a message to let other nodes know that
    // no further transaction can be processed for each tensor.

    if (shutdownRequested && (state == running) && (error == vl::VLE_Success)) {
      LOG(3) << "sending final transaction for all tensors" ;
      for (int i = 0 ; i < pool.sharedSpace->tensors.size() ; ++i) {
        SharedTensorSpace::SharedTensorInstance & tensor = pool.sharedSpace->tensors[i] ;
        if (tensor.finalTransaction > tensor.transaction) {
          tensor.finalTransaction = tensor.transaction ;
          Message msg ;
          msg.type = Message::tensorFinalTransaction ;
          msg.tensorId = i ;
          msg.finalTransaction = tensor.finalTransaction ;
          for (int p = 0 ; p < peers.size() ; ++p) {
            error = send(msg, peers[p].lab) ;
            if (error != vl::VLE_Success) {
              LOGERROR
                << "error while sending a message to lab "
                << peers[p].lab ;
              break ;
            }
          }
        }
      }
    }

    // Check for other actions.
    if (shutdownRequested && (state == running) && (error == vl::VLE_Success)) {
      // Check if the children are also in shutdown mode
      bool allDone = true ;
      for (int p = (pool.lab > 0) ; p < peers.size() ; ++p) {
        allDone &= peers[p].shutdownRequested ;
      }
      if (allDone) {
        state = Supervisor::shuttingDown ; // avoid sending the same message again later
        if (pool.lab > 0) {
          LOG(2) << "subtree ready to shutdown, telling parent lab" ;
          Message msg ;
          msg.type = Message::requestShutdown ;
          error = send(msg, peers[0].lab) ;
        } else {
          // Other processes will stop when connections are broken.
          LOG(2) << "everyone requested shutdown, root lab quitting" ;
          break ; // out of poll loop
        }
      }
    }

  } // back to poll

  LOG(2) << "terminating supervisory thread loop (error = " << error << ')' ;
  delete [] polls ;
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

#pragma mark -

ProcessPool processPool ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */

void atExit()
{
  processPool.finalize() ;
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  enum Commands { init, stats, reset, push, pull } command ;
  bool inplace = false ;
  std::string tensorName ;
  std::string prefix = "mcn" ;
  mxArray const * arg ;
  vl::ErrorCode error = vl::VLE_Success ;
  size_t labIndex = 0 ;
  size_t numLabs = 0 ;

  verbosity = 0 ;

  mexAtExit(atExit) ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  if (nin < 1) {
    vlmxError(VLMXE_IllegalArgument, "Not enough input arguments.") ;
  }

  if (!vlmxIsString(in[0], -1)) {
    vlmxError(VLMXE_IllegalArgument, "COMMAND is not a string.") ;
  }

  if (vlmxCompareToStringI(in[0],"init") == 0) {
    command = init ;
    if (nin < 4) {
      vlmxError(VLMXE_IllegalArgument, "Less than three arguments passed to INIT.") ;
    }
    arg = in[1] ;
    if (!vlmxIsPlainScalar(in[2])) {
      vlmxError(VLMXE_IllegalArgument, "LABINDEX is not a plain scalar.") ;
    }
    labIndex = mxGetScalar(in[2]) ;
    if (labIndex < 1) {
      vlmxError(VLMXE_IllegalArgument, "LABINDEX must be an integer greater than 0.") ;
    }
    if (!vlmxIsPlainScalar(in[3])) {
      vlmxError(VLMXE_IllegalArgument, "NUMLABS is not a plain scalar.") ;
    }
    numLabs = mxGetScalar(in[3]) ;
    if (numLabs < labIndex) {
      vlmxError(VLMXE_IllegalArgument, "NUMLABS must be an integer greater or equal to LABINDEX.") ;
    }
    next = 4 ;
  } else if (vlmxCompareToStringI(in[0], "stats") == 0)  {
    command = stats ;
    next = 1 ;
  } else if (vlmxCompareToStringI(in[0], "reset") == 0)  {
    command = reset ;
    next = 1 ;
  } else if (vlmxCompareToStringI(in[0], "push") == 0) {
    if (nin < 3) {
      vlmxError(VLMXE_IllegalArgument, "Less than three arguments passed to PUSH.") ;
    }
    command = push ;
    VLMXErrorCode error = vlmxParseString(tensorName, in[1]) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "NAME is not a string.") ;
    }
    arg = in[2] ;
    next = 3 ;
  } else if (vlmxCompareToStringI(in[0], "pull") == 0) {
    if (nin < 2) {
      mexErrMsgTxt("Less than two arguments passed to PULL.") ;
    }
    command = pull ;
    VLMXErrorCode error = vlmxParseString(tensorName, in[1]) ;
    if (error != VLMXE_Success) {
      vlmxError(error, "NAME is not a string.") ;
    }
    next = 2 ;
  }
  else {
    vlmxError(VLMXE_IllegalArgument, "Unknown COMMAND.") ;
  }

  // optional arguments
  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {

      case opt_prefix : {
        if (!vlmxIsString(optarg, -1)) {
          vlmxError(VLMXE_IllegalArgument, "PREFIX is not a string.") ;
        }
        char str [512] ;
        mxGetString (optarg, str, sizeof(str)/sizeof(str[0])) ;
        prefix = str ;
        break ;
      }

      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_inplace :
        inplace = true ;
        break ;
    }
  }

  switch (command) {
    case init:
    {
      (verbosity >= 2) && mexPrintf("vl_tmove: command 'init'\n") ;

      // Initialize shared space. mexInit() may thorow a MEX error;
      // the auto_ptr should avoid a leak in this case.
      std::auto_ptr<SharedTensorSpace> sharedSpace(new SharedTensorSpace()) ;
      sharedSpace->mexInit(arg) ;

      // Initialize the pool, including attaching the shared space.
      // Now the shared space is owned by the process pool.
      error = processPool.init(prefix, labIndex - 1, numLabs, sharedSpace.release()) ;
      if (error != vl::VLE_Success) {
        mexErrMsgTxt("Could not initialize connections to other MATLAB labs.") ;
      }

      // At this point, sharedSpace is handled by the ProcessPool thread,
      // so we interact with it indirectly
      break ;
    }

    case stats :
      (verbosity >= 2) && mexPrintf("vl_tmove: command 'stats'\n") ;
      processPool.mexPrint() ;
      break ;

    case push :
      (verbosity >= 2) && mexPrintf("vl_tmove: command 'push' on tensor '%s'%s\n", tensorName.c_str(), inplace?" (inplace)":"") ;
      processPool.mexPush(tensorName, arg, inplace) ;
      break ;

    case pull :
      (verbosity >= 2) && mexPrintf("vl_tmove: command 'pull' on tensor '%s'%s\n", tensorName.c_str(),
                                    inplace?" (inplace)":"") ;
      out[0] = processPool.mexPull(tensorName, inplace) ;
      break ;

    case reset :
      (verbosity >= 2) && mexPrintf("vl_tmove: command 'reset'\n") ;
      processPool.shutdown() ; // gracefully (wait for others to finish)
      processPool.finalize() ; // no matter what
      break ;
  }
}


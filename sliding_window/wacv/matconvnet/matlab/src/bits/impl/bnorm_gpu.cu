// @file   bnorm_gpu.cu
// @brief  Batch normalization implementation (GPU)
// @author Sebastien Ehrhardt
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Sebastien Ehrhardt and Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bnorm.hpp"
#include "../datacu.hpp"
#include "blashelper.hpp"
#include "sharedmem.cuh"
#include <assert.h>
#include <float.h>
#include <stdint.h>

// MSB_WARP = log2(WARP_SIZE)
#define WARP_SIZE 32
#define MSB_WARP 5

// macro function
#define min(a,b) (a > b ? b : a);

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                                         Helpers	*/
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */

static inline int getBlockSize(int dataSize)
{
  int blockSize = VL_CUDA_NUM_THREADS / 2 ;
  if (dataSize < blockSize) {
    unsigned int numWarps = dataSize / WARP_SIZE ;
    if (numWarps < 4) {
      blockSize = 2 * WARP_SIZE ;
    }
    else if (numWarps < 8) {
      blockSize = 4 * WARP_SIZE ;
    }
    else {
      blockSize = 8 * WARP_SIZE ;
    }
  }
  return blockSize ;
}

// get the smallest x which is a multiple of factor
static inline int nextMultipleOf(int x, int factor)
{
  return factor * ((x + factor - 1)/factor) ;
}

/*
 # Reduction over the whole batch

 `bnorm` works by accumulating statistics over planes (channels) and
 images in a batch. It then uses these statistics to renormalize the values.

 Summing over plens efficiently over planes is a little complex on the GPU.
 What we have are threads, block of threads, and a grid of blocks:

 * Warps (up to 32 threads). Highly coupled, and in fact *coalesced* and 
 run essentially in a single stream of vector instructions on the GPU, 
 which also means that they stay syncrhonized implicitly.

 * Blocks (up to 512 threads). Blocks are assigned to a SM, and the SM
 breaks them down into warps for execution. Threads in the same block
 can be synchronised explicity using __syncthreads(). They all run
 concurrently in the same SM.

 * Grid. A grid is an array of blocks that are scheduled onto multiple SMs. 
 Threads in a grid can only  be synchronised implicitly at the end of a kernel.

 Given these constraints, we explain next how operations are mapped to the
 blocks and the threads.

 The input data is organised in SIZE images, each of which is composed of 
 DEPTH planes. The goal is to compute the mean and std deviation of each
 plane (across images). In the follwing diagram, planes are enumerated 
 from left to right and top to bottom, first listing all the planes for 
 one image (a row) and then subsequent images (in different rows).

      +-------+   +-------+   +-------+   +-------+
      |plane 1|   |p 2    |   |p 3    |   |p 4    |  numPlanes = 12
      |ch 1   |   |c 2    |   |c 3    |   |c 4    |  depth = 4
      |image 1|   |i 1    |   |i 1    |   |i 1    |  planeArea = 28
  +---+block 1|   |b 2    |   |b 3    |   |b 4    |  planeStride = gridSize = 8
  |   +-------+   +-------+   +-------+   +-------+
  |
  |   +-------+   +-------+   +-------+   +-------+
  |   |p 5    |   |p 6    |   |p 7    |   |p 8    |
  |   |c 1    |   |c 2    |   |c 3    |   |c 4    |
  |   |i 2    |   |i 2    |   |i 2    |   |i 2    |
  |   |b 5    |   |b 6    |   |b 7    |   |b 8    |
  |   +-------+   +-------+   +-------+   +-------+
  |
  |   +-------+   +-------+   +-------+   +-------+
  |   |p 9    |   |p 10   |   |p 11   |   |p 12   |
  |   |c 1    |   |c 2    |   |c 3    |   |c 4    |
  |   |i 3    |   |i 3    |   |i 3    |   |i 3    |
  +-->+b 1    |   |b 2    |   |b 3    |   |b 4    |
      +-------+   +-------+   +-------+   +-------+

 We create a certain number of thread blocks. Call this number gridSize.
 Each block operates (sums) over a certain number of planes, with
 subsequent blocks taking over subsequent planes.

 Since there may be less blocks than planes overall, a single block 
 does more than one plane in general but skips over the ones that are 
 already processed by neighbour blocks. In the example, the thread block 1
 integrates plane 1 and plane 9).

 It is important to optimise how blocks access memory. This is organised
 in three phases:

 1. Blocks accumulate in a shared scratch space (of blockSize elements, 
 for each block) partial sums. In this manner, the scratch space of each block
 contains the statistics for a particular plane (feature channels) and subset 
 of the images.

 2. Blocks reduce the data in their scratch space using within-block reduction.

 3. This is still a partial result as blocks do not do in general all the images.
 A last pass accumulates the outputs from the individual blocks.

 # Sliding-window accumulation

 As blocks accumulate over different planes and images and these are not 
 necessarily aligned to nice memory boundaries, the problem is how to make 
 computations efficient.

 The trick is to look at the block as a jumping window, sliding over the memory
 that needs to be summed, but always aligned at good block boundaries. This means
 that occasionally threads in a block will access some data that needs to be discarded.

 +-------+ +-------+           +-------+ +-------+      aligned blocks (with two warps each)
 |   :   | |   :   |           |   :   | |   :   |      covering the data
 +-------+ +-------+           +-------+ +-------+
 +-------------+             +-------------+            data to sum

 +-------------------------------------------------------->
 increasing memory addresses

 As each block slides over the data, it accumulates partial results
 in a scratch buffer which has a number of elememts equal to the block size.
 Evenetually, block-level reduction is performed on this scratch buffer
 to get the total.

 # Per-block reduction

 Use a block of blockSize threads to accumulate all the values in the
 shared array mdata[], which has blockSize elements:

 mdata[0] <- mdata[0] + mdata[1] + ... + mdata[blockSize-1]

 blockSize is a power of two and less than the maxmimum allowed block
 size (usually 512). mdata[] has to be padded with zeros to allow
 summation over vectors whose dimension is less than blockSize.

 This is done as follows:

 1. First, the first half of the threads in the block accumulate
 the second half of mdata in the first half:

 tid=0:             mdata[0] = mdata[0] + mdata[blockSize/2]
 ...
 tid=blockSize/2-1: mdata[blockSize/2-1] = mdata[blockSize/2-1] + mdata[blockSize-1]

 Note that half of the threads are idle

 2. Then, the first quarter of the threads reduce the result further:

 tid=0:             mdata[0] = mdata[0] + mdata[blockSize/4]
 ...
 tid=blockSize/4-1: mdata[blockSize/4-1] = mdata[blockSize/4-1] + mdata[blockSize/2-1]

 3. This continues until only tid=0 operates:

 tid=0:             mdata[0] = mdata[0] + mdata[1]

 This is further divded into two regimes. In the first regime, tid
 may span threads in the same block but different warps. Here
 the code must be explicitly snychronized.

 In the second regime, tid < WARP_SIZE, and synchronization is not
 required as threads are coalesced.
 */

template<typename T>
__forceinline__ __device__ void blockReduce(volatile T * mdata,
                                            unsigned int tid,
                                            unsigned int blockSize,
                                            unsigned int maxDataSize)
{
  // todo: get rid of maxDataSize?
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >=512) { if (tid < 512) { mdata[tid] += mdata[tid + 512]; } __syncthreads(); } // mdata[0:511] = mdata[0:511] + mdata[512:1023]
  if (blockSize >= 512  && maxDataSize + WARP_SIZE >=256) { if (tid < 256) { mdata[tid] += mdata[tid + 256]; } __syncthreads(); } // mdata[0:255] = mdata[0:255] + mdata[256:511]
  if (blockSize >= 256  && maxDataSize + WARP_SIZE >=128) { if (tid < 128) { mdata[tid] += mdata[tid + 128]; } __syncthreads(); } // mdata[0:127] = mdata[0:127] + mdata[128:255]
  if (blockSize >= 128  && maxDataSize + WARP_SIZE >=64 ) { if (tid <  64) { mdata[tid] += mdata[tid + 64];  } __syncthreads(); } // mdata[0:63]  = mdata[0:63]  + mdata[64:127]
  if (tid < 32) {
    // now enter warp
    if (blockSize >=  64) { mdata[tid] += mdata[tid + 32]; } // mdata[0:31] = mdata[0:31] + mdata[32:63]
    if (blockSize >=  32) { mdata[tid] += mdata[tid + 16]; } // mdata[0:15] = mdata[0:15] + mdata[16:31]
    if (blockSize >=  16) { mdata[tid] += mdata[tid +  8]; } // mdata[0:7]  = mdata[0:7]  + mdata[7:15]
    if (blockSize >=   8) { mdata[tid] += mdata[tid +  4]; } // mdata[0:3]  = mdata[0:3]  + mdata[4:7]
    if (blockSize >=   4) { mdata[tid] += mdata[tid +  2]; } // mdata[0:1]  = mdata[0:1]  + mdata[2:3]
    if (blockSize >=   2) { mdata[tid] += mdata[tid +  1]; } // mdata[0]    = mdata[0]    + mdata[1]
  }
}

template<typename T>
__forceinline__ __device__ void blockReduce2(volatile T * mdata,
                                             volatile T * sdata,
                                             unsigned int tid,
                                             unsigned int blockSize,
                                             unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >=512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; } __syncthreads(); }
  if (blockSize >= 512  && maxDataSize + WARP_SIZE >=256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256  && maxDataSize + WARP_SIZE >=128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128  && maxDataSize + WARP_SIZE >=64)  { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  } __syncthreads(); }
  if (tid < 32) {
    if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; }
    if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; }
    if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; }
    if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; }
    if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; }
    if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; }
  }
}

template<typename T>
__forceinline__ __device__ void blockReduce4(volatile T * sdata,
                                             volatile T * mdata,
                                             volatile T * rdata,
                                             volatile T * tdata,
                                             unsigned int tid,
                                             unsigned int blockSize,
                                             unsigned int maxDataSize)
{
  __syncthreads();
  if (blockSize >= 1024 && maxDataSize + WARP_SIZE >= 512) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; mdata[tid] += mdata[tid + 512]; rdata[tid] += rdata[tid + 512]; tdata[tid] += tdata[tid + 512];} __syncthreads(); }
  if (blockSize >= 512 && maxDataSize + WARP_SIZE >= 256) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; mdata[tid] += mdata[tid + 256]; rdata[tid] += rdata[tid + 256]; tdata[tid] += tdata[tid + 256];} __syncthreads(); }
  if (blockSize >= 256 && maxDataSize + WARP_SIZE >= 128) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; mdata[tid] += mdata[tid + 128]; rdata[tid] += rdata[tid + 128]; tdata[tid] += tdata[tid + 128];} __syncthreads(); }
  if (blockSize >= 128 && maxDataSize + WARP_SIZE >= 64) { if (tid <  64) { sdata[tid] += sdata[tid + 64];  mdata[tid] += mdata[tid + 64];  rdata[tid] += rdata[tid + 64]; tdata[tid] += tdata[tid + 64];} __syncthreads(); }
  if (tid < 32) {
    if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; mdata[tid] += mdata[tid + 32]; rdata[tid] += rdata[tid + 32]; tdata[tid] += tdata[tid + 32];}
    if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; mdata[tid] += mdata[tid + 16]; rdata[tid] += rdata[tid + 16]; tdata[tid] += tdata[tid + 16];}
    if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; mdata[tid] += mdata[tid +  8]; rdata[tid] += rdata[tid +  8]; tdata[tid] += tdata[tid +  8];}
    if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; mdata[tid] += mdata[tid +  4]; rdata[tid] += rdata[tid +  4]; tdata[tid] += tdata[tid +  4];}
    if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; mdata[tid] += mdata[tid +  2]; rdata[tid] += rdata[tid +  2]; tdata[tid] += tdata[tid +  2];}
    if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; mdata[tid] += mdata[tid +  1]; rdata[tid] += rdata[tid +  1]; tdata[tid] += tdata[tid +  1];}
  }
}

// Get largest memory address that is aligned to a warp worth of T
// and smaller than x.

template<typename T>
__forceinline__ __device__ uintptr_t getBlockBeginning(void const * x)
{
  return (uintptr_t)(x) & (~((uintptr_t)(WARP_SIZE*sizeof(T)) - 1)) ;
}

// Use the current block of thread to sum over a given column of a matrix. The selected
// column is given by the thread block index in the block grid.
//
// This function uses an amoutn of scratch memory equal to blockSize*sizeof(T)
// where blockSize=blockDim.x.

template<typename T>
__forceinline__ __device__ T matrixSumHelper(T const * matrix, int numRows)
{
  // One thread block per column to sum
  // Shared memory is per-block, it holds blockSize intermediate reults
  //extern __shared__ T scratch [] ;
  SharedMemory<T> smem ;
  T * scratch = smem.getPointer() ;
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  int blockSize = blockDim.x ;

  // Note that scratch is different for different blocks, hence
  // matrix columns. Now fill scratch with partial sums, in a sliding-window
  // manner.
  scratch[tid] = 0 ;
  T const * columnBegin = matrix + column * numRows ;
  T const * columnEnd = columnBegin + numRows ;
  T const * block = (T const*) getBlockBeginning<T>(columnBegin) + tid ;
  while (block < columnEnd) {
    if (block >= columnBegin) {
      scratch[tid] += *block ;
    }
    block += blockSize ;
  }

  // Now scratch[] has blockSize partial sums for this column
  // Finish by reducing and saving
  blockReduce<T>(scratch, tid, blockSize, numRows) ;

  return scratch[0] ;
}

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                                  compute_moments	*/
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */

// This kernel accumulates means and variances for the data.
// Each block of thread sums over one or more data planes, resulting
// in an array accumulator[] of dimension numChunks x 2*numChannels.
//
// If each thread block scans all the images, then numChunks = 1.
// However, for efficiency different thread blocks do different
// subset of images, resulting in numChunks partial results to be summed
// later by a second kernel.
//
// The first part accumulator[:,0:numChannels-1] stores the data for the mean
// and the second part accumulator[:,numChannels,2*numChannels-1] the data
// for the sigmas.
//
// This function uses the sliding-window summing technique described
// above. It requires
//
//    2*sizeof(T)*blockSize
//
// bytes of shared scratch memory to hold to hold partial sums for
// means and sigmas.

template<typename T>
__global__ void accumulate_moments_partial(T * accumulator,
                                           T const * data,
                                           int planeArea,
                                           int numPlanes,
                                           int numChannels,
                                           int numChunks)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  //extern __shared__ T s [] ;
  SharedMemory<T> smem ;
  T * s = smem.getPointer() ;
  T * mdata = s ;
  T * sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        T x = *block ;
        mdata[tid] += x ;
        sdata[tid] += x * x ;
      }
      block += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce2<T>(sdata, mdata, tid, blockSize, planeArea) ;

  if (tid == 0) {
    int chunk = blockIdx.x / numChannels ;
    int i = chunk + channel * numChunks ;
    accumulator[i] = mdata[0];
    accumulator[i + gridDim.x] = sdata[0];
  }
}

// This kernel sums over the accumulator computed by the function
// above to obtain the moments.
//
// This kernel uses matrixSumHelper() defined above. Hence:
//
// 1. The block grid must be set to have a block
//    for each column of accumulator[]. There are here 2*numChannels columns.
//
// 2. There can be any (reasonable) blockSize. Blocks will iterate
//    over rows as needed to compte the operation.
//
// 3. It must be called with `blockSize*sizeof(T)` shared
//    scratch space.

template<typename T>
__global__ void accumulate_moments_finish(T * moments,
                                          T const * accumulator,
                                          int numRows)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  T x = matrixSumHelper(accumulator, numRows) ;
  if (tid == 0) {
    moments[column] = x ;
  }
}

// After accumulation, we need to renormalize the moments.
//
// 1. It shoudl be called with enough threads to cover all
//    numChannels in the moments.
//
// 2. The actual number of blocks is determined based on the block
//    size to satisfy condition (2).

template<typename T>
__global__ void normalize_moments(T * moments,
                                  unsigned int numChannels,
                                  T mass,
                                  T epsilon)
{
  int unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
  if (i < numChannels){
    // max(0, __) is for numerical issues
    T mean = moments[i] / mass ;
    T sigma2 = max((T).0, moments[i + numChannels]/mass - mean*mean) ;
    moments[i] = mean ;
    moments[i + numChannels] = sqrt(sigma2 + epsilon);
  }
}

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                                     compute_ders */
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */

// Same as accumulate_moments above. Call with:
//
// 1. 2*sizeof(T)*blockSize scratch space
// 2.
//
// bytes of shared scratch memory to hold to hold partial sums for
// means and sigmas.
//
// Below, either accumulator is not NULL and derMultipliers, derBiases,
// and moments are, or the function is run in a `final' mode,
// with accumulator set to NULL, and the other points set to their
// `final' destination.

template<typename T>
__global__ void accumulate_ders_partial
(T * accumulator,
 T * derMultipliers,
 T * derBiases,
 T const * data,
 T const * derOutput,
 int planeArea,
 int numPlanes,
 int numChannels,
 int numChunks)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;
  //extern __shared__ T s[] ;
  SharedMemory<T> smem ;
  T * s = smem.getPointer() ;

  T * mdata = s ;
  T * sdata = mdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    T const * dblock = derOutput + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        T x = *block ;
        T dy = *dblock ;
        mdata[tid] += x * dy ;
        sdata[tid] += dy ;
      }
      block += blockSize ;
      dblock += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce2<T>(sdata, mdata, tid, blockSize, planeArea);

  if (tid == 0) {
    if (numChannels == gridDim.x) {
      // Final output ready
      derMultipliers[blockIdx.x] = mdata[0];
      derBiases[blockIdx.x] = sdata[0];
    } else {
      // Partially accumulated outut
      int chunk = blockIdx.x / numChannels ;
      int i = chunk + channel * numChunks ;
      accumulator[i] = mdata[0]; // derMultipliers
      accumulator[i + gridDim.x] = sdata[0]; // derBiases
    }
  }
}

template<typename T>
__global__ void accumulate_ders_finish(T * derMultipliers,
                                       T * derBiases,
                                       T const * accumulator,
                                       int numChunks,
                                       int numChannels)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  T x = matrixSumHelper(accumulator, numChunks) ;
  if (tid == 0) {
    // Recall that the matrix stores in order [derMultipliers derBiases means sigmas]
    // containing four types of data
    int type = column / numChannels ;
    int channel = column % numChannels ;

    if (type == 0) {
      derMultipliers[channel] = x ;
    }
    else {
      derBiases[channel] = x ;
    }
  }
}

template<typename T>
__global__ void normalize_ders(T * derMultipliers,
                               T const * derBiases,
                               T const * moments,
                               unsigned int numChannels,
                               T mass,
                               T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < numChannels){
    T mean = moments[idx] ;
    T sigma = moments[idx + numChannels] ;
    derMultipliers[idx] = (derMultipliers[idx] - mean*derBiases[idx]) / sigma ;
  }
}

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                         compute_ders_and_moments	*/
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */

// Same as accumulate_moments above. Call with:
//
// 1. 4*sizeof(T)*blockSize scratch space
// 2.
//
// bytes of shared scratch memory to hold to hold partial sums for
// means and sigmas.
//
// Below, either accumulator is not NULL and derMultipliers, derBiases,
// and moments are, or the function is run in a `final' mode,
// with accumulator set to NULL, and the other points set to their
// `final' destination.

template<typename T>
__global__ void accumulate_ders_and_moments_partial
(T * accumulator,
 T * derMultipliers,
 T * derBiases,
 T * moments,
 T const * data,
 T const * derOutput,
 int planeArea,
 int numPlanes,
 int numChannels,
 int numChunks)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;
  //extern __shared__ T s[] ;
  SharedMemory<T> smem ;
  T * s = smem.getPointer() ;

  T * mdata = s ;
  T * sdata = mdata + blockSize ;
  T * rdata = sdata + blockSize ;
  T * tdata = rdata + blockSize ;

  mdata[tid] = 0 ;
  sdata[tid] = 0 ;
  rdata[tid] = 0 ;
  tdata[tid] = 0 ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    T const * dblock = derOutput + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        T x = *block ;
        T dy = *dblock ;
        mdata[tid] += x * dy ;
        sdata[tid] += dy ;
        rdata[tid] += x * x ;
        tdata[tid] += x ;
      }
      block += blockSize ;
      dblock += blockSize ;
    }
    plane += planeStride ;
  }

  blockReduce4<T>(sdata, mdata, rdata, tdata, tid, blockSize, planeArea);

  if (tid == 0) {
    if (numChannels == gridDim.x) {
      // Final output ready
      derMultipliers[blockIdx.x] = mdata[0];
      derBiases[blockIdx.x] = sdata[0];
      moments[blockIdx.x] = tdata[0];
      moments[blockIdx.x+numChannels] = rdata[0];
    } else {
      // Partially accumulated outut
      int chunk = blockIdx.x / numChannels ;
      int i = chunk + channel * numChunks ;
      accumulator[i] = mdata[0]; // derMultipliers
      accumulator[i + gridDim.x] = sdata[0]; // derBiases
      accumulator[i + 2*gridDim.x] = tdata[0]; // means
      accumulator[i + 3*gridDim.x] = rdata[0]; // sigmas
    }
  }
}

template<typename T>
__global__ void accumulate_ders_and_moments_finish(T * derMultipliers,
                                                   T * derBiases,
                                                   T * moments,
                                                   T const * accumulator,
                                                   int numChunks,
                                                   int numChannels)
{
  int tid = threadIdx.x ;
  int column = blockIdx.x ;
  T x = matrixSumHelper(accumulator, numChunks) ;
  if (tid == 0) {
    // Recall that the matrix stores in order [derMultipliers derBiases means sigmas]
    // containing four types of data
    int type = column / numChannels ;
    int channel = column % numChannels ;

    if (type == 0) {
      derMultipliers[channel] = x ;
    }
    else if (type == 1) {
      derBiases[channel] = x ;
    }
    else if (type == 2) {
      moments[channel] = x ;
    }
    else {
      moments[channel + numChannels] = x ;
    }
  }
}

template<typename T>
__global__ void normalize_ders_and_moments(T * derMultipliers,
                                           T * derBiases,
                                           T * moments,
                                           unsigned int numChannels,
                                           T mass,
                                           T epsilon)
{
  unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
  if (idx < numChannels){
    T mean = moments[idx] / mass;
    T sigma2 = max((T).0, moments[idx + numChannels]/mass - mean*mean) ;
    T sigma = sqrt(sigma2 + epsilon);
    moments[idx] = mean ;
    moments[idx + numChannels] = sigma ;
    derMultipliers[idx] = (derMultipliers[idx]-mean*derBiases[idx]) / sigma ;
  }
}

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                             forward and backward */
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */

// Call this kernel like compute_moments, but it does not need a scratch sapce

template<typename T>
__global__ void batch_normalize_forward(T * outputData,
                                        T const * moments,
                                        T const * data,
                                        T const * multipliers,
                                        T const * biases,
                                        int planeArea,
                                        int numPlanes,
                                        int numChannels)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  // Not optimized for compute capability < 1.2
  T mean = moments[channel];
  T sigma = moments[channel+numChannels];
  T multiplier = multipliers[channel];
  T bias = biases[channel];
  T coefficient = multiplier / sigma ;

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T>(planeBegin) + tid ;
    T * oblock = outputData + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        *oblock = coefficient * (*block - mean) + bias ;
      }
      block += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}

template<typename T>
__global__ void batch_normalize_backward(T * derData,
                                         T const * moments,
                                         T const * data,
                                         T const * multipliers,
                                         T const * derMultipliers,
                                         T const * derBiases,
                                         T const * derOutput,
                                         int planeArea,
                                         int numPlanes,
                                         int numChannels,
                                         T mass)
{
  int tid = threadIdx.x ;
  int plane = blockIdx.x ;
  int blockSize = blockDim.x ;
  int planeStride = gridDim.x ;
  int channel = blockIdx.x % numChannels ;

  // Not optimized for compute capability < 1.2
  T mu = moments[channel];
  T sigma = moments[channel + numChannels] ;
  T multiplier = multipliers[channel] ;
  T derMultiplier = derMultipliers[channel] ;

  T muz = derBiases[channel] / mass;
  T G1 = multiplier / sigma ;
  T G2 = G1 * derMultiplier / (mass*sigma);

  while (plane < numPlanes) {
    T const * planeBegin = data + plane * planeArea ;
    T const * planeEnd = planeBegin + planeArea ;
    T const * block = (T const*) getBlockBeginning<T> (planeBegin) + tid ;
    T const * dblock = derOutput + (block - data) ;
    T * oblock = derData + (block - data) ;
    while (block < planeEnd) {
      if (block >= planeBegin) {
        *oblock = G1 * (*dblock - muz) - G2 * (*block - mu);
      }
      block += blockSize ;
      dblock += blockSize ;
      oblock += blockSize ;
    }
    plane += planeStride ;
  }
}

/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */
/*                                                  bnorm interface */
/* ---------------------------------------------------------------- */
/* ---------------------------------------------------------------- */

namespace vl { namespace impl {

  template<typename T>
  struct bnorm<vl::VLDT_GPU, T>
  {

    /* ------------------------------------------------------------ */
    /*                                                      forward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    forward(Context& context,
            T* output,
            T* moments,
            T const* data,
            T const* multipliers,
            T const* biases,
            size_t height, size_t width, size_t depth, size_t size,
            T epsilon)
    {
      cudaError_t status ;
      unsigned int planeArea = height * width ;
      unsigned int numPlanes = depth * size ;

      // # Block size
      //
      // The block size is a multiple of the warp size, and generally
      // as large as possible. However, we should avoid making the block
      // size too much larger than the area of a plane. In fact,
      // blocks process one plane at a time and would be required to discard
      // a lot of work in this case.

      unsigned int blockSize = getBlockSize(planeArea) ;

      // Each channel is processed by one or more blocks.
      // There are numChunks >= 1 blocks per channel, each working
      // on a subset of one or more images. There are
      //
      //     gridSize = numChunks * depth
      //
      // blocks in the grid.
      //
      // We select numChunks to satisfy the following constraints:
      //
      // 1. There must be at least one block per channel:
      //
      //       numChunks >= 1
      //
      // 2. There must be at most one block per image:
      //
      //       numChunks <= size
      //
      // 3. The grid size must be less than 65536 (CUDA limit)
      //
      //       numChunks <= 65536 / depth
      //
      // Note that constraints (1) and (3) can be satisfied only if
      // depth <= 65536. This is usually not a problem, but may fail
      // in general.
      //
      // In general, (1--3) can be satisfied by setting numChunks=1.
      // However, this is suboptimal if there are too many operations
      // per block.
      //
      // We would like to do at most
      //
      //       L = 10e3 * blockSize
      //
      // operations per block and each block does
      //
      //       (planeArea * size)/numChunks
      //
      // operation. Thus the target value for numChunks is
      //
      //       numChunks = ceil((planeArea * size) / L).
      //

      const unsigned int L = 10000 * blockSize ;
      unsigned int numChunks = (planeArea * size + L - 1) / L ;

      numChunks = min(numChunks, size) ;
      numChunks = min(numChunks, 65536 / depth) ;
      numChunks = max(numChunks, 1) ;
      numChunks = 1  ; // <--  to be removed
      unsigned int gridSize = depth * numChunks ;

      assert(numChunks >= 1) ;
      assert(numChunks <= size) ;
      assert(gridSize <= 65536) ;

      if (numChunks > 1) {

        // We need:
        //
        // * The `accumulator[]` buffer which has size (numChunks x 2*depth) = 2*gridSize
        //   elements to store the partial moments.
        //
        // * Potentially, space for moments[], which has size 2 x depth.

        unsigned int accumulatorSize = 2 * nextMultipleOf(gridSize, WARP_SIZE) ;
        unsigned int workspaceSize = accumulatorSize + (moments ? 0 : 2 * depth) ;
        T * workspace = (T*)context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(T)) ;

        T * accumulator = workspace;
        if (moments == NULL) {
          moments = workspace + accumulatorSize ;
        }

        // Accumulate partial moment summations
        accumulate_moments_partial <<<gridSize, blockSize, 2*blockSize*sizeof(T)>>>
        (accumulator,
         data,
         planeArea,
         numPlanes,
         depth,
         numChunks) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

        // Sum over the chunks (rows of accumulator[])
        int blockSizeSum = getBlockSize(numChunks) ;
        accumulate_moments_finish <<<2*depth, blockSizeSum, blockSizeSum*sizeof(T)>>>
        (moments, accumulator, numChunks) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

      } else {
        if (moments == NULL) {
          moments = (T*) context.getWorkspace(vl::VLDT_GPU, 2*depth * sizeof(T)) ;
        }

        accumulate_moments_partial <<<gridSize, blockSize, 2*blockSize*sizeof(T)>>>
        (moments,
         data,
         planeArea,
         numPlanes,
         depth,
         1) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;
      }

      T mass = planeArea*size;
      normalize_moments <<<divideAndRoundUp(depth,blockSize),blockSize>>>
      (moments, depth, mass, epsilon) ;

      // Finally, normalize the data
      batch_normalize_forward <<<gridSize, blockSize>>>
      (output,
       moments, data, multipliers, biases,
       planeArea,
       numPlanes,
       depth) ;

      status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    /* ------------------------------------------------------------ */
    /*                                        forward_given_moments */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    forward_given_moments(Context& context,
                          T* output,
                          T const* moments,
                          T const* data,
                          T const* multipliers,
                          T const* biases,
                          size_t height, size_t width, size_t depth, size_t size)
    {
      cudaError_t status ;
      unsigned int planeArea = height * width ;
      unsigned int numPlanes = depth * size ;

      unsigned int blockSize = getBlockSize(planeArea) ;
      const unsigned int L = 10000 * blockSize ;
      unsigned int numChunks = (planeArea * size + L - 1) / L ;

      numChunks = min(numChunks, size) ;
      numChunks = min(numChunks, 65536 / depth) ;
      numChunks = max(numChunks, 1) ;
      numChunks = 1  ; // <--  to be removed
      unsigned int gridSize = depth * numChunks ;

      assert(numChunks >= 1) ;
      assert(numChunks <= size) ;
      assert(gridSize <= 65536) ;

      batch_normalize_forward <<<gridSize, blockSize>>>
      (output,
       moments, data, multipliers, biases,
       planeArea,
       numPlanes,
       depth) ;

      status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    /* ------------------------------------------------------------ */
    /*                                                     backward */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward(Context& context,
             T* derData,
             T* derMultipliers,
             T* derBiases,
             T* moments,
             T const* data,
             T const* multipliers,
             T const* biases,
             T const* derOutput,
             size_t height, size_t width, size_t depth, size_t size,
             T epsilon)
    {
      cudaError_t status = cudaSuccess;
      unsigned int planeArea = height * width ;
      unsigned int numPlanes = depth * size ;

      unsigned int blockSize = getBlockSize(planeArea) ;
      const unsigned int L = 10000 * blockSize ;
      unsigned int numChunks = (planeArea * size + L - 1) / L ;

      numChunks = min(numChunks, size) ;
      numChunks = min(numChunks, 65536 / depth) ;
      numChunks = max(numChunks, 1) ;
      numChunks = 1  ; // <--  to be removed
      unsigned int gridSize = depth * numChunks ;

      assert(numChunks >= 1) ;
      assert(numChunks <= size) ;
      assert(gridSize <= 65536) ;

      if (numChunks > 1) {

        unsigned int accumulatorSize = 4 * nextMultipleOf(gridSize, WARP_SIZE) ;
        unsigned int workspaceSize = accumulatorSize + (moments ? 0 : 2 * depth) ;
        T * workspace = (T*)context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(T)) ;

        T * accumulator = workspace;
        if (moments == NULL) {
          moments = workspace + accumulatorSize ;
        }

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

        // Mean, variance, derMultipliers and derBiases computation
        accumulate_ders_and_moments_partial<T> <<<gridSize, blockSize, 4*blockSize*sizeof(T)>>>
        (accumulator,
         NULL, NULL, NULL,
         data,
         derOutput,
         planeArea,
         numPlanes,
         depth,
         numChunks) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

        // Sum over the chunks (rows of accumulator[])
        int blockSizeSum = getBlockSize(numChunks) ;
        accumulate_ders_and_moments_finish<T> <<<4*depth, blockSizeSum, blockSizeSum*sizeof(T)>>>
        (derMultipliers, derBiases, moments, accumulator, numChunks, depth) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

      } else {
        if (moments == NULL) {
          moments = (T*) context.getWorkspace(vl::VLDT_GPU, 2*depth * sizeof(T)) ;
        }

        accumulate_ders_and_moments_partial<T> <<<gridSize, blockSize, 4*blockSize*sizeof(T)>>>
        (NULL,
         derMultipliers, derBiases, moments,
         data,
         derOutput,
         planeArea,
         numPlanes,
         depth,
         1) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;
      }

      T mass = planeArea*size;
      normalize_ders_and_moments<T> <<<divideAndRoundUp(depth,blockSize),blockSize>>>
      (derMultipliers, derBiases, moments, depth, mass, epsilon) ;

      // Compute output
      batch_normalize_backward<T> <<<gridSize, blockSize>>>
      (derData,
       moments, data,
       multipliers, derMultipliers, derBiases, derOutput,
       planeArea, numPlanes, depth,
       mass) ;

      status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

    /* ------------------------------------------------------------ */
    /*                                       backward_given_moments */
    /* ------------------------------------------------------------ */

    static vl::ErrorCode
    backward_given_moments(Context& context,
                           T* derData,
                           T* derMultipliers,
                           T* derBiases,
                           T const* moments,
                           T const* data,
                           T const* multipliers,
                           T const* biases,
                           T const* derOutput,
                           size_t height, size_t width, size_t depth, size_t size,
                           T epsilon)
    {
      cudaError_t status;
      unsigned int planeArea = height * width ;
      unsigned int numPlanes = depth * size ;

      unsigned int blockSize = getBlockSize(planeArea) ;
      const unsigned int L = 10000 * blockSize ;
      unsigned int numChunks = (planeArea * size + L - 1) / L ;

      numChunks = min(numChunks, size) ;
      numChunks = min(numChunks, 65536 / depth) ;
      numChunks = max(numChunks, 1) ;
      numChunks = 1  ; // <--  to be removed
      unsigned int gridSize = depth * numChunks ;

      assert(numChunks >= 1) ;
      assert(numChunks <= size) ;
      assert(gridSize <= 65536) ;

      if (numChunks > 1) {

        unsigned int workspaceSize = 2 * nextMultipleOf(gridSize, WARP_SIZE) ;
        T * accumulator = (T*)context.getWorkspace(vl::VLDT_GPU, workspaceSize * sizeof(T)) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

        // Mean, variance, derMultipliers and derBiases computation
        accumulate_ders_partial<T> <<<gridSize, blockSize, 2*blockSize*sizeof(T)>>>
        (accumulator,
         NULL, NULL,
         data,
         derOutput,
         planeArea,
         numPlanes,
         depth,
         numChunks) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

        // Sum over the chunks (rows of accumulator[])
        int blockSizeSum = getBlockSize(numChunks) ;
        accumulate_ders_finish<T> <<<2*depth, blockSizeSum, blockSizeSum*sizeof(T)>>>
        (derMultipliers, derBiases, accumulator, numChunks, depth) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;

      } else {
        accumulate_ders_partial<T> <<<gridSize, blockSize, 2*blockSize*sizeof(T)>>>
        (NULL,
         derMultipliers, derBiases,
         data,
         derOutput,
         planeArea,
         numPlanes,
         depth,
         1) ;

        status = cudaPeekAtLastError() ;
        if (status != cudaSuccess) return vl::VLE_Cuda ;
      }

      T mass = planeArea*size;
      normalize_ders<T> <<<divideAndRoundUp(depth,blockSize),blockSize>>>
      (derMultipliers, derBiases, moments, depth, mass, epsilon) ;

      // Compute output
      batch_normalize_backward<T> <<<gridSize, blockSize>>>
      (derData,
       moments, data,
       multipliers, derMultipliers, derBiases, derOutput,
       planeArea, numPlanes, depth,
       mass) ;

      status = cudaPeekAtLastError() ;
      return (status == cudaSuccess) ? vl::VLE_Success : vl::VLE_Cuda ;
    }

  } ; // struct bnorm
} } // namespace vl::impl

template struct vl::impl::bnorm<vl::VLDT_GPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::bnorm<vl::VLDT_GPU, double> ;
#endif

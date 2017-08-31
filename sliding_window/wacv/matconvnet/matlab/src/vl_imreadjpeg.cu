/** @file vl_imreadjpeg.cu
 ** @brief Load and transform images asynchronously
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <assert.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>

#include "bits/impl/tinythread.h"
#include "bits/impl/blashelper.hpp"
#include "bits/imread.hpp"
#include "bits/impl/imread_helpers.hpp"

#include "bits/datamex.hpp"
#include "bits/mexutils.h"

static int verbosity = 0 ;

/* option codes */
enum {
  opt_num_threads = 0,
  opt_prefetch,
  opt_resize,
  opt_pack,
  opt_gpu,
  opt_verbose,
  opt_subtract_average,
  opt_crop_size,
  opt_crop_location,
  opt_crop_anisotropy,
  opt_flip,
  opt_contrast,
  opt_saturation,
  opt_brightness,
  opt_interpolation,
} ;

/* options */
VLMXOption  options [] = {
  {"NumThreads",       1,   opt_num_threads        },
  {"Prefetch",         0,   opt_prefetch           },
  {"Verbose",          0,   opt_verbose            },
  {"Resize",           1,   opt_resize             },
  {"Pack",             0,   opt_pack               },
  {"GPU",              0,   opt_gpu                },
  {"SubtractAverage",  1,   opt_subtract_average   },
  {"CropAnisotropy",   1,   opt_crop_anisotropy    },
  {"CropSize",         1,   opt_crop_size          },
  {"CropLocation",     1,   opt_crop_location      },
  {"Flip",             0,   opt_flip               },
  {"Brightness",       1,   opt_brightness         },
  {"Contrast",         1,   opt_contrast           },
  {"Saturation",       1,   opt_saturation         },
  {"Interpolation",    1,   opt_interpolation      },
  {0,                  0,   0                      }
} ;

enum {
  IN_FILENAMES = 0, IN_END
} ;

enum {
  OUT_IMAGES = 0, OUT_END
} ;

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
<<"[info] "<<__func__<<"::"

#define LOG(level) \
if (verbosity < level) { } \
else vl::Logger().getStream() \
<<"[info] "<<__func__<<"::"

/* ---------------------------------------------------------------- */
/*                                                            Batch */
/* ---------------------------------------------------------------- */

class Batch
{
public:
  struct Item
  {
    enum State {
      prefetch,
      fetch,
      ready
    } state ;

    Batch const & batch ;
    std::string name ;
    vl::ImageShape shape ;
    mxArray * array ;
    vl::ErrorCode error ;
    char errorMessage [512] ;
    bool borrowed ;
    vl::MexTensor cpuArray ;
    vl::MexTensor gpuArray ;
    int index ;

    size_t outputWidth ;
    size_t outputHeight ;
    size_t outputNumChannels ;
    size_t cropWidth ;
    size_t cropHeight ;
    size_t cropOffsetX ;
    size_t cropOffsetY ;
    bool flip ;
    vl::impl::ImageResizeFilter::FilterType filterType ;

    float brightnessShift [3] ;
    float contrastShift ;
    float saturationShift ;

    Item(Batch const & batch) ;
    mxArray * relinquishArray() ;
  } ;

  enum ResizeMethod {
    noResize,
    resizeShortestSide,
    fixedSize
  } ;

  enum PackingMethod {
    individualArrays,
    singleArray
  };

  enum CropLocation {
    cropCenter,
    cropRandom
  } ;

  Batch(vl::MexContext & context) ;
  ~Batch() ;
  vl::ErrorCode init() ;
  void finalize() ;
  vl::ErrorCode registerItem(std::string const & name) ;

  size_t getNumberOfItems() const ;
  Item * getItem(int index) ;
  void clear() ;
  void sync() const ;
  vl::ErrorCode prefetch() ;
  mxArray * relinquishArray() ;

  void setGpuMode(bool gpu) ;
  void setPackingMethod(PackingMethod method) ;
  void setResizeMethod(ResizeMethod method, int height, int width) ;

  void setAverage(double average []) ;
  void setAverageImage(float const * image) ;
  void setColorDeviation(double brightness [], double contrast, double saturation) ;
  void setFlipMode(bool x) ;
  void setCropAnisotropy(double minAnisotropy, double maxAnisotropy) ;
  void setCropSize(double minSize, double maxSize) ;
  void setCropLocation(CropLocation location) ;
  void setFilterType(vl::impl::ImageResizeFilter::FilterType type) ;
  PackingMethod getPackingMethod() const  ;

  Item * borrowNextItem() ;
  void returnItem(Item * item) ;

private:
  vl::MexContext & context ;

  tthread::mutex mutable mutex ;
  tthread::condition_variable mutable waitNextItemToBorrow ;
  tthread::condition_variable mutable waitCompletion ;
  bool quit ;
  typedef std::vector<Item*> items_t ;
  items_t items ;
  int nextItem ;
  int numReturnedItems ;

  enum PackingMethod packingMethod ;
  enum ResizeMethod resizeMethod ;
  int resizeHeight ;
  int resizeWidth ;
  bool gpuMode ;

  double average [3] ;
  float * averageImage ;

  double contrastDeviation ;
  double saturationDeviation ;
  double brightnessDeviation [9] ;
  double minCropAnisotropy ;
  double maxCropAnisotropy ;
  double minCropSize ;
  double maxCropSize ;
  CropLocation cropLocation ;
  bool flipMode ;

  vl::impl::ImageResizeFilter::FilterType filterType ;

  vl::MexTensor cpuPack ;
  vl::MexTensor gpuPack ;
  friend class ReaderTask ;
  int gpuDevice ;
#if ENABLE_GPU
  bool cudaStreamInitialized ;
  cudaStream_t cudaStream ;
  float * cpuPinnedPack ;
  size_t cpuPinnedPackSize ;
#endif
} ;

Batch::Item::Item(Batch const & batch)
: batch(batch),
  cpuArray(batch.context),
  gpuArray(batch.context),
  borrowed(false),
  error(vl::VLE_Success),
  state(ready),
  flip(false)
{
  memset(errorMessage,sizeof(errorMessage),0) ;
}

mxArray * Batch::Item::relinquishArray()
{
  if (batch.gpuMode) {
    return gpuArray.relinquish() ;
  } else {
    return cpuArray.relinquish() ;
  }
}

mxArray * Batch::relinquishArray()
{
  if (gpuMode) {
    return gpuPack.relinquish() ;
  } else {
    return cpuPack.relinquish() ;
  }
}

Batch::Batch(vl::MexContext & context)
: context(context),
  cpuPack(context),
  gpuPack(context),
  quit(true),
  resizeMethod(noResize),
  packingMethod(individualArrays),
  gpuMode(false),
  numReturnedItems(0),
  averageImage(NULL)
#if ENABLE_GPU
, cpuPinnedPack(NULL),
  cpuPinnedPackSize(0)
#endif
{ }

Batch::~Batch()
{
  finalize() ;
}

size_t Batch::getNumberOfItems() const
{
  return items.size() ;
}

Batch::Item * Batch::getItem(int index)
{
  return items[index] ;
}

vl::ErrorCode Batch::init()
{
  finalize() ;
  LOG(2)<<"beginning batch" ;
  quit = false ;
  nextItem = 0 ;
  numReturnedItems = 0 ;

  // Restore defaults
  memset(brightnessDeviation, 0, sizeof(brightnessDeviation)) ;
  contrastDeviation = 0. ;
  saturationDeviation = 0. ;
  memset(average, 0, sizeof(average)) ;
  averageImage = NULL ;

  cropLocation = cropCenter ;
  minCropSize = 1. ;
  maxCropSize = 1. ;
  minCropAnisotropy = 1. ;
  maxCropAnisotropy = 1. ;
  flipMode = false ;

  filterType = vl::impl::ImageResizeFilter::kBilinear ;

  packingMethod = individualArrays ;
  resizeMethod = noResize ;
  gpuMode = false ;
  gpuDevice = -1 ;
#if ENABLE_GPU
  if (cudaStreamInitialized) {
    cudaStreamDestroy(cudaStream) ;
    cudaStreamInitialized = false ;
  }
#endif
  return vl::VLE_Success ;
}

void Batch::finalize()
{
  LOG(2)<<"finalizing batch" ;

  // Clear current batch
  clear() ;

  // Release memory
#if ENABLE_GPU
  if (cpuPinnedPack) {
    cudaFreeHost(cpuPinnedPack) ;
    cpuPinnedPack = 0 ;
    cpuPinnedPackSize = 0 ;
  }
#endif

  // Signal waiting threads that we are quitting
  {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    quit = true ;
    waitNextItemToBorrow.notify_all() ;
  }
}

Batch::Item * Batch::borrowNextItem()
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;
  while (true) {
    if (quit) { return NULL ; }
    if (nextItem < items.size()) {
      Item * item = items[nextItem] ;
      if (item->state != Item::ready) {
        item->borrowed = true ;
        nextItem ++  ;
        return item ;
      }
    }
    waitNextItemToBorrow.wait(mutex) ;
  }
}

void Batch::returnItem(Batch::Item * item)
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;
  numReturnedItems ++ ;
  if (item->state == Item::fetch &&
      numReturnedItems == items.size() &&
      packingMethod == singleArray &&
      gpuMode) {
#if ENABLE_GPU
    LOG(2) << "push to GPU the pack" ;
    cudaError_t cerror ;
    cerror = cudaMemcpyAsync (gpuPack.getMemory(),
                              cpuPinnedPack,
                              gpuPack.getNumElements() * sizeof(float),
                              cudaMemcpyHostToDevice,
                              cudaStream) ;
    if (cerror != cudaSuccess) {
      item->error = vl::VLE_Cuda ;
      snprintf(item->errorMessage, sizeof(item->errorMessage),
              "cudaMemcpyAsnyc : '%s'", cudaGetErrorString(cerror)) ;
    }
#endif
  }
  item->borrowed = false ;
  item->state = Batch::Item::ready ;
  waitCompletion.notify_all() ;
}

void Batch::setAverageImage(float const * image)
{
  if (image == NULL) {
    if (averageImage) {
      free(averageImage) ;
      averageImage = NULL ;
    }
    return ;
  }
  assert (resizeMethod == fixedSize) ;
  averageImage = (float*)malloc(sizeof(float) * resizeHeight * resizeWidth * 3) ;
  memcpy(averageImage, image, sizeof(float) * resizeHeight * resizeWidth * 3) ;
}

void Batch::clear()
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;

  // Stop threads from getting more tasks. After this any call to borrowItem() by a worker will
  // stop in a waiting state. Thus, we simply wait for all of them to return their items.
  nextItem = (int)items.size() ;

  // Wait for all thread to return their items
  for (int i = 0 ; i < items.size() ; ++i) {
    while (items[i]->borrowed) {
      waitCompletion.wait(mutex) ;
    }
  }
  for (int i = 0 ; i < items.size() ; ++i) {
    delete items[i] ;
  }
  items.clear() ;

  // Clear average image
  setAverageImage(NULL) ;

  // At the end of the current (empty) list
  nextItem = 0 ;
  numReturnedItems = 0 ;
}

void Batch::sync() const
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;

  // Wait for threads to complete work for all items.
  // Note that it is not enough to check that threads are all in a
  // "done" state as this does not mean that all work has been done yet.
  // Instead, we look at the number of items returned.
  while (numReturnedItems < items.size()) {
    waitCompletion.wait(mutex) ;
  }

  if (gpuMode) {
#if ENABLE_GPU
    cudaError_t cerror ;
    cerror = cudaStreamSynchronize(cudaStream) ;
    if (cerror != cudaSuccess) {
      LOGERROR << "CUDA error while synchronizing a stream: '" << cudaGetErrorString(cerror) << '\'' ;
    }
#endif
  }
}

vl::ErrorCode Batch::registerItem(std::string const & name)
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;
  Item * item = new Item(*this) ;
  item->index = (int)items.size() ;
  item->name = name ;
  item->state = Item::prefetch ;
  items.push_back(item) ;
  return vl::VLE_Success ;
}

void Batch::setGpuMode(bool gpu)
{
  tthread::lock_guard<tthread::mutex> lock(mutex) ;
#if ENABLE_GPU
  if (gpu) {
    cudaGetDevice(&gpuDevice) ;
    if (!cudaStreamInitialized) {
      cudaError_t cerror ;
      cerror = cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking) ;
      if (cerror != cudaSuccess) {
        LOGERROR
        << "CUDA error while creating a stream '"
        << cudaGetErrorString(cerror) << '\"' ;
      } else {
        cudaStreamInitialized = true ;
      }
    }
  }
#endif
  gpuMode = gpu ;
}

void Batch::setResizeMethod(Batch::ResizeMethod method, int height, int width)
{
  resizeMethod = method ;
  resizeHeight = height ;
  resizeWidth = width ;
}

void Batch::setPackingMethod(Batch::PackingMethod method)
{
  assert(method == individualArrays || method == singleArray) ;
  packingMethod = method ;
}

Batch::PackingMethod Batch::getPackingMethod() const
{
  return packingMethod ;
}

void Batch::setAverage(double average [])
{
  ::memcpy(this->average, average, sizeof(this->average)) ;
}

void Batch::setColorDeviation(double brightness [], double contrast, double saturation)
{
  ::memcpy(brightnessDeviation, brightness, sizeof(brightnessDeviation)) ;
  contrastDeviation = contrast ;
  saturationDeviation = saturation ;
}

void Batch::setFilterType(vl::impl::ImageResizeFilter::FilterType type)
{
  filterType = type ;
}

void Batch::setFlipMode(bool x)
{
  flipMode = x ;
}

void Batch::setCropAnisotropy(double minAnisotropy, double maxAnisotropy)
{
  assert(minAnisotropy <= maxAnisotropy) ;
  assert(0.0 <= minAnisotropy && minAnisotropy <= 1.0) ;
  minCropAnisotropy = minAnisotropy ;
  maxCropAnisotropy = maxAnisotropy ;
}

void Batch::setCropSize(double minSize, double maxSize)
{
  assert(minSize <= maxSize) ;
  assert(0.0 <= minSize && minSize <= 1.0) ;
  assert(0.0 <= maxSize && maxSize <= 1.0) ;
  minCropSize = minSize ;
  maxCropSize = maxSize ;
}

void Batch::setCropLocation(CropLocation location)
{
  assert(location == cropCenter || location == cropRandom) ;
  cropLocation = location ;
}

//void Batch::getItemTransformation(Item * item)
//{
//
//}

vl::ErrorCode Batch::prefetch()
{
  // Prod and then wait for reader threads to initialize the shape of the images
  // and then perform the requried allocations.
  waitNextItemToBorrow.notify_all() ;
  sync() ;

  // In packing mode, preallocate all memory here.
  if (packingMethod == singleArray) {
    assert(resizeMethod == fixedSize) ;
    vl::TensorShape shape(resizeHeight, resizeWidth, 3, getNumberOfItems()) ;
    if (gpuMode) {
#if ENABLE_GPU
      gpuPack.init(vl::VLDT_GPU, vl::VLDT_Float, shape) ;
      gpuPack.makePersistent() ;
      size_t memSize = shape.getNumElements() * sizeof(float) ;
      if (cpuPinnedPackSize < memSize) {
        if (cpuPinnedPack) {
          cudaFreeHost(cpuPinnedPack) ;
        }
        cudaMallocHost(&cpuPinnedPack, memSize) ;
        cpuPinnedPackSize = memSize ;
      }
#endif
    } else {
      cpuPack.init(vl::VLDT_CPU, vl::VLDT_Float, shape) ;
      cpuPack.makePersistent() ;
    }
  }

  // Get ready to reprocess all items.
  nextItem = 0 ;
  numReturnedItems = 0 ;

  for (int i = 0 ; i < getNumberOfItems() ; ++ i) {
    Batch::Item * item = getItem(i) ;
    if (item->error == vl::VLE_Success) {
      if (verbosity >= 2) {
        mexPrintf("%20s: %d x %d x %d\n", item->name.c_str(), item->shape.width, item->shape.height, item->shape.depth) ;
      }
    } else {
      mexPrintf("%20s: error '%s'\n", item->name.c_str(), item->errorMessage) ;
    }

    // Determine the shape of (height and width) of the output image. This is either
    // the same as the input image, or with a fixed size for the shortest side,
    // or a fixed size for both sides.

    int outputHeight ;
    int outputWidth ;
    double cropHeight ;
    double cropWidth ;
    int dx ;
    int dy ;

    switch (resizeMethod) {
      case noResize:
        outputHeight = (int)item->shape.height ;
        outputWidth = (int)item->shape.width ;
        break ;

      case resizeShortestSide: {
        double scale1 = (double)resizeHeight / item->shape.width ;
        double scale2 = (double)resizeHeight / item->shape.height ;
        double scale = std::max(scale1, scale2) ;
        outputHeight = (int)std::max(1.0, round(scale * item->shape.height)) ;
        outputWidth = (int)std::max(1.0, round(scale * item->shape.width)) ;
        break ;
      }

      case fixedSize:
        outputHeight = resizeHeight ;
        outputWidth = resizeWidth ;
        break ;
    }

    // Determine the aspect ratio of the crop in the input image.
    {
      double anisotropyRatio = 1.0 ;
      if (minCropAnisotropy == 0 || maxCropAnisotropy == 0) {
        // Stretch crop to have the same shape as the input.
        double inputAspect = (double)item->shape.width / item->shape.height ;
        double outputAspect = (double)outputWidth / outputHeight ;
        anisotropyRatio = inputAspect / outputAspect ;
      } else {
        double z = (double)rand() / RAND_MAX ;
        double a = log(maxCropAnisotropy) ;
        double b = log(minCropAnisotropy) ;
        anisotropyRatio = exp(z * (b - a) + a) ;
      }
      cropWidth = outputWidth * sqrt(anisotropyRatio) ;
      cropHeight = outputHeight / sqrt(anisotropyRatio) ;
    }

    // Determine the crop size.
    {
      double scale = std::min(item->shape.width / cropWidth,
                              item->shape.height / cropHeight) ;
      double z = (double)rand() / RAND_MAX ;
#if 1
      double a = maxCropSize * maxCropSize ;
      double b = minCropSize * minCropSize ;
      double size = sqrt(z * (b - a) + a) ;
#else
      double a = maxCropSize ;
      double b = minCropSize ;
      double size = z * (b - a) + a ;
#endif
      cropWidth *= scale * size ;
      cropHeight *= scale * size ;
    }

    int cropWidth_i = (int)std::min(round(cropWidth), (double)item->shape.width) ;
    int cropHeight_i = (int)std::min(round(cropHeight), (double)item->shape.height) ;

    // Determine the crop location.
    {
      dx = (int)item->shape.width - cropWidth_i ;
      dy = (int)item->shape.height - cropHeight_i ;
      switch (cropLocation) {
        case cropCenter:
          dx /= 2 ;
          dy /= 2 ;
          break ;
        case cropRandom:
          dx = rand() % (dx + 1) ;
          dy = rand() % (dy + 1) ;
          break ;
        default:
          LOGERROR << "cropLocation not set" ;
      }
    }

    // Save.
    item->outputWidth = outputWidth ;
    item->outputHeight = outputHeight ;
    item->outputNumChannels = (packingMethod == individualArrays) ? item->shape.depth : 3 ;

    item->cropWidth = cropWidth_i ;
    item->cropHeight = cropHeight_i ;
    item->cropOffsetX = dx ;
    item->cropOffsetY = dy ;
    item->flip = flipMode && (rand() > RAND_MAX/2) ;
    item->filterType = filterType ;

    // Color processing.
    item->saturationShift = (float)(1. + saturationDeviation * (2.*(double)rand()/RAND_MAX - 1.)) ;
    item->contrastShift = (float)(1. + contrastDeviation * (2.*(double)rand()/RAND_MAX - 1.)) ;
    {
      int numChannels = (int)item->outputNumChannels ;
      double w [3] ;
      for (int i = 0 ; i < numChannels ; ++i) { w[i] = vl::randn() ; }
      for (int i = 0 ; i < numChannels ; ++i) {
        item->brightnessShift[i] = 0.f ;
        for (int j = 0 ; j < numChannels ; ++j) {
          item->brightnessShift[i] += (float)(brightnessDeviation[i + 3*j] * w[i]) ;
        }
      }
    }

    LOG(2)
    << "input ("  << item->shape.width << " x " << item->shape.height << " x " << item->shape.depth << ") "
    << "output (" << item->outputWidth << " x " << item->outputHeight << " x " << item->outputNumChannels << ") "
    << "crop ("   << item->cropWidth   << " x " << item->cropHeight   << ") "
    << "offset (" << item->cropOffsetX << ", "  << item->cropOffsetY  << ")" ;

    if (packingMethod == individualArrays) {
      vl::TensorShape shape(outputHeight, outputWidth, item->outputNumChannels, 1) ;
      item->cpuArray.init(vl::VLDT_CPU, vl::VLDT_Float, shape) ;
      item->cpuArray.makePersistent() ;
      if (gpuMode) {
        item->gpuArray.init(vl::VLDT_GPU, vl::VLDT_Float, shape) ;
        item->gpuArray.makePersistent() ;
      }
    }

    // Ready to fetch
    item->state = Item::fetch ;
  }

  // Notify that we are ready to fetch
  {
    tthread::lock_guard<tthread::mutex> lock(mutex) ;
    waitNextItemToBorrow.notify_all() ;
  }

  return vl::VLE_Success ;
}


/* ---------------------------------------------------------------- */
/*                                                       ReaderTask */
/* ---------------------------------------------------------------- */

class ReaderTask
{
public:
  ReaderTask() ;
  ~ReaderTask() { finalize() ; }
  vl::ErrorCode init(Batch * batch, int index) ;
  void finalize() ;

private:
  int index ;
  Batch * batch ;
  tthread::thread * thread ;
  vl::ImageReader * reader ;
  static void threadEntryPoint(void * thing) ;
  void entryPoint() ;
  void * getBuffer(int index, size_t size) ;
  int gpuDevice ;

private:
  ReaderTask(ReaderTask const &) ;
  ReaderTask & operator= (ReaderTask const &) ;

  struct Buffer {
    void * memory ;
    size_t size ;
  } buffers [2] ;
} ;

void ReaderTask::threadEntryPoint(void * thing)
{
  ((ReaderTask*)thing)->entryPoint() ;
}

ReaderTask::ReaderTask()
: batch(NULL), thread(NULL), reader(NULL)
{
  memset(buffers, 0, sizeof(buffers)) ;
}

void * ReaderTask::getBuffer(int index, size_t size)
{
  if (buffers[index].size < size) {
    if (buffers[index].memory) {
      free(buffers[index].memory) ;
    }
    buffers[index].memory = malloc(size) ;
    buffers[index].size = size ;
  }
  return buffers[index].memory ;
}

void ReaderTask::entryPoint()
{
  LOG(2) << "reader " << index << " task staring" ;

  while (true) {
#if ENABLE_GPU
    if (batch->gpuMode && batch->gpuDevice != gpuDevice) {
      LOG(2) << "reader " << index << " setting GPU device" ;
      cudaSetDevice(batch->gpuDevice) ;
      cudaGetDevice(&gpuDevice) ;
    }
#endif

    Batch::Item * item = batch->borrowNextItem() ;
    LOG(3) << "borrowed " << item ;
    if (item == NULL) { break ; }
    if (item->error != vl::VLE_Success) {
      batch->returnItem(item) ;
      continue ;
    }

    switch (item->state) {
      case Batch::Item::prefetch: {
        item->error = reader->readShape(item->shape, item->name.c_str()) ;
        if (item->error != vl::VLE_Success) {
          snprintf(item->errorMessage, sizeof(item->errorMessage), "%s", reader->getLastErrorMessage()) ;
        }
        break ;
      }

      case Batch::Item::fetch: {
        // Get the CPU buffer that will hold the pixels.
        float * outputPixels;
        if (batch->getPackingMethod() == Batch::individualArrays) {
          outputPixels = (float*)item->cpuArray.getMemory() ;
        } else {
          if (batch->gpuMode) {
#if ENABLE_GPU
            outputPixels = batch->cpuPinnedPack ;
#else
            snprintf(item->errorMessage, sizeof(item->errorMessage), "GPU support not compiled.") ;
            break;
#endif
          } else {
            outputPixels = (float*)batch->cpuPack.getMemory() ;
          }
          outputPixels += item->outputHeight*item->outputWidth*3*item->index ;
        }

        // Read full image.
        float * inputPixels = (float*)getBuffer(0,
                                                item->shape.height *
                                                item->shape.width *
                                                item->shape.depth * sizeof(float)) ;
        item->error = reader->readPixels(inputPixels, item->name.c_str()) ;
        if (item->error != vl::VLE_Success) {
          snprintf(item->errorMessage, sizeof(item->errorMessage), "%s", reader->getLastErrorMessage()) ;
          break ;
        }

        // Crop.
        float * temp = (float*)getBuffer(1,
                                         item->outputHeight *
                                         item->shape.width *
                                         item->shape.depth * sizeof(float)) ;

        vl::impl::imageResizeVertical(temp, inputPixels,
                                      item->outputHeight,
                                      item->shape.height,
                                      item->shape.width,
                                      item->shape.depth,
                                      item->cropHeight,
                                      item->cropOffsetY,
                                      false, // flip
                                      item->filterType) ;

        vl::impl::imageResizeVertical(outputPixels, temp,
                                      item->outputWidth,
                                      item->shape.width,
                                      item->outputHeight,
                                      item->shape.depth,
                                      item->cropWidth,
                                      item->cropOffsetX,
                                      item->flip,
                                      item->filterType) ;

        // Postprocess colors.
        {
          size_t inputNumChannels = item->shape.depth ;
          size_t K = item->outputNumChannels ;
          size_t n = item->outputHeight*item->outputWidth ;
          if (batch->averageImage) {
            // If there is an average image, then subtract it now.
            // Grayscale images are expanded here to color if needed.
            // Withouth an average image,
            // they are expanded later.

            for (size_t k = inputNumChannels ; k < K ; ++k) {
              ::memcpy(outputPixels + n*k, outputPixels, sizeof(float) * n) ;
            }

            vl::impl::blas<vl::VLDT_CPU,vl::VLDT_Float>::axpy
              (batch->context,
               n * item->outputNumChannels,
               -1.0f,
               batch->averageImage, 1,
               outputPixels, 1) ;

            inputNumChannels = K ;
          }
          float dv [3] ;
          float * channels [3] ;
          for (int k = 0 ; k < K ; ++k) {
            channels[k] = outputPixels + n * k ;
          }
          for (int k = 0 ; k < inputNumChannels ; ++k) {
            dv[k] = item->brightnessShift[k] - batch->average[k] ;
            if (item->contrastShift != 1.) {
              double mu = 0. ;
              float const * pixel = channels[k] ;
              float const * end = channels[k] + n ;
              while (pixel != end) { mu += (double)(*pixel++) ; }
              mu /= (double)n ;
              dv[k] += (float)((1.0 - (double)item->contrastShift) * mu) ;
            }
          }
          {
            float mu = 0.f ;
            for (int k = 0 ; k < inputNumChannels ; ++k) {
              mu += dv[k] ;
            }
            float a = item->saturationShift ;
            float b = (1. - item->saturationShift) / inputNumChannels ;
            for (int k = 0 ; k < inputNumChannels ; ++k) {
              dv[k] = a * dv[k] + b * mu ;
            }
          }
          {
            float const * end = channels[0] + n ;
            float v [3] ;
            if (K == 3 && inputNumChannels == 3) {
              float const a = item->contrastShift * item->saturationShift ;
              float const b = item->contrastShift * (1.f - item->saturationShift) / K ;
              while (channels[0] != end) {
                float mu = 0.f ;
                v[0] = *channels[0] ; mu += v[0] ;
                v[1] = *channels[1] ; mu += v[1] ;
                v[2] = *channels[2] ; mu += v[2] ;
                *channels[0]++ = a * v[0] + b * mu + dv[0] ;
                *channels[1]++ = a * v[1] + b * mu + dv[1] ;
                *channels[2]++ = a * v[2] + b * mu + dv[2] ;
              }
            } else if (K == 3 && inputNumChannels == 1) {
              float const a = item->contrastShift * item->saturationShift ;
              float const b = item->contrastShift * (1.f - item->saturationShift) / K ;
              while (channels[0] != end) {
                float mu = 0.f ;
                v[0] = *channels[0] ; mu += v[0] ;
                v[1] = *channels[0] ; mu += v[1] ;
                v[2] = *channels[0] ; mu += v[2] ;
                *channels[0]++ = a * v[0] + b * mu + dv[0] ;
                *channels[1]++ = a * v[1] + b * mu + dv[0] ;
                *channels[2]++ = a * v[2] + b * mu + dv[0] ;
              }
            } else {
              float const a = item->contrastShift ;
              while (channels[0] != end) {
                float v = *channels[0] ;
                *channels[0]++ = a * v + dv[0] ;
              }
            }
          }
        }

        // Copy to GPU.
        if (batch->getPackingMethod() == Batch::individualArrays && batch->gpuMode) {
#if ENABLE_GPU
          cudaError_t cerror ;
          cerror = cudaMemcpyAsync (item->gpuArray.getMemory(),
                                    outputPixels,
                                    item->gpuArray.getNumElements() * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    batch->cudaStream) ;
          if (cerror != cudaSuccess) {
            item->error = vl::VLE_Cuda ;
            snprintf(item->errorMessage, sizeof(item->errorMessage),
                     "CUDA error while copying memory from host to device: '%s'", cudaGetErrorString(cerror)) ;
            break ;
          }
#endif
        }
        break ;
      }

      case Batch::Item::ready:
        break ;
    }
    batch->returnItem(item) ;
  }
  LOG(2) << "reader " << index << " task quitting" ;
}

void ReaderTask::finalize()
{
  LOG(2)<<"finalizing reader " << index ;
  if (thread) {
    if (thread->joinable()) {
      thread->join() ;
    }
    delete thread ;
    thread = NULL ;
  }
  for (int i = 0 ; i < sizeof(buffers)/sizeof(Buffer) ; ++i) {
    if (buffers[i].memory) {
      free(buffers[i].memory) ;
      buffers[i].memory = NULL ;
      buffers[i].size = 0 ;
    }
  }
  if (reader) {
    delete reader ;
    reader = NULL ;
  }
  index = -1 ;
  batch = NULL ;
}

vl::ErrorCode ReaderTask::init(Batch * batch, int index)
{
  finalize() ;
  this->batch = batch ;
  this->index = index ;
  thread = new tthread::thread(threadEntryPoint, this) ;
  reader = new vl::ImageReader() ;
  return vl::VLE_Success ;
}

/* ---------------------------------------------------------------- */
/*                                                            Cache */
/* ---------------------------------------------------------------- */

vl::MexContext context ;
Batch batch(context) ;
bool batchIsInitialized = false ;
typedef std::vector<ReaderTask*> readers_t ;
readers_t readers ;

void atExit()
{
  if (batchIsInitialized) {
    batch.finalize() ;
    batchIsInitialized = false ;
  }
  for (int r = 0 ; r < readers.size() ; ++r) {
    readers[r]->finalize() ;
    delete readers[r] ;
  }
  readers.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                            Cache */
/* ---------------------------------------------------------------- */

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool prefetch = false ;
  bool gpuMode = false ;
  int requestedNumThreads = (int)readers.size() ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  Batch::PackingMethod packingMethod = Batch::individualArrays ;
  Batch::ResizeMethod resizeMethod = Batch::noResize ;
  int resizeWidth = -1 ;
  int resizeHeight = -1 ;
  vl::ErrorCode error ;

  double average [3] = {0.} ;
  vl::MexTensor averageImage(context) ;
  double brightnessDeviation [9] = {0.} ;
  double saturationDeviation = 0. ;
  double contrastDeviation = 0. ;
  bool flipMode = false ;
  Batch::CropLocation cropLocation = Batch::cropCenter ;
  double minCropSize = 1.0, maxCropSize = 1.0 ;
  double minCropAnisotropy = 1.0, maxCropAnisotropy = 1.0 ;

  vl::impl::ImageResizeFilter::FilterType filterType = vl::impl::ImageResizeFilter::kBilinear ;

  verbosity = 0 ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 1) {
    vlmxError(VLMXE_IllegalArgument, "There is less than one argument.") ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_prefetch :
        prefetch = true ;
        break ;

      case opt_pack :
        packingMethod = Batch::singleArray ;
        break ;

      case opt_gpu :
#ifndef ENABLE_GPU
        vlmxError(VLMXE_IllegalArgument, "Not compiled with GPU support.") ;
#endif
        gpuMode = true ;
        break ;

      case opt_num_threads :
        requestedNumThreads = (int)mxGetScalar(optarg) ;
        break ;

      case opt_resize :
        if (!vlmxIsPlainVector(optarg, -1)) {
          vlmxError(VLMXE_IllegalArgument, "RESIZE is not a plain vector.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1 :
            resizeMethod = Batch::resizeShortestSide ;
            resizeHeight = (int)mxGetPr(optarg)[0] ;
            resizeWidth = (int)mxGetPr(optarg)[0] ;
            break ;
          case 2 :
            resizeMethod = Batch::fixedSize ;
            resizeHeight = (int)mxGetPr(optarg)[0] ;
            resizeWidth = (int)mxGetPr(optarg)[1] ;
            break;
          default:
            vlmxError(VLMXE_IllegalArgument, "RESIZE does not have one or two dimensions.") ;
            break ;
        }
        if (resizeHeight < 1 || resizeWidth < 1) {
          vlmxError(VLMXE_IllegalArgument, "An element of RESIZE is smaller than one.") ;
        }
        break ;

      case opt_brightness: {
        if (!vlmxIsPlainMatrix(optarg, -1, -1)) {
          vlmxError(VLMXE_IllegalArgument, "BRIGHTNESS is not a plain matrix.") ;
        }
        size_t n = mxGetNumberOfElements(optarg) ;
        memset(brightnessDeviation, 0, sizeof(brightnessDeviation)) ;
        if (n == 1) {
          double x = mxGetPr(optarg)[0] ;
          brightnessDeviation[0] = x;
          brightnessDeviation[3] = x;
          brightnessDeviation[8] = x;
        } else if (n == 3) {
          double const* x = mxGetPr(optarg) ;
          brightnessDeviation[0] = x[0];
          brightnessDeviation[3] = x[1];
          brightnessDeviation[8] = x[2];
        } else if (n == 9) {
          memcpy(brightnessDeviation, mxGetPr(optarg), sizeof(brightnessDeviation)) ;
        } else {
          vlmxError(VLMXE_IllegalArgument, "BRIGHTNESS does not have 1, 3, or 9 elements.") ;
        }
        break ;
      }

      case opt_saturation: {
        if (!vlmxIsPlainScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "SATURATION is not a plain scalar.") ;
        }
        double x = mxGetPr(optarg)[0] ;
        if (x < 0 || x > 1.0) {
          vlmxError(VLMXE_IllegalArgument, "SATURATION is not in the [0,1] range..") ;
        }
        saturationDeviation = x ;
        break ;
      }

      case opt_contrast: {
        if (!vlmxIsPlainScalar(optarg)) {
          vlmxError(VLMXE_IllegalArgument, "CONTRAST is not a plain scalar.") ;
        }
        double x = mxGetPr(optarg)[0] ;
        if (x < 0 || x > 1.0) {
          vlmxError(VLMXE_IllegalArgument, "CONTRAST is not in the [0,1] range..") ;
        }
        contrastDeviation = x ;
        break ;
      }

      case opt_crop_anisotropy: {
        if (!vlmxIsPlainScalar(optarg) && !vlmxIsPlainVector(optarg, 2)) {
          vlmxError(VLMXE_IllegalArgument, "CROPANISOTROPY is not a plain scalar or vector with two components.") ;
        }
        minCropAnisotropy =  mxGetPr(optarg)[0] ;
        maxCropAnisotropy =  mxGetPr(optarg)[std::min((mwSize)1, mxGetNumberOfElements(optarg)-1)] ;
        if (minCropAnisotropy < 0.0 || minCropAnisotropy > maxCropAnisotropy) {
          vlmxError(VLMXE_IllegalArgument, "CROPANISOTROPY values are not in the legal range.") ;
        }
        break ;
      }

      case opt_crop_size: {
        if (!vlmxIsPlainScalar(optarg) && !vlmxIsPlainVector(optarg, 2)) {
          vlmxError(VLMXE_IllegalArgument, "CROPSIZE is not a plain scalar or vector with two components.") ;
        }
        minCropSize = mxGetPr(optarg)[0] ;
        maxCropSize = mxGetPr(optarg)[std::min((mwSize)1, mxGetNumberOfElements(optarg)-1)] ;
        if (minCropSize < 0.0 || minCropSize > maxCropSize || maxCropSize > 1.0) {
          vlmxError(VLMXE_IllegalArgument, "CROPSIZE values are not in the legal range.") ;

        }
        break ;
      }

      case opt_crop_location: {
        if (!vlmxIsString(optarg, -1)) {
          vlmxError(VLMXE_IllegalArgument, "CROPLOCATION is not a string") ;
        }
        if (vlmxCompareToStringI(optarg, "random") == 0) {
          cropLocation = Batch::cropRandom ;
        } else if (vlmxCompareToStringI(optarg, "center") == 0) {
          cropLocation = Batch::cropCenter ;
        } else {
          vlmxError(VLMXE_IllegalArgument, "CROPLOCATION value unknown.") ;
        }
        break ;
      }

      case opt_subtract_average: {
        if (vlmxIsVector(optarg,1) || vlmxIsVector(optarg, 3)) {
          size_t n = mxGetNumberOfElements(optarg) ;
          switch (mxGetClassID(optarg)) {
          case mxSINGLE_CLASS: {
            float * x = (float*)mxGetData(optarg) ;
            average[0] = x[std::min((size_t)0,n-1)] ;
            average[1] = x[std::min((size_t)1,n-1)] ;
            average[2] = x[std::min((size_t)2,n-1)] ;
            break ;
          }
          case mxDOUBLE_CLASS: {
            double * x = mxGetPr(optarg) ;
            average[0] = (float)x[std::min((size_t)0,n-1)] ;
            average[1] = (float)x[std::min((size_t)1,n-1)] ;
            average[2] = (float)x[std::min((size_t)2,n-1)] ;
            break ;
          }
          default:
            vlmxError(VLMXE_IllegalArgument, "SUBTRACTAVERAGE is not SINGLE or DOUBLE vector.") ;
          }
        } else {
          if (mxGetClassID(optarg) != mxSINGLE_CLASS ||
              mxGetNumberOfDimensions(optarg) > 3) {
            vlmxError(VLMXE_IllegalArgument, "SUBTRACTAVERAGE is not a SINGLE image of a compatible shape.") ;
          }
          averageImage.init(optarg) ;
        }
        break ;
      }

      case opt_flip: {
        flipMode = true ;
        break ;
      }

      case opt_interpolation: {
        if (!vlmxIsString(optarg,-1)) {
          vlmxError(VLMXE_IllegalArgument, "INTERPOLATION is not a string.") ;
        }
        if (vlmxIsEqualToStringI(optarg, "box")) {
          filterType = vl::impl::ImageResizeFilter::kBox ;
        } else if (vlmxIsEqualToStringI(optarg, "bilinear")) {
          filterType = vl::impl::ImageResizeFilter::kBilinear ;
        } else if (vlmxIsEqualToStringI(optarg, "bicubic")) {
          filterType = vl::impl::ImageResizeFilter::kBicubic ;
        } else if (vlmxIsEqualToStringI(optarg, "lanczos2")) {
          filterType = vl::impl::ImageResizeFilter::kLanczos2 ;
        } else if (vlmxIsEqualToStringI(optarg, "lanczos3")) {
          filterType = vl::impl::ImageResizeFilter::kLanczos3 ;
        } else {
          vlmxError(VLMXE_IllegalArgument, "INTERPOLATION is not a supported method.") ;
        }
        break ;
      }
    }
  }

  if (averageImage) {
    if (resizeMethod != Batch::fixedSize) {
      vlmxError(VLMXE_IllegalArgument, "Cannot subtract an average image unless RESIZE is used to set the size of the output.") ;
    }
    if (averageImage.getNumDimensions() != 3 ||
        averageImage.getHeight() != resizeHeight ||
        averageImage.getWidth() != resizeWidth ||
        averageImage.getDepth() !=3) {
      vlmxError(VLMXE_IllegalArgument, "The average image is not a RESIZEHEIGHT x RESIZEWIDTH x 3 array.") ;
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  if (!mxIsCell(in[IN_FILENAMES])) {
    vlmxError(VLMXE_IllegalArgument, "FILENAMES is not a cell array of strings.") ;
  }

  // If the requested number of threads changes, finalize everything
  requestedNumThreads = (std::max)(requestedNumThreads, 1) ;
  if (readers.size() != requestedNumThreads) {
    atExit() ; // Delete threads and current batch
  }

  // Prepare batch.
  if (!batchIsInitialized) {
    error = batch.init() ;
    if (error != vl::VLE_Success) {
      vlmxError(VLMXE_Execution, "Could not initialize a batch structure") ;
    }
    batchIsInitialized = true ;
  }

  // Prepare reader tasks.
  for (size_t r = readers.size() ; r < requestedNumThreads ; ++r) {
    readers.push_back(new ReaderTask()) ;
    vl::ErrorCode error = readers[r]->init(&batch, r) ;
    if (error != vl::VLE_Success) {
      vlmxError(VLMXE_Execution, "Could not create the requested number of threads") ;
    }
  }

  // Extract filenames as strings.
  bool sameAsPrefeteched = true ;
  std::vector<std::string> filenames ;
  for (int i = 0 ; i < (int)mxGetNumberOfElements(in[IN_FILENAMES]) ; ++i) {
    mxArray* filenameArray = mxGetCell(in[IN_FILENAMES], i) ;
    if (!vlmxIsString(filenameArray,-1)) {
      vlmxError(VLMXE_IllegalArgument, "FILENAMES contains an entry that is not a string.") ;
    }
    char filename [512] ;
    mxGetString (filenameArray, filename, sizeof(filename)/sizeof(char)) ;
    filenames.push_back(std::string(filename)) ;
    sameAsPrefeteched &= (i < batch.getNumberOfItems() && batch.getItem(i)->name == filenames[i]) ;
  }

  // If the list of names is not the same as the prefetched ones,
  // start a new cycle.
  if (!sameAsPrefeteched) {
    batch.clear() ;

    // Check compatibility of options
    if (packingMethod == Batch::singleArray && resizeMethod != Batch::fixedSize) {
      vlmxError(VLMXE_IllegalArgument, "PACK must be used in combination with resizing to a fixed size.") ;
    }

    if (verbosity >= 2) {
      mexPrintf("vl_imreadjpeg: gpu mode: %s\n", gpuMode?"yes":"no") ;
      mexPrintf("vl_imreadjpeg: crop anisotropy: [%.1g, %.1g]\n",
                minCropAnisotropy, maxCropAnisotropy) ;
      mexPrintf("vl_imreadjpeg: crop size: [%.1g, %.1g]\n",
                minCropSize, maxCropSize) ;
      mexPrintf("vl_imreadjpeg: num_threads: %d requested %d readers\n",
                requestedNumThreads, readers.size());
    }


    batch.setResizeMethod(resizeMethod, resizeHeight, resizeWidth) ;
    batch.setPackingMethod(packingMethod) ;
    batch.setGpuMode(gpuMode) ;

    batch.setFlipMode(flipMode) ;
    batch.setCropLocation(cropLocation) ;
    batch.setCropAnisotropy(minCropAnisotropy, maxCropAnisotropy) ;
    batch.setCropSize(minCropSize, maxCropSize) ;
    batch.setColorDeviation(brightnessDeviation,
                            contrastDeviation,
                            saturationDeviation) ;

    batch.setAverage(average) ;
    if (averageImage) {
      batch.setAverageImage((float const*)averageImage.getMemory()) ;
    }

    batch.setFilterType(filterType) ;

    for (int i = 0 ; i < filenames.size() ; ++ i) {
      batch.registerItem(filenames[i]) ;
    }

    batch.prefetch() ;
  }

  // Done if prefetching only.
  if (prefetch) { return ; }

  // Return result.
  batch.sync() ;

  switch (batch.getPackingMethod()) {
    case Batch::singleArray: {
      mwSize dims [] = {1,1} ;
      out[OUT_IMAGES] = mxCreateCellArray(2, dims) ;
      mxSetCell(out[OUT_IMAGES], 0, batch.relinquishArray()) ;
      break ;
    }

    case Batch::individualArrays:
      out[OUT_IMAGES] = mxCreateCellArray(mxGetNumberOfDimensions(in[IN_FILENAMES]),
                                          mxGetDimensions(in[IN_FILENAMES])) ;
      for (int i = 0 ; i < batch.getNumberOfItems() ; ++i) {
        Batch::Item * item = batch.getItem(i) ;
        if (item->error != vl::VLE_Success) {
          vlmxWarning(VLMXE_Execution, "could not read image '%s' because '%s'",
                      item->name.c_str(),
                      item->errorMessage) ;
        } else {
          mxSetCell(out[OUT_IMAGES], i, item->relinquishArray()) ;
        }
      }
      break ;
  }

  // Finalize.
  batch.clear() ;
}

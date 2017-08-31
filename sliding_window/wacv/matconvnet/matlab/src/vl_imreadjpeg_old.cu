/** @file vl_imreadjpeg.cu
 ** @brief Load images asynchronously
 ** @author Andrea Vedaldi
 **/

/*
Copyright (C) 2014-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/impl/tinythread.h"
#include "bits/imread.hpp"
#include "bits/impl/imread_helpers.hpp"

#include <vector>
#include <string>
#include <algorithm>

#include "bits/data.hpp"
#include "bits/mexutils.h"

/* option codes */
enum {
  opt_num_threads = 0,
  opt_prefetch,
  opt_resize,
  opt_verbose,
} ;

/* options */
VLMXOption  options [] = {
  {"NumThreads",       1,   opt_num_threads        },
  {"Prefetch",         0,   opt_prefetch           },
  {"Verbose",          0,   opt_verbose            },
  {"Resize",           1,   opt_resize             },
  {0,                  0,   0                      }
} ;

enum {
  IN_FILENAMES = 0, IN_END
} ;

enum {
  OUT_IMAGES = 0, OUT_END
} ;

enum ResizeMode
{
  kResizeNone,
  kResizeAnisotropic,
  kResizeIsotropic,
} ;

/* ---------------------------------------------------------------- */
/*                                                           Caches */
/* ---------------------------------------------------------------- */

class ImageBuffer : public vl::Image
{
public:
  ImageBuffer()
  : vl::Image(), hasMatlabMemory(false), isMemoryOwner(false)
  { }

  ImageBuffer(ImageBuffer const & im)
  : vl::Image(im), hasMatlabMemory(im.hasMatlabMemory), isMemoryOwner(false)
  { }

  ~ImageBuffer()
  {
    clear() ;
  }

  ImageBuffer & operator = (ImageBuffer const & imb)
  {
    clear() ;
    vl::Image::operator=(imb) ;
    hasMatlabMemory = imb.hasMatlabMemory ;
    isMemoryOwner = false ;
    return *this ;
  }

  void clear()
  {
    if (isMemoryOwner && memory) {
      if (hasMatlabMemory) {
        mxFree(memory) ;
      } else {
        free(memory) ;
      }
    }
    isMemoryOwner = false ;
    hasMatlabMemory = false ;
    vl::Image::clear() ;
  }

  float * relinquishMemory()
  {
    float * memory_ = memory ;
    isMemoryOwner = false ;
    clear() ;
    return memory_ ;
  }

  vl::ErrorCode init(vl::ImageShape const & shape_, bool matlab_)
  {
    clear() ;
    shape = shape_ ;
    isMemoryOwner = true ;
    if (matlab_) {
      memory = (float*)mxMalloc(sizeof(float)*shape.getNumElements()) ;
      mexMakeMemoryPersistent(memory) ;
      hasMatlabMemory = true ;
    } else {
      memory = (float*)malloc(sizeof(float)*shape.getNumElements()) ;
      hasMatlabMemory = false ;
    }
    return vl::VLE_Success ;
  }

  bool hasMatlabMemory ;
  bool isMemoryOwner ;
} ;

#define TASK_ERROR_MSG_MAX_LEN 1024

struct Task
{
  std::string name ;
  bool done ;
  ImageBuffer resizedImage ;
  ImageBuffer inputImage ;
  vl::ErrorCode error ;
  bool requireResize ;
  char errorMessage [TASK_ERROR_MSG_MAX_LEN] ;

  Task() { }

private:
  Task(Task const &) ;
  Task & operator= (Task const &) ;
} ;

typedef std::vector<Task*> Tasks ;
Tasks tasks ;
tthread::mutex tasksMutex ;
tthread::condition_variable tasksCondition ;
tthread::condition_variable completedCondition ;
int nextTaskIndex = 0 ;
int numTasksCompleted = 0 ;

typedef std::pair<tthread::thread*,vl::ImageReader*> reader_t ;
typedef std::vector<reader_t> readers_t ;
readers_t readers ;
bool terminateReaders = true ;

/* ---------------------------------------------------------------- */
/*                                                Tasks and readers */
/* ---------------------------------------------------------------- */

void reader_function(void* reader_)
{
  vl::ImageReader* reader = (vl::ImageReader*) reader_ ;
  int taskIndex ;

  tasksMutex.lock() ;
  while (true) {
    // wait for next task
    while ((nextTaskIndex >= tasks.size()) && ! terminateReaders) {
      tasksCondition.wait(tasksMutex);
    }
    if (terminateReaders) {
      break ;
    }
    taskIndex = nextTaskIndex++ ;
    Task & thisTask = *tasks[taskIndex] ;

    tasksMutex.unlock() ;
    if (thisTask.error == vl::VLE_Success) {
      // the memory has been pre-allocated
      thisTask.error = reader->readPixels(thisTask.inputImage.getMemory(), thisTask.name.c_str()) ;
      if (thisTask.error != vl::VLE_Success) {
        strncpy(thisTask.errorMessage, reader->getLastErrorMessage(), TASK_ERROR_MSG_MAX_LEN) ;
      }
    }

    if ((thisTask.error == vl::VLE_Success) && thisTask.requireResize) {
      vl::impl::resizeImage(thisTask.resizedImage, thisTask.inputImage) ;
    }

    tasksMutex.lock() ;
    thisTask.done = true ;
    numTasksCompleted ++ ;
    completedCondition.notify_all() ;
  }
  tasksMutex.unlock() ;
}

void delete_readers()
{
  tasksMutex.lock() ;
  terminateReaders = true ;
  tasksMutex.unlock() ;
  tasksCondition.notify_all() ;
  for (int r = 0 ; r < (int)readers.size() ; ++r) {
    readers[r].first->join() ;
    delete readers[r].first ;
    delete readers[r].second ;
  }
  readers.clear() ;
}

void create_readers(int num, int verbosity)
{
  if (num <= 0) {
    num = (std::max)(1, (int)readers.size()) ;
  }
  if (readers.size() == num) {
    return ;
  }
  if (verbosity > 1) { mexPrintf("vl_imreadjpeg: flushing reader threads\n") ; }
  delete_readers() ;

  terminateReaders = false ;
  for (int r = 0 ; r < num ; ++r) {
    vl::ImageReader * reader = new vl::ImageReader() ;
    tthread::thread * readerThread = new tthread::thread(reader_function, reader) ;
    readers.push_back(reader_t(readerThread, reader)) ;
  }
  if (verbosity > 1) { mexPrintf("vl_imreadjpeg: created %d reader threads\n", readers.size()) ; }
}

void delete_tasks() {
  for (int t = 0 ; t < (int)tasks.size() ; ++t) {
    if (tasks[t]) { delete tasks[t] ; }
  }
  tasks.clear() ;
}

void flush_tasks() {
  // wait until all tasks in the current list are complete
  tasksMutex.lock() ;
  while (numTasksCompleted < (int)tasks.size()) {
    completedCondition.wait(tasksMutex);
  }

  // now delete them
  delete_tasks() ;
  numTasksCompleted = 0 ;
  nextTaskIndex = 0 ;
  tasksMutex.unlock() ;
}

void atExit()
{
  delete_readers() ;
  delete_tasks() ;
}

/* ---------------------------------------------------------------- */
/*                                                            Cache */
/* ---------------------------------------------------------------- */

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool prefetch = false ;
  int requestedNumThreads = -1 ;
  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;
  int i ;
  ResizeMode resizeMode = kResizeNone ;
  int resizeWidth = 1 ;
  int resizeHeight = 1 ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 1) {
    mexErrMsgTxt("There is less than one argument.") ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_prefetch :
        prefetch = true ;
        break ;

      case opt_resize :
        if (!vlmxIsPlainVector(optarg, -1)) {
          mexErrMsgTxt("RESIZE is not a plain vector.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1 :
            resizeMode = kResizeIsotropic ;
            resizeHeight = (int)mxGetPr(optarg)[0] ;
            //  resizeWidth other has the dummy value 1
            break ;
          case 2 :
            resizeMode = kResizeAnisotropic ;
            resizeHeight = (int)mxGetPr(optarg)[0] ;
            resizeWidth = (int)mxGetPr(optarg)[1] ;
            break;
          default:
            mexErrMsgTxt("RESIZE does not have one or two dimensions.") ;
            break ;
        }
        if (resizeHeight < 1 || resizeWidth < 1) {
          mexErrMsgTxt("An element of RESIZE is smaller than one.") ;
        }
        break ;

      case opt_num_threads :
        requestedNumThreads = (int)mxGetScalar(optarg) ;
        break ;
    }
  }

  if (!mxIsCell(in[IN_FILENAMES])) {
    mexErrMsgTxt("FILENAMES is not a cell array of strings.") ;
  }

  // prepare reader tasks
  create_readers(requestedNumThreads, verbosity) ;

  if (verbosity) {
    mexPrintf("vl_imreadjpeg: numThreads = %d, prefetch = %d\n",
              readers.size(), prefetch) ;
    switch (resizeMode) {
      case kResizeIsotropic:
        mexPrintf("vl_imreadjpeg: isotropic resize to x%d\n", resizeHeight) ;
        break ;
      case kResizeAnisotropic:
        mexPrintf("vl_imreadjpeg: anisotropic resize to %dx%d\n", resizeHeight, resizeWidth) ;
        break ;
      default:
        break ;
    }
  }

  // extract filenames as strings
  std::vector<std::string> filenames ;
  for (i = 0 ; i < (int)mxGetNumberOfElements(in[IN_FILENAMES]) ; ++i) {
    mxArray* filename_array = mxGetCell(in[IN_FILENAMES], i) ;
    if (!vlmxIsString(filename_array,-1)) {
      mexErrMsgTxt("FILENAMES contains an entry that is not a string.") ;
    }
    char filename [4096] ;
    mxGetString (filename_array, filename, sizeof(filename)/sizeof(char)) ;
    filenames.push_back(std::string(filename)) ;
  }

  // check if the cached tasks match the new ones
  bool match = true ;
  for (int t = 0 ; match & (t < (signed)filenames.size()) ; ++t) {
    if (t >= (signed)tasks.size()) {
      match = false ;
      break ;
    }
    match &= (tasks[t]->name == filenames[t]) ;
  }

  // if there is no match, then flush tasks and start over
  if (!match) {
    if (verbosity > 1) {
      mexPrintf("vl_imreadjpeg: flushing tasks\n") ;
    }
    flush_tasks() ;
    tasksMutex.lock() ;
    for (int t = 0 ; t < (signed)filenames.size() ; ++t) {
      Task* newTask(new Task()) ;
      newTask->name = filenames[t] ;
      newTask->done = false ;
      ImageBuffer & inputImage = newTask->inputImage ;
      ImageBuffer & resizedImage = newTask->resizedImage ;

      vl::ImageShape shape ;
      newTask->error = readers[0].second->readShape(shape, filenames[t].c_str()) ;
      if (newTask->error == vl::VLE_Success) {
        vl::ImageShape resizedShape = shape ;
        switch (resizeMode) {
          case kResizeAnisotropic:
            resizedShape.height = resizeHeight ;
            resizedShape.width = resizeWidth ;
            break ;
          case kResizeIsotropic:
          {
            // note: not a bug below, resizeHeight contains the only resize param
            float scale = (std::max)((float)resizeHeight / shape.width,
                                     (float)resizeHeight / shape.height);
            resizedShape.height = roundf(resizedShape.height * scale) ;
            resizedShape.width = roundf(resizedShape.width * scale) ;
            break ;
          }
          default:
            break ;
        }
        newTask->requireResize = ! (resizedShape == shape) ;
        if (newTask->requireResize) {
          newTask->error = inputImage.init(shape, false) ;
          if (newTask->error == vl::VLE_Success) {
            newTask->error = resizedImage.init(resizedShape, true) ;
          }
        } else {
          newTask->error = resizedImage.init(shape, true) ;
          // alias: remark: resized image will be asked to release memory so it *must* be the owner
          inputImage  = resizedImage ;
        }
      } else {
        strncpy(newTask->errorMessage, readers[0].second->getLastErrorMessage(), TASK_ERROR_MSG_MAX_LEN) ;
        char message [1024*2] ;
        int offset = snprintf(message, sizeof(message)/sizeof(char),
                              "could not read the header of image '%s'", newTask->name.c_str()) ;
        if (strlen(newTask->errorMessage) > 0) {
          snprintf(message + offset, sizeof(message)/sizeof(char) - offset,
                   " [%s]", newTask->errorMessage) ;
        }
        mexWarnMsgTxt(message) ;
      }
      tasks.push_back(newTask) ;
    }
    tasksMutex.unlock() ;
    tasksCondition.notify_all() ;
  }

  // done if prefetching only
  if (prefetch) { return ; }

  // return
  out[OUT_IMAGES] = mxCreateCellArray(mxGetNumberOfDimensions(in[IN_FILENAMES]),
                                      mxGetDimensions(in[IN_FILENAMES])) ;

  for (int t = 0 ; t < tasks.size() ; ++t) {
    tasksMutex.lock() ;
    while (!tasks[t]->done) {
      completedCondition.wait(tasksMutex);
    }
    ImageBuffer & image = tasks[t]->resizedImage ;
    tasksMutex.unlock() ;

    if (tasks[t]->error == vl::VLE_Success) {
      vl::ImageShape const & shape = image.getShape() ;
      mwSize dimensions [3] = {
        (mwSize)shape.height,
        (mwSize)shape.width,
        (mwSize)shape.depth} ;
      mwSize dimensions_ [3] = {0} ;
      mxArray * image_array = mxCreateNumericArray(3, dimensions_, mxSINGLE_CLASS, mxREAL) ;
      mxSetDimensions(image_array, dimensions, 3) ;
      mxSetData(image_array, image.relinquishMemory()) ;
      mxSetCell(out[OUT_IMAGES], t, image_array) ;
    } else {
      strncpy(tasks[t]->errorMessage, readers[0].second->getLastErrorMessage(), TASK_ERROR_MSG_MAX_LEN) ;
      char message [1024*2] ;
      int offset = snprintf(message, sizeof(message)/sizeof(char),
                            "could not read image '%s'", tasks[t]->name.c_str()) ;
      if (strlen(tasks[t]->errorMessage) > 0) {
        snprintf(message + offset, sizeof(message)/sizeof(char) - offset,
                 " [%s]", tasks[t]->errorMessage) ;
      }
      mexWarnMsgTxt(message) ;
    }
  }
  flush_tasks() ;
}

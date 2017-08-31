// @file imread.hpp
// @brief Image reader
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl__imread__
#define __vl__imread__

#include "data.hpp"

namespace vl {

#define VL_IMAGE_ERROR_MSG_MAX_LENGTH 256

  struct ImageShape
  {
    size_t height ;
    size_t width ;
    size_t depth ;

    ImageShape() ;
    ImageShape(size_t height, size_t width, size_t depth) ;
    ImageShape(ImageShape const & im) ;
    ImageShape & operator = (ImageShape const & im) ;
    bool operator == (ImageShape const & im) ;

    size_t getNumElements() const ;
    void clear() ;
  } ;

  class Image
  {
  public:
    Image() ;
    Image(Image const & im) ;
    Image(ImageShape const & shape, float * memory = NULL) ;
    ImageShape const & getShape() const ;
    float * getMemory() const ;
    void clear() ;

  protected:
    ImageShape shape ;
    float * memory ;
  } ;

  class ImageReader
  {
  public:
    ImageReader() ;
    ~ImageReader() ;
    vl::ErrorCode readShape(ImageShape & image, char const * fileName) ;
    vl::ErrorCode readPixels(float * memory, char const * fileName) ;
    char const * getLastErrorMessage() const ;

  private:
    class Impl ;
    Impl * impl ;
  } ;
}

#endif

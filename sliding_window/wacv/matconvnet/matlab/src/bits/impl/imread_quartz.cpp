// @file imread_quartz.cpp
// @brief Image reader based on Apple Quartz (Core Graphics).
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../imread.hpp"
#include "imread_helpers.hpp"

#import <ImageIO/ImageIO.h>
#include <algorithm>

#include <iostream>

/* ---------------------------------------------------------------- */
/*                                     Quartz reader implementation */
/* ---------------------------------------------------------------- */

#define check(x) \
if (!(x)) { error = vl::VLE_Unknown ; goto done ; }

#define ERR_MSG_MAX_LEN 1024

struct vl::ImageReader::Impl
{
  char lastErrorMessage [ERR_MSG_MAX_LEN] ;
  Impl() { lastErrorMessage[0] = 0 ; }
} ;

vl::ImageReader::ImageReader()
: impl(new Impl())
{ }

vl::ImageReader::~ImageReader()
{
  delete(impl) ;
}

const char * vl::ImageReader::getLastErrorMessage() const
{
  return impl->lastErrorMessage ;
}

vl::ErrorCode
vl::ImageReader::readPixels(float * memory, const char * fileName)
{
  vl::ErrorCode error = vl::VLE_Success ;

  // intermediate buffer
  char unsigned * pixels = NULL ;
  int bytesPerPixel ;
  int bytesPerRow ;

  // Core graphics
  CGBitmapInfo bitmapInfo ;
  CFURLRef url = NULL ;
  CGImageSourceRef imageSourceRef = NULL ;
  CGImageRef imageRef = NULL ;
  CGContextRef contextRef = NULL ;
  CGColorSpaceRef sourceColorSpaceRef = NULL ;
  CGColorSpaceRef colorSpaceRef = NULL ;

  // initialize the image as null
  ImageShape shape ;
  shape.width = 0 ;
  shape.height = 0 ;
  shape.depth = 0 ;

  // get file
  url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, (const UInt8 *)fileName, strlen(fileName), false) ;
  check(url) ;

  // get image source from file
  imageSourceRef = CGImageSourceCreateWithURL(url, NULL) ;
  check(imageSourceRef) ;

  // get image from image source
  imageRef = CGImageSourceCreateImageAtIndex(imageSourceRef, 0, NULL);
  check(imageRef) ;

  sourceColorSpaceRef = CGImageGetColorSpace(imageRef) ;
  check(sourceColorSpaceRef) ;

  shape.width = CGImageGetWidth(imageRef);
  shape.height = CGImageGetHeight(imageRef);
  shape.depth = CGColorSpaceGetNumberOfComponents(sourceColorSpaceRef) ;
  check(shape.depth == 1 || shape.depth == 3) ;

  // decode image to L (8 bits per pixel) or RGBA (32 bits per pixel)
  switch (shape.depth) {
    case 1:
      colorSpaceRef = CGColorSpaceCreateDeviceGray();
      bytesPerPixel = 1 ;
      bitmapInfo = kCGImageAlphaNone ;
      break ;

    case 3:
      colorSpaceRef = CGColorSpaceCreateDeviceRGB();
      bytesPerPixel = 4 ;
      bitmapInfo = kCGImageAlphaPremultipliedLast || kCGBitmapByteOrder32Big ;
      /* this means
       pixels[0] = R
       pixels[1] = G
       pixels[2] = B
       pixels[3] = A
       */
      break ;
  }
  check(colorSpaceRef) ;

  bytesPerRow = shape.width * bytesPerPixel ;
  pixels = (char unsigned*)malloc(shape.height * bytesPerRow) ;
  check(pixels) ;

  contextRef = CGBitmapContextCreate(pixels,
                                     shape.width, shape.height,
                                     8, bytesPerRow,
                                     colorSpaceRef,
                                     bitmapInfo) ;
  check(contextRef) ;

  CGContextDrawImage(contextRef, CGRectMake(0, 0, shape.width, shape.height), imageRef);

  // copy pixels to MATLAB format
  {
    Image image(shape, memory) ;
    switch (shape.depth) {
      case 3:
        vl::impl::imageFromPixels<impl::pixelFormatRGBA>(image, pixels, shape.width * bytesPerPixel) ;
        break ;
      case 1:
        vl::impl::imageFromPixels<impl::pixelFormatL>(image, pixels, shape.width * bytesPerPixel) ;
        break ;
    }
  }

done:
  if (pixels) { free(pixels) ; }
  if (contextRef) { CFRelease(contextRef) ; }
  if (colorSpaceRef) { CFRelease(colorSpaceRef) ; }
  if (imageRef) { CFRelease(imageRef) ; }
  if (imageSourceRef) { CFRelease(imageSourceRef) ; }
  if (url) { CFRelease(url) ; }
  return error ;
}

vl::ErrorCode
vl::ImageReader::readShape(vl::ImageShape & shape, const char * fileName)
{
  vl::ErrorCode error = vl::VLE_Success ;

  // intermediate buffer
  char unsigned * rgba = NULL ;

  // Core graphics
  CFURLRef url = NULL ;
  CGImageSourceRef imageSourceRef = NULL ;
  CGImageRef imageRef = NULL ;
  CGColorSpaceRef sourceColorSpaceRef = NULL ;

  // initialize the image as null
  shape.clear() ;

  // get file
  url = CFURLCreateFromFileSystemRepresentation(kCFAllocatorDefault, (const UInt8 *)fileName, strlen(fileName), false) ;
  check(url) ;

  // get image source from file
  imageSourceRef = CGImageSourceCreateWithURL(url, NULL) ;
  check(imageSourceRef) ;

  // get image from image source
  imageRef = CGImageSourceCreateImageAtIndex(imageSourceRef, 0, NULL);
  check(imageRef) ;

  sourceColorSpaceRef = CGImageGetColorSpace(imageRef) ;
  check(sourceColorSpaceRef) ;

  shape.width = CGImageGetWidth(imageRef);
  shape.height = CGImageGetHeight(imageRef);
  shape.depth = CGColorSpaceGetNumberOfComponents(sourceColorSpaceRef) ;
  check(shape.depth == 1 || shape.depth == 3) ;

done:
  if (imageRef) { CFRelease(imageRef) ; }
  if (imageSourceRef) { CFRelease(imageSourceRef) ; }
  if (url) { CFRelease(url) ; }
  return error ;
}

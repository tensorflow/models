// @file imread_libjpeg.cpp
// @brief Image reader based on libjpeg.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../imread.hpp"
#include "imread_helpers.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>
extern "C" {
#include <jpeglib.h>
#include <setjmp.h>
}

/* ---------------------------------------------------------------- */
/*                                    LibJPEG reader implementation */
/* ---------------------------------------------------------------- */

#define ERR_MSG_MAX_LEN 1024

class vl::ImageReader::Impl
{
public:
  Impl() ;
  ~Impl() ;
  struct jpeg_error_mgr jpegErrorManager ; /* must be the first element */
  struct jpeg_decompress_struct decompressor ;
  jmp_buf onJpegError ;
  char jpegLastErrorMsg [JMSG_LENGTH_MAX] ;
  char lastErrorMessage [ERR_MSG_MAX_LEN] ;

  vl::ErrorCode readPixels(float * memory, char const * filename) ;
  vl::ErrorCode readShape(vl::ImageShape & shape, char const * filename) ;

  static void reader_jpeg_error (j_common_ptr cinfo)
  {
    vl::ImageReader::Impl* self = (vl::ImageReader::Impl*) cinfo->err ;
    (*(cinfo->err->format_message)) (cinfo, self->jpegLastErrorMsg) ;
    longjmp(self->onJpegError, 1) ;
  }
} ;

vl::ImageReader::Impl::Impl()
{
  lastErrorMessage[0] = 0 ;
  decompressor.err = jpeg_std_error(&jpegErrorManager) ;
  jpegErrorManager.error_exit = reader_jpeg_error ;
  jpeg_create_decompress(&decompressor) ;
}

vl::ImageReader::Impl::~Impl()
{
  jpeg_destroy_decompress(&decompressor) ;
}

vl::ErrorCode
vl::ImageReader::Impl::readPixels(float * memory, char const * filename)
{
  vl::ErrorCode error = vl::VLE_Success ;
  char unsigned * pixels = NULL ;
  JSAMPARRAY scanlines = NULL ;
  bool requiresAbort = false ;

  /* initialize the image as null */
  ImageShape shape ;

  /* open file */
  FILE* fp = fopen(filename, "r") ;
  if (fp == NULL) {
    error = vl::VLE_Unknown ;
    std::snprintf(lastErrorMessage,  sizeof(lastErrorMessage),
                  "imread_libjpeg: unable to open %s", filename) ;
    return error ;
  }

  /* handle LibJPEG errors */
  if (setjmp(onJpegError)) {
    requiresAbort = true;
    error = vl::VLE_Unknown ;
    std::snprintf(lastErrorMessage,  sizeof(lastErrorMessage),
                  "libjpeg: %s", jpegLastErrorMsg) ;
    goto done ;
  }

  /* set which file to read */
  jpeg_stdio_src(&decompressor, fp);

  /* read image metadata */
  jpeg_read_header(&decompressor, TRUE) ;
  requiresAbort = true ;

  /* figure out if the image is grayscale (depth = 1) or color (depth = 3) */
  decompressor.quantize_colors = FALSE ;
  if (decompressor.jpeg_color_space == JCS_GRAYSCALE) {
    shape.depth = 1 ;
    decompressor.out_color_space = JCS_GRAYSCALE ;
  }  else {
    shape.depth = 3 ;
    decompressor.out_color_space = JCS_RGB ;
  }

  /* get the output dimension */
  jpeg_calc_output_dimensions(&decompressor) ;
  shape.width = decompressor.output_width ;
  shape.height = decompressor.output_height ;

  /* allocate scaline buffer */
  pixels = (char unsigned*)malloc(sizeof(char) * shape.width * shape.height * shape.depth) ;
  if (pixels == NULL) {
    error = vl::VLE_Unknown ;
    goto done ;
  }
  scanlines = (char unsigned**)malloc(sizeof(char*) * shape.height) ;
  if (scanlines == NULL) {
    error = vl::VLE_Unknown ;
    goto done ;
  }
  for (int y = 0 ; y < shape.height ; ++y) {
    scanlines[y] = pixels + shape.depth * shape.width * y ;
  }

  /* decompress each scanline and transpose the result into MATLAB format */
  jpeg_start_decompress(&decompressor);
  while(decompressor.output_scanline < shape.height) {
    jpeg_read_scanlines(&decompressor,
                        scanlines + decompressor.output_scanline,
                        shape.height - decompressor.output_scanline);
  }
  {
    Image image(shape, memory) ;
    switch (shape.depth) {
    case 3 : vl::impl::imageFromPixels<impl::pixelFormatRGB>(image, pixels, shape.width*3) ; break ;
    case 1 : vl::impl::imageFromPixels<impl::pixelFormatL>(image, pixels, shape.width*1) ; break ;
    default : error = vl::VLE_Unknown ; goto done ;
    }
    jpeg_finish_decompress(&decompressor) ;
    requiresAbort = false ;
  }

done:
  if (requiresAbort) { jpeg_abort((j_common_ptr)&decompressor) ; }
  if (scanlines) free(scanlines) ;
  if (pixels) free(pixels) ;
  fclose(fp) ;
  return error ;
}

vl::ErrorCode
vl::ImageReader::Impl::readShape(vl::ImageShape & shape, char const * filename)
{
  vl::ErrorCode error = vl::VLE_Success ;

  int row_stride ;
  const int blockSize = 32 ;
  char unsigned * pixels = NULL ;
  JSAMPARRAY scanlines ;
  bool requiresAbort = false ;

  // open file
  FILE* fp = fopen(filename, "r") ;
  if (fp == NULL) {
    error = vl::VLE_Unknown ;
    std::snprintf(lastErrorMessage,  sizeof(lastErrorMessage),
                  "imread_libjpeg: unable to open %s", filename) ;
    return error ;
  }

  // handle LibJPEG errors
  if (setjmp(onJpegError)) {
    error = vl::VLE_Unknown ;
    std::snprintf(lastErrorMessage,  sizeof(lastErrorMessage),
                  "libjpeg: %s", jpegLastErrorMsg) ;
    goto done ;
  }

  /* set which file to read */
  jpeg_stdio_src(&decompressor, fp);

  /* read image metadata */
  jpeg_read_header(&decompressor, TRUE) ;
  requiresAbort = true ;

  /* figure out if the image is grayscale (depth = 1) or color (depth = 3) */
  if (decompressor.jpeg_color_space == JCS_GRAYSCALE) {
    shape.depth = 1 ;
  }  else {
    shape.depth = 3 ;
  }

  /* get the output dimension (this may differ from the input if we were to scale the image) */
  jpeg_calc_output_dimensions(&decompressor) ;
  shape.width = decompressor.output_width ;
  shape.height = decompressor.output_height ;

done:
  if (requiresAbort) { jpeg_abort((j_common_ptr)&decompressor) ; }
  fclose(fp) ;
  return error ;
}

/* ---------------------------------------------------------------- */
/*                                                   LibJPEG reader */
/* ---------------------------------------------------------------- */

vl::ImageReader::ImageReader()
: impl(NULL)
{
  impl = new vl::ImageReader::Impl() ;
}

vl::ImageReader::~ImageReader()
{
  delete impl ;
}

vl::ErrorCode
vl::ImageReader::readPixels(float * memory, char const * filename)
{
  return impl->readPixels(memory, filename) ;
}

vl::ErrorCode
vl::ImageReader::readShape(vl::ImageShape & shape, char const * filename)
{
  return impl->readShape(shape, filename) ;
}

char const *
vl::ImageReader::getLastErrorMessage() const
{
  return impl->lastErrorMessage  ;
}

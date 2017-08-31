// @file imread_gdiplus.cpp
// @brief Image reader based on Windows GDI+.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "../imread.hpp"
#include "imread_helpers.hpp"

#include <windows.h>
#include <gdiplus.h>
#include <algorithm>

#include <mex.h>

#include "../mexutils.h"

using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

/* ---------------------------------------------------------------- */
/*                                       GDI+ reader implementation */
/* ---------------------------------------------------------------- */

#define check(x) \
if (!x) { image.error = 1 ; goto done ; }

#define ERR_MAX_LEN 1024

static const char * GdiErrMsg[] = {
  "Ok",
  "GenericError",
  "InvalidParameter (or image file does not exist)",
  "OutOfMemory",
  "ObjectBusy",
  "InsufficientBuffer",
  "NotImplemented",
  "Win32Error",
  "WrongState",
  "Aborted",
  "FileNotFound",
  "ValueOverflow",
  "AccessDenied",
  "UnknownImageFormat",
  "FontFamilyNotFound",
  "FontStyleNotFound",
  "NotTrueTypeFont",
  "UnsupportedGdiplusVersion",
  "GdiplusNotInitialized",
  "PropertyNotFound",
  "PropertyNotSupported",
  "ProfileNotFound"
};

class vl::ImageReader::Impl
{
public:
  Impl() ;
  ~Impl() ;
  GdiplusStartupInput gdiplusStartupInput;
  ULONG_PTR           gdiplusToken;
  vl::ErrorCode readPixels(float * memory, char const * filename) ;
  vl::ErrorCode readShape(vl::ImageShape & shape, char const * filename) ;
  char lastErrorMessage[ERR_MAX_LEN];
} ;

vl::ImageReader::Impl::Impl()
{
  lastErrorMessage[0] = 0;
  GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
}

vl::ImageReader::Impl::~Impl()
{
  GdiplusShutdown(gdiplusToken);
}

static void getImagePropertiesHelper(vl::ImageShape & shape, Gdiplus::Bitmap & bitmap)
{
  bool grayscale = (bool)(bitmap.GetFlags() & ImageFlagsColorSpaceGRAY) ;
  shape.width = bitmap.GetWidth() ;
  shape.height = bitmap.GetHeight() ;
  shape.depth = grayscale ? 1 : 3 ;
}

vl::ErrorCode
vl::ImageReader::Impl::readPixels(float * memory, char const * filename)
{
  vl::ErrorCode error = vl::VLE_Success ;
  vl::ImageShape shape ;
  Status status ;
  Rect rect ;

  wchar_t filenamew [1024*4] ;
  size_t n = 0 ;
  mbstowcs_s(&n, filenamew, sizeof(filenamew)/sizeof(wchar_t), filename, _TRUNCATE);

  BitmapData data ;
  Bitmap bitmap(filenamew);
  status = bitmap.GetLastStatus();
  if (status != Ok) {
    snprintf(lastErrorMessage,  sizeof(lastErrorMessage),
                  "gdi+: %s", GdiErrMsg[(int)status]) ;
    error = vl::VLE_Unknown ;
    mexPrintf(lastErrorMessage) ;
    return error;
  }

  getImagePropertiesHelper(shape, bitmap) ;

  // get the pixels
  // by default let GDIplus read as 32bpp RGB, unless the image is indexed 8bit grayscale
  Image image(shape, memory);

  Gdiplus::PixelFormat targetPixelFormat = PixelFormat32bppRGB ;

  if (shape.depth == 1) {
    Gdiplus::PixelFormat gdiPixelFormat = bitmap.GetPixelFormat();
    if (gdiPixelFormat == PixelFormat8bppIndexed) {
      int paletteSize = bitmap.GetPaletteSize() ;
      Gdiplus::ColorPalette * palette =
        reinterpret_cast<Gdiplus::ColorPalette *>(new char[paletteSize]) ;
      bitmap.GetPalette(palette, paletteSize) ;
      bool isStandardGrayscale = (palette->Count == 256) ;
      if (isStandardGrayscale) {
        for (int c = 0 ; c < 256 ; ++c) {
          isStandardGrayscale &= palette->Entries[c] == Color::MakeARGB(255,c,c,c) ;
          // mexPrintf("c%d: %d %d\n",c, palette->Entries[c], Color::MakeARGB(255,c,c,c)) ;
        }
      }
      delete[] reinterpret_cast<char *>(palette) ;
      if (isStandardGrayscale) {
        targetPixelFormat = PixelFormat8bppIndexed ;
      }
    }
  }

  rect = Rect(0, 0, (int)shape.width, (int)shape.height);
  status = bitmap.LockBits(&rect,
                           ImageLockModeRead,
                           targetPixelFormat,
                           &data) ;
  if (status != Ok) {
    snprintf(lastErrorMessage,  sizeof(lastErrorMessage),
                  "gdi+: %s", GdiErrMsg[(int)status]) ;
    error = vl::VLE_Unknown;
    return error;
  }

  // copy RGB to MATLAB format
  switch (shape.depth) {
	case 3:
	  vl::impl::imageFromPixels<impl::pixelFormatBGRA>(image, (char unsigned const *)data.Scan0, data.Stride) ;
    break ;
	case 1:
    switch (targetPixelFormat) {
    case PixelFormat8bppIndexed:
      vl::impl::imageFromPixels<impl::pixelFormatL>(image, (char unsigned const *)data.Scan0, data.Stride) ;
      break ;
    default:
      vl::impl::imageFromPixels<impl::pixelFormatBGRAasL>(image, (char unsigned const *)data.Scan0, data.Stride) ;
      break ;
    }
	  break ;
  }

  bitmap.UnlockBits(&data) ;

  return error;
}

vl::ErrorCode
vl::ImageReader::Impl::readShape(vl::ImageShape & shape, char const * filename)
{
  vl::ErrorCode error = vl::VLE_Success ;
  Status status ;
  wchar_t filenamew [1024*4] ;
  size_t n = 0 ;
  mbstowcs_s(&n, filenamew, sizeof(filenamew)/sizeof(wchar_t), filename, _TRUNCATE);

  Bitmap bitmap(filenamew);
  status = bitmap.GetLastStatus();
  if (status != Ok) {
    snprintf(lastErrorMessage,  sizeof(lastErrorMessage),
                  "gdi+: %s", GdiErrMsg[(int)status]) ;
    error = vl::VLE_Unknown ;
    return error;
  }

  getImagePropertiesHelper(shape, bitmap) ;
}

/* ---------------------------------------------------------------- */
/*                                                      GDI+ reader */
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
  return impl->lastErrorMessage ;
}

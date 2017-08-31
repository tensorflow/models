// @file imread_helpers.cpp
// @brief Image reader helper functions.
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#ifdef __SSSE3__
#include <tmmintrin.h>
#endif

#include "../data.hpp"

namespace vl { namespace impl {

  enum pixelFormatId {
    pixelFormatL,
    pixelFormatRGB,
    pixelFormatRGBA,
    pixelFormatBGR,
    pixelFormatBGRA,
    pixelFormatBGRAasL
  };

#ifndef __SSSE3__
#ifdef _MSC_VER
#pragma message ( "SSSE3 instruction set not enabled. Using slower image conversion routines." )
#else
#warning "SSSE3 instruction set not enabled. Using slower image conversion routines."
#endif

  template<int pixelFormat> void
  imageFromPixels(vl::Image & image, char unsigned const * rgb, int rowStride)
  {
    vl::ImageShape const & shape = image.getShape() ;
    int blockSizeX ;
    int blockSizeY ;
    int pixelStride ;
    int imagePlaneStride = (int)shape.width * (int)shape.height ;
    switch (pixelFormat) {
      case pixelFormatL:
        pixelStride = 1 ;
        blockSizeX = 16 ;
        blockSizeY = 4 ;
        break ;
      case pixelFormatBGR:
      case pixelFormatRGB:
        pixelStride = 3 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        break ;
      case pixelFormatRGBA:
      case pixelFormatBGRA:
      case pixelFormatBGRAasL:
        pixelStride = 4 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        break ;
      default:
        assert(false) ;
    }

    // we pull out these values as otherwise the compiler
    // will assume that the reference &image can be aliased
    // and recompute silly multiplications in the inner loop

    float * const  __restrict imageMemory = image.getMemory() ;
    int const imageHeight = (int)shape.height ;
    int const imageWidth = (int)shape.width ;

    for (int x = 0 ; x < imageWidth ; x += blockSizeX) {
      float * __restrict imageMemoryX = imageMemory + x * imageHeight ;
      int bsx = (std::min)(imageWidth - x, blockSizeX) ;

      for (int y = 0 ; y < imageHeight ; y += blockSizeY) {
        int bsy = (std::min)(imageHeight - y, blockSizeY) ;
        float * __restrict r ;
        float * rend ;
        for (int dx = 0 ; dx < bsx ; ++dx) {
          char unsigned const * __restrict pixel = rgb + y * rowStride + (x + dx) * pixelStride ;
          r = imageMemoryX + y + dx * imageHeight ;
          rend = r + bsy ;
          while (r != rend) {
            switch (pixelFormat) {
              case pixelFormatRGBA:
              case pixelFormatRGB:
                r[0 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[2 * imagePlaneStride] = (float) pixel[2] ;
                break ;
              case pixelFormatBGR:
              case pixelFormatBGRA:
                r[2 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[0 * imagePlaneStride] = (float) pixel[2] ;
                break;
              case pixelFormatBGRAasL:
              case pixelFormatL:
                r[0] = (float) pixel[0] ;
                break ;
            }
            r += 1 ;
            pixel += rowStride ;
          }
        }
      }
    }
  }

#else
#ifdef _MSC_VER
#pragma message ( "SSSE3 instruction set enabled." )
#endif
  /* SSSE3 optimised version */

  template<int pixelFormat> void
  imageFromPixels(vl::Image & image, char unsigned const * rgb, int rowStride)
  {
    vl::ImageShape const & shape = image.getShape() ;
    int blockSizeX ;
    int blockSizeY ;
    int pixelStride ;
    int imagePlaneStride = (int)shape.width * (int)shape.height ;
    __m128i shuffleRgb ;
    __m128i const shuffleL = _mm_set_epi8(0xff, 0xff, 0xff,  3,
                                          0xff, 0xff, 0xff,  2,
                                          0xff, 0xff, 0xff,  1,
                                          0xff, 0xff, 0xff,  0) ;
    __m128i const mask = _mm_set_epi32(0xff, 0xff, 0xff, 0xff) ;

    switch (pixelFormat) {
      case pixelFormatL:
        pixelStride = 1 ;
        blockSizeX = 16 ;
        blockSizeY = 4 ;
        break ;
      case pixelFormatBGR:
      case pixelFormatRGB:
        pixelStride = 3 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        assert(shape.depth == 3) ;
        break ;
      case pixelFormatRGBA:
      case pixelFormatBGRA:
      case pixelFormatBGRAasL:
        pixelStride = 4 ;
        blockSizeX = 4 ;
        blockSizeY = 4 ;
        assert(shape.depth == 3) ;
        break ;
      default:
        assert(false) ;
    }

    switch (pixelFormat) {
      case pixelFormatL:
        break ;

      case pixelFormatRGB:
        shuffleRgb = _mm_set_epi8(0xff, 11, 10,  9,
                                  0xff,  8,  7,  6,
                                  0xff,  5,  4,  3,
                                  0xff,  2,  1,  0) ;
        break ;

      case pixelFormatRGBA:
        shuffleRgb = _mm_set_epi8(0xff, 14, 13, 12,
                                  0xff, 10,  9,  8,
                                  0xff,  6,  5,  4,
                                  0xff,  2,  1,  0) ;
        break ;

      case pixelFormatBGR:
        shuffleRgb = _mm_set_epi8(0xff,  9, 10, 11,
                                  0xff,  6,  7,  8,
                                  0xff,  3,  4,  4,
                                  0xff,  0,  1,  2) ;
        break ;

      case pixelFormatBGRA:
        shuffleRgb = _mm_set_epi8(0xff, 12, 13, 14,
                                  0xff,  8,  9, 10,
                                  0xff,  4,  5,  6,
                                  0xff,  0,  1,  2) ;
        break ;

      case pixelFormatBGRAasL:
        shuffleRgb = _mm_set_epi8(0xff, 0xff, 0xff, 12,
                                  0xff, 0xff, 0xff, 8,
                                  0xff, 0xff, 0xff, 4,
                                  0xff, 0xff, 0xff, 0) ;
        break ;
    }

    // we pull out these values as otherwise the compiler
    // will assume that the reference &image can be aliased
    // and recompute silly multiplications in the inner loop
    float *  const __restrict imageMemory = image.getMemory() ;
    int const imageHeight = (int)shape.height ;
    int const imageWidth = (int)shape.width ;

    for (int x = 0 ; x < imageWidth ; x += blockSizeX) {
      int y = 0 ;
      float * __restrict imageMemoryX = imageMemory + x * imageHeight ;
      int bsx = (std::min)(imageWidth - x, blockSizeX) ;
      if (bsx < blockSizeX) goto boundary ;

      for ( ; y < imageHeight - blockSizeY + 1 ; y += blockSizeY) {
        char unsigned const * __restrict pixel = rgb + y * rowStride + x * pixelStride ;
        float * __restrict r = imageMemoryX + y ;
        __m128i p0, p1, p2, p3, T0, T1, T2, T3 ;

        /* convert a blockSizeX x blockSizeY block in the input image */
        switch (pixelFormat) {
          case pixelFormatRGB :
          case pixelFormatRGBA :
          case pixelFormatBGR :
          case pixelFormatBGRA :
          case pixelFormatBGRAasL :
            // load 4x4 RGB pixels
            p0 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;
            p1 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;
            p2 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;
            p3 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)pixel), shuffleRgb) ; pixel += rowStride ;

            // transpose pixels as 32-bit integers (see also below)
            T0 = _mm_unpacklo_epi32(p0, p1);
            T1 = _mm_unpacklo_epi32(p2, p3);
            T2 = _mm_unpackhi_epi32(p0, p1);
            T3 = _mm_unpackhi_epi32(p2, p3);
            p0 = _mm_unpacklo_epi64(T0, T1);
            p1 = _mm_unpackhi_epi64(T0, T1);
            p2 = _mm_unpacklo_epi64(T2, T3);
            p3 = _mm_unpackhi_epi64(T2, T3);

            // store r
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p0, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p1, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p2, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p3, mask))) ;

            if (pixelFormat == pixelFormatBGRAasL) break ;

            // store g
            r += (imageWidth - 3) * imageHeight ;
            p0 = _mm_srli_epi32 (p0, 8) ;
            p1 = _mm_srli_epi32 (p1, 8) ;
            p2 = _mm_srli_epi32 (p2, 8) ;
            p3 = _mm_srli_epi32 (p3, 8) ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p0, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p1, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p2, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p3, mask))) ;

            // store b
            r += (imageWidth - 3) * imageHeight ;
            p0 = _mm_srli_epi32 (p0, 8) ;
            p1 = _mm_srli_epi32 (p1, 8) ;
            p2 = _mm_srli_epi32 (p2, 8) ;
            p3 = _mm_srli_epi32 (p3, 8) ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p0, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p1, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p2, mask))) ; r += imageHeight ;
            _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_and_si128(p3, mask))) ;
            break ;

          case pixelFormatL:
            // load 4x16 L pixels
            p0 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;
            p1 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;
            p2 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;
            p3 = _mm_loadu_si128((__m128i*)pixel) ; pixel += rowStride ;

            /*
             Pixels are collected in little-endian order: the first pixel
             is at the `right' (least significant byte of p0:

             p[0] = a, p[1] = b, ...

             p0: [ ... | ... | ... | d c b a ]
             p1: [ ... | ... | ... | h g f e ]
             p2: [ ... | ... | ... | l k j i ]
             p3: [ ... | ... | ... | p o n m ]

             The goal is to transpose four 4x4 subblocks in the
             4 x 16 pixel array. The first step interlaves individual
             pixels in p0 and p1:

             T0: [ ... | ... | h d g c | f b e a ]
             T1: [ ... | ... | p l o k | n j m i ]
             T2: [ ... | ... | ... | ... ]
             T3: [ ... | ... | ... | ... ]

             The second step interleaves groups of two pixels:

             p0: [pl hd | ok gc | nj fb | mi ea] (pixels in the rightmost 4x4 subblock)
             p1: ...
             p2: ...
             p3: ...

             The third step interlevaes groups of four pixels:

             T0: [ ... | njfb | ... | miea ]
             T1: ...
             T2: ...
             T3: ...

             The last step interleaves groups of eight pixels:

             p0: [ ... | ... | ... | miea ]
             p1: [ ... | ... | ... | njfb ]
             p2: [ ... | ... | ... | okgc ]
             p3: [ ... | ... | ... | dklp ]

             */

            T0 = _mm_unpacklo_epi8(p0, p1);
            T1 = _mm_unpacklo_epi8(p2, p3);
            T2 = _mm_unpackhi_epi8(p0, p1);
            T3 = _mm_unpackhi_epi8(p2, p3);
            p0 = _mm_unpacklo_epi16(T0, T1);
            p1 = _mm_unpackhi_epi16(T0, T1);
            p2 = _mm_unpacklo_epi16(T2, T3);
            p3 = _mm_unpackhi_epi16(T2, T3);
            T0 = _mm_unpacklo_epi32(p0, p1);
            T1 = _mm_unpacklo_epi32(p2, p3);
            T2 = _mm_unpackhi_epi32(p0, p1);
            T3 = _mm_unpackhi_epi32(p2, p3);
            p0 = _mm_unpacklo_epi64(T0, T1);
            p1 = _mm_unpackhi_epi64(T0, T1);
            p2 = _mm_unpacklo_epi64(T2, T3);
            p3 = _mm_unpackhi_epi64(T2, T3);

            // store four 4x4 subblock
            for (int i = 0 ; i < 4 ; ++i) {
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p0, shuffleL))) ; r += imageHeight ;
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p1, shuffleL))) ; r += imageHeight ;
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p2, shuffleL))) ; r += imageHeight ;
              _mm_storeu_ps(r, _mm_cvtepi32_ps(_mm_shuffle_epi8(p3, shuffleL))) ; r += imageHeight ;
              p0 = _mm_srli_si128 (p0, 4) ;
              p1 = _mm_srli_si128 (p1, 4) ;
              p2 = _mm_srli_si128 (p2, 4) ;
              p3 = _mm_srli_si128 (p3, 4) ;
            }
            break ;
        }
      } /* next y */

    boundary:
      /* special case if there is not a full 4x4 block to process */
      for ( ; y < imageHeight ; y += blockSizeY) {
        int bsy = (std::min)(imageHeight - y, blockSizeY) ;
        float * __restrict r ;
        float * rend ;
        for (int dx = 0 ; dx < bsx ; ++dx) {
          char unsigned const * __restrict pixel = rgb + y * rowStride + (x + dx) * pixelStride ;
          r = imageMemoryX + y + dx * imageHeight ;
          rend = r + bsy ;
          while (r != rend) {
            switch (pixelFormat) {
              case pixelFormatRGBA:
              case pixelFormatRGB:
                r[0 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[2 * imagePlaneStride] = (float) pixel[2] ;
                break ;
              case pixelFormatBGR:
              case pixelFormatBGRA:
                r[2 * imagePlaneStride] = (float) pixel[0] ;
                r[1 * imagePlaneStride] = (float) pixel[1] ;
                r[0 * imagePlaneStride] = (float) pixel[2] ;
                break;
              case pixelFormatBGRAasL:
              case pixelFormatL:
                r[0] = (float) pixel[0] ;
                break ;
            }
            r += 1 ;
            pixel += rowStride ;
          }
        }
      }
    }
  }

#endif

  struct ImageResizeFilter
  {
    float * weights ;
    int * starts ;
    int filterSize ;
    enum FilterType { kBox, kBilinear, kBicubic, kLanczos2, kLanczos3 } ;

    ~ImageResizeFilter() {
      free(weights) ;
      free(starts) ;
    }

    ImageResizeFilter(size_t outputWidth, size_t inputWidth, size_t cropWidth, size_t cropOffset, FilterType filterType = kBilinear)
    {
      filterSize = 0 ;
      switch (filterType) {
        case kBox      : filterSize = 1 ; break ;
        case kBilinear : filterSize = 2 ; break ;
        case kBicubic  : filterSize = 4 ; break ;
        case kLanczos2 : filterSize = 4 ; break ;
        case kLanczos3 : filterSize = 6 ; break ;
      }

      /* 
       Find reverse mapping u = alpha v + beta where v is in the target
       domain and u in the source domain.
       */
      float alpha = (float)cropWidth / outputWidth ;
      float beta = 0.5f * (alpha - 1) + cropOffset ;
      float filterSupport = (float)filterSize ;

      /* 
       The filter is virtually applied in the target domain u. This is transferred
       to the input domain v. If the image is shrunk, the filter is
       spread over several input pixels, which is a simple form
       of antialisaing. Note that this could be switched off.
       */
      if (alpha > 1) {
        filterSupport *= alpha ;
        filterSize = (int)ceilf(filterSupport) ;
      }

      weights = (float*)calloc(filterSize * outputWidth, sizeof(float)) ;
      starts = (int*)malloc(sizeof(int) * outputWidth) ;
      float * filter = weights ;

      /* the filter extends in the interval (-filterSize/2, filterSize/2)
        (extrema not included)
       */
      for (int v = 0 ; v < outputWidth ; ++v, filter += filterSize) {
        /* y(v) = sum_k h(k - u) x(k),  u = alpha * v + beta */
        /* for uniformity we assume that the sum is non-zero for
         u - filterSize/2 <= k < u + filterSize/2
         so that there are always filerWidth elements to sum on */
        float u = alpha * v + beta ;
        float mass = 0 ;
        int skip = filterSize ;

        starts[v] = (int)std::ceil(u - filterSupport / 2) ;

        for (int r = 0 ; r < filterSize ; ++r) {
          int k = r + starts[v] ;
          float h ;
          float delta = u - k ;
          if (alpha > 1) {
            delta /= alpha ;
          }
          switch (filterType) {
            case kBox:
              h = (float)((-0.5f <= delta) & (delta < 0.5f)) ;
              break ;
            case kBilinear:
              h = (std::max)(0.0f, 1.0f - fabsf(delta)) ;
              break ;
            case kBicubic: {
              float adelta = fabsf(delta) ;
              float adelta2 = adelta*adelta ;
              float adelta3 = adelta*adelta2 ;
              if (adelta <= 1.0f) {
                h = 1.5f * adelta3 - 2.5f * adelta2 + 1.f ;
              } else if (adelta <= 2.0f) {
                h = -0.5f * adelta3 + 2.5f * adelta2 - 4.f * adelta + 2.f ;
              } else {
                h = 0.f ;
              }
              break ;
            }
            case kLanczos2: {
              if (fabsf(delta) < 2) {
                const float eps = 1e-5f ;
                h = (sin(VL_M_PI_F * delta) *
                     sin(VL_M_PI_F * delta / 2.f) + eps) /
                ((VL_M_PI_F*VL_M_PI_F * delta*delta / 2.f) + eps);
              } else {
                h = 0.f ;
              }
              break ;
            }
            case kLanczos3:
              if (fabsf(delta) < 3) {
                const float eps = 1e-5f ;
                h = (sin(VL_M_PI_F * delta) *
                     sin(VL_M_PI_F * delta / 3.f) + eps) /
                ((VL_M_PI_F*VL_M_PI_F * delta*delta / 3.f) + eps);
              } else {
                h = 0.f ;
              }
              break ;
            default:
              assert(false) ;
              break ;
          }
          {
            // MATLAB uses a slightly different method for resizing
            // the borders: it mirrors-pad them. This is a bit more
            // difficult to obtain with our data structure. Instead,
            // we repeat the first/last pixel.
            int q = r ;
            if (k < 0) {
              q = r - k ;
            } else if (k >= (signed)inputWidth) {
              q = r - (k - (signed)inputWidth + 1) ;
            }
            filter[q] += h ;
            mass += h ;
            if (h) {
              skip = (std::min)(skip, q) ;
            }
          }
        }
        {
          int r = 0 ;
          starts[v] += skip ;
          for (r = 0 ; r < filterSize - skip ; ++r) {
            filter[r] = filter[r + skip] / mass ;
          }
          for ( ;  r < filterSize ; ++r) {
            filter[r] = 0.f ;
          }
        }
      }
    }
  } ;

  inline void imageResizeVertical(float * output, float const * input,
                                  size_t outputHeight,
                                  size_t height, size_t width, size_t depth,
                                  size_t cropHeight,
                                  size_t cropOffset,
                                  bool flip = false,
                                  vl::impl::ImageResizeFilter::FilterType filterType = vl::impl::ImageResizeFilter::kBilinear)
  {
    ImageResizeFilter filters(outputHeight, height, cropHeight, cropOffset, filterType) ;
    int filterSize = filters.filterSize ;
    for (int d = 0 ; d < (int)depth ; ++d) {
      for (int x = 0 ; x < (int)width ; ++x) {
        for (int y = 0 ; y < (int)outputHeight ; ++y) {
          float z = 0 ;
          int begin = filters.starts[y] ;
          float const * weights = filters.weights + filterSize * y ;
          for (int k = begin ; k < begin + filterSize ; ++k) {
            float w = *weights++ ;
            if (w == 0.f) break ;
            //if ((0 <= k) & (k < (signed)height)) {
              z += input[k] * w ;
            //}
          }
          if (!flip) {
            output[x + y * width] = z ; // transpose
          } else {
            output[x + ((int)outputHeight - 1 - y) * width] = z ; // flip and transpose
          }
        }
        input += height ;
      }
      output += outputHeight * width ;
    }
  }

  inline void resizeImage(vl::Image & output, vl::Image const & input)
  {
    vl::ImageShape const & inputShape = input.getShape() ;
    vl::ImageShape const & outputShape = output.getShape() ;
    assert(outputShape.depth == inputShape.depth) ;
    float * temp = (float*)malloc(sizeof(float) * outputShape.height * inputShape.width * inputShape.depth) ;
    imageResizeVertical(temp, input.getMemory(),
                        outputShape.height,
                        inputShape.height, inputShape.width, inputShape.depth,
                        inputShape.height, 0) ;
    imageResizeVertical(output.getMemory(), temp,
                        outputShape.width,
                        inputShape.width, outputShape.height, inputShape.depth,
                        inputShape.width, 0) ;
    free(temp) ;
  }
  
} }

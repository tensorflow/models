// @file copy_cpu.cpp
// @brief Copy and other data operations (CPU)
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "copy.hpp"
#include <string.h>

namespace vl { namespace impl {

  template <typename type>
  struct operations<vl::VLDT_CPU, type>
  {
    typedef type data_type ;

    static vl::ErrorCode
    copy(data_type * dest,
         data_type const * src,
         size_t numElements)
    {
      memcpy(dest, src, numElements * sizeof(data_type)) ;
      return VLE_Success ;
    }

    static vl::ErrorCode
    fill(data_type * dest,
         size_t numElements,
         data_type value)
    {
      for (size_t k = 0 ; k < numElements ; ++k) {
        dest[k] = value ;
      }
      return VLE_Success ;
    }
  } ;

} }

template struct vl::impl::operations<vl::VLDT_CPU, float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::operations<vl::VLDT_CPU, double> ;
#endif



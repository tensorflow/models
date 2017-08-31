#ifndef COMPAT_H
#define COMPAT_H

#ifdef _MSC_VER
#define snprintf _snprintf
#define vsnprintf _vsnprintf
#define __func__ __FUNCTION__
#undef max
#undef min

#ifdef  _WIN64
typedef signed __int64 ssize_t;
#else
typedef signed int ssize_t;
#endif // _WIN64

#if _MSC_VER < 1700
#define false 0
#define true 1
#elif _MSC_VER > 1700
#include <stdbool.h>
#endif // _MSC_VER < 1700

#if _MSC_VER < 1800
// Add some missing functions from C99
#define isnan(x) _isnan(x)
#define isinf(x) (!_finite(x))
#define round(x) (x >= 0.0 ? (double)(int)(x + 0.5) : (double)(int)(x - 0.5))
#define roundf(x) (x >= 0.0f ? (float)(int)(x + 0.5f) : (float)(int)(x - 0.5f))
#endif

#endif // _MSC_VER


#endif // COMPAT_H

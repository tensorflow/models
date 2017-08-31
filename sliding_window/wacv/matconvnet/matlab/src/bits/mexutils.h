/** @file    mexutils.h
 ** @brief   MEX utilities
 ** @author  Andrea Vedaldi
 **/

/*
Copyright (C) 2007-16 Andrea Vedaldi and Brian Fulkerson.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef MEXUTILS_H
#define MEXUTILS_H

#include "mex.h"

#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>

#include <ctype.h>
#include <string.h>
#include <assert.h>

#include "impl/compat.h"

#define VL_INLINE static __inline

/** @brief Access MEX input argument */
#undef IN
#define IN(x) (in[IN_ ## x])

/** @brief Access MEX output argument */
#undef OUT
#define OUT(x) (out[OUT_ ## x])

/** @{
 ** @name Error handling
 **/

/** @brief Error codes */
typedef enum {
  VLMXE_Success = 0,
  VLMXE_Alloc,
  VLMXE_IllegalArgument,
  VLMXE_NotEnoughInputArguments,
  VLMXE_TooManyInputArguments,
  VLMXE_NotEnoughOutputArguments,
  VLMXE_TooManyOutputArguments,
  VLMXE_InvalidOption,
  VLMXE_InconsistentData,
  VLMXE_Execution,
  VLMXE_TimeOut,
  VLMXE_Undefined,
  VLMXE_NumCodes // must be last
} VLMXErrorCode ;

static char const * vlmxErrorMessageTable [] = {
  "Success",                 "Success.",
  "Alloc",                   "Allocation failed.",
  "InvalidArgument",         "Invalid argument.",
  "NotEnoughInputArguments", "Not enough input arguments.",
  "TooManyInputArguments",   "Too many input arguments.",
  "NotEnoughOutputArguments","Not enough output arguments.",
  "TooManyOutputArguments",  "Too many output arguments.",
  "InvalidOption",           "Invalid option.",
  "InconsistentData",        "Inconsistent data.",
  "execution",               "Execution error.",
  "Eimeout",                 "Timeout."
  "UndefinedError",          "Undefined error.",
} ;

static inline void
vlmxErrorHelper(bool isError, VLMXErrorCode errorCode,
                char const * errorMessage,  va_list args)
{
  char const * errorString ;
  char formattedErrorCode [512] ;
  char formattedErrorMessage [1024] ;

  if (errorCode < 0 || errorCode >= VLMXE_NumCodes) {
    errorCode = VLMXE_Undefined ;
  }

  errorString = vlmxErrorMessageTable[2*errorCode] ;
  errorMessage || (errorMessage = vlmxErrorMessageTable[2*errorCode+1]) ;

  snprintf(formattedErrorCode,
           sizeof(formattedErrorCode)/sizeof(char),
           "VLMX:%s", errorString) ;

  vsnprintf(formattedErrorMessage,
            sizeof(formattedErrorMessage)/sizeof(char),
            errorMessage, args) ;

  if (isError) {
    mexErrMsgIdAndTxt (formattedErrorCode, formattedErrorMessage) ;
  } else {
    mexWarnMsgIdAndTxt (formattedErrorCode, formattedErrorMessage) ;
  }
}

/** @brief Throw a MEX error
 ** @param errorCode error ID string.
 ** @param errorMessage error message C-style format string.
 ** @param ... format string arguments.
 **
 ** The function internally calls @c mxErrMsgTxtAndId, which causes
 ** the MEX file to abort.
 **/

#if defined(VL_COMPILER_GNUC) & ! defined(__DOXYGEN__)
static void __attribute__((noreturn))
#else
static void
#endif
vlmxError (VLMXErrorCode errorCode, char const * errorMessage, ...)
{
  va_list args ;
  va_start(args, errorMessage) ;
  vlmxErrorHelper(true, errorCode, errorMessage, args) ;
  va_end(args) ;
}

/** @brief Throw a MEX warning
 ** @param errorCode error ID string.
 ** @param errorMessage error message C-style format string.
 ** @param ... format string arguments.
 **
 ** The function internally calls @c mxWarnMsgTxtAndId.
 **/

static void
vlmxWarning (VLMXErrorCode errorCode, char const * errorMessage, ...)
{
  va_list args ;
  va_start(args, errorMessage) ;
  vlmxErrorHelper(false, errorCode, errorMessage, args) ;
  va_end(args) ;
}

/** @} */

/** @name Check for array attributes
 ** @{ */

/** @brief Check if a MATLAB array is of a prescribed class
 ** @param array MATLAB array.
 ** @param classId prescribed class of the array.
 ** @return ::true if the class is of the array is of the prescribed class.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE bool
vlmxIsOfClass (mxArray const * array, mxClassID classId)
{
  return mxGetClassID (array) == classId ;
}

/** @brief Check if a MATLAB array is real
 ** @param array MATLAB array.
 ** @return ::true if the array is real.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE bool
vlmxIsReal (mxArray const * array)
{
  return mxIsNumeric (array) && ! mxIsComplex (array) ;
}

/** @} */

/** @name Check for scalar, vector and matrix arrays
 ** @{ */

/** @brief Check if a MATLAB array is scalar
 ** @param array MATLAB array.
 ** @return ::true if the array is scalar.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE bool
vlmxIsScalar (mxArray const * array)
{
  return (! mxIsSparse (array)) && (mxGetNumberOfElements (array) == 1)  ;
}

/** @brief Check if a MATLAB array is a vector.
 ** @param array MATLAB array.
 ** @param numElements number of elements (negative for any).
 ** @return ::true if the array is a vecotr of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

static bool
vlmxIsVector (mxArray const * array, ssize_t numElements)
{
  size_t numDimensions = (unsigned) mxGetNumberOfDimensions (array) ;
  mwSize const * dimensions = mxGetDimensions (array) ;
  size_t di ;

  /* check that it is not sparse */
  if (mxIsSparse (array)) {
    return false ;
  }

  /* check that the number of elements is the prescribed one */
  if ((numElements >= 0) && ((unsigned) mxGetNumberOfElements (array) !=
                             (unsigned) numElements)) {
    return false ;
  }

  /* check that all but at most one dimension is singleton */
  for (di = 0 ;  di < numDimensions ; ++ di) {
    if (dimensions[di] != 1) break ;
  }
  for (++ di ; di < numDimensions ; ++di) {
    if (dimensions[di] != 1) return false ;
  }
  return true ;
}

/** @brief Check if a MATLAB array is a matrix.
 ** @param array MATLAB array.
 ** @param M number of rows (negative for any).
 ** @param N number of columns (negative for any).
 ** @return ::true if the array is a matrix of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

static bool
vlmxIsMatrix (mxArray const * array, ssize_t M, ssize_t N)
{
  size_t numDimensions = (unsigned) mxGetNumberOfDimensions (array) ;
  mwSize const * dimensions = mxGetDimensions (array) ;
  size_t di ;

  /* check that it is not sparse */
  if (mxIsSparse (array)) {
    return false ;
  }

  /* check that the number of elements is the prescribed one */
  if ((M >= 0) && ((unsigned) mxGetM (array) != (unsigned) M)) {
    return false;
  }
  if ((N >= 0) && ((unsigned) mxGetN (array) != (unsigned) N)) {
    return false;
  }

  /* ok if empty and either M = 0 or N = 0 */
  if ((mxGetNumberOfElements (array) == 0) && (mxGetM (array) == 0 || mxGetN (array) == 0)) {
    return true ;
  }

  /* ok if any dimension beyond the first two is singleton */
  for (di = 2 ; ((unsigned)dimensions[di] == 1) && di < numDimensions ; ++ di) ;
  return di == numDimensions ;
}

/** @brief Check if the MATLAB array has the specified dimensions.
 ** @param array array to check.
 ** @param numDimensions number of dimensions.
 ** @param dimensions dimensions.
 ** @return true the test succeeds.
 **
 ** The test is true if @a numDimensions < 0. If not, it is false if
 ** the array has not @a numDimensions. Otherwise it is true is @a
 ** dimensions is @c NULL or if each entry of @a dimensions is
 ** either negative or equal to the corresponding array dimension.
 **/

static bool
vlmxIsArray (mxArray const * array, ssize_t numDimensions, ssize_t* dimensions)
{
  if (numDimensions >= 0) {
    ssize_t d ;
    mwSize const * actualDimensions = mxGetDimensions (array) ;

    if ((unsigned) mxGetNumberOfDimensions (array) != (unsigned) numDimensions) {
      return false ;
    }

    if(dimensions != NULL) {
      for(d = 0 ; d < numDimensions ; ++d) {
        if (dimensions[d] >= 0 && (unsigned) dimensions[d] != (unsigned) actualDimensions[d])
          return false ;
      }
    }
  }
  return true ;
}

/** @} */

/** @name Check for plain arrays
 ** @{ */

/** @brief Check if a MATLAB array is plain
 ** @param array MATLAB array.
 ** @return ::true if the array is plain.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE bool
vlmxIsPlain (mxArray const * array)
{
  return
  vlmxIsReal (array) &&
  vlmxIsOfClass (array, mxDOUBLE_CLASS) ;
}


/** @brief Check if a MATLAB array is plain scalar
 ** @param array MATLAB array.
 ** @return ::true if the array is plain scalar.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE bool
vlmxIsPlainScalar (mxArray const * array)
{
  return vlmxIsPlain (array) && vlmxIsScalar (array) ;
}

/** @brief Check if a MATLAB array is a plain vector.
 ** @param array MATLAB array.
 ** @param numElements number of elements (negative for any).
 ** @return ::true if the array is a plain vecotr of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE bool
vlmxIsPlainVector (mxArray const * array, ssize_t numElements)
{
  return vlmxIsPlain (array) && vlmxIsVector (array, numElements) ;
}

/** @brief Check if a MATLAB array is a plain matrix.
 ** @param array MATLAB array.
 ** @param M number of rows (negative for any).
 ** @param N number of columns (negative for any).
 ** @return ::true if the array is a plain matrix of the prescribed size.
 ** @sa @ref mexutils-array-test
 **/

VL_INLINE bool
vlmxIsPlainMatrix (mxArray const * array, ssize_t M, ssize_t N)
{
  return vlmxIsPlain (array) && vlmxIsMatrix (array, M, N) ;
}

/** @brief Check if the array is a string
 ** @param array array to test.
 ** @param length string length.
 ** @return true if the array is a string of the specified length
 **
 ** The array @a array satisfies the test if:
 ** - its storage class is CHAR;
 ** - it has two dimensions but only one row;
 ** - @a length < 0 or the array has @a length columns.
 **/

static int
vlmxIsString (const mxArray* array, ssize_t length)
{
  mwSize M = (mwSize) mxGetM (array) ;
  mwSize N = (mwSize) mxGetN (array) ;

  return
  mxIsChar(array) &&
  mxGetNumberOfDimensions(array) == 2 &&
  (M == 1 || (M == 0 && N == 0)) &&
  (length < 0 || (signed)N == length) ;
}


/** @} */

/** @brief Create a MATLAB array which is a plain scalar
 ** @param x scalar value.
 ** @return the new array.
 **/

static mxArray *
vlmxCreatePlainScalar (double x)
{
  mxArray * array = mxCreateDoubleMatrix (1,1,mxREAL) ;
  *mxGetPr(array) = x ;
  return array ;
}

/** @brief Case insensitive string comparison
 ** @param s1 first string.
 ** @param s2 second string.
 ** @return comparison result.
 **
 ** The comparison result is equal to 0 if the strings are equal, >0
 ** if the first string is greater than the second (in lexicographical
 ** order), and <0 otherwise.
 **/

static int
vlmxCompareStringsI(const char *s1, const char *s2)
{
  /*
   Since tolower has an int argument, characters must be unsigned
   otherwise will be sign-extended when converted to int.
   */
  while (tolower((unsigned char)*s1) == tolower((unsigned char)*s2))
  {
    if (*s1 == 0) return 0 ; /* implies *s2 == 0 */
    s1++;
    s2++;
  }
  return tolower((unsigned char)*s1) - tolower((unsigned char)*s2) ;
}

/** @brief Case insensitive string comparison with array
 ** @param array first string (as a MATLAB array).
 ** @param string second string.
 ** @return comparison result.
 **
 ** The comparison result is equal to 0 if the strings are equal, >0
 ** if the first string is greater than the second (in lexicographical
 ** order), and <0 otherwise.
 **/

static int
vlmxCompareToStringI(mxArray const * array, char const  * string)
{
  mxChar const * s1 = (mxChar const *) mxGetData(array) ;
  char unsigned const * s2 = (char unsigned const*) string ;
  size_t n = mxGetNumberOfElements(array) ;

  /*
   Since tolower has an int argument, characters must be unsigned
   otherwise will be sign-extended when converted to int.
   */
  while (n && tolower((unsigned)*s1) == tolower(*s2)) {
    if (*s2 == 0) return 1 ; /* s2 terminated on 0, but s1 did not terminate yet */
    s1 ++ ;
    s2 ++ ;
    n -- ;
  }
  return tolower(n ? (unsigned)*s1 : 0) - tolower(*s2) ;
}

/** @brief Case insensitive string equality test with array
 ** @param array first string (as a MATLAB array).
 ** @param string second string.
 ** @return true if the strings are equal.
 **/

static int
vlmxIsEqualToStringI(mxArray const * array, char const  * string)
{
  return vlmxCompareToStringI(array, string) == 0 ;
}

/* ---------------------------------------------------------------- */
/*                        Options handling                          */
/* ---------------------------------------------------------------- */

/** @brief MEX option */

typedef struct
{
  const char *name ; /**< option name */
  int hasArgument ;  /**< has argument? */
  int value ;        /**< value to return */
} VLMXOption ;


/** @brief Parse the next option
 ** @param args     MEX argument array.
 ** @param nargs    MEX argument array length.
 ** @param options  List of option definitions.
 ** @param next     Pointer to the next option (input and output).
 ** @param optarg   Pointer to the option optional argument (output).
 ** @return the code of the next option, or -1 if there are no more options.
 **
 ** The function parses the array @a args for options. @a args is
 ** expected to be a sequence alternating option names and option
 ** values, in the form of @a nargs instances of @c mxArray. The
 ** function then scans the option starting at position @a next in the
 ** array.  The option name is matched (case insensitive) to the table
 ** of options @a options, a pointer to the option value is stored in
 ** @a optarg, @a next is advanced to the next option, and the option
 ** code is returned.
 **
 ** The function is typically used in a loop to parse all the available
 ** options. @a next is initialized to zero, and then the function
 ** is called until the special code -1 is returned.
 **
 ** If the option name cannot be matched to the available options,
 ** either because the option name is not a string array or because
 ** the name is unknown, the function throws a MEX error.
 **/

static int
vlmxNextOption (mxArray const *args[], int nargs,
                VLMXOption  const *options,
                int *next,
                mxArray const **optarg)
{
  char name [1024] ;
  int opt = -1, i;

  if (*next >= nargs) {
    return opt ;
  }

  /* check the array is a string */
  if (! vlmxIsString (args [*next], -1)) {
    vlmxError (VLMXE_InvalidOption,
               "The option name is not a string (argument number %d)",
               *next + 1) ;
  }

  /* retrieve option name */
  if (mxGetString (args [*next], name, sizeof(name))) {
    vlmxError (VLMXE_InvalidOption,
               "The option name is too long (argument number %d)",
               *next + 1) ;
  }

  /* advance argument list */
  ++ (*next) ;

  /* now lookup the string in the option table */
  for (i = 0 ; options[i].name != 0 ; ++i) {
    if (vlmxCompareStringsI(name, options[i].name) == 0) {
      opt = options[i].value ;
      break ;
    }
  }

  /* unknown argument */
  if (opt < 0) {
    vlmxError (VLMXE_InvalidOption, "Unknown option '%s'.", name) ;
  }

  /* no argument */
  if (! options [i].hasArgument) {
    if (optarg) *optarg = 0 ;
    return opt ;
  }

  /* argument */
  if (*next >= nargs) {
    vlmxError(VLMXE_InvalidOption,
              "Option '%s' requires an argument.", options[i].name) ;
  }

  if (optarg) *optarg = args [*next] ;
  ++ (*next) ;
  return opt ;
}

/* -------------------------------------------------------------------
 *                                                     VLMXEnumeration
 * ---------------------------------------------------------------- */

/** @name String enumerations
 ** @{ */

/** @brief A member of an enumeration */
typedef struct _VLMXEnumerationItem
{
  char const *name ; /**< enumeration member name. */
  int value ;        /**< enumeration member value. */
} VLMXEnumerationItem ;

/** @brief Get a member of an enumeration by name
 ** @param enumeration array of ::VLMXEnumerator objects.
 ** @param name the name of the desired member.
 ** @return enumerator matching @a name.
 **
 ** If @a name is not found in the enumeration, then the value
 ** @c NULL is returned.
 **
 ** @sa vl-stringop-enumeration
 **/

static VLMXEnumerationItem *
vlmxEnumerationGet (VLMXEnumerationItem const *enumeration, char const *name)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (strcmp(name, enumeration->name) == 0) {
      return (VLMXEnumerationItem*)enumeration ;
    }
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get a member of an enumeration by name (case insensitive)
 ** @param enumeration array of ::VLMXEnumerator objects.
 ** @param name the name of the desired member.
 ** @return enumerator matching @a name.
 **
 ** If @a name is not found in the enumeration, then the value
 ** @c NULL is returned. @a string is matched case insensitive.
 **
 ** @sa vl-stringop-enumeration
 **/

static VLMXEnumerationItem *
vlmxEnumerationGetCaseI (VLMXEnumerationItem const *enumeration, char const *name)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (vlmxCompareStringsI(name, enumeration->name) == 0) {
      return (VLMXEnumerationItem*)enumeration ;
    }
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get a member of an enumeration by value
 ** @param enumeration array of ::VLMXEnumerator objects.
 ** @param value value of the desired member.
 ** @return enumerator matching @a value.
 **
 ** If @a value is not found in the enumeration, then the value
 ** @c NULL is returned.
 **
 ** @sa vl-stringop-enumeration
 **/

static VLMXEnumerationItem *
vlmxEnumerationGetByValue (VLMXEnumerationItem const *enumeration, int value)
{
  assert(enumeration) ;
  while (enumeration->name) {
    if (enumeration->value == value) {
      return (VLMXEnumerationItem*)enumeration ;
    }
    enumeration ++ ;
  }
  return NULL ;
}

/** @brief Get an emumeration item by name
 ** @param enumeration the enumeration to decode.
 ** @param name_array member name as a MATLAB string array.
 ** @param caseInsensitive if @c true match the string case-insensitive.
 ** @return the corresponding enumeration member, or @c NULL if any.
 **
 ** The function throws a MEX error if @a name_array is not a string or
 ** if the name is not found in the enumeration.
 **/

static VLMXEnumerationItem *
vlmxDecodeEnumeration (mxArray const *name_array,
                       VLMXEnumerationItem const *enumeration,
                       bool caseInsensitive)
{
  char name [1024] ;

  /* check the array is a string */
  if (! vlmxIsString (name_array, -1)) {
    vlmxError (VLMXE_IllegalArgument, "The array is not a string.") ;
  }

  /* retrieve option name */
  if (mxGetString (name_array, name, sizeof(name))) {
    vlmxError (VLMXE_IllegalArgument, "The string array is too long.") ;
  }

  if (caseInsensitive) {
    return vlmxEnumerationGetCaseI(enumeration, name) ;
  } else {
    return vlmxEnumerationGet(enumeration, name) ;
  }
}

/* MEXUTILS_H */
#endif

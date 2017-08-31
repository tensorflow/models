// @file data.hpp
// @brief Basic data structures
// @author Andrea Vedaldi

/*
Copyright (C) 2015-16 Andrea Vedaldi.
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#ifndef __vl_data_hpp__
#define __vl_data_hpp__

#include <cstddef>
#include <string>

#include "impl/compat.h"

#define STRINGIZE(x) STRINGIZE_HELPER(x)
#define STRINGIZE_HELPER(x) #x
#define FILELINE STRINGIZE(__FILE__) ":" STRINGIZE(__LINE__)
#define divides(a,b) ((b) == (b)/(a)*(a))

#if ENABLE_DOUBLE
#define IF_DOUBLE(x) x
#else
#define IF_DOUBLE(x)
#endif

#define VL_M_PI 3.14159265358979323846
#define VL_M_PI_F 3.14159265358979323846f

namespace vl {

  /// Error codes
  enum ErrorCode {
    VLE_Success = 0,
    VLE_Unsupported,
    VLE_Cuda,
    VLE_Cudnn,
    VLE_Cublas,
    VLE_OutOfMemory,
    VLE_OutOfGPUMemeory,
    VLE_IllegalArgument,
    VLE_Unknown,
    VLE_Timeout,
    VLE_NoData,
    VLE_IllegalMessage,
    VLE_Interrupted
  } ;

  /// Get an error message for a given code
  const char * getErrorMessage(ErrorCode error) ;

  /// Type of device: CPU or GPU
  enum DeviceType {
    VLDT_CPU = 0,
    VLDT_GPU
  }  ;

  /// Type of data (char, float, double, ...)
  enum DataType {
    VLDT_Char,
    VLDT_Float,
    VLDT_Double
  } ;

  template <vl::DataType id> struct DataTypeTraits { } ;
  template <> struct DataTypeTraits<VLDT_Char> { typedef char type ; } ;
  template <> struct DataTypeTraits<VLDT_Float> { typedef float type ; } ;
  template <> struct DataTypeTraits<VLDT_Double> { typedef double type ; } ;

  template <typename type> struct BuiltinToDataType {} ;
  template <> struct BuiltinToDataType<char> { enum { dataType = VLDT_Char } ; } ;
  template <> struct BuiltinToDataType<float> { enum { dataType = VLDT_Float } ; } ;
  template <> struct BuiltinToDataType<double> { enum { dataType = VLDT_Double } ; } ;

  inline size_t getDataTypeSizeInBytes(DataType dataType) {
    switch (dataType) {
      case VLDT_Char:   return sizeof(DataTypeTraits<VLDT_Char>::type) ;
      case VLDT_Float:  return sizeof(DataTypeTraits<VLDT_Float>::type) ;
      case VLDT_Double: return sizeof(DataTypeTraits<VLDT_Double>::type) ;
      default:          return 0 ;
    }
  }

  class CudaHelper ;

  /* -----------------------------------------------------------------
   * Helpers
   * -------------------------------------------------------------- */

  /// Computes the smallest multiple of @a b which is greater
  /// or equal to @a a.
  inline int divideAndRoundUp(int a, int b)
  {
    return (a + b - 1) / b ;
  }

  inline size_t divideAndRoundUp(size_t a, size_t b)
  {
    return (a + b - 1) / b ;
  }

  /// Compute the greatest common divisor g of non-negative integers
  /// @a a and @a b as well as two integers @a u and @a v such that
  /// $au + bv = g$ (Bezout's coefficients).
  int gcd(int a, int b, int &u, int& v) ;

  /// Draw a Normally-distributed scalar
  double randn() ;

  /// Get realtime monotnic clock in microseconds
  size_t getTime() ;

  namespace impl {
    class Buffer
    {
    public:
      Buffer() ;
      vl::ErrorCode init(DeviceType deviceType, DataType dataType, size_t size) ;
      void * getMemory() ;
      int getNumReallocations() const ;
      void clear() ;
      void invalidateGpu() ;
    protected:
      DeviceType deviceType ;
      DataType dataType ;
      size_t size ;
      void * memory ;
      int numReallocations ;
    } ;
  }

  /* -----------------------------------------------------------------
   * Context
   * -------------------------------------------------------------- */

  class Context
  {
  public:
    Context() ;
    ~Context() ;

    void * getWorkspace(DeviceType device, size_t size) ;
    void clearWorkspace(DeviceType device) ;
    void * getAllOnes(DeviceType device, DataType type, size_t size) ;
    void clearAllOnes(DeviceType device) ;
    CudaHelper& getCudaHelper() ;

    void clear() ; // do a reset
    void invalidateGpu() ; // drop CUDA memory and handles

    vl::ErrorCode passError(vl::ErrorCode error, char const * message = NULL) ;
    vl::ErrorCode setError(vl::ErrorCode error, char const * message = NULL) ;
    void resetLastError() ;
    vl::ErrorCode getLastError() const ;
    std::string const& getLastErrorMessage() const ;

  private:
    impl::Buffer workspace[2] ;
    impl::Buffer allOnes[2] ;

    ErrorCode lastError ;
    std::string lastErrorMessage ;

    CudaHelper * cudaHelper ;
  } ;

  /* -----------------------------------------------------------------
   * TensorShape
   * -------------------------------------------------------------- */

#define VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS 8

  class TensorShape
  {
  public:
    TensorShape() ;
    TensorShape(TensorShape const& t) ;
    TensorShape(size_t height, size_t width, size_t depth, size_t size) ;
    TensorShape(size_t const * dimensions, size_t numDimensions) ;

    void clear() ; // set to empty (numDimensions = 0)
    void setDimension(size_t num, size_t dimension) ;
    void setDimensions(size_t const * dimensions, size_t numDimensions) ;
    void setHeight(size_t x) ;
    void setWidth(size_t x) ;
    void setDepth(size_t x) ;
    void setSize(size_t x) ;
    void reshape(size_t numDimensions) ; // squash or stretch to numDimensions
    void reshape(TensorShape const & shape) ; // same as operator=

    size_t getDimension(size_t num) const ;
    size_t const * getDimensions() const ;
    size_t getNumDimensions() const ;
    size_t getHeight() const ;
    size_t getWidth() const ;
    size_t getDepth() const ;
    size_t getSize() const ;

    size_t getNumElements() const ;
    bool isEmpty() const ;

  protected:
    size_t dimensions [VL_TENSOR_SHAPE_MAX_NUM_DIMENSIONS] ;
    size_t numDimensions ;
  } ;

  bool operator == (TensorShape const & a, TensorShape const & b) ;

  inline bool operator != (TensorShape const & a, TensorShape const & b)
  {
    return ! (a == b) ;
  }

  /* -----------------------------------------------------------------
   * Tensor
   * -------------------------------------------------------------- */

  class Tensor : public TensorShape
  {
  public:
    Tensor() ;
    Tensor(Tensor const &) ;
    Tensor(TensorShape const & shape, DataType dataType,
           DeviceType deviceType, void * memory, size_t memorySize) ;
    void * getMemory() ;
    DeviceType getDeviceType() const ;
    TensorShape getShape() const ;
    DataType getDataType() const ;
    operator bool() const ;
    bool isNull() const ;
    void setMemory(void * x) ;

  protected:
    DeviceType deviceType ;
    DataType dataType ;
    void * memory ;
    size_t memorySize ;
  } ;

  inline Tensor::Tensor(Tensor const& t)
  : TensorShape(t), dataType(t.dataType), deviceType(t.deviceType),
  memory(t.memory), memorySize(t.memorySize)
  { }

  inline bool areCompatible(Tensor const & a, Tensor const & b)
  {
    return
    (a.isEmpty() || a.isNull()) ||
    (b.isEmpty() || b.isNull()) ||
    ((a.getDeviceType() == b.getDeviceType()) & (a.getDataType() == b.getDataType())) ;
  }
}

#endif

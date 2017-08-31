/* -*- mode: c++; tab-width: 2; indent-tabs-mode: nil; -*-
Copyright (c) 2010-2012 Marcus Geelnard

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

#ifndef _TINYTHREAD_H_
#define _TINYTHREAD_H_

/// @file
/// @mainpage TinyThread++ API Reference
///
/// @section intro_sec Introduction
/// TinyThread++ is a minimal, portable implementation of basic threading
/// classes for C++.
///
/// They closely mimic the functionality and naming of the C++11 standard, and
/// should be easily replaceable with the corresponding std:: variants.
///
/// @section port_sec Portability
/// The Win32 variant uses the native Win32 API for implementing the thread
/// classes, while for other systems, the POSIX threads API (pthread) is used.
///
/// @section class_sec Classes
/// In order to mimic the threading API of the C++11 standard, subsets of
/// several classes are provided. The fundamental classes are:
/// @li tthread::thread
/// @li tthread::mutex
/// @li tthread::recursive_mutex
/// @li tthread::condition_variable
/// @li tthread::lock_guard
/// @li tthread::fast_mutex
///
/// @section misc_sec Miscellaneous
/// The following special keywords are available: #thread_local.
///
/// For more detailed information (including additional classes), browse the
/// different sections of this documentation. A good place to start is:
/// tinythread.h.

// Which platform are we on?
#if !defined(_TTHREAD_PLATFORM_DEFINED_)
  #if defined(_WIN32) || defined(__WIN32__) || defined(__WINDOWS__)
    #define _TTHREAD_WIN32_
  #else
    #define _TTHREAD_POSIX_
  #endif
  #define _TTHREAD_PLATFORM_DEFINED_
#endif

// Platform specific includes
#if defined(_TTHREAD_WIN32_)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #define __UNDEF_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #ifdef __UNDEF_LEAN_AND_MEAN
    #undef WIN32_LEAN_AND_MEAN
    #undef __UNDEF_LEAN_AND_MEAN
  #endif
#else
  #include <pthread.h>
  #include <signal.h>
  #include <sched.h>
  #include <unistd.h>
#endif

// Generic includes
#include <ostream>

/// TinyThread++ version (major number).
#define TINYTHREAD_VERSION_MAJOR 1
/// TinyThread++ version (minor number).
#define TINYTHREAD_VERSION_MINOR 1
/// TinyThread++ version (full version).
#define TINYTHREAD_VERSION (TINYTHREAD_VERSION_MAJOR * 100 + TINYTHREAD_VERSION_MINOR)

// Do we have a fully featured C++11 compiler?
#if (__cplusplus > 199711L) || (defined(__STDCXX_VERSION__) && (__STDCXX_VERSION__ >= 201001L))
  #define _TTHREAD_CPP11_
#endif

// ...at least partial C++11?
#if defined(_TTHREAD_CPP11_) || defined(__GXX_EXPERIMENTAL_CXX0X__) || defined(__GXX_EXPERIMENTAL_CPP0X__)
  #define _TTHREAD_CPP11_PARTIAL_
#endif

// Macro for disabling assignments of objects.
#ifdef _TTHREAD_CPP11_PARTIAL_
  #define _TTHREAD_DISABLE_ASSIGNMENT(name) \
      name(const name&) = delete; \
      name& operator=(const name&) = delete;
#else
  #define _TTHREAD_DISABLE_ASSIGNMENT(name) \
      name(const name&); \
      name& operator=(const name&);
#endif

/// @def thread_local
/// Thread local storage keyword.
/// A variable that is declared with the @c thread_local keyword makes the
/// value of the variable local to each thread (known as thread-local storage,
/// or TLS). Example usage:
/// @code
/// // This variable is local to each thread.
/// thread_local int variable;
/// @endcode
/// @note The @c thread_local keyword is a macro that maps to the corresponding
/// compiler directive (e.g. @c __declspec(thread)). While the C++11 standard
/// allows for non-trivial types (e.g. classes with constructors and
/// destructors) to be declared with the @c thread_local keyword, most pre-C++11
/// compilers only allow for trivial types (e.g. @c int). So, to guarantee
/// portable code, only use trivial types for thread local storage.
/// @note This directive is currently not supported on Mac OS X (it will give
/// a compiler error), since compile-time TLS is not supported in the Mac OS X
/// executable format. Also, some older versions of MinGW (before GCC 4.x) do
/// not support this directive.
/// @hideinitializer

#if !defined(_TTHREAD_CPP11_) && !defined(thread_local)
 #if defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__SUNPRO_CC) || defined(__IBMCPP__)
  #define thread_local __thread
 #else
  #define thread_local __declspec(thread)
 #endif
#endif


/// Main name space for TinyThread++.
/// This namespace is more or less equivalent to the @c std namespace for the
/// C++11 thread classes. For instance, the tthread::mutex class corresponds to
/// the std::mutex class.
namespace tthread {

/// Mutex class.
/// This is a mutual exclusion object for synchronizing access to shared
/// memory areas for several threads. The mutex is non-recursive (i.e. a
/// program may deadlock if the thread that owns a mutex object calls lock()
/// on that object).
/// @see recursive_mutex
class mutex {
  public:
    /// Constructor.
    mutex()
#if defined(_TTHREAD_WIN32_)
      : mAlreadyLocked(false)
#endif
    {
#if defined(_TTHREAD_WIN32_)
      InitializeCriticalSection(&mHandle);
#else
      pthread_mutex_init(&mHandle, NULL);
#endif
    }

    /// Destructor.
    ~mutex()
    {
#if defined(_TTHREAD_WIN32_)
      DeleteCriticalSection(&mHandle);
#else
      pthread_mutex_destroy(&mHandle);
#endif
    }

    /// Lock the mutex.
    /// The method will block the calling thread until a lock on the mutex can
    /// be obtained. The mutex remains locked until @c unlock() is called.
    /// @see lock_guard
    inline void lock()
    {
#if defined(_TTHREAD_WIN32_)
      EnterCriticalSection(&mHandle);
      while(mAlreadyLocked) Sleep(1000); // Simulate deadlock...
      mAlreadyLocked = true;
#else
      pthread_mutex_lock(&mHandle);
#endif
    }

    /// Try to lock the mutex.
    /// The method will try to lock the mutex. If it fails, the function will
    /// return immediately (non-blocking).
    /// @return @c true if the lock was acquired, or @c false if the lock could
    /// not be acquired.
    inline bool try_lock()
    {
#if defined(_TTHREAD_WIN32_)
      bool ret = (TryEnterCriticalSection(&mHandle) ? true : false);
      if(ret && mAlreadyLocked)
      {
        LeaveCriticalSection(&mHandle);
        ret = false;
      }
      return ret;
#else
      return (pthread_mutex_trylock(&mHandle) == 0) ? true : false;
#endif
    }

    /// Unlock the mutex.
    /// If any threads are waiting for the lock on this mutex, one of them will
    /// be unblocked.
    inline void unlock()
    {
#if defined(_TTHREAD_WIN32_)
      mAlreadyLocked = false;
      LeaveCriticalSection(&mHandle);
#else
      pthread_mutex_unlock(&mHandle);
#endif
    }

    _TTHREAD_DISABLE_ASSIGNMENT(mutex)

  private:
#if defined(_TTHREAD_WIN32_)
    CRITICAL_SECTION mHandle;
    bool mAlreadyLocked;
#else
    pthread_mutex_t mHandle;
#endif

    friend class condition_variable;
};

/// Recursive mutex class.
/// This is a mutual exclusion object for synchronizing access to shared
/// memory areas for several threads. The mutex is recursive (i.e. a thread
/// may lock the mutex several times, as long as it unlocks the mutex the same
/// number of times).
/// @see mutex
class recursive_mutex {
  public:
    /// Constructor.
    recursive_mutex()
    {
#if defined(_TTHREAD_WIN32_)
      InitializeCriticalSection(&mHandle);
#else
      pthread_mutexattr_t attr;
      pthread_mutexattr_init(&attr);
      pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
      pthread_mutex_init(&mHandle, &attr);
#endif
    }

    /// Destructor.
    ~recursive_mutex()
    {
#if defined(_TTHREAD_WIN32_)
      DeleteCriticalSection(&mHandle);
#else
      pthread_mutex_destroy(&mHandle);
#endif
    }

    /// Lock the mutex.
    /// The method will block the calling thread until a lock on the mutex can
    /// be obtained. The mutex remains locked until @c unlock() is called.
    /// @see lock_guard
    inline void lock()
    {
#if defined(_TTHREAD_WIN32_)
      EnterCriticalSection(&mHandle);
#else
      pthread_mutex_lock(&mHandle);
#endif
    }

    /// Try to lock the mutex.
    /// The method will try to lock the mutex. If it fails, the function will
    /// return immediately (non-blocking).
    /// @return @c true if the lock was acquired, or @c false if the lock could
    /// not be acquired.
    inline bool try_lock()
    {
#if defined(_TTHREAD_WIN32_)
      return TryEnterCriticalSection(&mHandle) ? true : false;
#else
      return (pthread_mutex_trylock(&mHandle) == 0) ? true : false;
#endif
    }

    /// Unlock the mutex.
    /// If any threads are waiting for the lock on this mutex, one of them will
    /// be unblocked.
    inline void unlock()
    {
#if defined(_TTHREAD_WIN32_)
      LeaveCriticalSection(&mHandle);
#else
      pthread_mutex_unlock(&mHandle);
#endif
    }

    _TTHREAD_DISABLE_ASSIGNMENT(recursive_mutex)

  private:
#if defined(_TTHREAD_WIN32_)
    CRITICAL_SECTION mHandle;
#else
    pthread_mutex_t mHandle;
#endif

    friend class condition_variable;
};

/// Lock guard class.
/// The constructor locks the mutex, and the destructor unlocks the mutex, so
/// the mutex will automatically be unlocked when the lock guard goes out of
/// scope. Example usage:
/// @code
/// mutex m;
/// int counter;
///
/// void increment()
/// {
///   lock_guard<mutex> guard(m);
///   ++ counter;
/// }
/// @endcode

template <class T>
class lock_guard {
  public:
    typedef T mutex_type;

    lock_guard() : mMutex(0) {}

    /// The constructor locks the mutex.
    explicit lock_guard(mutex_type &aMutex)
    {
      mMutex = &aMutex;
      mMutex->lock();
    }

    /// The destructor unlocks the mutex.
    ~lock_guard()
    {
      if(mMutex)
        mMutex->unlock();
    }

  private:
    mutex_type * mMutex;
};

/// Condition variable class.
/// This is a signalling object for synchronizing the execution flow for
/// several threads. Example usage:
/// @code
/// // Shared data and associated mutex and condition variable objects
/// int count;
/// mutex m;
/// condition_variable cond;
///
/// // Wait for the counter to reach a certain number
/// void wait_counter(int targetCount)
/// {
///   lock_guard<mutex> guard(m);
///   while(count < targetCount)
///     cond.wait(m);
/// }
///
/// // Increment the counter, and notify waiting threads
/// void increment()
/// {
///   lock_guard<mutex> guard(m);
///   ++ count;
///   cond.notify_all();
/// }
/// @endcode
class condition_variable {
  public:
    /// Constructor.
#if defined(_TTHREAD_WIN32_)
    condition_variable();
#else
    condition_variable()
    {
      pthread_cond_init(&mHandle, NULL);
    }
#endif

    /// Destructor.
#if defined(_TTHREAD_WIN32_)
    ~condition_variable();
#else
    ~condition_variable()
    {
      pthread_cond_destroy(&mHandle);
    }
#endif

    /// Wait for the condition.
    /// The function will block the calling thread until the condition variable
    /// is woken by @c notify_one(), @c notify_all() or a spurious wake up.
    /// @param[in] aMutex A mutex that will be unlocked when the wait operation
    ///   starts, an locked again as soon as the wait operation is finished.
    template <class _mutexT>
    inline void wait(_mutexT &aMutex)
    {
#if defined(_TTHREAD_WIN32_)
      // Increment number of waiters
      EnterCriticalSection(&mWaitersCountLock);
      ++ mWaitersCount;
      LeaveCriticalSection(&mWaitersCountLock);

      // Release the mutex while waiting for the condition (will decrease
      // the number of waiters when done)...
      aMutex.unlock();
      _wait();
      aMutex.lock();
#else
      pthread_cond_wait(&mHandle, &aMutex.mHandle);
#endif
    }

    /// Notify one thread that is waiting for the condition.
    /// If at least one thread is blocked waiting for this condition variable,
    /// one will be woken up.
    /// @note Only threads that started waiting prior to this call will be
    /// woken up.
#if defined(_TTHREAD_WIN32_)
    void notify_one();
#else
    inline void notify_one()
    {
      pthread_cond_signal(&mHandle);
    }
#endif

    /// Notify all threads that are waiting for the condition.
    /// All threads that are blocked waiting for this condition variable will
    /// be woken up.
    /// @note Only threads that started waiting prior to this call will be
    /// woken up.
#if defined(_TTHREAD_WIN32_)
    void notify_all();
#else
    inline void notify_all()
    {
      pthread_cond_broadcast(&mHandle);
    }
#endif

    _TTHREAD_DISABLE_ASSIGNMENT(condition_variable)

  private:
#if defined(_TTHREAD_WIN32_)
    void _wait();
    HANDLE mEvents[2];                  ///< Signal and broadcast event HANDLEs.
    unsigned int mWaitersCount;         ///< Count of the number of waiters.
    CRITICAL_SECTION mWaitersCountLock; ///< Serialize access to mWaitersCount.
#else
    pthread_cond_t mHandle;
#endif
};


/// Thread class.
class thread {
  public:
#if defined(_TTHREAD_WIN32_)
    typedef HANDLE native_handle_type;
#else
    typedef pthread_t native_handle_type;
#endif

    class id;

    /// Default constructor.
    /// Construct a @c thread object without an associated thread of execution
    /// (i.e. non-joinable).
    thread() : mHandle(0), mNotAThread(true)
#if defined(_TTHREAD_WIN32_)
    , mWin32ThreadID(0)
#endif
    {}

    /// Thread starting constructor.
    /// Construct a @c thread object with a new thread of execution.
    /// @param[in] aFunction A function pointer to a function of type:
    ///          <tt>void fun(void * arg)</tt>
    /// @param[in] aArg Argument to the thread function.
    /// @note This constructor is not fully compatible with the standard C++
    /// thread class. It is more similar to the pthread_create() (POSIX) and
    /// CreateThread() (Windows) functions.
    thread(void (*aFunction)(void *), void * aArg);

    /// Destructor.
    /// @note If the thread is joinable upon destruction, @c std::terminate()
    /// will be called, which terminates the process. It is always wise to do
    /// @c join() before deleting a thread object.
    ~thread();

    /// Wait for the thread to finish (join execution flows).
    /// After calling @c join(), the thread object is no longer associated with
    /// a thread of execution (i.e. it is not joinable, and you may not join
    /// with it nor detach from it).
    void join();

    /// Check if the thread is joinable.
    /// A thread object is joinable if it has an associated thread of execution.
    bool joinable() const;

    /// Detach from the thread.
    /// After calling @c detach(), the thread object is no longer assicated with
    /// a thread of execution (i.e. it is not joinable). The thread continues
    /// execution without the calling thread blocking, and when the thread
    /// ends execution, any owned resources are released.
    void detach();

    /// Return the thread ID of a thread object.
    id get_id() const;

    /// Get the native handle for this thread.
    /// @note Under Windows, this is a @c HANDLE, and under POSIX systems, this
    /// is a @c pthread_t.
    inline native_handle_type native_handle()
    {
      return mHandle;
    }

    /// Determine the number of threads which can possibly execute concurrently.
    /// This function is useful for determining the optimal number of threads to
    /// use for a task.
    /// @return The number of hardware thread contexts in the system.
    /// @note If this value is not defined, the function returns zero (0).
    static unsigned hardware_concurrency();

    _TTHREAD_DISABLE_ASSIGNMENT(thread)

  private:
    native_handle_type mHandle;   ///< Thread handle.
    mutable mutex mDataMutex;     ///< Serializer for access to the thread private data.
    bool mNotAThread;             ///< True if this object is not a thread of execution.
#if defined(_TTHREAD_WIN32_)
    unsigned int mWin32ThreadID;  ///< Unique thread ID (filled out by _beginthreadex).
#endif

    // This is the internal thread wrapper function.
#if defined(_TTHREAD_WIN32_)
    static unsigned WINAPI wrapper_function(void * aArg);
#else
    static void * wrapper_function(void * aArg);
#endif
};

/// Thread ID.
/// The thread ID is a unique identifier for each thread.
/// @see thread::get_id()
class thread::id {
  public:
    /// Default constructor.
    /// The default constructed ID is that of thread without a thread of
    /// execution.
    id() : mId(0) {};

    id(unsigned long int aId) : mId(aId) {};

    id(const id& aId) : mId(aId.mId) {};

    inline id & operator=(const id &aId)
    {
      mId = aId.mId;
      return *this;
    }

    inline friend bool operator==(const id &aId1, const id &aId2)
    {
      return (aId1.mId == aId2.mId);
    }

    inline friend bool operator!=(const id &aId1, const id &aId2)
    {
      return (aId1.mId != aId2.mId);
    }

    inline friend bool operator<=(const id &aId1, const id &aId2)
    {
      return (aId1.mId <= aId2.mId);
    }

    inline friend bool operator<(const id &aId1, const id &aId2)
    {
      return (aId1.mId < aId2.mId);
    }

    inline friend bool operator>=(const id &aId1, const id &aId2)
    {
      return (aId1.mId >= aId2.mId);
    }

    inline friend bool operator>(const id &aId1, const id &aId2)
    {
      return (aId1.mId > aId2.mId);
    }

    inline friend std::ostream& operator <<(std::ostream &os, const id &obj)
    {
      os << obj.mId;
      return os;
    }

  private:
    unsigned long int mId;
};


// Related to <ratio> - minimal to be able to support chrono.
typedef long long __intmax_t;

/// Minimal implementation of the @c ratio class. This class provides enough
/// functionality to implement some basic @c chrono classes.
template <__intmax_t N, __intmax_t D = 1> class ratio {
  public:
    static double _as_double() { return double(N) / double(D); }
};

/// Minimal implementation of the @c chrono namespace.
/// The @c chrono namespace provides types for specifying time intervals.
namespace chrono {
  /// Duration template class. This class provides enough functionality to
  /// implement @c this_thread::sleep_for().
  template <class _Rep, class _Period = ratio<1> > class duration {
    private:
      _Rep rep_;
    public:
      typedef _Rep rep;
      typedef _Period period;

      /// Construct a duration object with the given duration.
      template <class _Rep2>
        explicit duration(const _Rep2& r) : rep_(r) {}

      /// Return the value of the duration object.
      rep count() const
      {
        return rep_;
      }
  };

  // Standard duration types.
  typedef duration<__intmax_t, ratio<1, 1000000000> > nanoseconds; ///< Duration with the unit nanoseconds.
  typedef duration<__intmax_t, ratio<1, 1000000> > microseconds;   ///< Duration with the unit microseconds.
  typedef duration<__intmax_t, ratio<1, 1000> > milliseconds;      ///< Duration with the unit milliseconds.
  typedef duration<__intmax_t> seconds;                            ///< Duration with the unit seconds.
  typedef duration<__intmax_t, ratio<60> > minutes;                ///< Duration with the unit minutes.
  typedef duration<__intmax_t, ratio<3600> > hours;                ///< Duration with the unit hours.
}

/// The namespace @c this_thread provides methods for dealing with the
/// calling thread.
namespace this_thread {
  /// Return the thread ID of the calling thread.
  thread::id get_id();

  /// Yield execution to another thread.
  /// Offers the operating system the opportunity to schedule another thread
  /// that is ready to run on the current processor.
  inline void yield()
  {
#if defined(_TTHREAD_WIN32_)
    Sleep(0);
#else
    sched_yield();
#endif
  }

  /// Blocks the calling thread for a period of time.
  /// @param[in] aTime Minimum time to put the thread to sleep.
  /// Example usage:
  /// @code
  /// // Sleep for 100 milliseconds
  /// this_thread::sleep_for(chrono::milliseconds(100));
  /// @endcode
  /// @note Supported duration types are: nanoseconds, microseconds,
  /// milliseconds, seconds, minutes and hours.
  template <class _Rep, class _Period> void sleep_for(const chrono::duration<_Rep, _Period>& aTime)
  {
#if defined(_TTHREAD_WIN32_)
    Sleep(int(double(aTime.count()) * (1000.0 * _Period::_as_double()) + 0.5));
#else
    usleep(int(double(aTime.count()) * (1000000.0 * _Period::_as_double()) + 0.5));
#endif
  }
}

}

// Define/macro cleanup
#undef _TTHREAD_DISABLE_ASSIGNMENT

#endif // _TINYTHREAD_H_

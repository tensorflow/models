/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/shared_store.h"

#include <string>

#include <gmock/gmock.h>
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/core/threadpool.h"

using ::testing::_;

namespace syntaxnet {

struct NoArgs {
  NoArgs() {
    LOG(INFO) << "Calling NoArgs()";
  }
};

struct OneArg {
  string name;
  explicit OneArg(const string &n) : name(n) {
    LOG(INFO) << "Calling OneArg(" << name << ")";
  }
};

struct TwoArgs {
  string name;
  int age;
  TwoArgs(const string &n, int a) : name(n), age(a) {
    LOG(INFO) << "Calling TwoArgs(" << name << ", " << age << ")";
  }
};

struct Slow {
  string lengthy;
  Slow() {
    LOG(INFO) << "Calling Slow()";
    lengthy.assign(50 << 20, 'L');  // 50MB of the letter 'L'
  }
};

struct CountCalls {
  CountCalls() {
    LOG(INFO) << "Calling CountCalls()";
    ++constructor_calls;
  }

  ~CountCalls() {
    LOG(INFO) << "Calling ~CountCalls()";
    ++destructor_calls;
  }

  static void Reset() {
    constructor_calls = 0;
    destructor_calls = 0;
  }

  static int constructor_calls;
  static int destructor_calls;
};

int CountCalls::constructor_calls = 0;
int CountCalls::destructor_calls = 0;

class PointerSet {
 public:
  PointerSet() { }

  void Add(const void *p) {
    mutex_lock l(mu_);
    pointers_.insert(p);
  }

  int size() {
    mutex_lock l(mu_);
    return pointers_.size();
  }

 private:
  mutex mu_;
  unordered_set<const void *> pointers_;
};

class SharedStoreTest : public testing::Test {
 protected:
  ~SharedStoreTest() {
    // Clear the shared store after each test, otherwise objects created
    // in one test may interfere with other tests.
    SharedStore::Clear();
  }
};

// Verify that we can call constructors with varying numbers and types of args.
TEST_F(SharedStoreTest, ConstructorArgs) {
  SharedStore::Get<NoArgs>("no args");
  SharedStore::Get<OneArg>("one arg", "Fred");
  SharedStore::Get<TwoArgs>("two args", "Pebbles", 2);
}

// Verify that an object with a given key is created only once.
TEST_F(SharedStoreTest, Shared) {
  const NoArgs *ob1 = SharedStore::Get<NoArgs>("first");
  const NoArgs *ob2 = SharedStore::Get<NoArgs>("second");
  const NoArgs *ob3 = SharedStore::Get<NoArgs>("first");
  EXPECT_EQ(ob1, ob3);
  EXPECT_NE(ob1, ob2);
  EXPECT_NE(ob2, ob3);
}

// Verify that objects with the same name but different types do not collide.
TEST_F(SharedStoreTest, DifferentTypes) {
  const NoArgs *ob1 = SharedStore::Get<NoArgs>("same");
  const OneArg *ob2 = SharedStore::Get<OneArg>("same", "foo");
  const TwoArgs *ob3 = SharedStore::Get<TwoArgs>("same", "bar", 5);
  EXPECT_NE(static_cast<const void *>(ob1), static_cast<const void *>(ob2));
  EXPECT_NE(static_cast<const void *>(ob1), static_cast<const void *>(ob3));
  EXPECT_NE(static_cast<const void *>(ob2), static_cast<const void *>(ob3));
}

// Factory method to make a OneArg.
OneArg *MakeOneArg(const string &n) {
  return new OneArg(n);
}

TEST_F(SharedStoreTest, ClosureGet) {
  std::function<OneArg *()> closure1 = std::bind(MakeOneArg, "Al");
  std::function<OneArg *()> closure2 = std::bind(MakeOneArg, "Al");
  const OneArg *ob1 = SharedStore::ClosureGet("first", &closure1);
  const OneArg *ob2 = SharedStore::ClosureGet("first", &closure2);
  EXPECT_EQ("Al", ob1->name);
  EXPECT_EQ(ob1, ob2);
}

TEST_F(SharedStoreTest, PermanentCallback) {
  std::function<OneArg *()> closure = std::bind(MakeOneArg, "Al");
  const OneArg *ob1 = SharedStore::ClosureGet("first", &closure);
  const OneArg *ob2 = SharedStore::ClosureGet("first", &closure);
  EXPECT_EQ("Al", ob1->name);
  EXPECT_EQ(ob1, ob2);
}

// Factory method to "make" a NoArgs by simply returning an input pointer.
NoArgs *BogusMakeNoArgs(NoArgs *ob) {
  return ob;
}

// Create a CountCalls object, pretend it failed, and return null.
CountCalls *MakeFailedCountCalls() {
  CountCalls *ob = new CountCalls;
  delete ob;
  return nullptr;
}

// Verify that ClosureGet() only calls the closure for a given key once,
// even if the closure fails.
TEST_F(SharedStoreTest, FailedClosureGet) {
  CountCalls::Reset();
  std::function<CountCalls *()> closure1(MakeFailedCountCalls);
  std::function<CountCalls *()> closure2(MakeFailedCountCalls);
  const CountCalls *ob1 = SharedStore::ClosureGet("first", &closure1);
  const CountCalls *ob2 = SharedStore::ClosureGet("first", &closure2);
  EXPECT_EQ(nullptr, ob1);
  EXPECT_EQ(nullptr, ob2);
  EXPECT_EQ(1, CountCalls::constructor_calls);
}

typedef SharedStoreTest SharedStoreDeathTest;

TEST_F(SharedStoreDeathTest, ClosureGetOrDie) {
  NoArgs *empty = nullptr;
  std::function<NoArgs *()> closure = std::bind(BogusMakeNoArgs, empty);
  EXPECT_DEATH(SharedStore::ClosureGetOrDie("first", &closure), "nullptr");
}

TEST_F(SharedStoreTest, Release) {
  const OneArg *ob1 = SharedStore::Get<OneArg>("first", "Fred");
  const OneArg *ob2 = SharedStore::Get<OneArg>("first", "Fred");
  EXPECT_EQ(ob1, ob2);
  EXPECT_TRUE(SharedStore::Release(ob1));      // now refcount = 1
  EXPECT_TRUE(SharedStore::Release(ob1));      // now object is deleted
  EXPECT_FALSE(SharedStore::Release(ob1));     // now object is not found
  EXPECT_TRUE(SharedStore::Release(nullptr));  // release(nullptr) returns true
}

TEST_F(SharedStoreTest, Clear) {
  CountCalls::Reset();

  SharedStore::Get<CountCalls>("first");
  SharedStore::Get<CountCalls>("second");
  SharedStore::Get<CountCalls>("first");

  // Test that the constructor and destructor are each called exactly once
  // for each key in the shared store.
  SharedStore::Clear();
  EXPECT_EQ(2, CountCalls::constructor_calls);
  EXPECT_EQ(2, CountCalls::destructor_calls);
}

void GetSharedObject(PointerSet *ps) {
  // Gets a shared object whose constructor takes a long time.
  const Slow *ob = SharedStore::Get<Slow>("first");

  // Collects the pointer we got. Later, we'll check whether SharedStore
  // mistakenly called the constructor more than once.
  ps->Add(static_cast<const void *>(ob));
}

// If multiple parallel threads all access an object with the same key,
// only one object is created.
TEST_F(SharedStoreTest, ThreadSafety) {
  const int kNumThreads = 20;
  tensorflow::thread::ThreadPool *pool = new tensorflow::thread::ThreadPool(
      tensorflow::Env::Default(), "ThreadSafetyPool", kNumThreads);
  PointerSet ps;
  for (int i = 0; i < kNumThreads; ++i) {
    std::function<void()> closure = std::bind(GetSharedObject, &ps);
    pool->Schedule(closure);
  }

  // Waits for closures to finish, then delete the pool.
  delete pool;

  // Expects only one object to have been created across all threads.
  EXPECT_EQ(1, ps.size());
}

}  // namespace syntaxnet

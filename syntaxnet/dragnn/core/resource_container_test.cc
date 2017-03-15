// Tests the methods of ResourceContainer.
//
// NOTE(danielandor): For all tests: ResourceContainer is derived from
// RefCounted, which requires the use of Unref to reduce the ref count
// to zero and automatically delete the pointer.
#include "dragnn/core/resource_container.h"

#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

class MockDatatype {};

TEST(ResourceContainerTest, Get) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  MockDatatype *data_ptr = data.get();
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  EXPECT_EQ(data_ptr, container->get());
  container->Unref();
}

TEST(ResourceContainerTest, Release) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  MockDatatype *data_ptr = data.get();
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  std::unique_ptr<MockDatatype> data_again = container->release();
  container->Unref();
  EXPECT_EQ(data_ptr, data_again.get());
}

TEST(ResourceContainerTest, NullptrOnGetAfterRelease) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  container->release();
  EXPECT_EQ(nullptr, container->get());
  container->Unref();
}

TEST(ResourceContainerTest, DebugString) {
  std::unique_ptr<MockDatatype> data(new MockDatatype());
  auto *container = new ResourceContainer<MockDatatype>(std::move(data));
  EXPECT_EQ("ResourceContainer", container->DebugString());
  container->Unref();
}

}  // namespace dragnn
}  // namespace syntaxnet

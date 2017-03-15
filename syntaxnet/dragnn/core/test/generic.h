#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_GENERIC_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_GENERIC_H_

#include <utility>

#include <gmock/gmock.h>

#include "syntaxnet/base.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace test {

MATCHER_P(EqualsProto, a, "Protos are not equivalent:") {
  return a.DebugString() == arg.DebugString();
}

// Returns the prefix for where the test data is stored.
string GetTestDataPrefix();

}  // namespace test
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_TEST_GENERIC_H_

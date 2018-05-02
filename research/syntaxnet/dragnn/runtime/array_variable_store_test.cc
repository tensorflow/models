// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "dragnn/runtime/array_variable_store.h"

#include <string.h>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/file_array_variable_store.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/mmap_array_variable_store.h"
#include "dragnn/runtime/test/helpers.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

template <class T>
void ExpectBlockedData(BlockedMatrix<T> matrix,
                       const std::vector<std::vector<T>> &data) {
  EXPECT_EQ(matrix.num_vectors(), data.size());

  // The indices don't really have semantic names, so we just use `i` and `j`.
  // See BlockedMatrixFormat for details.
  for (int i = 0; i < matrix.num_vectors(); ++i) {
    EXPECT_EQ(matrix.block_size(), data[i].size());
    for (int j = 0; j < data[i].size(); ++j) {
      EXPECT_EQ(matrix.vector(i)[j], data[i][j]);
    }
  }
}

// Returns an ArrayVariableStoreSpec parsed from the |text|.
ArrayVariableStoreSpec MakeSpec(const string &text) {
  ArrayVariableStoreSpec spec;
  CHECK(TextFormat::ParseFromString(text, &spec));
  return spec;
}

// Returns an ArrayVariableStoreSpec that has proper top-level settings and
// whose variables are parsed from the |variables_text|.
ArrayVariableStoreSpec MakeSpecWithVariables(const string &variables_text) {
  return MakeSpec(tensorflow::strings::StrCat(
      "version: 0 alignment_bytes: ", internal::kAlignmentBytes,
      " is_little_endian: ", tensorflow::port::kLittleEndian, " ",
      variables_text));
}

// Tests that kLittleEndian actually means little-endian.
TEST(ArrayVariableStoreTest, EndianDetection) {
  static_assert(sizeof(uint32) == 4 * sizeof(uint8), "Unexpected int sizes");
  const uint32 foo = 0xdeadbeef;
  uint8 foo_bytes[4];
  memcpy(foo_bytes, &foo, 4 * sizeof(uint8));
  if (tensorflow::port::kLittleEndian) {
    EXPECT_EQ(foo_bytes[3], 0xde);
    EXPECT_EQ(foo_bytes[2], 0xad);
    EXPECT_EQ(foo_bytes[1], 0xbe);
    EXPECT_EQ(foo_bytes[0], 0xef);
  } else {
    EXPECT_EQ(foo_bytes[0], 0xde);
    EXPECT_EQ(foo_bytes[1], 0xad);
    EXPECT_EQ(foo_bytes[2], 0xbe);
    EXPECT_EQ(foo_bytes[3], 0xef);
  }
}

// Tests that the store checks for missing fields.
TEST(ArrayVariableStoreTest, MissingRequiredField) {
  for (const string kSpec :
       {"version: 0 alignment_bytes: 0", "version: 0 is_little_endian: true",
        "alignment_bytes: 0 is_little_endian: true"}) {
    ArrayVariableStore store;
    EXPECT_THAT(store.Reset(MakeSpec(kSpec), AlignedView()),
                test::IsErrorWithSubstr(
                    "ArrayVariableStoreSpec is missing a required field"));
  }
}

// Tests that the store checks for a matching version number.
TEST(ArrayVariableStoreTest, VersionMismatch) {
  const string kSpec = "version: 999 alignment_bytes: 0 is_little_endian: true";
  ArrayVariableStore store;
  EXPECT_THAT(store.Reset(MakeSpec(kSpec), AlignedView()),
              test::IsErrorWithSubstr("ArrayVariableStoreSpec.version (999) "
                                      "does not match the binary (0)"));
}

// Tests that the store checks for a matching alignment requirement.
TEST(ArrayVariableStoreTest, AlignmentMismatch) {
  const string kSpec = "version: 0 alignment_bytes: 1 is_little_endian: true";
  ArrayVariableStore store;
  EXPECT_THAT(store.Reset(MakeSpec(kSpec), AlignedView()),
              test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                  "ArrayVariableStoreSpec.alignment_bytes (1) does not match "
                  "the binary (", internal::kAlignmentBytes, ")")));
}

// Tests that the store checks for matching endian-ness.
TEST(ArrayVariableStoreTest, EndiannessMismatch) {
  const string kSpec = tensorflow::strings::StrCat(
      "version: 0 alignment_bytes: ", internal::kAlignmentBytes,
      " is_little_endian: ", !tensorflow::port::kLittleEndian);
  ArrayVariableStore store;
  EXPECT_THAT(
      store.Reset(MakeSpec(kSpec), AlignedView()),
      test::IsErrorWithSubstr(tensorflow::strings::StrCat(
          "ArrayVariableStoreSpec.is_little_endian (",
          !tensorflow::port::kLittleEndian, ") does not match the binary (",
          tensorflow::port::kLittleEndian, ")")));
}

// Tests that the store rejects FORMAT_UNKNOWN variables.
TEST(ArrayVariableStoreTest, RejectFormatUnknown) {
  const string kVariables = "variable { format: FORMAT_UNKNOWN }";
  ArrayVariableStore store;
  EXPECT_THAT(store.Reset(MakeSpecWithVariables(kVariables), AlignedView()),
              test::IsErrorWithSubstr("Unknown variable format"));
}

// Tests that the store rejects FORMAT_FLAT variables with too few sub-views.
TEST(ArrayVariableStoreTest, TooFewViewsForFlatVariable) {
  const string kVariables = "variable { format: FORMAT_FLAT num_views: 0 }";
  ArrayVariableStore store;
  EXPECT_THAT(
      store.Reset(MakeSpecWithVariables(kVariables), AlignedView()),
      test::IsErrorWithSubstr("Flat variables must have 1 view"));
}

// Tests that the store rejects FORMAT_FLAT variables with too many sub-views.
TEST(ArrayVariableStoreTest, TooManyViewsForFlatVariable) {
  const string kVariables = "variable { format: FORMAT_FLAT num_views: 2 }";
  ArrayVariableStore store;
  EXPECT_THAT(
      store.Reset(MakeSpecWithVariables(kVariables), AlignedView()),
      test::IsErrorWithSubstr("Flat variables must have 1 view"));
}

// Tests that the store accepts FORMAT_ROW_MAJOR_MATRIX variables with one
// sub-view.
TEST(ArrayVariableStoreTest, MatrixWithOneRow) {
  const string kVariables =
      "variable { format: FORMAT_ROW_MAJOR_MATRIX num_views: 1 view_size: 0 }";
  ArrayVariableStore store;
  TF_EXPECT_OK(store.Reset(MakeSpecWithVariables(kVariables), AlignedView()));
}

// Tests that the store rejects variables that overrun the main byte array.
TEST(ArrayVariableStoreTest, VariableOverrunsMainByteArray) {
  const string kVariables =
      "variable { format: FORMAT_FLAT num_views: 1 view_size: 1024 }";
  AlignedView data;
  TF_ASSERT_OK(data.Reset(nullptr, 1023));

  ArrayVariableStore store;
  EXPECT_THAT(
      store.Reset(MakeSpecWithVariables(kVariables), data),
      test::IsErrorWithSubstr("Variable would overrun main byte array"));
}

// Tests that the store rejects duplicate variables.
TEST(ArrayVariableStoreTest, DuplicateVariables) {
  const string kVariables = R"(
    variable { name: 'x' format: FORMAT_FLAT num_views: 1 view_size: 1024 }
    variable { name: 'y' format: FORMAT_FLAT num_views: 1 view_size: 2048 }
    variable { name: 'x' format: FORMAT_FLAT num_views: 1 view_size: 4096 }
  )";
  AlignedView data;
  TF_ASSERT_OK(data.Reset(nullptr, 1 << 20));  // 1MB

  ArrayVariableStore store;
  EXPECT_THAT(store.Reset(MakeSpecWithVariables(kVariables), data),
              test::IsErrorWithSubstr("Duplicate variable"));
}

// Tests that the store rejects sets of variables that do not completely cover
// the main byte array.
TEST(ArrayVariableStoreTest, LeftoverBytesInMainByteArray) {
  const string kVariables = R"(
    variable { name: 'x' format: FORMAT_FLAT num_views: 1 view_size: 1024 }
    variable { name: 'y' format: FORMAT_FLAT num_views: 1 view_size: 2048 }
    variable { name: 'z' format: FORMAT_FLAT num_views: 1 view_size: 4096 }
  )";
  AlignedView data;
  TF_ASSERT_OK(data.Reset(nullptr, 1 << 20));  // 1MB

  ArrayVariableStore store;
  EXPECT_THAT(store.Reset(MakeSpecWithVariables(kVariables), data),
              test::IsErrorWithSubstr(
                  "Variables do not completely cover main byte array"));
}

// The fast matrix-vector routines do not support padding.
TEST(ArrayVariableStoreTest, PaddingInBlockedMatrix) {
  const string kVariables = R"(
    variable {
      name: "baz"
      format: FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX
      num_views: 4
      view_size: 16
      dimension: 2
      dimension: 4
      dimension: 2
    }
  )";
  AlignedView data;
  TF_ASSERT_OK(data.Reset(nullptr, 1 << 20));  // 1MB

  ArrayVariableStore store;
  EXPECT_THAT(store.Reset(MakeSpecWithVariables(kVariables), data),
              test::IsErrorWithSubstr(
                  "Currently, fast matrix-vector operations do not support "
                  "padded blocked matrices"));
}

// Tests that the store cannot retrieve variables when it is uninitialized.
TEST(ArrayVariableStoreTest, LookupWhenUninitialized) {
  ArrayVariableStore store;
  Vector<float> vector;
  EXPECT_THAT(store.Lookup("foo", &vector),
              test::IsErrorWithSubstr("ArrayVariableStore not initialized"));
}

// Tests that the store can use an empty byte array when there are no variables.
TEST(ArrayVariableStoreTest, EmptyByteArrayWorksIfNoVariables) {
  ArrayVariableStore store;
  TF_EXPECT_OK(store.Reset(MakeSpecWithVariables(""), AlignedView()));

  // The store contains nothing.
  Vector<float> vector;
  EXPECT_THAT(
      store.Lookup("foo", &vector),
      test::IsErrorWithSubstr("ArrayVariableStore has no variable with name "
                              "'foo' and format FORMAT_FLAT"));
}

// Tests that the store fails if it is closed before it has been initialized.
TEST(ArrayVariableStoreTest, CloseBeforeReset) {
  ArrayVariableStore store;
  EXPECT_THAT(store.Close(),
              test::IsErrorWithSubstr("ArrayVariableStore not initialized"));
}

// Tests that the store can be closed (once) after it has been initialized.
TEST(ArrayVariableStoreTest, CloseAfterReset) {
  ArrayVariableStore store;
  TF_ASSERT_OK(store.Reset(MakeSpecWithVariables(""), AlignedView()));
  TF_EXPECT_OK(store.Close());

  // Closing twice is still an error.
  EXPECT_THAT(store.Close(),
              test::IsErrorWithSubstr("ArrayVariableStore not initialized"));
}

// Templated on an ArrayVariableStore subclass.
template <class Subclass>
class ArrayVariableStoreSubclassTest : public ::testing::Test {};

typedef ::testing::Types<FileArrayVariableStore, MmapArrayVariableStore>
    Subclasses;
TYPED_TEST_CASE(ArrayVariableStoreSubclassTest, Subclasses);

// Tests that the store fails to load a non-existent file.
TYPED_TEST(ArrayVariableStoreSubclassTest, NonExistentFile) {
  // Paths to the spec and data produced by array_variable_store_builder_test.
  const string kDataPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(), "dragnn/runtime/testdata/non_existent_file");

  TypeParam store;
  EXPECT_THAT(store.Reset(MakeSpecWithVariables(""), kDataPath),
              test::IsErrorWithSubstr(""));
}

// Tests that the store can load an empty file if there are no variables.
TYPED_TEST(ArrayVariableStoreSubclassTest, EmptyFile) {
  // Paths to the spec and data produced by array_variable_store_builder_test.
  const string kDataPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(), "dragnn/runtime/testdata/empty_file");

  TypeParam store;
  TF_ASSERT_OK(store.Reset(MakeSpecWithVariables(""), kDataPath));

  Vector<float> vector;
  Matrix<float> row_major_matrix;
  EXPECT_THAT(store.Lookup("foo", &vector),
              test::IsErrorWithSubstr("ArrayVariableStore has no variable with "
                                      "name 'foo' and format FORMAT_FLAT"));
  EXPECT_THAT(
      store.Lookup("bar", &row_major_matrix),
      test::IsErrorWithSubstr("ArrayVariableStore has no variable with name "
                              "'bar' and format FORMAT_ROW_MAJOR_MATRIX"));
}

// Tests that the store, when loading a pre-built byte array, produces the same
// variables that the builder converted.
TYPED_TEST(ArrayVariableStoreSubclassTest, RegressionTest) {
  // Paths to the spec and data produced by array_variable_store_builder_test.
  const string kSpecPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(),
      "dragnn/runtime/testdata/array_variable_store_spec");
  const string kDataPath = tensorflow::io::JoinPath(
      test::GetTestDataPrefix(),
      "dragnn/runtime/testdata/array_variable_store_data");

  ArrayVariableStoreSpec spec;
  TF_CHECK_OK(
      tensorflow::ReadTextProto(tensorflow::Env::Default(), kSpecPath, &spec));

  TypeParam store;
  TF_ASSERT_OK(store.Reset(spec, kDataPath));

  Matrix<float> foo;
  TF_ASSERT_OK(store.Lookup("foo", &foo));

  // NB: These assertions must be kept in sync with the variables defined in
  // array_variable_store_builder_test.cc.
  ExpectMatrix(foo, {{0.0, 0.5, 1.0},  //
                     {1.5, 2.0, 2.5},  //
                     {3.0, 3.5, 4.0},  //
                     {4.5, 5.0, 5.5}});

  // Blocked formats.
  BlockedMatrix<double> baz;
  TF_ASSERT_OK(store.Lookup("baz", &baz));
  EXPECT_EQ(baz.num_rows(), 2);
  EXPECT_EQ(baz.num_columns(), 8);
  EXPECT_EQ(baz.block_size(), 4);
  ExpectBlockedData(baz, {{1.0, 2.0, 2.0, 2.0},  //
                          {3.0, 4.0, 4.0, 4.0},  //
                          {5.0, 6.0, 6.0, 6.0},  //
                          {7.0, 8.0, 8.0, 8.0}});

  // Try versions of "foo" and "baz" with the wrong format.
  Vector<float> vector;
  Matrix<float> row_major_matrix;
  EXPECT_THAT(store.Lookup("foo", &vector),
              test::IsErrorWithSubstr("ArrayVariableStore has no variable with "
                                      "name 'foo' and format FORMAT_FLAT"));
  EXPECT_THAT(store.Lookup("baz", &vector),
              test::IsErrorWithSubstr("ArrayVariableStore has no variable with "
                                      "name 'baz' and format FORMAT_FLAT"));
  EXPECT_THAT(
      store.Lookup("baz", &row_major_matrix),
      test::IsErrorWithSubstr("ArrayVariableStore has no variable with name "
                              "'baz' and format FORMAT_ROW_MAJOR_MATRIX"));

  // Try totally unknown variables.
  EXPECT_THAT(store.Lookup("missing", &vector),
              test::IsErrorWithSubstr("ArrayVariableStore has no variable with "
                                      "name 'missing' and format FORMAT_FLAT"));
  EXPECT_THAT(
      store.Lookup("missing", &row_major_matrix),
      test::IsErrorWithSubstr("ArrayVariableStore has no variable with name "
                              "'missing' and format FORMAT_ROW_MAJOR_MATRIX"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

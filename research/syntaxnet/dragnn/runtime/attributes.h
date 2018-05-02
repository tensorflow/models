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

// Utils for parsing configuration attributes from (name,value) string pairs as
// typed values.  Intended for parsing RegisteredModuleSpec.parameters, similar
// to get_attrs_with_defaults() in network_units.py.  Example usage:
//
// // Create a subclass of Attributes.
// struct MyComponentAttributes : public Attributes {
//   // Mandatory attribute with type and name.  The "this" allows the attribute
//   // to register itself in its container---i.e., MyComponentAttributes.
//   Mandatory<float> coefficient{"coefficient", this};
//
//   // Optional attributes with type, name, and default value.
//   Optional<bool> ignore_case{"ignore_case", true, this};
//   Optional<std::vector<int32>> layer_sizes{"layer_sizes", {1, 2, 3}, this};
//
//   // Ignored attribute, which does not parse any value.
//   Ignored dropout_keep_prob{"dropout_keep_prob", this};
// };
//
// // Initialize an instance of the subclass from a string-to-string mapping.
// RegisteredModuleSpec spec;
// MyComponentAttributes attributes;
// TF_RETURN_IF_ERROR(attributes.Reset(spec.parameters()));
//
// // Access the attributes as accessors.
// bool ignore_case = attributes.ignore_case();
// float coefficient = attributes.coefficient();
// const std::vector<int32> &layer_sizes = attributes.layer_sizes();
//
// See the unit test for additional usage examples.
//
// TODO(googleuser): Build typed attributes into the RegisteredModuleSpec and
// get rid of this module.

#ifndef DRAGNN_RUNTIME_ATTRIBUTES_H_
#define DRAGNN_RUNTIME_ATTRIBUTES_H_

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for sets of attributes.  Use as indicated in the file comment.
class Attributes {
 public:
  // Untyped mapping from which typed attributes are parsed.
  using Mapping = tensorflow::protobuf::Map<string, string>;

  // Forbids copying, which would invalidate the pointers in |attributes_|.
  Attributes(const Attributes &that) = delete;
  Attributes &operator=(const Attributes &that) = delete;

  // Parses registered attributes from the name-to-value |mapping|.  On error,
  // returns non-OK.  Errors include unknown names in |mapping|, string-to-value
  // parsing failures, and missing mandatory attributes.
  tensorflow::Status Reset(const Mapping &mapping);

 protected:
  // Implementations of the supported kinds of attributes, defined below.
  class Ignored;
  template <class T>
  class Optional;
  template <class T>
  class Mandatory;

  // Forbids lifecycle management except via subclasses.
  Attributes() = default;
  virtual ~Attributes() = default;

 private:
  // Base class for an individual attribute, defined below.
  class Attribute;

  // Registers the |attribute| with the |name|, which must be unique.
  void Register(const string &name, Attribute *attribute);

  // Parses the string |str| into the |value| object.
  static tensorflow::Status ParseValue(const string &str, string *value);
  static tensorflow::Status ParseValue(const string &str, bool *value);
  static tensorflow::Status ParseValue(const string &str, int32 *value);
  static tensorflow::Status ParseValue(const string &str, int64 *value);
  static tensorflow::Status ParseValue(const string &str, size_t *value);
  static tensorflow::Status ParseValue(const string &str, float *value);
  template <class Element>
  static tensorflow::Status ParseValue(const string &str,
                                       std::vector<Element> *value);

  // Registered attributes, keyed by name.
  std::map<string, Attribute *> attributes_;
};

// Implementation details below.

// Base class for individual attributes.
class Attributes::Attribute {
 public:
  Attribute() = default;
  Attribute(const Attribute &that) = delete;
  Attribute &operator=(const Attribute &that) = delete;
  virtual ~Attribute() = default;

  // Parses the |value| string into a typed object.  On error, returns non-OK.
  virtual tensorflow::Status Parse(const string &value) = 0;

  // Returns true if this is a mandatory attribute.  Defaults to optional.
  virtual bool IsMandatory() const { return false; }
};

// Implements an ignored attribute.
class Attributes::Ignored : public Attribute {
 public:
  // Registers this in the |attributes| with the |name|.
  Ignored(const string &name, Attributes *attributes) {
    attributes->Register(name, this);
  }

  // Ignores the |value|.
  tensorflow::Status Parse(const string &value) override {
    return tensorflow::Status::OK();
  }
};

// Implements an optional attribute.
template <class T>
class Attributes::Optional : public Attribute {
 public:
  // Registers this in the |attributes| with the |name| and |default_value|.
  Optional(const string &name, const T &default_value, Attributes *attributes)
      : value_(default_value) {
    attributes->Register(name, this);
  }

  // Parses the |value| into the |value_|.
  tensorflow::Status Parse(const string &value) override {
    return ParseValue(value, &value_);
  }

  // Returns the parsed |value_|.  Overloading operator() allows a struct member
  // to be called like an accessor.
  const T &operator()() const { return value_; }

 private:
  // The parsed value, or the default value if not explicitly specified.
  T value_;
};

// Implements a mandatory attribute.
template <class T>
class Attributes::Mandatory : public Optional<T> {
 public:
  // Registers this in the |attributes| with the |name|.
  Mandatory(const string &name, Attributes *attributes)
      : Optional<T>(name, T(), attributes) {}

  // Returns true since this is mandatory.
  bool IsMandatory() const override { return true; }

 private:
  // The parsed value, or the default value if not explicitly specified.
  T value_;
};

template <class Element>
tensorflow::Status Attributes::ParseValue(const string &str,
                                          std::vector<Element> *value) {
  value->clear();
  if (!str.empty()) {
    for (const string &element_str : tensorflow::str_util::Split(str, ",")) {
      value->emplace_back();
      TF_RETURN_IF_ERROR(ParseValue(element_str, &value->back()));
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_ATTRIBUTES_H_

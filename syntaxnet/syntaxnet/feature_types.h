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

// Common feature types for parser components.

#ifndef SYNTAXNET_FEATURE_TYPES_H_
#define SYNTAXNET_FEATURE_TYPES_H_

#include <algorithm>
#include <map>
#include <string>
#include <utility>

#include "syntaxnet/utils.h"

namespace syntaxnet {

// Use the same type for feature values as is used for predicated.
typedef int64 Predicate;
typedef Predicate FeatureValue;

// Each feature value in a feature vector has a feature type. The feature type
// is used for converting feature type and value pairs to predicate values. The
// feature type can also return names for feature values and calculate the size
// of the feature value domain. The FeatureType class is abstract and must be
// specialized for the concrete feature types.
class FeatureType {
 public:
  // Initializes a feature type.
  explicit FeatureType(const string &name)
      : name_(name), base_(0) {}

  virtual ~FeatureType() {}

  // Converts a feature value to a name.
  virtual string GetFeatureValueName(FeatureValue value) const = 0;

  // Returns the size of the feature values domain.
  virtual int64 GetDomainSize() const = 0;

  // Returns the feature type name.
  const string &name() const { return name_; }

  Predicate base() const { return base_; }
  void set_base(Predicate base) { base_ = base; }

 private:
  // Feature type name.
  string name_;

  // "Base" feature value: i.e. a "slot" in a global ordering of features.
  Predicate base_;
};

// Templated generic resource based feature type. This feature type delegates
// look up of feature value names to an unknown resource class, which is not
// owned. Optionally, this type can also store a mapping of extra values which
// are not in the resource.
//
// Note: this class assumes that Resource->GetFeatureValueName() will return
// successfully for values ONLY in the range [0, Resource->NumValues()) Any
// feature value not in the extra value map and not in the above range of
// Resource will result in a ERROR and return of "<INVALID>".
template<class Resource>
class ResourceBasedFeatureType : public FeatureType {
 public:
  // Creates a new type with given name, resource object, and a mapping of
  // special values. The values must be greater or equal to
  // resource->NumValues() so as to avoid collisions; this is verified with
  // CHECK at creation.
  ResourceBasedFeatureType(const string &name, const Resource *resource,
                           const map<FeatureValue, string> &values)
      : FeatureType(name), resource_(resource), values_(values) {
    max_value_ = resource->NumValues() - 1;
    for (const auto &pair : values) {
      CHECK_GE(pair.first, resource->NumValues()) << "Invalid extra value: "
               << pair.first << "," << pair.second;
      max_value_ = pair.first > max_value_ ? pair.first : max_value_;
    }
  }

  // Creates a new type with no special values.
  ResourceBasedFeatureType(const string &name, const Resource *resource)
      : ResourceBasedFeatureType(name, resource, {}) {}

  // Returns the feature name for a given feature value. First checks the values
  // map, then checks the resource to look up the name.
  string GetFeatureValueName(FeatureValue value) const override {
    if (values_.find(value) != values_.end()) {
      return values_.find(value)->second;
    }
    if (value >= 0 && value < resource_->NumValues()) {
      return resource_->GetFeatureValueName(value);
    } else {
      LOG(ERROR) << "Invalid feature value " << value << " for " << name();
      return "<INVALID>";
    }
  }

  // Returns the number of possible values for this feature type. This is the
  // based on the largest value that was observed in the extra values.
  FeatureValue GetDomainSize() const override { return max_value_ + 1; }

 protected:
  // Shared resource. Not owned.
  const Resource *resource_ = nullptr;

  // Maximum possible value this feature could take.
  FeatureValue max_value_;

  // Mapping for extra feature values not in the resource.
  map<FeatureValue, string> values_;
};

// Feature type that is defined using an explicit map from FeatureValue to
// string values.  This can reduce some of the boilerplate when defining
// features that generate enum values.  Example usage:
//
//   class BeverageSizeFeature : public FeatureFunction<Beverage>
//     enum FeatureValue { SMALL, MEDIUM, LARGE };  // values for this feature
//     void Init(TaskContext *context) override {
//       set_feature_type(new EnumFeatureType("beverage_size",
//           {{SMALL, "SMALL"}, {MEDIUM, "MEDIUM"}, {LARGE, "LARGE"}});
//     }
//     [...]
//   };
class EnumFeatureType : public FeatureType {
 public:
  EnumFeatureType(const string &name,
                  const map<FeatureValue, string> &value_names)
      : FeatureType(name), value_names_(value_names) {
    for (const auto &pair : value_names) {
      CHECK_GE(pair.first, 0)
          << "Invalid feature value: " << pair.first << ", " << pair.second;
      domain_size_ = std::max(domain_size_, pair.first + 1);
    }
  }

  // Returns the feature name for a given feature value.
  string GetFeatureValueName(FeatureValue value) const override {
    auto it = value_names_.find(value);
    if (it == value_names_.end()) {
      LOG(ERROR)
          << "Invalid feature value " << value << " for " << name();
      return "<INVALID>";
    }
    return it->second;
  }

  // Returns the number of possible values for this feature type. This is one
  // greater than the largest value in the value_names map.
  FeatureValue GetDomainSize() const override { return domain_size_; }

 protected:
  // Maximum possible value this feature could take.
  FeatureValue domain_size_ = 0;

  // Names of feature values.
  map<FeatureValue, string> value_names_;
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_FEATURE_TYPES_H_

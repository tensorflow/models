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
// Generic feature functions. These feature functions are independent of the
// feature function template types.
//
// The generic features should be instantiated and registered using the
// REGISTER_SYNTAXNET_GENERIC_FEATURES() macro:
//
// typedef GenericFeatures<Foo, int> GenericFooFeatures;
// REGISTER_SYNTAXNET_GENERIC_FEATURES(GenericFooFeatures);
//

#ifndef SYNTAXNET_GENERIC_FEATURES_H_
#define SYNTAXNET_GENERIC_FEATURES_H_

#include <string>
#include <utility>
#include <vector>

#include "syntaxnet/base.h"
#include "syntaxnet/feature_extractor.h"

namespace syntaxnet {

class TaskContext;
class WorkspaceSet;

// A class encapsulating all generic feature types.
class GenericFeatureTypes {
 public:
  // Base class for tuple feature types.
  class TupleFeatureTypeBase : public FeatureType {
   public:
    // Creates a tuple whose elements are defined by the sub-types.  This does
    // not take ownership of the sub-types, which must remain live while this
    // is in use.
    TupleFeatureTypeBase(const string &prefix,
                         const std::vector<FeatureType *> &sub_types);

    // Returns a string representation of the tuple value.
    string GetFeatureValueName(FeatureValue value) const override;

    // Returns the domain size of this feature.
    FeatureValue GetDomainSize() const override;

   protected:
    // Sets the feature domain sizes and computes the total domain size of the
    // tuple.  Derived classes should call this method from their constructor.
    void InitDomainSizes(vector<FeatureValue> *sizes);

   private:
    // Returns a string name for a type using the prefix and sub-types.
    static string CreateTypeName(const string &prefix,
                                 const std::vector<FeatureType *> &sub_types);

    // The types of the sub-features.  Not owned.
    const std::vector<const FeatureType *> types_;

    // The domain size of the tuple.
    FeatureValue size_ = 0;
  };

  // Feature type for tuples of fixed size.
  template <int kNumElements>
  class StaticTupleFeatureType : public TupleFeatureTypeBase {
   public:
    static_assert(kNumElements >= 2, "At least two elements required");

    // Creates a fixed-size tuple of sub-types.  This does not take ownership
    // of the sub-types, which must remain live while this is in use.
    StaticTupleFeatureType(const string &prefix,
                           const std::vector<FeatureType *> &sub_types)
        : TupleFeatureTypeBase(prefix, sub_types) {
      CHECK_EQ(sub_types.size(), kNumElements);
      sizes_.resize(kNumElements);
      InitDomainSizes(&sizes_);
    }

    // Returns the conjoined tuple value for a list of sub-values.  The range
    // values[0,kNumElements) must be valid and non-absent.
    FeatureValue Conjoin(const FeatureValue *values) const {
      DCHECK_GE(values[kNumElements - 1], 0);
      DCHECK_LT(values[kNumElements - 1], sizes_[kNumElements - 1]);
      DCHECK_NE(values[kNumElements - 1], GenericFeatureFunction::kNone);
      FeatureValue conjoined = values[kNumElements - 1];
      for (int i = kNumElements - 2; i >= 0; --i) {
        DCHECK_GE(values[i], 0);
        DCHECK_LT(values[i], sizes_[i]);
        DCHECK_NE(values[i], GenericFeatureFunction::kNone);
        conjoined = values[i] + conjoined * sizes_[i];
      }
      return conjoined;
    }

   private:
    // The domain sizes of the sub-types.
    vector<FeatureValue> sizes_;
  };

  // Feature type for tuples of dynamic size.
  class DynamicTupleFeatureType : public TupleFeatureTypeBase {
   public:
    // Creates a tuple of sub-types.  This does not take ownership of the
    // sub-types, which must remain live while this is in use.
    DynamicTupleFeatureType(const string &prefix,
                            const std::vector<FeatureType *> &sub_types);

    // Returns the conjoined tuple value for a list of sub-values, which must
    // be the same size as the number of elements and non-absent.
    FeatureValue Conjoin(const std::vector<FeatureValue> &values) const {
      DCHECK_EQ(values.size(), sizes_.size());
      DCHECK_GE(values.back(), 0);
      DCHECK_LT(values.back(), sizes_.back());
      DCHECK_NE(values.back(), GenericFeatureFunction::kNone);
      FeatureValue conjoined = values.back();
      for (int i = static_cast<int>(sizes_.size()) - 2; i >= 0; --i) {
        DCHECK_GE(values[i], 0);
        DCHECK_LT(values[i], sizes_[i]);
        DCHECK_NE(values[i], GenericFeatureFunction::kNone);
        conjoined = values[i] + conjoined * sizes_[i];
      }
      return conjoined;
    }

   private:
    // The domain sizes of the sub-types.
    std::vector<FeatureValue> sizes_;
  };

  // A wrapper which simply delegates to the sub-type.  This does not take
  // ownership of the sub-type, which must remain live while this is in use.
  class WrappedFeatureType : public FeatureType {
   public:
    explicit WrappedFeatureType(FeatureType *sub_type)
        : FeatureType(sub_type->name()), sub_type_(sub_type) {}

    string GetFeatureValueName(FeatureValue value) const override {
      return sub_type_->GetFeatureValueName(value);
    }

    FeatureValue GetDomainSize() const override {
      return sub_type_->GetDomainSize();
    }

   private:
    FeatureType *sub_type_;
  };
};

// A class encapsulating all generic feature functions.
template <class OBJ, class... ARGS>
class GenericFeatures {
 public:
  // Base class for feature functions.
  typedef FeatureFunction<OBJ, ARGS...> Base;

  // Base class for nested feature functions: these still have their own feature
  // type, so make sure not to pass to the nested ones.
  class MetaBase : public MetaFeatureFunction<OBJ, ARGS...> {
   public:
    // Don't use the nested logic for feature types by default.
    void GetFeatureTypes(std::vector<FeatureType *> *types) const override {
      GenericFeatureFunction::GetFeatureTypes(types);
    }
  };

  // Feature function that adds a bias value to the feature vector.
  class Bias : public Base {
    enum BiasFeatureValue { ON };

   public:
    // Initializes the feature.
    void Init(TaskContext *context) override {
      this->set_feature_type(
          new EnumFeatureType(this->name(), {{BiasFeatureValue::ON, "ON"}}));
    }

    // Returns the bias value.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      return 0;
    }
  };

  // Feature function that returns a constant value.
  class Constant : public Base {
   public:
    // Initializes the feature.
    void Init(TaskContext *context) override {
      value_ = this->GetIntParameter("value", 0);
      this->set_feature_type(new NumericFeatureType(this->name(), value_ + 1));
    }

    // Returns the constant's value.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      return value_;
    }

   private:
    int value_ = 0;
  };

  // A feature function that tests equality between two nested features.  This
  // can be used, for example, to check morphological agreement.
  class Equals : public MetaBase {
    enum EqualsFeatureValue { DIFFERENT, EQUAL };

   public:
    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      const auto &nested = this->nested();
      CHECK_EQ(nested.size(), 2)
          << "The 'equals' feature requires two nested features.";
      this->set_feature_type(new EnumFeatureType(
          this->name(), {{EqualsFeatureValue::DIFFERENT, "DIFFERENT"},
                         {EqualsFeatureValue::EQUAL, "EQUAL"}}));
    }

    // Returns the equality value, or kNone if either value is absent.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      const auto &nested = this->nested();
      const FeatureValue a =
          nested[0]->Compute(workspaces, object, args..., fv);
      if (a == Base::kNone) return Base::kNone;
      const FeatureValue b =
          nested[1]->Compute(workspaces, object, args..., fv);
      if (b == Base::kNone) return Base::kNone;
      return a == b ? 1 : 0;
    }
  };

  // Abstract base class for features that compare a nested feature's value
  // to a target value (specified via the 'value' parameter).
  //
  // Subclasses must implement InitTypes() and ComputeValue().
  class CompareValue : public MetaBase {
   public:
    // Initialize the type information.
    virtual void InitTypes() = 0;

    // Compute the feature value given the nested feature value and the target
    // value (i.e., what was passed as the 'value' parameter).
    virtual FeatureValue ComputeValue(FeatureValue nested_feature_value,
                                      FeatureValue target_value) const = 0;

    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      string value_str = this->GetParameter("value");
      CHECK_GT(value_str.size(), 0)
          << "The '" << this->FunctionName()
          << "' feature requires a 'value' parameter.";

      const auto &nested = this->nested();
      CHECK_EQ(nested.size(), 1) << "The '" << this->FunctionName()
                                 << "' feature requires one nested feature.";

      // Only allow nested features with exactly one feature type.
      FeatureType *nested_feature_type =
          CHECK_NOTNULL(nested.front()->GetFeatureType());

      for (int i = 0; i < nested_feature_type->GetDomainSize(); ++i) {
        if (nested_feature_type->GetFeatureValueName(i) == value_str) {
          value_ = i;
          break;
        }
      }

      CHECK_NE(value_, -1) << "Unknown feature value specified: " << value_str
                           << ".";

      InitTypes();
    }

    // Extracts the nested feature value, and delegates computation of the
    // final feature value to ComputeValue().
    // Returns kNone if the nested feature value is absent.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      const auto &nested = this->nested();
      FeatureValue feature_value =
          nested.front()->Compute(workspaces, object, args..., fv);
      if (feature_value == Base::kNone) return Base::kNone;
      return ComputeValue(feature_value, value_);
    }

   private:
    // The value to compare the feature against.
    int value_ = -1;
  };

  // A feature function that fires if and only if the nested feature has the
  // given value.
  class Filter : public CompareValue {
    enum FilterFeatureValue { ON };

   public:
    void InitTypes() override {
      this->set_feature_type(
          new EnumFeatureType(this->name(), {{FilterFeatureValue::ON, "ON"}}));
    }

    FeatureValue ComputeValue(FeatureValue nested_feature_value,
                              FeatureValue target_value) const override {
      return nested_feature_value == target_value ? 0 : Base::kNone;
    }
  };

  // A feature function that tests equality between a feature and a value.
  class Is : public CompareValue {
    enum IsFeatureValue { FALSE, TRUE };

   public:
    void InitTypes() override {
      this->set_feature_type(new EnumFeatureType(
          this->name(),
          {{IsFeatureValue::FALSE, "FALSE"}, {IsFeatureValue::TRUE, "TRUE"}}));
    }

    FeatureValue ComputeValue(FeatureValue nested_feature_value,
                              FeatureValue target_value) const override {
      return nested_feature_value == target_value;
    }
  };

  // A feature function that forwards the nested feature value, unless it equals
  // the target value (in which case, the feature doesn't fire).
  class Ignore : public CompareValue {
   public:
    void InitTypes() override {
      this->set_feature_type(new GenericFeatureTypes::WrappedFeatureType(
          this->nested().front()->GetFeatureType()));
    }

    FeatureValue ComputeValue(FeatureValue nested_feature_value,
                              FeatureValue target_value) const override {
      return nested_feature_value == target_value
                 ? GenericFeatureFunction::kNone
                 : nested_feature_value;
    }
  };

  // Abstract base class for features that reduce several binary values to a
  // to a single binary value.
  //
  // Subclasses must implement Compute().
  class BinaryReduce : public MetaBase {
    enum BinaryReduceFeatureValue { FALSE, TRUE };

   public:
    // Initializes the feature.
    // Checks that all the nested features are binary, and sets the output
    // feature type to binary.
    void InitNested(TaskContext *context) override {
      for (const Base *function : this->nested()) {
        FeatureType *nested_type = CHECK_NOTNULL(function->GetFeatureType());
        CHECK_EQ(nested_type->GetDomainSize(), 2)
            << this->name() << " requires nested binary feature types only.";
      }
      this->set_feature_type(new EnumFeatureType(
          this->name(), {{BinaryReduceFeatureValue::FALSE, "FALSE"},
                         {BinaryReduceFeatureValue::TRUE, "TRUE"}}));
    }
  };

  // A feature function that takes any number of binary nested features, and
  // returns whether they all evaluate to 1.
  class All : public BinaryReduce {
   public:
    // Returns whether all nested feature values are 1, or kNone if any of them
    // are unavailable.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      for (const Base *function : this->nested()) {
        const FeatureValue value =
            function->Compute(workspaces, object, args..., fv);
        if (value == Base::kNone) return Base::kNone;
        if (value == 0) return 0;
      }
      return 1;
    }
  };

  // A feature function that takes any number of binary nested features, and
  // returns whether any of them evaluate to 1.
  class Any : public BinaryReduce {
   public:
    // Returns whether any nested feature values are 1, or kNone if any of them
    // are unavailable.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      for (const Base *function : this->nested()) {
        const FeatureValue value =
            function->Compute(workspaces, object, args..., fv);
        if (value == Base::kNone) return Base::kNone;
        if (value == 1) return 1;
      }
      return 0;
    }
  };

  // A feature function that computes a fixed-size tuple.
  template <int kNumElements>
  class StaticTuple : public MetaBase {
   public:
    // The associated fixed-size tuple type.
    typedef GenericFeatureTypes::StaticTupleFeatureType<kNumElements> Type;

    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      std::vector<FeatureType *> sub_types;
      for (const Base *function : this->nested()) {
        sub_types.push_back(CHECK_NOTNULL(function->GetFeatureType()));
      }
      this->set_feature_type(new Type(this->SubPrefix(), sub_types));
    }

    // Returns the tuple value, or kNone if any sub-value is unavailable.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      const auto &nested = this->nested();
      FeatureValue values[kNumElements];
      for (int i = 0; i < kNumElements; ++i) {
        const FeatureValue value =
            nested[i]->Compute(workspaces, object, args..., fv);
        if (value == Base::kNone) return Base::kNone;
        values[i] = value;
      }
      return static_cast<Type *>(this->feature_type())->Conjoin(values);
    }
  };

  // Convenience aliases for common fixed-size tuples.
  typedef StaticTuple<2> Pair;
  typedef StaticTuple<3> Triple;
  typedef StaticTuple<4> Quad;
  typedef StaticTuple<5> Quint;

  // A feature function that computes a dynamically-sized tuple.
  class Tuple : public MetaBase {
   public:
    // The associated tuple type.
    typedef GenericFeatureTypes::DynamicTupleFeatureType Type;

    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      std::vector<FeatureType *> sub_types;
      for (const Base *function : this->nested()) {
        sub_types.push_back(CHECK_NOTNULL(function->GetFeatureType()));
      }
      this->set_feature_type(new Type(this->SubPrefix(), sub_types));
    }

    // Returns the tuple value, or kNone if any sub-value is unavailable.
    FeatureValue Compute(const WorkspaceSet &workspaces, const OBJ &object,
                         ARGS... args, const FeatureVector *fv) const override {
      std::vector<FeatureValue> values;
      for (const Base *function : this->nested()) {
        const FeatureValue value =
            function->Compute(workspaces, object, args..., fv);
        if (value == Base::kNone) return Base::kNone;
        values.push_back(value);
      }
      return static_cast<Type *>(this->feature_type())->Conjoin(values);
    }
  };

  // A feature function that creates all pairs of the features extracted by the
  // nested feature functions. All the nested feature functions must return
  // single valued features.
  //
  // Parameters:
  // bool unary (false):
  //   If true, then unary features are also emitted.
  class Pairs : public MetaBase {
   public:
    // The pair feature type.
    typedef GenericFeatureTypes::StaticTupleFeatureType<2> Type;

    // Discards the pair types.
    ~Pairs() override {
      for (Type *type : pairs_) delete type;
    }

    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      unary_ = this->GetParameter("unary") == "true";
      const auto &nested = this->nested();
      CHECK_GE(nested.size(), 2)
          << "The 'pairs' feature requires at least two sub-features.";

      // Get the types of all nested features.
      types_.clear();
      for (const Base *function : nested) {
        types_.push_back(CHECK_NOTNULL(function->GetFeatureType()));
      }

      // Initialize the pair types for all features.
      pairs_.resize(NumPairs(nested.size()));
      for (int right = 1; right < nested.size(); ++right) {
        for (int left = 0; left < right; ++left) {
          pairs_[PairIndex(left, right)] =
              new Type(this->SubPrefix(), {types_[left], types_[right]});
        }
      }
    }

    // Produces all feature types.
    void GetFeatureTypes(std::vector<FeatureType *> *types) const override {
      if (unary_) types->insert(types->end(), types_.begin(), types_.end());
      types->insert(types->end(), pairs_.begin(), pairs_.end());
    }

    // Evaluates the feature.
    void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
                  ARGS... args, FeatureVector *result) const override {
      const auto &nested = this->nested();

      // Collect all active feature sub-values.
      std::vector<FeatureValue> values(nested.size());
      std::vector<int> active_indices;
      active_indices.reserve(nested.size());
      for (int i = 0; i < nested.size(); ++i) {
        values[i] = nested[i]->Compute(workspaces, object, args..., result);
        if (values[i] != Base::kNone) active_indices.push_back(i);
      }

      // Optionally generate unary features.
      if (unary_) {
        for (int index : active_indices) {
          result->add(types_[index], values[index]);
        }
      }

      // Generate all feature pairs.
      FeatureValue pair_values[2];
      for (int right = 1; right < active_indices.size(); ++right) {
        int right_index = active_indices[right];
        pair_values[1] = values[right_index];
        for (int left = 0; left < right; ++left) {
          int left_index = active_indices[left];
          pair_values[0] = values[left_index];
          Type *type = pairs_[PairIndex(left_index, right_index)];
          result->add(type, type->Conjoin(pair_values));
        }
      }
    }

   private:
    // Returns the number of pairs (i,j) where 0 <= i < j < size.
    static int NumPairs(int size) {
      DCHECK_GE(size, 0);
      return (size * (size - 1)) / 2;
    }

    // Returns the index for a pair (left,right) where left < right.  The
    // indices are suitable for densely linearizing pairs into an array.
    static int PairIndex(int left, int right) {
      DCHECK_LE(0, left);
      DCHECK_LT(left, right);
      return left + NumPairs(right);
    }

    // Whether to also emit unary features.
    bool unary_ = false;

    // Feature types for all nested features.  Not owned.
    std::vector<FeatureType *> types_;

    // Feature types for all pairs.  Indexed according to PairIndex().  Owned.
    std::vector<Type *> pairs_;
  };

  // Feature function for conjoining the first sub-feature with each of the
  // rest of the sub-features.
  //
  // Parameters:
  // bool unary (false):
  //   If true, then unary features are also emitted.
  class Conjoin : public MetaBase {
   public:
    // The pair feature type.
    typedef GenericFeatureTypes::StaticTupleFeatureType<2> Type;

    // Discards the pair types.
    ~Conjoin() override {
      for (Type *type : pairs_) delete type;
    }

    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      unary_ = this->GetParameter("unary") == "true";
      const auto &nested = this->nested();
      CHECK_GE(nested.size(), 2)
          << "The 'conjoin' feature requires at least two sub-features.";

      // Get the types of the rest of the nested features.
      types_.clear();
      for (const Base *function : nested) {
        types_.push_back(CHECK_NOTNULL(function->GetFeatureType()));
      }

      // Initialize the pair types.
      pairs_.assign(1, nullptr);
      for (int i = 1; i < types_.size(); ++i) {
        pairs_.push_back(new Type(this->SubPrefix(), {types_[0], types_[i]}));
      }
    }

    // Produces all feature types.
    void GetFeatureTypes(std::vector<FeatureType *> *types) const override {
      if (unary_) types->insert(types->end(), types_.begin() + 1, types_.end());
      types->insert(types->end(), pairs_.begin() + 1, pairs_.end());
    }

    // Evaluates the feature.
    void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
                  ARGS... args, FeatureVector *result) const override {
      const auto &nested = this->nested();
      FeatureValue values[2];
      values[0] = nested[0]->Compute(workspaces, object, args..., result);

      // Stop early if the first feature is absent.
      if (values[0] == Base::kNone) {
        if (unary_) {
          for (int i = 1; i < nested.size(); ++i) {
            values[1] = nested[i]->Compute(workspaces, object, args..., result);
            if (values[1] == Base::kNone) continue;
            result->add(types_[i], values[1]);
          }
        }
        return;
      }

      // Otherwise, the first feature exists; conjoin it with the rest.
      for (int i = 1; i < nested.size(); ++i) {
        values[1] = nested[i]->Compute(workspaces, object, args..., result);
        if (values[1] == Base::kNone) continue;
        if (unary_) result->add(types_[i], values[1]);
        result->add(pairs_[i], pairs_[i]->Conjoin(values));
      }
    }

   private:
    // Whether to also emit unary features.
    bool unary_ = false;

    // Feature types for all nested features.  Not owned.
    std::vector<FeatureType *> types_;

    // Feature types for all pairs.  The first element is null, in order to
    // align this list with types_.  Owned.
    std::vector<Type *> pairs_;
  };

  // Feature function for creating pairs of multi-valued features.  By default,
  // the feature computes the Cartesian product of the extracted sub-features,
  // but a parallel product can be specified via the options.
  //
  // Parameters:
  // bool parallel (false):
  //   If true, output features for parallel pairs, like a dot product.  The
  //   two sub-features must produce identical numbers of features.
  class MultiPair : public MetaBase {
   public:
    // The pair feature type.
    typedef GenericFeatureTypes::StaticTupleFeatureType<2> Type;

    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      parallel_ = this->GetParameter("parallel") == "true";
      std::vector<FeatureType *> sub_types;
      for (const Base *function : this->nested()) {
        sub_types.push_back(CHECK_NOTNULL(function->GetFeatureType()));
      }
      this->set_feature_type(new Type(this->SubPrefix(), sub_types));
    }

    // Evaluates the feature.
    void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
                  ARGS... args, FeatureVector *result) const override {
      const auto &nested = this->nested();
      const int orig_size = result->size();

      // Extract features from left half.  Values are extracted directly into
      // the result so that optimized variable references are handled properly.
      nested[0]->Evaluate(workspaces, object, args..., result);
      if (orig_size == result->size()) return;  // no left features
      std::vector<FeatureValue> left;
      for (int i = orig_size; i < result->size(); ++i) {
        left.push_back(result->value(i));
      }
      result->Truncate(orig_size);

      // Extract features from right half.
      nested[1]->Evaluate(workspaces, object, args..., result);
      if (orig_size == result->size()) return;  // no right features
      std::vector<FeatureValue> right;
      for (int i = orig_size; i < result->size(); ++i) {
        right.push_back(result->value(i));
      }
      result->Truncate(orig_size);

      // Compute the pair values.
      FeatureValue values[2];
      Type *type = static_cast<Type *>(this->feature_type());
      if (parallel_) {
        // Produce parallel pairs.
        CHECK_EQ(left.size(), right.size());
        for (int i = 0; i < left.size(); ++i) {
          values[0] = left[i];
          values[1] = right[i];
          result->add(type, type->Conjoin(values));
        }
      } else {
        // Produce all pairs.
        for (const FeatureValue left_value : left) {
          values[0] = left_value;
          for (const FeatureValue right_value : right) {
            values[1] = right_value;
            result->add(type, type->Conjoin(values));
          }
        }
      }
    }

   private:
    // Whether to do a parallel product instead of a Cartesian product.
    bool parallel_ = false;
  };

  // Feature function for conjoining the first multi-valued sub-feature with
  // each of the rest of the multi-valued sub-features.
  class MultiConjoin : public MetaBase {
   public:
    // The pair feature type.
    typedef GenericFeatureTypes::StaticTupleFeatureType<2> Type;

    // Discards the pair types.
    ~MultiConjoin() override {
      for (Type *type : pairs_) delete type;
    }

    // Initializes the feature.
    void InitNested(TaskContext *context) override {
      const auto &nested = this->nested();
      CHECK_GE(nested.size(), 2)
          << "The 'multiconjoin' feature requires at least two sub-features.";

      // Get the types of the rest of the nested features.
      std::vector<FeatureType *> types;
      types.reserve(nested.size());
      for (const Base *function : nested) {
        types.push_back(CHECK_NOTNULL(function->GetFeatureType()));
      }

      // Initialize the pair types.
      pairs_.clear();
      for (int i = 1; i < types.size(); ++i) {
        pairs_.push_back(new Type(this->SubPrefix(), {types[0], types[i]}));
      }
    }

    // Produces all feature types.
    void GetFeatureTypes(std::vector<FeatureType *> *types) const override {
      types->insert(types->end(), pairs_.begin(), pairs_.end());
    }

    // Evaluates the feature.
    void Evaluate(const WorkspaceSet &workspaces, const OBJ &object,
                  ARGS... args, FeatureVector *result) const override {
      const auto &nested = this->nested();
      const int orig_size = result->size();

      // Gather the lists of sub-values for each nested feature.  Sub-values
      // are extracted directly into the result so that optimized variable
      // references are handled properly.
      std::vector<std::vector<FeatureValue> > sub_values(nested.size());
      for (int i = 0; i < nested.size(); ++i) {
        nested[i]->Evaluate(workspaces, object, args..., result);
        if (orig_size == result->size()) {
          if (i == 0) {
            return;  // no first values; nothing will be extracted
          } else {
            continue;  // no non-first values; skip to next feature
          }
        }
        std::vector<FeatureValue> &values = sub_values[i];
        for (int j = orig_size; j < result->size(); ++j) {
          values.push_back(result->value(j));
        }
        result->Truncate(orig_size);
      }

      // Produce conjoined features.
      const std::vector<FeatureValue> &first_values = sub_values[0];
      FeatureValue values[2];
      for (int i = 1; i < sub_values.size(); ++i) {
        const std::vector<FeatureValue> &other_values = sub_values[i];
        if (other_values.empty()) continue;
        Type *type = pairs_[i - 1];
        for (const FeatureValue first_value : first_values) {
          values[0] = first_value;
          for (const FeatureValue other_value : other_values) {
            values[1] = other_value;
            result->add(type, type->Conjoin(values));
          }
        }
      }
    }

   private:
    // Feature types for all pairs.  Owned.
    std::vector<Type *> pairs_;
  };
};

#define REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, name, type) \
  typedef generics::type __##type##generics;                     \
  REGISTER_SYNTAXNET_FEATURE_FUNCTION(generics::Base, name, __##type##generics)

#define REGISTER_SYNTAXNET_GENERIC_FEATURES(generics)                   \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "bias", Bias);           \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "constant", Constant);   \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "equals", Equals);       \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "filter", Filter);       \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "is", Is);               \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "all", All);             \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "any", Any);             \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "pair", Pair);           \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "triple", Triple);       \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "quad", Quad);           \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "quint", Quint);         \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "tuple", Tuple);         \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "pairs", Pairs);         \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "conjoin", Conjoin);     \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "multipair", MultiPair); \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "ignore", Ignore);       \
  REGISTER_SYNTAXNET_GENERIC_FEATURE(generics, "multiconjoin", MultiConjoin)

}  // namespace syntaxnet

#endif  // SYNTAXNET_GENERIC_FEATURES_H_

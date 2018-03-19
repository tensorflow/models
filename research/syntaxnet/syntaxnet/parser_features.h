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

// Sentence-based features for the transition parser.

#ifndef SYNTAXNET_PARSER_FEATURES_H_
#define SYNTAXNET_PARSER_FEATURES_H_

#include <string>

#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/feature_types.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/workspace.h"

namespace syntaxnet {

// A union used to represent discrete and continuous feature values.
union FloatFeatureValue {
 public:
  explicit FloatFeatureValue(FeatureValue v) : discrete_value(v) {}
  FloatFeatureValue(uint32 i, float w) : id(i), weight(w) {}
  FeatureValue discrete_value;
  struct {
    uint32 id;
    float weight;
  };
};

typedef FeatureFunction<ParserState> ParserFeatureFunction;

// Feature function for the transition parser based on a parser state object and
// a token index. This typically extracts information from a given token.
typedef FeatureFunction<ParserState, int> ParserIndexFeatureFunction;

// Utilities to register the two types of parser features.
#define REGISTER_PARSER_FEATURE_FUNCTION(name, component) \
  REGISTER_SYNTAXNET_FEATURE_FUNCTION(ParserFeatureFunction, name, component)

#define REGISTER_PARSER_IDX_FEATURE_FUNCTION(name, component)           \
  REGISTER_SYNTAXNET_FEATURE_FUNCTION(ParserIndexFeatureFunction, name, \
                                      component)

// Alias for locator type that takes a parser state, and produces a focus
// integer that can be used on nested ParserIndexFeature objects.
template<class DER>
using ParserLocator = FeatureAddFocusLocator<DER, ParserState, int>;

// Alias for Locator type features that take (ParserState, int) signatures and
// call other ParserIndexFeatures.
template<class DER>
using ParserIndexLocator = FeatureLocator<DER, ParserState, int>;

// Feature extractor for the transition parser based on a parser state object.
typedef FeatureExtractor<ParserState> ParserFeatureExtractor;

// A simple wrapper FeatureType that adds a special "<ROOT>" type.
class RootFeatureType : public FeatureType {
 public:
  // Creates a RootFeatureType that wraps a given type and adds the special
  // "<ROOT>" value in root_value.
  RootFeatureType(const string &name, const FeatureType &wrapped_type,
                  int root_value);

  // Returns the feature value name, but with the special "<ROOT>" value.
  string GetFeatureValueName(FeatureValue value) const override;

  // Returns the original number of features plus one for the "<ROOT>" value.
  FeatureValue GetDomainSize() const override;

 private:
  // A wrapped type that handles everything else besides "<ROOT>".
  const FeatureType &wrapped_type_;

  // The reserved root value.
  int root_value_;
};

// Simple feature function that wraps a Sentence based feature
// function. It adds a "<ROOT>" feature value that is triggered whenever the
// focus is the special root token. This class is sub-classed based on the
// extracted arguments of the nested function.
template<class F>
class ParserSentenceFeatureFunction : public ParserIndexFeatureFunction {
 public:
  // Instantiates and sets up the nested feature.
  void Setup(TaskContext *context) override {
    this->feature_.set_descriptor(this->descriptor());
    this->feature_.set_prefix(this->prefix());
    this->feature_.set_extractor(this->extractor());
    feature_.Setup(context);
  }

  // Initializes the nested feature and sets feature type.
  void Init(TaskContext *context) override {
    feature_.Init(context);
    num_base_values_ = feature_.GetFeatureType()->GetDomainSize();
    set_feature_type(new RootFeatureType(
        name(), *feature_.GetFeatureType(), RootValue()));
  }

  // Passes workspace requests and preprocessing to the nested feature.
  void RequestWorkspaces(WorkspaceRegistry *registry) override {
    feature_.RequestWorkspaces(registry);
  }

  void Preprocess(WorkspaceSet *workspaces, ParserState *state) const override {
    feature_.Preprocess(workspaces, state->mutable_sentence());
  }

 protected:
  // Returns the special value to represent a root token.
  FeatureValue RootValue() const { return num_base_values_; }

  // Store the number of base values from the wrapped function so compute the
  // root value.
  int num_base_values_;

  // The wrapped feature.
  F feature_;
};

// Specialization of ParserSentenceFeatureFunction that calls the nested feature
// with (Sentence, int) arguments based on the current integer focus.
template<class F>
class BasicParserSentenceFeatureFunction :
      public ParserSentenceFeatureFunction<F> {
 public:
  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
    if (focus == -1) return this->RootValue();
    return this->feature_.Compute(workspaces, state.sentence(), focus, result);
  }
};

// Registry for the parser feature functions.
DECLARE_SYNTAXNET_CLASS_REGISTRY("parser feature function",
                                 ParserFeatureFunction);

// Registry for the parser state + token index feature functions.
DECLARE_SYNTAXNET_CLASS_REGISTRY("parser+index feature function",
                                 ParserIndexFeatureFunction);

}  // namespace syntaxnet

#endif  // SYNTAXNET_PARSER_FEATURES_H_

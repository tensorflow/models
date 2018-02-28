/* Copyright 2017 Google Inc. All Rights Reserved.

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

// Features for whole Sentence objects.  Contrast with SentenceFeature, which
// operates on tokens within Sentences.

#ifndef SYNTAXNET_WHOLE_SENTENCE_FEATURES_H_
#define SYNTAXNET_WHOLE_SENTENCE_FEATURES_H_

#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/registry.h"

namespace syntaxnet {

// Type of feature functions whose focus is a whole sentence.
typedef FeatureFunction<Sentence> WholeSentenceFeatureFunction;

// Utilities to register the two types of parser features.
#define REGISTER_WHOLE_SENTENCE_FEATURE_FUNCTION(name, type) \
  REGISTER_SYNTAXNET_FEATURE_FUNCTION(WholeSentenceFeatureFunction, name, type)

DECLARE_SYNTAXNET_CLASS_REGISTRY("whole sentence feature function",
                                 WholeSentenceFeatureFunction);

}  // namespace syntaxnet

#endif  // SYNTAXNET_WHOLE_SENTENCE_FEATURES_H_

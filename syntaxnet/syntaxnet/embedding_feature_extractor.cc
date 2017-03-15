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

#include "syntaxnet/embedding_feature_extractor.h"

#include <vector>

#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/parser_features.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/utils.h"

namespace syntaxnet {

void GenericEmbeddingFeatureExtractor::Setup(TaskContext *context) {
  // Don't use version to determine how to get feature FML.
  const string features = context->Get(
      tensorflow::strings::StrCat(ArgPrefix(), "_", "features"), "");
  const string embedding_names =
      context->Get(GetParamName("embedding_names"), "");
  const string embedding_dims =
      context->Get(GetParamName("embedding_dims"), "");
  LOG(INFO) << "Features: " << features;
  LOG(INFO) << "Embedding names: " << embedding_names;
  LOG(INFO) << "Embedding dims: " << embedding_dims;
  embedding_fml_ = utils::Split(features, ';');
  add_strings_ = context->Get(GetParamName("add_varlen_strings"), false);
  embedding_names_ = utils::Split(embedding_names, ';');
  for (const string &dim : utils::Split(embedding_dims, ';')) {
    embedding_dims_.push_back(utils::ParseUsing<int>(dim, utils::ParseInt32));
  }
}

void GenericEmbeddingFeatureExtractor::Init(TaskContext *context) {
}

std::vector<string> GenericEmbeddingFeatureExtractor::GetMappingsForEmbedding(
    const string &embedding_name) {
  const auto name = std::find(embedding_names_.begin(), embedding_names_.end(),
                              embedding_name);
  if (name == embedding_names_.end()) {
    return {};
  }
  const int embedding_idx = std::distance(embedding_names_.begin(), name);
  const auto &feature_extractor = generic_feature_extractor(embedding_idx);
  const auto *type = feature_extractor.feature_type(0);
  const int domain_size = type->GetDomainSize();
  for (int i = 1; i < feature_extractor.feature_types(); ++i) {
    // A sanity check that ensures all feature types have the same domain size.
    CHECK_EQ(domain_size, feature_extractor.feature_type(i)->GetDomainSize())
        << "FeatureType:" << feature_extractor.feature_type(i)
        << " (embedding_name:" << type->name() << ")"
        << " actual domain_size:"
        << feature_extractor.feature_type(i)->GetDomainSize()
        << " expected domain size:" << domain_size;
  }
  std::vector<string> mapped_feature_values(domain_size);
  for (Predicate p = 0; p < type->GetDomainSize(); ++p) {
    const string name = type->GetFeatureValueName(p);

    // Check for a unique mapping.
    CHECK_EQ(mapped_feature_values[p].size(), 0)
        << embedding_name << " \"" << name << "\" maps to predicate" << p
        << ", but collides with \"" << mapped_feature_values[p] << "\"";
    mapped_feature_values[p] = name;
  }

  return mapped_feature_values;
}

std::vector<std::vector<SparseFeatures>>
GenericEmbeddingFeatureExtractor::ConvertExample(
    const std::vector<FeatureVector> &feature_vectors) const {
  // Extract the features.
  std::vector<std::vector<SparseFeatures>> sparse_features(
      feature_vectors.size());
  for (size_t i = 0; i < feature_vectors.size(); ++i) {
    // Convert the nlp_parser::FeatureVector to dist belief format.
    sparse_features[i] = std::vector<SparseFeatures>(
        generic_feature_extractor(i).feature_types());

    for (int j = 0; j < feature_vectors[i].size(); ++j) {
      const FeatureType &feature_type = *feature_vectors[i].type(j);
      const FeatureValue value = feature_vectors[i].value(j);
      const bool is_continuous = feature_type.name().find("continuous") == 0;
      const int64 id = is_continuous ? FloatFeatureValue(value).id : value;
      const int base = feature_type.base();
      if (id >= 0) {
        sparse_features[i][base].add_id(id);
        if (is_continuous) {
          sparse_features[i][base].add_weight(FloatFeatureValue(value).weight);
        }
        if (add_strings_) {
          sparse_features[i][base].add_description(tensorflow::strings::StrCat(
              feature_type.name(), "=", feature_type.GetFeatureValueName(id)));
        }
      }
    }
  }

  return sparse_features;
}

}  // namespace syntaxnet

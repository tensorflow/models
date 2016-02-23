#include "neurosis/embedding_feature_extractor.h"

#include <vector>

#include "neurosis/feature_extractor.h"
#include "neurosis/parser_features.h"
#include "task_context.h"
#include "neurosis/utils.h"

namespace neurosis {

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

vector<vector<SparseFeatures>> GenericEmbeddingFeatureExtractor::ConvertExample(
    const vector<FeatureVector> &feature_vectors) const {
  // Extract the features.
  vector<vector<SparseFeatures>> sparse_features(feature_vectors.size());
  for (size_t i = 0; i < feature_vectors.size(); ++i) {
    // Convert the nlp_parser::FeatureVector to dist belief format.
    sparse_features[i] =
        vector<SparseFeatures>(generic_feature_extractor(i).feature_types());

    for (int j = 0; j < feature_vectors[i].size(); ++j) {
      const FeatureType &feature_type = *feature_vectors[i].type(j);
      const FeatureValue value = feature_vectors[i].value(j);

      // TODO(chrisalberti): remove this string operation. This is too costly to
      // be done every time we extract a feature.
      const bool is_continuous =
          feature_type.name().find("continuous") != string::npos;
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

}  // namespace neurosis

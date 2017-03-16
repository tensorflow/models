#include "dragnn/components/syntaxnet/syntaxnet_component.h"

#include <vector>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/component_registry.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/sparse.pb.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

using tensorflow::strings::StrCat;

namespace {

// Returns a new step in a trace based on a ComponentSpec.
ComponentStepTrace GetNewStepTrace(const ComponentSpec &spec,
                                   const TransitionState &state) {
  ComponentStepTrace step;
  for (auto &linked_spec : spec.linked_feature()) {
    auto &channel_trace = *step.add_linked_feature_trace();
    channel_trace.set_name(linked_spec.name());
    channel_trace.set_source_component(linked_spec.source_component());
    channel_trace.set_source_translator(linked_spec.source_translator());
    channel_trace.set_source_layer(linked_spec.source_layer());
  }
  for (auto &fixed_spec : spec.fixed_feature()) {
    step.add_fixed_feature_trace()->set_name(fixed_spec.name());
  }
  step.set_html_representation(state.HTMLRepresentation());
  return step;
}

// Returns the last step in the trace.
ComponentStepTrace *GetLastStepInTrace(ComponentTrace *trace) {
  CHECK_GT(trace->step_trace_size(), 0) << "Trace has no steps added yet";
  return trace->mutable_step_trace(trace->step_trace_size() - 1);
}

}  // anonymous namespace

SyntaxNetComponent::SyntaxNetComponent()
    : feature_extractor_("brain_parser"),
      rewrite_root_labels_(false),
      max_beam_size_(1),
      input_data_(nullptr) {}

void SyntaxNetComponent::InitializeComponent(const ComponentSpec &spec) {
  // Save off the passed spec for future reference.
  spec_ = spec;

  // Create and populate a TaskContext for the underlying parser.
  TaskContext context;

  // Add the specified resources.
  for (const Resource &resource : spec_.resource()) {
    auto *input = context.GetInput(resource.name());
    for (const Part &part : resource.part()) {
      auto *input_part = input->add_part();
      input_part->set_file_pattern(part.file_pattern());
      input_part->set_file_format(part.file_format());
      input_part->set_record_format(part.record_format());
    }
  }

  // Add the specified task args to the transition system.
  for (const auto &param : spec_.transition_system().parameters()) {
    context.SetParameter(param.first, param.second);
  }

  // Set the arguments for the feature extractor.
  std::vector<string> names;
  std::vector<string> dims;
  std::vector<string> fml;
  std::vector<string> predicate_maps;

  for (const FixedFeatureChannel &channel : spec.fixed_feature()) {
    names.push_back(channel.name());
    fml.push_back(channel.fml());
    predicate_maps.push_back(channel.predicate_map());
    dims.push_back(StrCat(channel.embedding_dim()));
  }

  context.SetParameter("neurosis_feature_syntax_version", "2");
  context.SetParameter("brain_parser_embedding_dims", utils::Join(dims, ";"));
  context.SetParameter("brain_parser_predicate_maps",
                       utils::Join(predicate_maps, ";"));
  context.SetParameter("brain_parser_features", utils::Join(fml, ";"));
  context.SetParameter("brain_parser_embedding_names", utils::Join(names, ";"));

  names.clear();
  dims.clear();
  fml.clear();
  predicate_maps.clear();

  std::vector<string> source_components;
  std::vector<string> source_layers;
  std::vector<string> source_translators;

  for (const LinkedFeatureChannel &channel : spec.linked_feature()) {
    names.push_back(channel.name());
    fml.push_back(channel.fml());
    dims.push_back(StrCat(channel.embedding_dim()));
    source_components.push_back(channel.source_component());
    source_layers.push_back(channel.source_layer());
    source_translators.push_back(channel.source_translator());
    predicate_maps.push_back("none");
  }

  context.SetParameter("link_embedding_dims", utils::Join(dims, ";"));
  context.SetParameter("link_predicate_maps", utils::Join(predicate_maps, ";"));
  context.SetParameter("link_features", utils::Join(fml, ";"));
  context.SetParameter("link_embedding_names", utils::Join(names, ";"));
  context.SetParameter("link_source_layers", utils::Join(source_layers, ";"));
  context.SetParameter("link_source_translators",
                       utils::Join(source_translators, ";"));
  context.SetParameter("link_source_components",
                       utils::Join(source_components, ";"));

  context.SetParameter("parser_transition_system",
                       spec.transition_system().registered_name());

  // Set up the fixed feature extractor.
  feature_extractor_.Setup(&context);
  feature_extractor_.Init(&context);
  feature_extractor_.RequestWorkspaces(&workspace_registry_);

  // Set up the underlying transition system.
  transition_system_.reset(ParserTransitionSystem::Create(
      context.Get("parser_transition_system", "arc-standard")));
  transition_system_->Setup(&context);
  transition_system_->Init(&context);

  // Create label map.
  string path = TaskContext::InputFile(*context.GetInput("label-map"));
  label_map_ =
      SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(path, 0, 0);

  // Set up link feature extractors.
  if (spec.linked_feature_size() > 0) {
    link_feature_extractor_.Setup(&context);
    link_feature_extractor_.Init(&context);
    link_feature_extractor_.RequestWorkspaces(&workspace_registry_);
  }

  // Get the legacy flag for simulating old parser processor behavior. If the
  // flag is not set, default to 'false'.
  rewrite_root_labels_ = context.Get("rewrite_root_labels", false);
}

std::unique_ptr<Beam<SyntaxNetTransitionState>> SyntaxNetComponent::CreateBeam(
    int max_size) {
  std::unique_ptr<Beam<SyntaxNetTransitionState>> beam(
      new Beam<SyntaxNetTransitionState>(max_size));
  auto permission_function = [this](SyntaxNetTransitionState *state,
                                    int action) {
    VLOG(3) << "permission_function action:" << action
            << " is_allowed:" << this->IsAllowed(state, action);
    return this->IsAllowed(state, action);
  };
  auto finality_function = [this](SyntaxNetTransitionState *state) {
    VLOG(2) << "finality_function is_final:" << this->IsFinal(state);
    return this->IsFinal(state);
  };
  auto oracle_function = [this](SyntaxNetTransitionState *state) {
    VLOG(2) << "oracle_function action:" << this->GetOracleLabel(state);
    return this->GetOracleLabel(state);
  };
  auto beam_ptr = beam.get();
  auto advance_function = [this, beam_ptr](SyntaxNetTransitionState *state,
                                           int action) {
    VLOG(2) << "advance_function beam ptr:" << beam_ptr << " action:" << action;
    this->Advance(state, action, beam_ptr);
  };
  beam->SetFunctions(permission_function, finality_function, advance_function,
                     oracle_function);

  return beam;
}

void SyntaxNetComponent::InitializeData(
    const std::vector<std::vector<const TransitionState *>> &parent_states,
    int max_beam_size, InputBatchCache *input_data) {
  // Save off the input data object.
  input_data_ = input_data;

  // If beam size has changed, change all beam sizes for existing beams.
  if (max_beam_size_ != max_beam_size) {
    CHECK_GT(max_beam_size, 0)
        << "Requested max beam size must be greater than 0.";
    VLOG(2) << "Adjusting max beam size from " << max_beam_size_ << " to "
            << max_beam_size;
    max_beam_size_ = max_beam_size;
    for (auto &beam : batch_) {
      beam->SetMaxSize(max_beam_size_);
    }
  }

  SentenceInputBatch *sentences = input_data->GetAs<SentenceInputBatch>();

  // Expect that the sentence data is the same size as the input states batch.
  if (!parent_states.empty()) {
    CHECK_EQ(parent_states.size(), sentences->data()->size());
  }

  // Adjust the beam vector so that it is the correct size for this batch.
  if (batch_.size() < sentences->data()->size()) {
    VLOG(1) << "Batch size is increased to " << sentences->data()->size()
            << " from " << batch_.size();
    for (int i = batch_.size(); i < sentences->data()->size(); ++i) {
      batch_.push_back(CreateBeam(max_beam_size));
    }
  } else if (batch_.size() > sentences->data()->size()) {
    VLOG(1) << "Batch size is decreased to " << sentences->data()->size()
            << " from " << batch_.size();
    batch_.erase(batch_.begin() + sentences->data()->size(), batch_.end());

  } else {
    VLOG(1) << "Batch size is constant at " << sentences->data()->size();
  }
  CHECK_EQ(batch_.size(), sentences->data()->size());

  // Fill the beams with the relevant data for that batch.
  for (int batch_index = 0; batch_index < sentences->data()->size();
       ++batch_index) {
    // Create a vector of states for this component's beam.
    std::vector<std::unique_ptr<SyntaxNetTransitionState>> initial_states;
    if (parent_states.empty()) {
      // If no states have been passed in, create a single state to seed the
      // beam.
      initial_states.push_back(
          CreateState(&(sentences->data()->at(batch_index))));
    } else {
      // If states have been passed in, seed the beam with them up to the max
      // beam size.
      int num_states =
          std::min(batch_.at(batch_index)->max_size(),
                   static_cast<int>(parent_states.at(batch_index).size()));
      VLOG(2) << "Creating a beam using " << num_states << " initial states";
      for (int i = 0; i < num_states; ++i) {
        std::unique_ptr<SyntaxNetTransitionState> state(
            CreateState(&(sentences->data()->at(batch_index))));
        state->Init(*parent_states.at(batch_index).at(i));
        initial_states.push_back(std::move(state));
      }
    }
    batch_.at(batch_index)->Init(std::move(initial_states));
  }
}

bool SyntaxNetComponent::IsReady() const { return input_data_ != nullptr; }

string SyntaxNetComponent::Name() const {
  return "SyntaxNet-backed beam parser";
}

int SyntaxNetComponent::BatchSize() const { return batch_.size(); }

int SyntaxNetComponent::BeamSize() const { return max_beam_size_; }

int SyntaxNetComponent::StepsTaken(int batch_index) const {
  return batch_.at(batch_index)->num_steps();
}

int SyntaxNetComponent::GetBeamIndexAtStep(int step, int current_index,
                                           int batch) const {
  return batch_.at(batch)->FindPreviousIndex(current_index, step);
}

int SyntaxNetComponent::GetSourceBeamIndex(int current_index, int batch) const {
  return batch_.at(batch)->FindPreviousIndex(current_index, 0);
}

std::function<int(int, int, int)> SyntaxNetComponent::GetStepLookupFunction(
    const string &method) {
  if (method == "shift-reduce-step") {
    // TODO(googleuser): Describe this function.
    return [this](int batch_index, int beam_index, int value) {
      SyntaxNetTransitionState *state =
          batch_.at(batch_index)->beam_state(beam_index);
      return state->step_for_token(value);
    };
  } else if (method == "reduce-step") {
    // TODO(googleuser): Describe this function.
    return [this](int batch_index, int beam_index, int value) {
      SyntaxNetTransitionState *state =
          batch_.at(batch_index)->beam_state(beam_index);
      return state->parent_step_for_token(value);
    };
  } else if (method == "parent-shift-reduce-step") {
    // TODO(googleuser): Describe this function.
    return [this](int batch_index, int beam_index, int value) {
      SyntaxNetTransitionState *state =
          batch_.at(batch_index)->beam_state(beam_index);
      return state->step_for_token(state->parent_step_for_token(value));
    };
  } else if (method == "reverse-token") {
    // TODO(googleuser): Describe this function.
    return [this](int batch_index, int beam_index, int value) {
      SyntaxNetTransitionState *state =
          batch_.at(batch_index)->beam_state(beam_index);
      int result = state->sentence()->sentence()->token_size() - value - 1;
      if (result >= 0 && result < state->sentence()->sentence()->token_size()) {
        return result;
      } else {
        return -1;
      }
    };
  } else {
    LOG(FATAL) << "Unable to find step lookup function " << method;
  }
}

void SyntaxNetComponent::AdvanceFromPrediction(const float transition_matrix[],
                                               int transition_matrix_length) {
  VLOG(2) << "Advancing from prediction.";
  int matrix_index = 0;
  int num_labels = transition_system_->NumActions(label_map_->Size());
  for (int i = 0; i < batch_.size(); ++i) {
    int max_beam_size = batch_.at(i)->max_size();
    int matrix_size = num_labels * max_beam_size;
    CHECK_LE(matrix_index + matrix_size, transition_matrix_length);
    if (!batch_.at(i)->IsTerminal()) {
      batch_.at(i)->AdvanceFromPrediction(&transition_matrix[matrix_index],
                                          matrix_size, num_labels);
    }
    matrix_index += num_labels * max_beam_size;
  }
}

void SyntaxNetComponent::AdvanceFromOracle() {
  VLOG(2) << "Advancing from oracle.";
  for (auto &beam : batch_) {
    beam->AdvanceFromOracle();
  }
}

bool SyntaxNetComponent::IsTerminal() const {
  VLOG(2) << "Checking terminal status.";
  for (const auto &beam : batch_) {
    if (!beam->IsTerminal()) {
      return false;
    }
  }
  return true;
}

std::vector<std::vector<const TransitionState *>>
SyntaxNetComponent::GetBeam() {
  std::vector<std::vector<const TransitionState *>> state_beam;
  for (auto &beam : batch_) {
    // Because this component only finalizes the data of the highest ranked
    // component in each beam, the next component should only be initialized
    // from the highest ranked component in that beam.
    state_beam.push_back({beam->beam().at(0)});
  }
  return state_beam;
}

int SyntaxNetComponent::GetFixedFeatures(
    std::function<int32 *(int)> allocate_indices,
    std::function<int64 *(int)> allocate_ids,
    std::function<float *(int)> allocate_weights, int channel_id) const {
  std::vector<SparseFeatures> features;

  const int channel_size = spec_.fixed_feature(channel_id).size();

  // For every beam in the batch...
  for (const auto &beam : batch_) {
    // For every element in the beam...
    for (int beam_idx = 0; beam_idx < beam->size(); ++beam_idx) {
      // Get the SparseFeatures from the feature extractor.
      auto state = beam->beam_state(beam_idx);
      const std::vector<std::vector<SparseFeatures>> sparse_features =
          feature_extractor_.ExtractSparseFeatures(
              *(state->sentence()->workspace()), *(state->parser_state()));

      // Hold the SparseFeatures for later processing.
      for (const SparseFeatures &f : sparse_features[channel_id]) {
        features.emplace_back(f);
        if (do_tracing_) {
          FixedFeatures fixed_features;
          for (const string &name : f.description()) {
            fixed_features.add_value_name(name);
          }
          fixed_features.set_feature_name("");
          auto *trace = GetLastStepInTrace(state->mutable_trace());
          auto *fixed_trace = trace->mutable_fixed_feature_trace(channel_id);
          *fixed_trace->add_value_trace() = fixed_features;
        }
      }
    }
    const int pad_amount = max_beam_size_ - beam->size();
    features.resize(features.size() + pad_amount * channel_size);
  }

  int feature_count = 0;
  for (const auto &feature : features) {
    feature_count += feature.id_size();
  }

  VLOG(2) << "Feature count is " << feature_count;
  int32 *indices_tensor = allocate_indices(feature_count);
  int64 *ids_tensor = allocate_ids(feature_count);
  float *weights_tensor = allocate_weights(feature_count);

  int array_index = 0;
  for (int feature_index = 0; feature_index < features.size();
       ++feature_index) {
    VLOG(2) << "Extracting for feature_index " << feature_index;
    const auto feature = features[feature_index];
    for (int sub_idx = 0; sub_idx < feature.id_size(); ++sub_idx) {
      indices_tensor[array_index] = feature_index;
      ids_tensor[array_index] = feature.id(sub_idx);
      if (sub_idx < feature.weight_size()) {
        weights_tensor[array_index] = feature.weight(sub_idx);
      } else {
        weights_tensor[array_index] = 1.0;
      }
      VLOG(2) << "Feature index: " << indices_tensor[array_index]
              << " id: " << ids_tensor[array_index]
              << " weight: " << weights_tensor[array_index];

      ++array_index;
    }
  }
  return feature_count;
}

int SyntaxNetComponent::BulkGetFixedFeatures(
    const BulkFeatureExtractor &extractor) {
  // Allocate a vector of SparseFeatures per channel.
  const int num_channels = spec_.fixed_feature_size();
  std::vector<int> channel_size(num_channels);
  for (int i = 0; i < num_channels; ++i) {
    channel_size[i] = spec_.fixed_feature(i).size();
  }
  std::vector<std::vector<SparseFeatures>> features(num_channels);
  std::vector<std::vector<int>> feature_indices(num_channels);
  std::vector<std::vector<int>> step_indices(num_channels);
  std::vector<std::vector<int>> element_indices(num_channels);
  std::vector<int> feature_counts(num_channels);
  int step_count = 0;

  while (!IsTerminal()) {
    int current_element = 0;

    // For every beam in the batch...
    for (const auto &beam : batch_) {
      // For every element in the beam...
      for (int beam_idx = 0; beam_idx < beam->size(); ++beam_idx) {
        // Get the SparseFeatures from the parser.
        auto state = beam->beam_state(beam_idx);
        const std::vector<std::vector<SparseFeatures>> sparse_features =
            feature_extractor_.ExtractSparseFeatures(
                *(state->sentence()->workspace()), *(state->parser_state()));

        for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
          int feature_count = 0;
          for (const SparseFeatures &f : sparse_features[channel_id]) {
            // Trace, if requested.
            if (do_tracing_) {
              FixedFeatures fixed_features;
              for (const string &name : f.description()) {
                fixed_features.add_value_name(name);
              }
              fixed_features.set_feature_name("");
              auto *trace = GetLastStepInTrace(state->mutable_trace());
              auto *fixed_trace =
                  trace->mutable_fixed_feature_trace(channel_id);
              *fixed_trace->add_value_trace() = fixed_features;
            }

            // Hold the SparseFeatures for later processing.
            features[channel_id].emplace_back(f);
            element_indices[channel_id].emplace_back(current_element);
            step_indices[channel_id].emplace_back(step_count);
            feature_indices[channel_id].emplace_back(feature_count);
            feature_counts[channel_id] += f.id_size();
            ++feature_count;
          }
        }
        ++current_element;
      }

      // Advance the current element to skip unused beam slots.
      // Pad the beam out to max_beam_size.
      int pad_amount = max_beam_size_ - beam->size();
      current_element += pad_amount;
    }
    AdvanceFromOracle();
    ++step_count;
  }

  const int total_steps = step_count;
  const int num_elements = batch_.size() * max_beam_size_;

  // This would be a good place to add threading.
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    int feature_count = feature_counts[channel_id];
    LOG(INFO) << "Feature count is " << feature_count << " for channel "
              << channel_id;
    int32 *indices_tensor =
        extractor.AllocateIndexMemory(channel_id, feature_count);
    int64 *ids_tensor = extractor.AllocateIdMemory(channel_id, feature_count);
    float *weights_tensor =
        extractor.AllocateWeightMemory(channel_id, feature_count);
    int array_index = 0;
    for (int feat_idx = 0; feat_idx < features[channel_id].size(); ++feat_idx) {
      const auto &feature = features[channel_id][feat_idx];
      int element_index = element_indices[channel_id][feat_idx];
      int step_index = step_indices[channel_id][feat_idx];
      int feature_index = feature_indices[channel_id][feat_idx];
      for (int sub_idx = 0; sub_idx < feature.id_size(); ++sub_idx) {
        indices_tensor[array_index] =
            extractor.GetIndex(total_steps, num_elements, feature_index,
                               element_index, step_index);
        ids_tensor[array_index] = feature.id(sub_idx);
        if (sub_idx < feature.weight_size()) {
          weights_tensor[array_index] = feature.weight(sub_idx);
        } else {
          weights_tensor[array_index] = 1.0;
        }
        ++array_index;
      }
    }
  }
  return step_count;
}

std::vector<LinkFeatures> SyntaxNetComponent::GetRawLinkFeatures(
    int channel_id) const {
  std::vector<LinkFeatures> features;
  const int channel_size = spec_.linked_feature(channel_id).size();
  std::unique_ptr<std::vector<string>> feature_names;
  if (do_tracing_) {
    feature_names.reset(new std::vector<string>);
    *feature_names = utils::Split(spec_.linked_feature(channel_id).fml(), ' ');
  }

  // For every beam in the batch...
  for (int batch_idx = 0; batch_idx < batch_.size(); ++batch_idx) {
    // For every element in the beam...
    const auto &beam = batch_[batch_idx];
    for (int beam_idx = 0; beam_idx < beam->size(); ++beam_idx) {
      // Get the raw link features from the linked feature extractor.
      auto state = beam->beam_state(beam_idx);
      std::vector<FeatureVector> raw_features(
          link_feature_extractor_.NumEmbeddings());
      link_feature_extractor_.ExtractFeatures(*(state->sentence()->workspace()),
                                              *(state->parser_state()),
                                              &raw_features);

      // Add the raw feature values to the LinkFeatures proto.
      CHECK_LT(channel_id, raw_features.size());
      for (int i = 0; i < raw_features[channel_id].size(); ++i) {
        features.emplace_back();
        features.back().set_feature_value(raw_features[channel_id].value(i));
        features.back().set_batch_idx(batch_idx);
        features.back().set_beam_idx(beam_idx);
        if (do_tracing_) {
          features.back().set_feature_name(feature_names->at(i));
        }
      }
    }

    // Pad the beam out to max_beam_size.
    int pad_amount = max_beam_size_ - beam->size();
    features.resize(features.size() + pad_amount * channel_size);
  }

  return features;
}

std::vector<std::vector<int>> SyntaxNetComponent::GetOracleLabels() const {
  std::vector<std::vector<int>> oracle_labels;
  for (const auto &beam : batch_) {
    oracle_labels.emplace_back();
    for (int beam_idx = 0; beam_idx < beam->size(); ++beam_idx) {
      // Get the raw link features from the linked feature extractor.
      auto state = beam->beam_state(beam_idx);
      oracle_labels.back().push_back(GetOracleLabel(state));
    }
  }
  return oracle_labels;
}

void SyntaxNetComponent::FinalizeData() {
  // This chooses the top-scoring member of the beam to annotate the underlying
  // document.
  VLOG(2) << "Finalizing data.";
  for (auto &beam : batch_) {
    if (beam->size() != 0) {
      auto top_state = beam->beam_state(0);
      VLOG(3) << "Finalizing for sentence: "
              << top_state->sentence()->sentence()->ShortDebugString();
      top_state->parser_state()->AddParseToDocument(
          top_state->sentence()->sentence(), rewrite_root_labels_);
      VLOG(3) << "Sentence is now: "
              << top_state->sentence()->sentence()->ShortDebugString();
    } else {
      LOG(WARNING) << "Attempting to finalize an empty beam for component "
                   << spec_.name();
    }
  }
}

void SyntaxNetComponent::ResetComponent() {
  for (auto &beam : batch_) {
    beam->Reset();
  }
  input_data_ = nullptr;
  max_beam_size_ = 0;
}

std::unique_ptr<SyntaxNetTransitionState> SyntaxNetComponent::CreateState(
    SyntaxNetSentence *sentence) {
  VLOG(3) << "Creating state for sentence "
          << sentence->sentence()->DebugString();
  std::unique_ptr<ParserState> parser_state(new ParserState(
      sentence->sentence(), transition_system_->NewTransitionState(false),
      label_map_));
  sentence->workspace()->Reset(workspace_registry_);
  feature_extractor_.Preprocess(sentence->workspace(), parser_state.get());
  link_feature_extractor_.Preprocess(sentence->workspace(), parser_state.get());
  std::unique_ptr<SyntaxNetTransitionState> transition_state(
      new SyntaxNetTransitionState(std::move(parser_state), sentence));
  return transition_state;
}

bool SyntaxNetComponent::IsAllowed(SyntaxNetTransitionState *state,
                                   int action) const {
  return transition_system_->IsAllowedAction(action, *(state->parser_state()));
}

bool SyntaxNetComponent::IsFinal(SyntaxNetTransitionState *state) const {
  return transition_system_->IsFinalState(*(state->parser_state()));
}

int SyntaxNetComponent::GetOracleLabel(SyntaxNetTransitionState *state) const {
  if (IsFinal(state)) {
    // It is not permitted to request an oracle label from a sentence that is
    // in a final state.
    return -1;
  } else {
    return transition_system_->GetNextGoldAction(*(state->parser_state()));
  }
}

void SyntaxNetComponent::Advance(SyntaxNetTransitionState *state, int action,
                                 Beam<SyntaxNetTransitionState> *beam) {
  auto parser_state = state->parser_state();
  auto sentence_size = state->sentence()->sentence()->token_size();
  const int num_steps = beam->num_steps();

  if (transition_system_->SupportsActionMetaData()) {
    const int parent_idx =
        transition_system_->ParentIndex(*parser_state, action);
    constexpr int kShiftAction = -1;
    if (parent_idx == kShiftAction) {
      if (parser_state->Next() < sentence_size && parser_state->Next() >= 0) {
        // if we have already consumed all the input then it is not a shift
        // action. We just skip it.
        state->set_step_for_token(parser_state->Next(), num_steps);
      }
    } else if (parent_idx >= 0) {
      VLOG(2) << spec_.name() << ": Updating pointer: " << parent_idx << " -> "
              << num_steps;
      state->set_step_for_token(parent_idx, num_steps);
      const int child_idx =
          transition_system_->ChildIndex(*parser_state, action);
      assert(child_idx >= 0 && child_idx < sentence_size);
      state->set_parent_for_token(child_idx, parent_idx);

      VLOG(2) << spec_.name() << ": Updating parent for child: " << parent_idx
              << " -> " << child_idx;
      state->set_parent_step_for_token(child_idx, num_steps);
    } else {
      VLOG(2) << spec_.name() << ": Invalid parent index: " << parent_idx;
    }
  }
  if (do_tracing_) {
    auto *trace = state->mutable_trace();
    auto *last_step = GetLastStepInTrace(trace);

    // Add action to the prior step.
    last_step->set_caption(
        transition_system_->ActionAsString(action, *parser_state));
    last_step->set_step_finished(true);
  }

  transition_system_->PerformAction(action, parser_state);

  if (do_tracing_) {
    // Add info for the next step.
    *state->mutable_trace()->add_step_trace() = GetNewStepTrace(spec_, *state);
  }
}

void SyntaxNetComponent::InitializeTracing() {
  do_tracing_ = true;
  CHECK(IsReady()) << "Cannot initialize trace before InitializeData().";

  // Initialize each element of the beam with a new trace.
  for (auto &beam : batch_) {
    for (int beam_idx = 0; beam_idx < beam->size(); ++beam_idx) {
      SyntaxNetTransitionState *state = beam->beam_state(beam_idx);
      std::unique_ptr<ComponentTrace> trace(new ComponentTrace());
      trace->set_name(spec_.name());
      *trace->add_step_trace() = GetNewStepTrace(spec_, *state);
      state->set_trace(std::move(trace));
    }
  }

  feature_extractor_.set_add_strings(true);
}

void SyntaxNetComponent::DisableTracing() {
  do_tracing_ = false;
  feature_extractor_.set_add_strings(false);
}

void SyntaxNetComponent::AddTranslatedLinkFeaturesToTrace(
    const std::vector<LinkFeatures> &features, int channel_id) {
  CHECK(do_tracing_) << "Tracing is not enabled.";
  int linear_idx = 0;
  const int channel_size = spec_.linked_feature(channel_id).size();

  // For every beam in the batch...
  for (const auto &beam : batch_) {
    // For every element in the beam...
    for (int beam_idx = 0; beam_idx < max_beam_size_; ++beam_idx) {
      for (int feature_idx = 0; feature_idx < channel_size; ++feature_idx) {
        if (beam_idx < beam->size()) {
          auto state = beam->beam_state(beam_idx);
          auto *trace = GetLastStepInTrace(state->mutable_trace());
          auto *link_trace = trace->mutable_linked_feature_trace(channel_id);
          if (features[linear_idx].feature_value() >= 0 &&
              features[linear_idx].step_idx() >= 0) {
            *link_trace->add_value_trace() = features[linear_idx];
          }
        }
        ++linear_idx;
      }
    }
  }
}

std::vector<std::vector<ComponentTrace>> SyntaxNetComponent::GetTraceProtos()
    const {
  std::vector<std::vector<ComponentTrace>> traces;

  // For every beam in the batch...
  for (const auto &beam : batch_) {
    std::vector<ComponentTrace> beam_trace;

    // For every element in the beam...
    for (int beam_idx = 0; beam_idx < beam->size(); ++beam_idx) {
      auto state = beam->beam_state(beam_idx);
      beam_trace.push_back(*state->mutable_trace());
    }
    traces.push_back(beam_trace);
  }
  return traces;
};

REGISTER_DRAGNN_COMPONENT(SyntaxNetComponent);

}  // namespace dragnn
}  // namespace syntaxnet

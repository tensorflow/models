#include "dragnn/core/compute_session_impl.h"

#include <algorithm>
#include <utility>

#include "dragnn/protos/data.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "syntaxnet/registry.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

ComputeSessionImpl::ComputeSessionImpl(
    int id,
    std::function<std::unique_ptr<Component>(const string &component_name,
                                             const string &backend_type)>
        component_builder)
    : component_builder_(std::move(component_builder)), id_(id) {}

void ComputeSessionImpl::Init(const MasterSpec &master_spec,
                              const GridPoint &hyperparams) {
  spec_ = master_spec;
  grid_point_ = hyperparams;

  VLOG(2) << "Creating components.";
  bool is_input = true;
  Component *predecessor;
  for (const ComponentSpec &spec : master_spec.component()) {
    // Construct the component using the specified backend.
    VLOG(2) << "Creating component '" << spec.name()
            << "' with backend: " << spec.backend().registered_name();
    auto component =
        component_builder_(spec.name(), spec.backend().registered_name());

    // Initializes the component.
    component->InitializeComponent(spec);

    // Adds a predecessor to non-input components.
    if (!is_input) {
      predecessors_.insert(
          std::pair<Component *, Component *>(component.get(), predecessor));
    }

    // The current component will be the predecessor component next time around.
    predecessor = component.get();

    // All components after the first are non-input components.
    is_input = false;

    // Move into components list.
    components_.insert(std::pair<string, std::unique_ptr<Component>>(
        spec.name(), std::move(component)));
  }
  VLOG(2) << "Done creating components.";

  VLOG(2) << "Adding translators.";
  for (const ComponentSpec &spec : master_spec.component()) {
    // First, get the component object for this spec.
    VLOG(2) << "Examining component: " << spec.name();
    auto map_result = components_.find(spec.name());
    CHECK(map_result != components_.end()) << "Unable to find component.";
    Component *start_component = map_result->second.get();

    if (spec.linked_feature_size() > 0) {
      VLOG(2) << "Adding " << spec.linked_feature_size() << " translators for "
              << spec.name();

      // Attach all the translators described in the spec.
      std::vector<IndexTranslator *> translator_set;
      for (const LinkedFeatureChannel &channel : spec.linked_feature()) {
        // For every translator, save off a non-unique ptr in the component name
        // to translator map, then push the unique ptr onto the management
        // vector.
        auto translator = CreateTranslator(channel, start_component);
        translator_set.push_back(translator.get());
        owned_translators_.push_back(std::move(translator));
      }

      // Once all translators have been created, associate this group of
      // translators with a component.
      translators_.insert(std::pair<string, std::vector<IndexTranslator *>>(
          spec.name(), std::move(translator_set)));
    } else {
      VLOG(2) << "No translators found for " << spec.name();
    }
  }
  VLOG(2) << "Done adding translators.";

  VLOG(2) << "Initialization complete.";
}

void ComputeSessionImpl::InitializeComponentData(const string &component_name,
                                                 int max_beam_size) {
  CHECK(input_data_ != nullptr) << "Attempted to access a component without "
                                   "providing input data for this session.";
  Component *component = GetComponent(component_name);

  // Try and find the source component. If one exists, check that it is terminal
  // and get its data; if not, pass in an empty vector for source data.
  auto source_result = predecessors_.find(component);
  if (source_result == predecessors_.end()) {
    VLOG(1) << "Source result not found. Using empty initialization vector for "
            << component_name;
    component->InitializeData({}, max_beam_size, input_data_.get());
  } else {
    VLOG(1) << "Source result found. Using prior initialization vector for "
            << component_name;
    auto source = source_result->second;
    CHECK(source->IsTerminal()) << "Source is not terminal for component '"
                                << component_name << "'. Exiting.";
    component->InitializeData(source->GetBeam(), max_beam_size,
                              input_data_.get());
  }
  if (do_tracing_) {
    component->InitializeTracing();
  }
}

int ComputeSessionImpl::BatchSize(const string &component_name) const {
  return GetReadiedComponent(component_name)->BatchSize();
}

int ComputeSessionImpl::BeamSize(const string &component_name) const {
  return GetReadiedComponent(component_name)->BeamSize();
}

const ComponentSpec &ComputeSessionImpl::Spec(
    const string &component_name) const {
  for (const auto &component : spec_.component()) {
    if (component.name() == component_name) {
      return component;
    }
  }
  LOG(FATAL) << "Missing component '" << component_name << "'. Exiting.";
}

int ComputeSessionImpl::SourceComponentBeamSize(const string &component_name,
                                                int channel_id) {
  const auto &translators = GetTranslators(component_name);
  return translators.at(channel_id)->path().back()->BeamSize();
}

void ComputeSessionImpl::AdvanceFromOracle(const string &component_name) {
  GetReadiedComponent(component_name)->AdvanceFromOracle();
}

void ComputeSessionImpl::AdvanceFromPrediction(const string &component_name,
                                               const float score_matrix[],
                                               int score_matrix_length) {
  GetReadiedComponent(component_name)
      ->AdvanceFromPrediction(score_matrix, score_matrix_length);
}

int ComputeSessionImpl::GetInputFeatures(
    const string &component_name, std::function<int32 *(int)> allocate_indices,
    std::function<int64 *(int)> allocate_ids,
    std::function<float *(int)> allocate_weights, int channel_id) const {
  return GetReadiedComponent(component_name)
      ->GetFixedFeatures(allocate_indices, allocate_ids, allocate_weights,
                         channel_id);
}

int ComputeSessionImpl::BulkGetInputFeatures(
    const string &component_name, const BulkFeatureExtractor &extractor) {
  return GetReadiedComponent(component_name)->BulkGetFixedFeatures(extractor);
}

std::vector<LinkFeatures> ComputeSessionImpl::GetTranslatedLinkFeatures(
    const string &component_name, int channel_id) {
  auto *component = GetReadiedComponent(component_name);
  auto features = component->GetRawLinkFeatures(channel_id);

  IndexTranslator *translator = GetTranslators(component_name).at(channel_id);
  for (int i = 0; i < features.size(); ++i) {
    LinkFeatures &feature = features[i];
    if (feature.has_feature_value()) {
      VLOG(2) << "Raw feature[" << i << "]: " << feature.ShortDebugString();
      IndexTranslator::Index index = translator->Translate(
          feature.batch_idx(), feature.beam_idx(), feature.feature_value());
      feature.set_step_idx(index.step_index);
      feature.set_batch_idx(index.batch_index);
      feature.set_beam_idx(index.beam_index);
    } else {
      VLOG(2) << "Raw feature[" << i << "]: PADDING (empty proto)";
    }
  }

  // Add the translated link features to the component's trace.
  if (do_tracing_) {
    component->AddTranslatedLinkFeaturesToTrace(features, channel_id);
  }

  return features;
}
std::vector<std::vector<int>> ComputeSessionImpl::EmitOracleLabels(
    const string &component_name) {
  return GetReadiedComponent(component_name)->GetOracleLabels();
}

bool ComputeSessionImpl::IsTerminal(const string &component_name) {
  return GetReadiedComponent(component_name)->IsTerminal();
}

void ComputeSessionImpl::SetTracing(bool tracing_on) {
  do_tracing_ = tracing_on;
  for (auto &component_pair : components_) {
    if (!tracing_on) {
      component_pair.second->DisableTracing();
    }
  }
}

void ComputeSessionImpl::FinalizeData(const string &component_name) {
  VLOG(2) << "Finalizing data for " << component_name;
  GetReadiedComponent(component_name)->FinalizeData();
}

std::vector<string> ComputeSessionImpl::GetSerializedPredictions() {
  VLOG(2) << "Geting serialized predictions.";
  return input_data_->SerializedData();
}

std::vector<MasterTrace> ComputeSessionImpl::GetTraceProtos() {
  std::vector<MasterTrace> traces;

  // First compute all possible traces for each component.
  std::map<string, std::vector<std::vector<ComponentTrace>>> component_traces;
  std::vector<string> pipeline;
  for (auto &component_spec : spec_.component()) {
    pipeline.push_back(component_spec.name());
    component_traces.insert(
        {component_spec.name(),
         GetComponent(component_spec.name())->GetTraceProtos()});
  }

  // Only output for the actual number of states in each beam.
  auto final_beam = GetComponent(pipeline.back())->GetBeam();
  for (int batch_idx = 0; batch_idx < final_beam.size(); ++batch_idx) {
    for (int beam_idx = 0; beam_idx < final_beam[batch_idx].size();
         ++beam_idx) {
      std::vector<int> beam_path;
      beam_path.push_back(beam_idx);

      // Trace components backwards, finding the source of each state in the
      // prior component.
      VLOG(2) << "Start trace: " << beam_idx;
      for (int i = pipeline.size() - 1; i > 0; --i) {
        const auto *component = GetComponent(pipeline[i]);
        int source_beam_idx =
            component->GetSourceBeamIndex(beam_path.back(), batch_idx);
        beam_path.push_back(source_beam_idx);

        VLOG(2) << "Tracing path: " << pipeline[i] << " = " << source_beam_idx;
      }

      // Trace the path from the *start* to the end.
      std::reverse(beam_path.begin(), beam_path.end());
      MasterTrace master_trace;
      for (int i = 0; i < pipeline.size(); ++i) {
        *master_trace.add_component_trace() =
            component_traces[pipeline[i]][batch_idx][beam_path[i]];
      }
      traces.push_back(master_trace);
    }
  }

  return traces;
}

void ComputeSessionImpl::SetInputData(const std::vector<string> &data) {
  input_data_.reset(new InputBatchCache(data));
}

void ComputeSessionImpl::ResetSession() {
  // Reset all component states.
  for (auto &component_pair : components_) {
    component_pair.second->ResetComponent();
  }

  // Reset the input data pointer.
  input_data_.reset();
}

int ComputeSessionImpl::Id() const { return id_; }

string ComputeSessionImpl::GetDescription(const string &component_name) const {
  return GetComponent(component_name)->Name();
}

const std::vector<const IndexTranslator *> ComputeSessionImpl::Translators(
    const string &component_name) const {
  auto translators = GetTranslators(component_name);
  std::vector<const IndexTranslator *> const_translators;
  for (const auto &translator : translators) {
    const_translators.push_back(translator);
  }
  return const_translators;
}

Component *ComputeSessionImpl::GetReadiedComponent(
    const string &component_name) const {
  auto component = GetComponent(component_name);
  CHECK(component->IsReady())
      << "Attempted to access component " << component_name
      << " without first initializing it.";
  return component;
}

Component *ComputeSessionImpl::GetComponent(
    const string &component_name) const {
  auto result = components_.find(component_name);
  if (result == components_.end()) {
    LOG(ERROR) << "Could not find component \"" << component_name
               << "\" in the component set. Current components are: ";
    for (const auto &component_pair : components_) {
      LOG(ERROR) << component_pair.first;
    }
    LOG(FATAL) << "Missing component. Exiting.";
  }

  auto component = result->second.get();
  return component;
}

const std::vector<IndexTranslator *> &ComputeSessionImpl::GetTranslators(
    const string &component_name) const {
  auto result = translators_.find(component_name);
  if (result == translators_.end()) {
    LOG(ERROR) << "Could not find component " << component_name
               << " in the translator set. Current components are: ";
    for (const auto &component_pair : translators_) {
      LOG(ERROR) << component_pair.first;
    }
    LOG(FATAL) << "Missing component. Exiting.";
  }
  return result->second;
}

std::unique_ptr<IndexTranslator> ComputeSessionImpl::CreateTranslator(
    const LinkedFeatureChannel &channel, Component *start_component) {
  const int num_components = spec_.component_size();
  VLOG(2) << "Channel spec: " << channel.ShortDebugString();

  // Find the linked feature's source component, if it exists.
  auto source_map_result = components_.find(channel.source_component());
  CHECK(source_map_result != components_.end())
      << "Unable to find source component " << channel.source_component();
  const Component *end_component = source_map_result->second.get();

  // Our goal here is to iterate up the source map from the
  // start_component to the end_component.
  Component *current_component = start_component;
  std::vector<Component *> path;
  path.push_back(current_component);
  while (current_component != end_component) {
    // Try to find the next link upwards in the source chain.
    auto source_result = predecessors_.find(current_component);

    // If this component doesn't have a source to find, that's an error.
    CHECK(source_result != predecessors_.end())
        << "No link to source " << channel.source_component();

    // If we jump more times than there are components in the graph, that
    // is an error state.
    CHECK_LT(path.size(), num_components) << "Too many jumps. Is there a "
                                             "loop in the MasterSpec "
                                             "component definition?";

    // Add the source to the vector and repeat.
    path.push_back(source_result->second);
    current_component = source_result->second;
  }

  // At this point, we have the source chain for the traslator and can
  // build it.
  std::unique_ptr<IndexTranslator> translator(
      new IndexTranslator(path, channel.source_translator()));
  return translator;
}

}  // namespace dragnn
}  // namespace syntaxnet

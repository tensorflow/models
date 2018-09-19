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

#include "dragnn/core/compute_session_impl.h"

#include <memory>
#include <utility>

#include "dragnn/components/util/bulk_feature_extractor.h"
#include "dragnn/core/component_registry.h"
#include "dragnn/core/compute_session.h"
#include "dragnn/core/compute_session_pool.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/input_batch.h"
#include "dragnn/core/test/fake_component_base.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_component.h"
#include "dragnn/core/test/mock_transition_state.h"
#include "dragnn/core/util/label.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

using syntaxnet::test::EqualsProto;
using testing::ElementsAre;
using testing::NotNull;
using testing::Return;
using testing::_;

// *****************************************************************************
// Test-internal class definitions.
// *****************************************************************************

class TestComponentType1 : public FakeComponentBase {
 public:
  int BeamSize() const override { return 3; }
  int BatchSize() const override { return 1; }
};

REGISTER_DRAGNN_COMPONENT(TestComponentType1);

class TestComponentType2 : public FakeComponentBase {
 public:
  int BeamSize() const override { return 4; }
  int BatchSize() const override { return 2; }
};

REGISTER_DRAGNN_COMPONENT(TestComponentType2);

// Define a component that returns false for IsReady and IsTerminal.
class UnreadyComponent : public FakeComponentBase {
 public:
  bool IsReady() const override { return false; }
  int BeamSize() const override { return 1; }
  int BatchSize() const override { return 2; }
  bool IsTerminal() const override { return false; }
};

REGISTER_DRAGNN_COMPONENT(UnreadyComponent);

class ComputeSessionImplTestPoolAccessor {
 public:
  static void SetComponentBuilder(
      ComputeSessionPool *pool,
      std::function<std::unique_ptr<Component>(const string &component_name,
                                               const string &backend_type)>
          component_builder_function) {
    pool->SetComponentBuilder(std::move(component_builder_function));
  }
};

// An InputBatch that uses the serialized data directly.
class IdentityBatch : public InputBatch {
 public:
  // Implements InputBatch.
  void SetData(const std::vector<string> &data) override { data_ = data; }
  int GetSize() const override { return data_.size(); }
  const std::vector<string> GetSerializedData() const override { return data_; }

 private:
  std::vector<string> data_;  // the batch data
};

// *****************************************************************************
// Tests begin here.
// *****************************************************************************

// Helper function to validate a translation path against a vector of expected
// component name strings.
void ValidatePath(const std::vector<string> &expected_path,
                  const std::vector<Component *> &path) {
  EXPECT_EQ(expected_path.size(), path.size());
  for (int i = 0; i < expected_path.size(); ++i) {
    EXPECT_EQ(expected_path.at(i), path.at(i)->Name());
  }
}

void AddComponentToSpec(const string &component_name,
                        const string &backend_name, MasterSpec *spec) {
  auto component_spec = spec->add_component();
  component_spec->set_name(component_name);
  auto backend = component_spec->mutable_backend();
  backend->set_registered_name(backend_name);
}

void AddTranslatorToSpec(const string &source_name, const string &dest_name,
                         const string &type, MasterSpec *spec) {
  // Find the destination component.
  ComponentSpec *dest_spec = nullptr;
  for (int i = 0; i < spec->component_size(); ++i) {
    if (spec->component(i).name() == dest_name) {
      dest_spec = spec->mutable_component(i);
      break;
    }
  }

  // Make sure it's not null...
  EXPECT_NE(dest_spec, nullptr);

  // Set up the translator.
  auto linked_feature = dest_spec->add_linked_feature();
  linked_feature->set_source_component(source_name);
  linked_feature->set_source_translator(type);
}

TEST(ComputeSessionImplTest, CreatesComponent) {
  // Define a spec that creates an instance of TestComponentType1.
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Make sure that the component exists and is of type TestComponentType1.
  string Type1ComponentDesc = "component_one";
  constexpr int kType1BatchSize = 1;
  EXPECT_EQ(Type1ComponentDesc, session->GetDescription("component_one"));
  EXPECT_EQ(kType1BatchSize, session->BatchSize("component_one"));
}

TEST(ComputeSessionImplTest, ReturnsComponentSpec) {
  // Define a spec that creates an instance of TestComponentType1 and
  // TestComponentType2.
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);
  AddComponentToSpec("component_two", "TestComponentType2", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);

  auto session = pool.GetSession();
  EXPECT_EQ(spec.component(1).DebugString(),
            session->Spec("component_two").DebugString());
}

TEST(ComputeSessionImplTest, CreatesMultipleComponents) {
  // Define a spec that creates an instance of TestComponentType1 and
  // TestComponentType2.
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);
  AddComponentToSpec("component_two", "TestComponentType2", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);

  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Make sure that the components exist and are the correct type.
  string Type1ComponentDesc = "component_one";
  constexpr int kType1BatchSize = 1;
  EXPECT_EQ(Type1ComponentDesc, session->GetDescription("component_one"));
  EXPECT_EQ(kType1BatchSize, session->BatchSize("component_one"));

  string Type2ComponentDesc = "component_two";
  constexpr int kType2BatchSize = 2;
  EXPECT_EQ(Type2ComponentDesc, session->GetDescription("component_two"));
  EXPECT_EQ(kType2BatchSize, session->BatchSize("component_two"));
}

TEST(ComputeSessionImplTest, InitializesComponents) {
  // Define a spec that creates an instance of TestComponentType1 and
  // TestComponentType2.
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);
  AddComponentToSpec("component_two", "TestComponentType2", &spec);

  // Create a map to hold references to mock components. Expect the correct
  // initialization call (with the appropriate proto passed in).
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    if (name == "component_one") {
      EXPECT_CALL(*component,
                  InitializeComponent(EqualsProto(spec.component(0))));
    } else {
      EXPECT_CALL(*component,
                  InitializeComponent(EqualsProto(spec.component(1))));
    }
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});
}

TEST(ComputeSessionImplTest, CreatesTranslator) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);
  AddComponentToSpec("component_two", "TestComponentType2", &spec);

  // Add a translator from component 1 to component 2.
  AddTranslatorToSpec("component_one", "component_two", "identity", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  auto linked_features = session->Translators("component_two");
  EXPECT_EQ(1, linked_features.size());
  ValidatePath({"component_two", "component_one"},
               linked_features.at(0)->path());
  EXPECT_EQ(linked_features.at(0)->method(), "identity");
}

TEST(ComputeSessionImplTest, CreatesTranslatorWithLongWalk) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);
  AddComponentToSpec("component_three", "TestComponentType2", &spec);

  // Add a translator from component 3 to component 1.
  AddTranslatorToSpec("component_one", "component_three", "identity", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Get and validate the linked feature vector for component 3.
  auto linked_features = session->Translators("component_three");
  EXPECT_EQ(1, linked_features.size());
  ValidatePath({"component_three", "component_two", "component_one"},
               linked_features.at(0)->path());
  EXPECT_EQ(linked_features.at(0)->method(), "identity");
}

TEST(ComputeSessionImplTest, CreatesTranslatorForMultipleComponents) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);
  AddComponentToSpec("component_three", "TestComponentType2", &spec);

  // Add a translator from component 3 to component 1.
  AddTranslatorToSpec("component_one", "component_three", "identity", &spec);

  // Add a translator from component 3 to component 2.
  AddTranslatorToSpec("component_two", "component_three", "history", &spec);

  // Add a translator from component 2 to component 1.
  AddTranslatorToSpec("component_one", "component_two", "history", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Get and validate the linked feature vector for component 3.
  auto linked_features = session->Translators("component_three");
  EXPECT_EQ(2, linked_features.size());
  ValidatePath({"component_three", "component_two", "component_one"},
               linked_features.at(0)->path());
  EXPECT_EQ(linked_features.at(0)->method(), "identity");
  ValidatePath({"component_three", "component_two"},
               linked_features.at(1)->path());
  EXPECT_EQ(linked_features.at(1)->method(), "history");

  // Get and validate the linked feature vector for component 2.
  auto linked_features_2 = session->Translators("component_two");
  EXPECT_EQ(1, linked_features_2.size());
  ValidatePath({"component_two", "component_one"},
               linked_features_2.at(0)->path());
  EXPECT_EQ(linked_features_2.at(0)->method(), "history");
}

TEST(ComputeSessionImplTest, CreatesMultipleTranslatorsBetweenSameComponents) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);

  // Add a translator from component 2 to component 1.
  AddTranslatorToSpec("component_one", "component_two", "identity", &spec);

  // Add a translator from component 2 to component 1.
  AddTranslatorToSpec("component_one", "component_two", "history", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Get and validate the linked feature vector for component 2.
  auto linked_features = session->Translators("component_two");
  EXPECT_EQ(2, linked_features.size());
  ValidatePath({"component_two", "component_one"},
               linked_features.at(0)->path());
  EXPECT_EQ(linked_features.at(0)->method(), "identity");
  ValidatePath({"component_two", "component_one"},
               linked_features.at(1)->path());
  EXPECT_EQ(linked_features.at(1)->method(), "history");
}

TEST(ComputeSessionImplTest, CreatesSelfReferentialTranslator) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);

  // Add a translator from component 1 to component 1.
  AddTranslatorToSpec("component_one", "component_one", "identity", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Get and validate the linked feature vector for component 1.
  auto linked_features = session->Translators("component_one");
  EXPECT_EQ(1, linked_features.size());
  ValidatePath({"component_one"}, linked_features.at(0)->path());
}

TEST(ComputeSessionImplTest, CreateTranslatorFailsWithWrongNameDeathTest) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);

  // Add a translator from a nonexistent component to component 1.
  AddTranslatorToSpec("NONEXISTENT_COMPONENT_THIS_WILL_DIE", "component_one",
                      "identity", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);

  EXPECT_DEATH(pool.GetSession(), "Unable to find source component");
}

TEST(ComputeSessionImplTest, GetsSourceComponentBeamSize) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);
  AddComponentToSpec("component_two", "TestComponentType2", &spec);

  // Add a translator from component 1 to component 2.
  AddTranslatorToSpec("component_one", "component_two", "identity", &spec);

  // Create a pool so we can get a session.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  constexpr int kChannelId = 0;
  constexpr int kType1BeamSize = 3;
  EXPECT_EQ(kType1BeamSize,
            session->SourceComponentBeamSize("component_two", kChannelId));
}

TEST(ComputeSessionImplTest, GetsTranslatedLinkFeatures) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);
  AddComponentToSpec("component_two", "TestComponentType2", &spec);

  // Add a translator from component 1 to component 2.
  AddTranslatorToSpec("component_one", "component_two", "identity", &spec);

  // Create a map to hold references to mock components.
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    EXPECT_CALL(*component, InitializeComponent(_));
    EXPECT_CALL(*component, IsReady()).WillRepeatedly(Return(true));
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool, substituting a mock component builder.
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Create a link features vector to return from the destination component.
  std::vector<LinkFeatures> features;
  LinkFeatures feature_one;
  feature_one.set_batch_idx(12);
  feature_one.set_beam_idx(23);
  feature_one.set_feature_value(34);
  features.push_back(feature_one);
  LinkFeatures feature_two;
  feature_two.set_batch_idx(45);
  feature_two.set_beam_idx(56);
  feature_two.set_feature_value(67);
  features.push_back(feature_two);

  // This link feature should remain empty.
  LinkFeatures padding_feature;
  features.push_back(padding_feature);

  // The session should request the raw link features for the specified channel.
  constexpr int kChannelId = 0;
  EXPECT_CALL(*mock_components["component_two"], GetRawLinkFeatures(kChannelId))
      .WillOnce(Return(features));

  // The session will request the source beam index for both features.
  constexpr int kSourceBeamOneIndex = 7;
  EXPECT_CALL(
      *mock_components["component_two"],
      GetSourceBeamIndex(feature_one.beam_idx(), feature_one.batch_idx()))
      .WillOnce(Return(kSourceBeamOneIndex));
  constexpr int kSourceBeamTwoIndex = 77;
  EXPECT_CALL(
      *mock_components["component_two"],
      GetSourceBeamIndex(feature_two.beam_idx(), feature_two.batch_idx()))
      .WillOnce(Return(kSourceBeamTwoIndex));

  // The translate call should use the 'identity' translator on the step index.
  // This means that the GetBeamIndexAtStep call will have the values from
  // the linked feature proto (since we also don't have an intermediate
  // component.)
  constexpr int kFeatureOneBeamIndex = 9;
  EXPECT_CALL(*mock_components["component_one"],
              GetBeamIndexAtStep(feature_one.feature_value(),
                                 kSourceBeamOneIndex, feature_one.batch_idx()))
      .WillOnce(Return(kFeatureOneBeamIndex));

  constexpr int kFeatureTwoBeamIndex = 99;
  EXPECT_CALL(*mock_components["component_one"],
              GetBeamIndexAtStep(feature_two.feature_value(),
                                 kSourceBeamTwoIndex, feature_two.batch_idx()))
      .WillOnce(Return(kFeatureTwoBeamIndex));

  auto translated_features =
      session->GetTranslatedLinkFeatures("component_two", kChannelId);

  auto translated_one = translated_features.at(0);
  EXPECT_EQ(translated_one.batch_idx(), feature_one.batch_idx());
  EXPECT_EQ(translated_one.beam_idx(), kFeatureOneBeamIndex);
  EXPECT_EQ(translated_one.step_idx(), feature_one.feature_value());

  auto translated_two = translated_features.at(1);
  EXPECT_EQ(translated_two.batch_idx(), feature_two.batch_idx());
  EXPECT_EQ(translated_two.beam_idx(), kFeatureTwoBeamIndex);
  EXPECT_EQ(translated_two.step_idx(), feature_two.feature_value());

  // The third feature is a padding feature, and so should be empty.
  auto translated_three = translated_features.at(2);
  EXPECT_FALSE(translated_three.has_batch_idx());
  EXPECT_FALSE(translated_three.has_beam_idx());
  EXPECT_FALSE(translated_three.has_step_idx());
  EXPECT_FALSE(translated_three.has_feature_value());
}

TEST(ComputeSessionImplTest, InitializesComponentDataWithNoSource) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);

  // Create a map to hold references to mock components.
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    EXPECT_CALL(*component, InitializeComponent(_));
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool, substituting a mock component builder.
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();

  // Set expectations and get a session, then get the component.
  // The initialization should be called with an empty state vector, but with
  // a non-null input batch cache pointer.
  constexpr int kMaxBeamSize = 11;
  EXPECT_CALL(*(mock_components["component_one"]),
              InitializeData(testing::IsEmpty(), kMaxBeamSize, NotNull()));
  session->SetInputData({"arbitrary_data"});
  session->InitializeComponentData("component_one", kMaxBeamSize);
}

TEST(ComputeSessionImplTest, InitializesComponentWithSource) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);

  // Create a map to hold references to mock components.
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    EXPECT_CALL(*component, InitializeComponent(_));
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool, substituting a mock component builder..
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Set expectations.
  constexpr int kMaxBeamSize = 11;
  MockTransitionState mock_transition_state;
  std::vector<std::vector<const TransitionState *>> beam(
      {{&mock_transition_state}});

  // Expect that the first component will report that it is terminal and return
  // a beam.
  EXPECT_CALL(*mock_components["component_one"], IsTerminal())
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_components["component_one"], GetBeam())
      .WillOnce(Return(beam));

  // Expect that the second component will receive that beam.
  EXPECT_CALL(*mock_components["component_two"],
              InitializeData(beam, kMaxBeamSize, NotNull()));

  // Attempt to initialize the component.
  session->InitializeComponentData("component_two", kMaxBeamSize);
}

TEST(ComputeSessionImplTest,
     InitializeDataFailsWhenInputDataNotProvidedDeathTest) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);

  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();

  constexpr int kMaxBeamSize = 3;
  EXPECT_DEATH(session->InitializeComponentData("component_one", kMaxBeamSize),
               "without providing input data");
}

TEST(ComputeSessionImplTest,
     InitializeDataFailsWhenComponentDoesNotExistdeathTest) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);

  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  constexpr int kMaxBeamSize = 3;
  EXPECT_DEATH(
      session->InitializeComponentData("DOES_NOT_EXIST_DIE", kMaxBeamSize),
      "Could not find component");
}

TEST(ComputeSessionImplTest,
     InitializeDataFailsWhenSourceIsNotTerminalDeathTest) {
  auto function_that_will_die = []() {
    MasterSpec spec;
    GridPoint hyperparams;

    AddComponentToSpec("component_one", "TestComponentType2", &spec);
    AddComponentToSpec("component_two", "TestComponentType1", &spec);

    // Create a map to hold references to mock components.
    std::map<string, MockComponent *> mock_components;
    auto builder_function = [&mock_components, spec](
                                const string &name,
                                const string &backend_type) {
      VLOG(2) << "Mocking for name: " << name;
      std::unique_ptr<MockComponent> component(new MockComponent());
      EXPECT_CALL(*component, InitializeComponent(_));
      mock_components[name] = component.get();
      return component;
    };

    // Create a pool, substituting a mock component builder.
    ComputeSessionPool pool(spec, hyperparams);
    ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                            builder_function);
    auto session = pool.GetSession();
    session->SetInputData({"arbitrary_data"});

    // Expect that the first component will report that it is not terminal
    EXPECT_CALL(*mock_components["component_one"], IsTerminal())
        .WillOnce(Return(false));

    // Attempt to initialize the component.
    constexpr int kMaxBeamSize = 11;
    session->InitializeComponentData("component_two", kMaxBeamSize);
  };

  // The death expectation is interacting strangely with this test, so I need
  // to wrap the function in a lambda.
  EXPECT_DEATH(function_that_will_die(), "is not terminal");
}

TEST(ComputeSessionImplTest, ResetSessionResetsAllComponents) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);

  // Create a map to hold references to mock components.
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    EXPECT_CALL(*component, InitializeComponent(_));
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool, substituting a mock component builder.
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Expect that the first component will report that it is not terminal
  EXPECT_CALL(*mock_components["component_one"], ResetComponent());
  EXPECT_CALL(*mock_components["component_two"], ResetComponent());

  session->ResetSession();
}

TEST(ComputeSessionImplTest, SetTracingPropagatesToAllComponents) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);

  // Add a translator from component 1 to component 2.
  AddTranslatorToSpec("component_one", "component_two", "identity", &spec);

  // Create a map to hold references to mock components.
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    EXPECT_CALL(*component, InitializeComponent(_));
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool, substituting a mock component builder.
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Enable tracing on the session.
  session->SetTracing(true);

  // Initialize the first component, along with its tracing.
  constexpr int kMaxBeamSize = 1;
  EXPECT_CALL(*mock_components["component_one"],
              InitializeData(testing::IsEmpty(), kMaxBeamSize, NotNull()));
  EXPECT_CALL(*mock_components["component_one"], InitializeTracing());
  session->InitializeComponentData("component_one", kMaxBeamSize);

  MockTransitionState mock_transition_state;
  std::vector<std::vector<const TransitionState *>> beam(
      {{&mock_transition_state}});
  EXPECT_CALL(*mock_components["component_one"], IsTerminal())
      .WillOnce(Return(true));
  EXPECT_CALL(*mock_components["component_one"], GetBeam())
      .WillOnce(Return(beam));

  // Expect that the second component will receive that beam, and then its
  // tracing will be initialized.
  EXPECT_CALL(*mock_components["component_two"],
              InitializeData(beam, kMaxBeamSize, NotNull()));
  EXPECT_CALL(*mock_components["component_two"], InitializeTracing());
  session->InitializeComponentData("component_two", kMaxBeamSize);

  // Expect that all components will see the tracing value.
  EXPECT_CALL(*mock_components["component_one"], IsReady())
      .WillRepeatedly(Return(true));
  EXPECT_CALL(*mock_components["component_two"], IsReady())
      .WillRepeatedly(Return(true));

  std::vector<LinkFeatures> features;
  LinkFeatures feature_one;
  feature_one.set_beam_idx(0);
  feature_one.set_batch_idx(0);
  feature_one.set_feature_value(34);
  features.push_back(feature_one);

  // Translated version: feature_value is copied to step_idx.
  std::vector<LinkFeatures> translated;
  feature_one.set_step_idx(feature_one.feature_value());
  translated.push_back(feature_one);

  // The session should request the raw link features for the specified channel.
  constexpr int kChannelId = 0;
  EXPECT_CALL(*mock_components["component_two"], GetRawLinkFeatures(kChannelId))
      .WillRepeatedly(Return(features));

  // Identity will not change the features.
  EXPECT_CALL(*mock_components["component_two"],
              AddTranslatedLinkFeaturesToTrace(
                  ElementsAre(EqualsProto(translated[0])), kChannelId));
  session->GetTranslatedLinkFeatures("component_two", kChannelId);

  // Now disable tracing.  This time we don't expect any tracing to be called.
  EXPECT_CALL(*mock_components["component_one"], DisableTracing());
  EXPECT_CALL(*mock_components["component_two"], DisableTracing());
  session->SetTracing(false);
  EXPECT_CALL(*mock_components["component_two"],
              AddTranslatedLinkFeaturesToTrace(
                  ElementsAre(EqualsProto(translated[0])), kChannelId))
      .Times(0);
  session->GetTranslatedLinkFeatures("component_two", kChannelId);
}

TEST(ComputeSessionImplTest, TraceSourceBeamPath) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType1", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);
  AddComponentToSpec("component_three", "TestComponentType1", &spec);

  // Create a map to hold references to mock components.
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    EXPECT_CALL(*component, InitializeComponent(_));
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool, substituting a mock component builder.
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  ComponentTrace trace;

  // Test logic: verify that the traces correspond only to the paths taken to
  // reach the final states in component 3. This requires backtracking to
  // retrace the path of the beam. In this case, we expect three paths:
  //
  // Component 0     -> Component 1     -> Component 2
  // batch 0, beam 6 -> batch 0, beam 4 -> batch 0, beam 0
  // batch 0, beam 6 -> batch 0, beam 3 -> batch 0, beam 1
  // batch 1, beam 0 -> batch 1, beam 2 -> batch 1, beam 0

  // Fill the component traces with some dummy values of the approach beam sizes
  // for each batch.

  // Component 1: batch 0 has beam size 7, batch 1 has beam size 2.
  std::vector<std::vector<ComponentTrace>> component_one_trace = {
      {trace, trace, trace, trace, trace, trace, trace}, {trace, trace}};

  // Component 2: batch 0 has beam size 5, batch 1 has beam size 3.
  std::vector<std::vector<ComponentTrace>> component_two_trace = {
      {trace, trace, trace, trace, trace}, {trace, trace, trace}};

  // Component 3: batch 0 has beam size 2, batch 1 has beam size 1.
  std::vector<std::vector<ComponentTrace>> component_three_trace = {
      {trace, trace}, {trace}};

  // The Session will get all traces from every component.
  EXPECT_CALL(*mock_components["component_one"], GetTraceProtos())
      .WillOnce(Return(component_one_trace));
  EXPECT_CALL(*mock_components["component_two"], GetTraceProtos())
      .WillOnce(Return(component_two_trace));
  EXPECT_CALL(*mock_components["component_three"], GetTraceProtos())
      .WillOnce(Return(component_three_trace));

  // Final beam has 2 states in batch 0, 1 state in batch 1. So we expect three
  // chains.
  MockTransitionState mock_transition_state;
  std::vector<std::vector<const TransitionState *>> beam(
      {{&mock_transition_state, &mock_transition_state},
       {&mock_transition_state}});

  EXPECT_CALL(*mock_components["component_three"], GetBeam())
      .WillOnce(Return(beam));

  // First test chain.
  EXPECT_CALL(*mock_components["component_three"], GetSourceBeamIndex(0, 0))
      .WillOnce(Return(4));
  EXPECT_CALL(*mock_components["component_two"], GetSourceBeamIndex(4, 0))
      .WillOnce(Return(6));

  // Second test chain.
  EXPECT_CALL(*mock_components["component_three"], GetSourceBeamIndex(1, 0))
      .WillOnce(Return(3));
  EXPECT_CALL(*mock_components["component_two"], GetSourceBeamIndex(3, 0))
      .WillOnce(Return(6));

  // Third test chain.
  EXPECT_CALL(*mock_components["component_three"], GetSourceBeamIndex(0, 1))
      .WillOnce(Return(2));
  EXPECT_CALL(*mock_components["component_two"], GetSourceBeamIndex(2, 1))
      .WillOnce(Return(1));

  // Execute the call's.
  session->GetTraceProtos();
}

TEST(ComputeSessionImplTest, InterfacePassesThrough) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "TestComponentType2", &spec);
  AddComponentToSpec("component_two", "TestComponentType1", &spec);

  // Create a map to hold references to mock components.
  std::map<string, MockComponent *> mock_components;
  auto builder_function = [&mock_components, spec](const string &name,
                                                   const string &backend_type) {
    VLOG(2) << "Mocking for name: " << name;
    std::unique_ptr<MockComponent> component(new MockComponent());
    EXPECT_CALL(*component, InitializeComponent(_));
    mock_components[name] = component.get();
    return component;
  };

  // Create a pool, substituting a mock component builder.
  ComputeSessionPool pool(spec, hyperparams);
  ComputeSessionImplTestPoolAccessor::SetComponentBuilder(&pool,
                                                          builder_function);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Expect that the first component will report that it is ready.
  EXPECT_CALL(*mock_components["component_one"], IsReady())
      .WillRepeatedly(Return(true));

  // BatchSize()
  int batch_size = 3;
  EXPECT_CALL(*mock_components["component_one"], BatchSize())
      .WillOnce(Return(batch_size));
  EXPECT_EQ(batch_size, session->BatchSize("component_one"));

  // BeamSize()
  int beam_size = 32;
  EXPECT_CALL(*mock_components["component_one"], BeamSize())
      .WillOnce(Return(beam_size));
  EXPECT_EQ(beam_size, session->BeamSize("component_one"));

  // AdvanceFromOracle()
  EXPECT_CALL(*mock_components["component_one"], AdvanceFromOracle());
  session->AdvanceFromOracle("component_one");

  // AdvanceFromPrediction()
  const int kNumActions = 1;
  const float score_matrix[] = {1.0, 2.3, 4.5};
  EXPECT_CALL(*mock_components["component_one"],
              AdvanceFromPrediction(score_matrix, batch_size, kNumActions));
  session->AdvanceFromPrediction("component_one", score_matrix, batch_size,
                                 kNumActions);

  // GetFixedFeatures
  auto allocate_indices = [](int size) -> int32 * { return nullptr; };
  auto allocate_ids = [](int size) -> int64 * { return nullptr; };
  auto allocate_weights = [](int size) -> float * { return nullptr; };
  constexpr int kChannelId = 3;
  EXPECT_CALL(*mock_components["component_one"],
              GetFixedFeatures(_, _, _, kChannelId))
      .WillOnce(Return(0));
  EXPECT_EQ(
      0, session->GetInputFeatures("component_one", allocate_indices,
                                   allocate_ids, allocate_weights, kChannelId));

  // BulkGetFixedFeatures
  BulkFeatureExtractor extractor(nullptr, nullptr, nullptr, false, 0, 0);
  EXPECT_CALL(*mock_components["component_one"], BulkGetFixedFeatures(_))
      .WillOnce(Return(0));
  EXPECT_EQ(0, session->BulkGetInputFeatures("component_one", extractor));

  // BulkEmbedFixedFeatures
  EXPECT_CALL(*mock_components["component_one"],
              BulkEmbedFixedFeatures(1, 2, 3, _, _));
  session->BulkEmbedFixedFeatures("component_one", 1, 2, 3, {nullptr}, nullptr);

  // EmitOracleLabels()
  // The size of oracle_labels is batch_size * beam_size * num_labels.
  const std::vector<std::vector<std::vector<Label>>> oracle_labels{
      {{{0, 1.f}}, {{1, 1.f}}}, {{{2, 1.f}}, {{3, 1.f}}}};

  EXPECT_CALL(*mock_components["component_one"], GetOracleLabels())
      .WillOnce(Return(oracle_labels));
  EXPECT_EQ(oracle_labels, session->EmitOracleLabels("component_one"));

  // IsTerminal()
  bool is_terminal = true;
  EXPECT_CALL(*mock_components["component_one"], IsTerminal())
      .WillOnce(Return(is_terminal));
  EXPECT_EQ(is_terminal, session->IsTerminal("component_one"));

  // FinalizeData()
  EXPECT_CALL(*mock_components["component_one"], FinalizeData());
  session->FinalizeData("component_one");
}

TEST(ComputeSessionImplTest, InterfaceRequiresReady) {
  MasterSpec spec;
  GridPoint hyperparams;

  AddComponentToSpec("component_one", "UnreadyComponent", &spec);

  // Create a pool, substituting a mock component builder.
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();
  session->SetInputData({"arbitrary_data"});

  // Call the functions which should die if the component isn't ready.
  EXPECT_DEATH(session->BatchSize("component_one"),
               "without first initializing it");
  EXPECT_DEATH(session->BeamSize("component_one"),
               "without first initializing it");
  EXPECT_DEATH(session->AdvanceFromOracle("component_one"),
               "without first initializing it");
  EXPECT_DEATH(session->EmitOracleLabels("component_one"),
               "without first initializing it");
  EXPECT_DEATH(session->IsTerminal("component_one"),
               "without first initializing it");
  EXPECT_DEATH(session->FinalizeData("component_one"),
               "without first initializing it");

  constexpr int kScoreMatrixLength = 3;
  const float score_matrix[kScoreMatrixLength] = {1.0, 2.3, 4.5};
  EXPECT_DEATH(session->AdvanceFromPrediction("component_one", score_matrix,
                                              kScoreMatrixLength, 1),
               "without first initializing it");
  constexpr int kArbitraryChannelId = 3;
  EXPECT_DEATH(session->GetInputFeatures("component_one", nullptr, nullptr,
                                         nullptr, kArbitraryChannelId),
               "without first initializing it");
  BulkFeatureExtractor extractor(nullptr, nullptr, nullptr, false, 0, 0);
  EXPECT_DEATH(session->BulkGetInputFeatures("component_one", extractor),
               "without first initializing it");
  EXPECT_DEATH(session->BulkEmbedFixedFeatures("component_one", 0, 0, 0,
                                               {nullptr}, nullptr),
               "without first initializing it");
  EXPECT_DEATH(
      session->GetTranslatedLinkFeatures("component_one", kArbitraryChannelId),
      "without first initializing it");
}

TEST(ComputeSessionImplTest, SetInputBatchCache) {
  // Use empty protos since we won't interact with components.
  MasterSpec spec;
  GridPoint hyperparams;
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();

  // Initialize a cached IdentityBatch.
  const std::vector<string> data = {"foo", "bar", "baz"};
  std::unique_ptr<InputBatchCache> input_batch_cache(new InputBatchCache(data));
  input_batch_cache->GetAs<IdentityBatch>();

  // Inject the cache into the session.
  session->SetInputBatchCache(std::move(input_batch_cache));

  // Check that the injected batch can be retrieved.
  EXPECT_EQ(session->GetSerializedPredictions(), data);
}

TEST(ComputeSessionImplTest, GetInputBatchCache) {
  // Use empty protos since we won't interact with components.
  MasterSpec spec;
  GridPoint hyperparams;
  ComputeSessionPool pool(spec, hyperparams);
  auto session = pool.GetSession();

  // No input data yet.
  EXPECT_EQ(session->GetInputBatchCache(), nullptr);

  // Set some data, expect some batch to be returned.
  session->SetInputData({"arbitrary_data"});
  EXPECT_NE(session->GetInputBatchCache(), nullptr);

  // Create a dummy batch.
  const std::vector<string> data = {"foo", "bar", "baz"};
  std::unique_ptr<InputBatchCache> input_batch_cache(new InputBatchCache(data));
  InputBatchCache *input_batch_cache_ptr = input_batch_cache.get();

  // Inject a batch, expect that batch to be returned.
  session->SetInputBatchCache(std::move(input_batch_cache));
  EXPECT_EQ(session->GetInputBatchCache(), input_batch_cache_ptr);
}

}  // namespace dragnn
}  // namespace syntaxnet

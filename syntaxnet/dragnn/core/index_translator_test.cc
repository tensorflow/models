#include "dragnn/core/index_translator.h"

#include "dragnn/core/test/mock_component.h"
#include "dragnn/core/test/mock_transition_state.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {

using testing::MockFunction;
using testing::Return;

TEST(IndexTranslatorTest, PerformsIdentityTranslation) {
  MockComponent mock_component;

  // We are testing the Identity lookup with a single component (so, self-
  // referencing) and thus we expect the translator to call GetBeamIndexAtStep
  // for the step and index we pass in.
  constexpr int kBeam = 4;
  constexpr int kFeature = 2;
  constexpr int kResultIndex = 3;
  constexpr int kBatch = 99;
  EXPECT_CALL(mock_component, GetBeamIndexAtStep(kFeature, kBeam, kBatch))
      .WillOnce(Return(kResultIndex));

  // Execute!
  IndexTranslator translator({&mock_component}, "identity");
  auto result = translator.Translate(kBatch, kBeam, kFeature);
  EXPECT_EQ(kResultIndex, result.beam_index);
  EXPECT_EQ(kFeature, result.step_index);
  EXPECT_EQ(kBatch, result.batch_index);
}

TEST(IndexTranslatorTest, PerformsHistoryTranslation) {
  MockComponent mock_component;

  // We are testing the History lookup with a single component (so, self-
  // referencing) and thus we expect the translator to call StepsTaken() to get
  // the number of steps taken and GetBeamIndexAtStep with (total-desired).
  constexpr int kBeam = 4;
  constexpr int kFeature = 2;
  constexpr int kTotalNumberSteps = 8;
  constexpr int kBatch = 99;

  // Here, the expected step result is two in from the final index, so
  // (8-1) - 2, or 5.
  constexpr int kExpectedResult = 5;
  constexpr int kResultIndex = 3;
  EXPECT_CALL(mock_component, StepsTaken(kBatch))
      .WillRepeatedly(Return(kTotalNumberSteps));
  EXPECT_CALL(mock_component,
              GetBeamIndexAtStep(kExpectedResult, kBeam, kBatch))
      .WillOnce(Return(kResultIndex));

  // Execute!
  IndexTranslator translator({&mock_component}, "history");
  auto result = translator.Translate(kBatch, kBeam, kFeature);
  EXPECT_EQ(kResultIndex, result.beam_index);
  EXPECT_EQ(kExpectedResult, result.step_index);
  EXPECT_EQ(kBatch, result.batch_index);
}

TEST(IndexTranslatorTest, TraversesPathToLookup) {
  MockComponent mock_component_a;
  MockComponent mock_component_b;
  MockComponent mock_component_c;
  constexpr int kBatch = 99;

  // The translator should request the source index from mock component A.
  constexpr int kBeam = 4;
  constexpr int kSourceBIndex = 3;
  EXPECT_CALL(mock_component_a, GetSourceBeamIndex(kBeam, kBatch))
      .WillOnce(Return(kSourceBIndex));

  // The translator should use the source index from A in a source index request
  // to component B.
  constexpr int kSourceCIndex = 17;
  EXPECT_CALL(mock_component_b, GetSourceBeamIndex(kSourceBIndex, kBatch))
      .WillOnce(Return(kSourceCIndex));

  // The translator should request the beam index at the requested step in
  // component C, using the beam index from the source index request to B.
  constexpr int kFeature = 2;
  constexpr int kResultIndex = 1157;

  // This is testing with an identity translator, so kFeature == kStep.
  EXPECT_CALL(mock_component_c,
              GetBeamIndexAtStep(kFeature, kSourceCIndex, kBatch))
      .WillOnce(Return(kResultIndex));

  // Execute!
  IndexTranslator translator(
      {&mock_component_a, &mock_component_b, &mock_component_c}, "identity");
  auto result = translator.Translate(kBatch, kBeam, kFeature);
  EXPECT_EQ(kResultIndex, result.beam_index);
  EXPECT_EQ(kFeature, result.step_index);
  EXPECT_EQ(kBatch, result.batch_index);
}

TEST(IndexTranslatorTest, RequestsArbitraryTranslationFunction) {
  MockComponent mock_component;
  MockFunction<int(int, int, int)> mock_function;

  // This test ensures that we can get an arbitrary translation function
  // from the component and execute it properly.
  constexpr int kBeam = 4;
  constexpr int kFeature = 2;
  constexpr int kFunctionResult = 10;
  constexpr int kResultIndex = 3;
  constexpr int kBatch = 99;

  // The arbitrary function should be called with the desired input.
  EXPECT_CALL(mock_function, Call(kBatch, kBeam, kFeature))
      .WillOnce(Return(kFunctionResult));

  // The translator should request the function from the component.
  EXPECT_CALL(mock_component, GetStepLookupFunction("arbitrary_function"))
      .WillOnce(Return(mock_function.AsStdFunction()));

  // The translator should call GetBeamIndexAtStep with the result of calling
  // the function.
  EXPECT_CALL(mock_component,
              GetBeamIndexAtStep(kFunctionResult, kBeam, kBatch))
      .WillOnce(Return(kResultIndex));

  // Execute!
  IndexTranslator translator({&mock_component}, "arbitrary_function");
  auto result = translator.Translate(kBatch, kBeam, kFeature);
  EXPECT_EQ(kResultIndex, result.beam_index);
  EXPECT_EQ(kFunctionResult, result.step_index);
  EXPECT_EQ(kBatch, result.batch_index);
}

// This test ensures that the translation function is queried with the beam
// index for that component, and that the translation function is taken from
// the correct component.
TEST(IndexTranslatorTest, RequestsArbitraryTranslationAcrossComponents) {
  MockComponent mock_component_a;
  MockComponent mock_component_b;
  MockFunction<int(int, int, int)> mock_function;

  // This test ensures that we can get an arbitrary translation function
  // from the component and execute it properly.
  constexpr int kFeature = 2;
  constexpr int kFunctionResult = 10;
  constexpr int kResultIndex = 3;
  constexpr int kBatch = 99;

  // The translator should request the source index from mock component A.
  constexpr int kBeam = 4;
  constexpr int kSourceBIndex = 3;
  EXPECT_CALL(mock_component_a, GetSourceBeamIndex(kBeam, kBatch))
      .WillOnce(Return(kSourceBIndex));

  // The translator should request the function from the component.
  EXPECT_CALL(mock_component_b, GetStepLookupFunction("arbitrary_function"))
      .WillOnce(Return(mock_function.AsStdFunction()));

  // The arbitrary function should be called with the desired input.
  EXPECT_CALL(mock_function, Call(kBatch, kSourceBIndex, kFeature))
      .WillOnce(Return(kFunctionResult));

  // The translator should call GetBeamIndexAtStep with the result of calling
  // the function.
  EXPECT_CALL(mock_component_b,
              GetBeamIndexAtStep(kFunctionResult, kSourceBIndex, kBatch))
      .WillOnce(Return(kResultIndex));

  // Execute!
  IndexTranslator translator({&mock_component_a, &mock_component_b},
                             "arbitrary_function");
  auto result = translator.Translate(kBatch, kBeam, kFeature);
  EXPECT_EQ(kResultIndex, result.beam_index);
  EXPECT_EQ(kFunctionResult, result.step_index);
  EXPECT_EQ(kBatch, result.batch_index);
}

}  // namespace dragnn
}  // namespace syntaxnet

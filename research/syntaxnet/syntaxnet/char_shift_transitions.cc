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

#include "syntaxnet/char_shift_transitions.h"

#include "syntaxnet/parser_features.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/sentence_features.h"
#include "syntaxnet/shared_store.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "util/utf8/unicodetext.h"

namespace syntaxnet {

ParserTransitionState *CharShiftTransitionState::Clone() const {
  CharShiftTransitionState *new_state =
      new CharShiftTransitionState(left_to_right_);
  new_state->num_chars_ = num_chars_;
  new_state->char_pos_map_ = char_pos_map_;
  new_state->char_len_map_ = char_len_map_;
  new_state->token_starts_ = token_starts_;
  new_state->token_ends_ = token_ends_;
  new_state->next_ = next_;
  return new_state;
}

void CharShiftTransitionState::Init(ParserState *state) {
  const Sentence &sentence = state->sentence();
  const string &text = sentence.text();
  const int num_tokens = sentence.token_size();
  const int start_byte = sentence.token(0).start();
  const int end_byte = sentence.token(num_tokens - 1).end();
  UnicodeText ut;
  ut.PointToUTF8(text.data() + start_byte, end_byte - start_byte + 1);
  int cur_byte = start_byte;
  int cur_token = 0;
  num_chars_ = 0;
  char_pos_map_.clear();
  char_len_map_.clear();
  token_starts_.clear();
  token_ends_.clear();
  for (auto i = ut.begin(); i != ut.end(); ++i) {
    const int char_len = i.utf8_length();
    char_pos_map_.push_back(cur_byte);
    char_len_map_.push_back(char_len);
    const bool is_start = sentence.token(cur_token).start() == cur_byte;
    const bool is_end =
        sentence.token(cur_token).end() == cur_byte + char_len - 1;
    token_starts_.push_back(is_start);
    token_ends_.push_back(is_end);
    if (is_end) ++cur_token;
    cur_byte += char_len;
    ++num_chars_;
  }

  next_ = left_to_right_ ? 0 : num_chars_ - 1;
  if (!left_to_right_) state->Advance(num_tokens - 1);
}

int CharShiftTransitionState::Next() const {
  DCHECK_GE(next_, -1);
  DCHECK_LE(next_, num_chars_);
  return next_;
}

int CharShiftTransitionState::Input(int offset) const {
  const int index = next_ + offset;
  return index >= -1 && index < num_chars_ ? index : -2;
}

string CharShiftTransitionState::GetChar(const ParserState &state,
                                         int i) const {
  const string &text = state.sentence().text();
  return (i >= 0 && i < num_chars_)
             ? text.substr(char_pos_map_[i], char_len_map_[i])
             : "";
}

void CharShiftTransitionState::Advance(int next) {
  DCHECK_LE(next, num_chars_);
  next_ = next;
}

bool CharShiftTransitionState::EndOfInput() const {
  return next_ == num_chars_;
}

bool CharShiftTransitionState::IsTokenStart(int i) const {
  return i >= 0 && i < num_chars_ && token_starts_[i];
}

bool CharShiftTransitionState::IsTokenEnd(int i) const {
  return i >= 0 && i < num_chars_ && token_ends_[i];
}

void CharShiftTransitionSystem::Setup(TaskContext *context) {
  left_to_right_ = context->Get("left-to-right", true);
}

bool CharShiftTransitionSystem::IsAllowedAction(
    ParserAction action, const ParserState &state) const {
  return !IsFinalState(state);
}

void CharShiftTransitionSystem::PerformActionWithoutHistory(
    ParserAction action, ParserState *state) const {
  DCHECK(IsAllowedAction(action, *state));
  CharShiftTransitionState *char_state =
      reinterpret_cast<CharShiftTransitionState *>(
          state->mutable_transition_state());
  int next = char_state->Next();

  // Updates token-level state if needed.
  const bool shift_token = left_to_right_ ? char_state->IsTokenStart(next + 1)
                                          : char_state->IsTokenEnd(next - 1);
  if (shift_token) {
    int token_next = state->Next();
    state->Push(token_next);
    token_next = left_to_right_ ? (token_next + 1) : (token_next - 1);
    state->Advance(token_next);
  }

  // Updates char-level state.
  next = left_to_right_ ? (next + 1) : (next - 1);
  char_state->Advance(next);
}

bool CharShiftTransitionSystem::IsFinalState(const ParserState &state) const {
  const CharShiftTransitionState *char_state =
      reinterpret_cast<const CharShiftTransitionState *>(
          state.transition_state());
  const bool is_final =
      left_to_right_ ? char_state->EndOfInput() : (char_state->Next() < 0);
  return is_final;
}

string CharShiftTransitionSystem::ActionAsString(
    ParserAction action, const ParserState &state) const {
  const Sentence &sentence = state.sentence();
  const CharShiftTransitionState *char_state =
      reinterpret_cast<const CharShiftTransitionState *>(
          state.transition_state());
  const string char_action = char_state->GetChar(state, char_state->Next());
  const string token_action = sentence.token(state.Next()).word();
  return tensorflow::strings::StrCat(char_action, ":", token_action);
}

ParserTransitionState *CharShiftTransitionSystem::NewTransitionState(
    bool training_mode) const {
  return new CharShiftTransitionState(left_to_right());
}

REGISTER_TRANSITION_SYSTEM("char-shift-only", CharShiftTransitionSystem);

// Feature locator for accessing the input characters in the char-shift-only
// transition state. It takes the offset relative to the current input character
// as an argument. Negative values represent characters to the left, positive
// values to the right and 0 (the default argument value) represents the current
// input character.
class CharInputLocator : public ParserLocator<CharInputLocator> {
 public:
  // Gets the new focus.
  int GetFocus(const WorkspaceSet &workspaces, const ParserState &state) const {
    const CharShiftTransitionState *char_state =
        reinterpret_cast<const CharShiftTransitionState *>(
            state.transition_state());
    return char_state->Input(argument());
  }
};

REGISTER_PARSER_FEATURE_FUNCTION("char-input", CharInputLocator);

// Port of the 'text-char' feature.
class TextCharFeature : public ParserIndexFeatureFunction {
 public:
  ~TextCharFeature() override {
    if (term_map_ != nullptr) {
      SharedStore::Release(term_map_);
      term_map_ = nullptr;
    }
  }

  // Requests the 'char-map' input.
  void Setup(TaskContext *context) override {
    context->GetInput(input_name_, "text", "");
  }

  // Initializes the table of characters.
  void Init(TaskContext *context) override {
    min_freq_ = GetIntParameter("min-freq", 0);
    max_num_terms_ = GetIntParameter("max-num-terms", 0);
    file_name_ = context->InputFile(*context->GetInput(input_name_));
    term_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        file_name_, min_freq_, max_num_terms_);
    set_feature_type(new ResourceBasedFeatureType<TextCharFeature>(
        name(), this,
        {{SpaceValue(), "<SPACE>"}, {UnknownValue(), "<UNKNOWN>"}}));
  }

  // Returns a unique name for the workspace.
  string WorkspaceName() const {
    return SharedStoreUtils::CreateDefaultName(
        "term-frequency-map", input_name_, min_freq_, max_num_terms_);
  }

  // Returns the total # of chars in the map.
  int64 NumValues() const { return term_map_->Size(); }

  // Convert the numeric value of the feature to a human readable string.
  string GetFeatureValueName(FeatureValue value) const {
    if (value >= 0 && value < term_map_->Size()) {
      return term_map_->GetTerm(value);
    }
    LOG(ERROR) << "Invalid feature value: " << value;
    return "<INVALID>";
  }

  // Stores the values of all chars in the sentence.
  void Preprocess(WorkspaceSet *workspaces, ParserState *state) const override {
    if (workspaces->Has<VectorIntWorkspace>(workspace_)) return;
    auto *sentence = state->mutable_sentence();
    const int start_byte = sentence->token(0).start();
    const int end_byte = sentence->token(sentence->token_size() - 1).end();
    const string &text = sentence->text();
    UnicodeText ut;
    ut.PointToUTF8(text.data() + start_byte, end_byte - start_byte + 1);
    const int num_char = distance(ut.begin(), ut.end());

    // Stores feature values into the main workspace.
    VectorIntWorkspace *workspace = new VectorIntWorkspace(num_char);
    int i = 0;
    for (auto it = ut.begin(); it != ut.end(); ++it) {
      string character = it.get_utf8_string();
      int value;
      if (SegmenterUtils::IsBreakChar(character)) {
        value = SpaceValue();
      } else {
        value = term_map_->LookupIndex(character, UnknownValue());
      }
      workspace->set_element(i++, value);
    }
    workspaces->Set<VectorIntWorkspace>(workspace_, workspace);
  }

  int SpaceValue() const { return term_map_->Size(); }

  int UnknownValue() const { return SpaceValue() + 1; }

  void RequestWorkspaces(WorkspaceRegistry *registry) override {
    workspace_ = registry->Request<VectorIntWorkspace>("text-char");
  }

  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
    auto &workspace = workspaces.Get<VectorIntWorkspace>(workspace_);
    if (focus < 0 || focus >= workspace.size()) return kNone;
    return workspace.element(focus);
  }

 private:
  // Shortcut pointer to shared map. Not owned.
  const TermFrequencyMap *term_map_ = nullptr;

  // Name of the input for the term map.
  string input_name_ = "char-map";

  // Filename of the underlying resource.
  string file_name_;

  // Minimum frequency for term map.
  int min_freq_;

  // Maximum number of terms for term map.
  int max_num_terms_;

  // Workspace ID.
  int workspace_;
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("text-char", TextCharFeature);

// Base class for a feature function which translates token-level focus to
// char-level focus. This is useful for downstream components, which want to
// use character offsets to index into steps in the char-shift-only transition
// system.
class CharFocusFeature : public ParserIndexFeatureFunction {
 public:
  // Constant for out of bound focus.
  static const int kOutOfBound = -1;

  // Initializes the feature function.
  void Init(TaskContext *context) override {
    constexpr int kUnused = 100;
    set_feature_type(new NumericFeatureType(name(), kUnused));
  }

  // Requests a workspace for storing results.
  void RequestWorkspaces(WorkspaceRegistry *registry) override {
    CHECK_NE(FunctionName(), "") << "Empty workspace names not allowed.";
    workspace_ = registry->Request<VectorIntWorkspace>(FunctionName());
  }

  // Translates token-level focus to byte-level focus.
  virtual int TokenToByteFocus(const WorkspaceSet &workspaces,
                               const ParserState &state,
                               int token_focus) const = 0;

  // Populates the workspace with character-level focus for each token-level
  // focus. Note that the text needs to be parsed as unicode here, even though
  // it is parsed in the char-shift-only transition state. The reason is that
  // the downstream transition system accessing the char-shift-only system do
  // not have access to the char-shift-only state.
  void Preprocess(WorkspaceSet *workspaces, ParserState *state) const override {
    if (workspaces->Has<VectorIntWorkspace>(workspace_)) return;
    const Sentence &sentence = state->sentence();
    const string &text = sentence.text();
    const int num_tokens = sentence.token_size();
    const int start_byte = sentence.token(0).start();
    const int end_byte = sentence.token(num_tokens - 1).end();
    VLOG(2) << "Preprocessing: " << num_tokens << " tokens, " << end_byte
            << " bytes";
    UnicodeText ut;
    ut.PointToUTF8(text.data() + start_byte, end_byte - start_byte + 1);

    // Populates the workspace.
    VectorIntWorkspace *workspace = new VectorIntWorkspace(num_tokens);
    int cur_byte = start_byte;
    int cur_char = 0;
    int cur_token = 0;
    for (auto i = ut.begin(); i != ut.end(); ++i) {
      const int char_len = i.utf8_length();
      const int byte_focus = TokenToByteFocus(*workspaces, *state, cur_token);
      if (byte_focus >= cur_byte && byte_focus < (cur_byte + char_len)) {
        VLOG(2) << "Setting token: " << cur_token << " = " << cur_char;
        workspace->set_element(cur_token, cur_char);
        if (++cur_token >= num_tokens) break;
      }
      ++cur_char;
      cur_byte += char_len;
    }
    workspaces->Set<VectorIntWorkspace>(workspace_, workspace);
  }

  // Returns kOutOfBound if the token focus is outside of the sentence.
  // Returns the character-level focus for the given token-level focus
  // otherwise.
  void Evaluate(const WorkspaceSet &workspaces, const ParserState &state,
                int focus, FeatureVector *result) const override {
    FeatureValue value = kOutOfBound;
    const VectorIntWorkspace &workspace =
        workspaces.Get<VectorIntWorkspace>(workspace_);
    if (focus >= 0 && focus < workspace.size()) {
      value = workspace.element(focus);
    }
    result->add(feature_type(), value);
  }

 protected:
  int workspace_ = -1;
};

// Feature function for translating focus on a token to focus on the first
// character of the token.
class FirstCharFocusFeature : public CharFocusFeature {
 public:
  int TokenToByteFocus(const WorkspaceSet &workspaces, const ParserState &state,
                       int token_focus) const override {
    const Sentence &sentence = state.sentence();
    return sentence.token(token_focus).start();
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("first-char-focus", FirstCharFocusFeature);

// Feature function for translating focus on a token to focus on the last char
// of the token.
class LastCharFocusFeature : public CharFocusFeature {
 public:
  int TokenToByteFocus(const WorkspaceSet &workspaces, const ParserState &state,
                       int token_focus) const override {
    const Sentence &sentence = state.sentence();
    return sentence.token(token_focus).end();
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("last-char-focus", LastCharFocusFeature);

}  // namespace syntaxnet

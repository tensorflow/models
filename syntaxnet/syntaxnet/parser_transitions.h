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

// Transition system for the transition-based dependency parser.

#ifndef SYNTAXNET_PARSER_TRANSITIONS_H_
#define SYNTAXNET_PARSER_TRANSITIONS_H_

#include <string>
#include <vector>

#include "syntaxnet/utils.h"
#include "syntaxnet/registry.h"

namespace tensorflow {
namespace io {
class RecordReader;
class RecordWriter;
}
}

namespace syntaxnet {

class Sentence;
class ParserState;
class TaskContext;

// Parser actions for the transition system are encoded as integers.
typedef int ParserAction;

// Label type for the parser action.
enum class LabelType {
  NO_LABEL = 0,
  LEFT_LABEL = 1,
  RIGHT_LABEL = 2,
};

// Transition system-specific state. Transition systems can subclass this to
// preprocess the parser state and/or to keep additional information during
// parsing.
class ParserTransitionState {
 public:
  virtual ~ParserTransitionState() {}

  // Clones the transition state.
  virtual ParserTransitionState *Clone() const = 0;

  // Initializes a parser state for the transition system.
  virtual void Init(ParserState *state) = 0;

  virtual void AddParseToDocument(const ParserState &state,
                                  bool rewrite_root_labels,
                                  Sentence *sentence) const {}

  // Whether a parsed token should be considered correct for evaluation.
  virtual bool IsTokenCorrect(const ParserState &state, int index) const = 0;

  // Returns a human readable string representation of this state.
  virtual string ToString(const ParserState &state) const = 0;
};

// A transition system is used for handling the parser state transitions. During
// training the transition system is used for extracting a canonical sequence of
// transitions for an annotated sentence. During parsing the transition system
// is used for applying the predicted transitions to the parse state and
// therefore build the parse tree for the sentence. Transition systems can be
// implemented by subclassing this abstract class and registered using the
// REGISTER_TRANSITION_SYSTEM macro.
class ParserTransitionSystem
    : public RegisterableClass<ParserTransitionSystem> {
 public:
  // Construction and cleanup.
  ParserTransitionSystem() {}
  virtual ~ParserTransitionSystem() {}

  // Sets up the transition system. If inputs are needed, this is the place to
  // specify them.
  virtual void Setup(TaskContext *context) {}

  // Initializes the transition system.
  virtual void Init(TaskContext *context) {}

  // Reads the transition system from disk.
  virtual void Read(tensorflow::io::RecordReader *reader) {}

  // Writes the transition system to disk.
  virtual void Write(tensorflow::io::RecordWriter *writer) const {}

  // Returns the number of action types.
  virtual int NumActionTypes() const = 0;

  // Returns the number of actions.
  virtual int NumActions(int num_labels) const = 0;

  // Internally creates the set of outcomes (when transition systems support a
  // variable number of actions).
  virtual void CreateOutcomeSet(int num_labels) {}

  // Returns the default action for a given state.
  virtual ParserAction GetDefaultAction(const ParserState &state) const = 0;

  // Returns the next gold action for the parser during training using the
  // dependency relations found in the underlying annotated sentence.
  virtual ParserAction GetNextGoldAction(const ParserState &state) const = 0;

  // Returns all next gold actions for the parser during training using the
  // dependency relations found in the underlying annotated sentence.
  virtual void GetAllNextGoldActions(const ParserState &state,
                                     std::vector<ParserAction> *actions) const {
    ParserAction action = GetNextGoldAction(state);
    *actions = {action};
  }

  // Internally counts all next gold actions from the current parser state.
  virtual void CountAllNextGoldActions(const ParserState &state) {}

  // Returns the number of atomic actions within the specified ParserAction.
  virtual int ActionLength(ParserAction action) const { return 1; }

  // Returns true if the action is allowed in the given parser state.
  virtual bool IsAllowedAction(ParserAction action,
                               const ParserState &state) const = 0;

  // Performs the specified action on a given parser state. The action is not
  // saved in the state's history.
  virtual void PerformActionWithoutHistory(ParserAction action,
                                           ParserState *state) const = 0;

  // Performs the specified action on a given parser state. The action is saved
  // in the state's history.
  void PerformAction(ParserAction action, ParserState *state) const;

  // Returns true if a given state is deterministic.
  virtual bool IsDeterministicState(const ParserState &state) const = 0;

  // Returns true if no more actions can be applied to a given parser state.
  virtual bool IsFinalState(const ParserState &state) const = 0;

  // Returns a string representation of a parser action.
  virtual string ActionAsString(ParserAction action,
                                const ParserState &state) const = 0;

  // Returns a new transition state that can be used to put additional
  // information in a parser state. By specifying if we are in training_mode
  // (true) or not (false), we can construct a different transition state
  // depending on whether we are training a model or parsing new documents. A
  // null return value means we don't need to add anything to the parser state.
  virtual ParserTransitionState *NewTransitionState(bool training_mode) const {
    return nullptr;
  }

  // Whether to back off to the best allowable transition rather than the
  // default action when the highest scoring action is not allowed.  Some
  // transition systems do not degrade gracefully to the default action and so
  // should return true for this function.
  virtual bool BackOffToBestAllowableTransition() const { return false; }

  // Whether the system returns multiple gold transitions from a single
  // configuration.
  virtual bool ReturnsMultipleGoldTransitions() const { return false; }

  // Whether the system allows non-projective trees.
  virtual bool AllowsNonProjective() const { return false; }

  // Action meta data: get pointers to token indices based on meta-info about
  // (state, action) pairs. NOTE: the following interface is somewhat
  // experimental and may be subject to change. Use with caution and ask
  // djweiss@ for details.

  // Whether or not the system supports computing meta-data about actions.
  virtual bool SupportsActionMetaData() const { return false; }

  // Get the index of the child that would be created by this action. -1 for
  // no child created.
  virtual int ChildIndex(const ParserState &state,
                         const ParserAction &action) const {
    return -1;
  }

  // Get the index of the parent that would gain a new child by this action. -1
  // for no parent modified.
  virtual int ParentIndex(const ParserState &state,
                          const ParserAction &action) const {
    return -1;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ParserTransitionSystem);
};

#define REGISTER_TRANSITION_SYSTEM(type, component) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(ParserTransitionSystem, type, component)

}  // namespace syntaxnet

#endif  // SYNTAXNET_PARSER_TRANSITIONS_H_

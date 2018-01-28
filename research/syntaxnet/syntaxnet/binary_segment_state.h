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

#ifndef SYNTAXNET_BINARY_SEGMENT_STATE_H_
#define SYNTAXNET_BINARY_SEGMENT_STATE_H_

#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"

namespace syntaxnet {

class Sentence;

// Parser state for binary segmentation transition system. The input of the
// system is a sequence of utf8 characters that are to be segmented into tokens.
// The system contains two type of transitions/actions:
//  -START: the token at input is the first character of a new word.
//  -MERGE: the token at input is to be merged with the its previous token.
//
// A BinarySegmentState is used to store segmentation histories that can be used
// as features. In addition, it also provides the functionality to add
// segmentation results to the document. The function assumes that sentences in
// a document are processed in left-to-right order. See also the comments of
// the FinishDocument function for explaination.
//
// Note on spaces:
// Spaces, or more generally break-characters, should never be any part of a
// word, and the START/MERGE of spaces would be ignored. In addition, if a space
// starts a new word, then the actual first char of that word is the first
// non-space token following the space.
// Some examples:
//  -chars:  ' ' A B
//  -tags:    S  M M
//  -result: 'AB'
//
//  -chars:  A ' ' B
//  -tags:   S  M  M
//  -result: 'AB'
//
//  -chars:  A ' ' B
//  -tags:   S  S  M
//  -result: 'AB'
//
//  -chars:  A  B  ' '
//  -tags:   S  S  M
//  -result: 'A', 'B'
class BinarySegmentState : public ParserTransitionState {
 public:
  ParserTransitionState *Clone() const override;
  void Init(ParserState *state) override {}

  // Returns the number of start tokens that have already been identified. In
  // other words, number of start tokens between the first token of the sentence
  // and state.Input(), with state.Input() excluded.
  static int NumStarts(const ParserState &state) {
    return state.StackSize();
  }

  // Returns the index of the k-th most recent start token.
  static int LastStart(int k, const ParserState &state) {
    DCHECK_GE(k, 0);
    DCHECK_LT(k, NumStarts(state));
    return state.Stack(k);
  }

  // Adds the token at given index as a new start token.
  static void AddStart(int index, ParserState *state) {
    state->Push(index);
  }

  // Adds segmentation results to the given sentence.
  void AddParseToDocument(const ParserState &state,
                          bool rewrite_root_labels,
                          Sentence *sentence) const override;

  // Whether a parsed token should be considered correct for evaluation.
  bool IsTokenCorrect(const ParserState &state, int index) const override {
    return true;
  }

  // Returns a human readable string representation of this state.
  string ToString(const ParserState &state) const override;
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_BINARY_SEGMENT_STATE_H_

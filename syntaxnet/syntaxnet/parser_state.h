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

// Parser state for the transition-based dependency parser.

#ifndef $TARGETDIR_PARSER_STATE_H_
#define $TARGETDIR_PARSER_STATE_H_

#include <string>
#include <vector>

#include "syntaxnet/utils.h"
#include "syntaxnet/kbest_syntax.pb.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/sentence.pb.h"

namespace syntaxnet {

class TermFrequencyMap;

// A ParserState object represents the state of the parser during the parsing of
// a sentence. The state consists of a pointer to the next input token and a
// stack of partially processed tokens. The parser state can be changed by
// applying a sequence of transitions. Some transitions also add relations
// to the dependency tree of the sentence. The parser state also records the
// (partial) parse tree for the sentence by recording the head of each token and
// the label of this relation. The state is used for both training and parsing.
class ParserState {
 public:
  // String representation of the root label.
  static const char kRootLabel[];

  // Default value for the root label in case it's not in the label map.
  static const int kDefaultRootLabel = -1;

  // Initializes the parser state for a sentence, using an additional transition
  // state for preprocessing and/or additional information specific to the
  // transition system. The transition state is allowed to be null, in which
  // case no additional work is performed. The parser state takes ownership of
  // the transition state. A label map is used for transforming between integer
  // and string representations of the labels.
  ParserState(Sentence *sentence,
              ParserTransitionState *transition_state,
              const TermFrequencyMap *label_map);

  // Deletes the parser state.
  ~ParserState();

  // Clones the parser state.
  ParserState *Clone() const;

  // Returns the root label.
  int RootLabel() const;

  // Returns the index of the next input token.
  int Next() const;

  // Returns the number of tokens in the sentence.
  int NumTokens() const { return num_tokens_; }

  // Returns the token index relative to the next input token. If no such token
  // exists, returns -2.
  int Input(int offset) const;

  // Advances to the next input token.
  void Advance();

  // Returns true if all tokens have been processed.
  bool EndOfInput() const;

  // Pushes an element to the stack.
  void Push(int index);

  // Pops the top element from stack and returns it.
  int Pop();

  // Returns the element from the top of the stack.
  int Top() const;

  // Returns the element at a certain position in the stack. Stack(0) is the top
  // stack element. If no such position exists, returns -2.
  int Stack(int position) const;

  // Returns the number of elements on the stack.
  int StackSize() const;

  // Returns true if the stack is empty.
  bool StackEmpty() const;

  // Returns the head index for a given token.
  int Head(int index) const;

  // Returns the label of the relation to head for a given token.
  int Label(int index) const;

  // Returns the parent of a given token 'n' levels up in the tree.
  int Parent(int index, int n) const;

  // Returns the leftmost child of a given token 'n' levels down in the tree. If
  // no such child exists, returns -2.
  int LeftmostChild(int index, int n) const;

  // Returns the rightmost child of a given token 'n' levels down in the tree.
  // If no such child exists, returns -2.
  int RightmostChild(int index, int n) const;

  // Returns the n-th left sibling of a given token. If no such sibling exists,
  // returns -2.
  int LeftSibling(int index, int n) const;

  // Returns the n-th right sibling of a given token. If no such sibling exists,
  // returns -2.
  int RightSibling(int index, int n) const;

  // Adds an arc to the partial dependency tree of the state.
  void AddArc(int index, int head, int label);

  // Returns the gold head index for a given token, based on the underlying
  // annotated sentence.
  int GoldHead(int index) const;

  // Returns the gold label for a given token, based on the underlying annotated
  // sentence.
  int GoldLabel(int index) const;

  // Get a reference to the underlying token at index. Returns an empty default
  // Token if accessing the root.
  const Token &GetToken(int index) const {
    if (index == -1) return kRootToken;
    return sentence().token(index);
  }

  // Annotates a document with the dependency relations built during parsing for
  // one of its sentences. If rewrite_root_labels is true, then all tokens with
  // no heads will be assigned the default root label "ROOT".
  void AddParseToDocument(Sentence *document, bool rewrite_root_labels) const;

  // As above, but uses the default of rewrite_root_labels = true.
  void AddParseToDocument(Sentence *document) const {
    AddParseToDocument(document, true);
  }

  // Whether a parsed token should be considered correct for evaluation.
  bool IsTokenCorrect(int index) const;

  // Returns the string representation of a dependency label, or an empty string
  // if the label is invalid.
  string LabelAsString(int label) const;

  // Returns a string representation of the parser state.
  string ToString() const;

  // Returns the underlying sentence instance.
  const Sentence &sentence() const { return *sentence_; }
  Sentence *mutable_sentence() const { return sentence_; }

  // Returns the transition system-specific state.
  const ParserTransitionState *transition_state() const {
    return transition_state_;
  }
  ParserTransitionState *mutable_transition_state() {
    return transition_state_;
  }

  // Gets/sets the flag which says that the state was obtained though gold
  // transitions only.
  bool is_gold() const { return is_gold_; }
  void set_is_gold(bool is_gold) { is_gold_ = is_gold; }

 private:
  // Empty constructor used for the cloning operation.
  ParserState() {}

  // Default value for the root token.
  const Token kRootToken;

  // Sentence to parse. Not owned.
  Sentence *sentence_ = nullptr;

  // Number of tokens in the sentence to parse.
  int num_tokens_;

  // Which alternative token analysis is used for tag/category/head/label
  // information. -1 means use default.
  int alternative_ = -1;

  // Transition system-specific state. Owned.
  ParserTransitionState *transition_state_ = nullptr;

  // Label map used for conversions between integer and string representations
  // of the dependency labels. Not owned.
  const TermFrequencyMap *label_map_ = nullptr;

  // Root label.
  int root_label_;

  // Index of the next input token.
  int next_;

  // Parse stack of partially processed tokens.
  vector<int> stack_;

  // List of head positions for the (partial) dependency tree.
  vector<int> head_;

  // List of dependency relation labels describing the (partial) dependency
  // tree.
  vector<int> label_;

  // Score of the parser state.
  double score_ = 0.0;

  // True if this is the gold standard sequence (used for structured learning).
  bool is_gold_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(ParserState);
};

}  // namespace syntaxnet

#endif  // $TARGETDIR_PARSER_STATE_H_

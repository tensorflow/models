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

// Feature modeling language (fml) parser.
//
// BNF grammar for fml:
//
// <feature model> ::= { <feature extractor> }
//
// <feature extractor> ::= <extractor spec> |
//                         <extractor spec> '.' <feature extractor> |
//                         <extractor spec> '{' { <feature extractor> } '}'
//
// <extractor spec> ::= <extractor type>
//                      [ '(' <parameter list> ')' ]
//                      [ ':' <extractor name> ]
//
// <parameter list> = ( <parameter> | <argument> ) { ',' <parameter> }
//
// <parameter> ::= <parameter name> '=' <parameter value>
//
// <extractor type> ::= NAME
// <extractor name> ::= NAME | STRING
// <argument> ::= NUMBER
// <parameter name> ::= NAME
// <parameter value> ::= NUMBER | STRING | NAME

#ifndef $TARGETDIR_FML_PARSER_H_
#define $TARGETDIR_FML_PARSER_H_

#include <string>

#include "syntaxnet/utils.h"
#include "syntaxnet/feature_extractor.pb.h"

namespace syntaxnet {

class FMLParser {
 public:
  // Parses fml specification into feature extractor descriptor.
  void Parse(const string &source, FeatureExtractorDescriptor *result);

 private:
  // Initializes the parser with the source text.
  void Initialize(const string &source);

  // Outputs error message and exits.
  void Error(const string &error_message);

  // Moves to the next input character.
  void Next();

  // Moves to the next input item.
  void NextItem();

  // Parses a feature descriptor.
  void ParseFeature(FeatureFunctionDescriptor *result);

  // Parses a parameter specification.
  void ParseParameter(FeatureFunctionDescriptor *result);

  // Returns true if end of source input has been reached.
  bool eos() { return current_ == source_.end(); }

  // Item types.
  enum ItemTypes {
    END = 0,
    NAME = -1,
    NUMBER = -2,
    STRING = -3,
  };

  // Source text.
  string source_;

  // Current input position.
  string::iterator current_;

  // Line number for current input position.
  int line_number_;

  // Start position for current item.
  string::iterator item_start_;

  // Start position for current line.
  string::iterator line_start_;

  // Line number for current item.
  int item_line_number_;

  // Item type for current item. If this is positive it is interpreted as a
  // character. If it is negative it is interpreted as an item type.
  int item_type_;

  // Text for current item.
  string item_text_;
};

}  // namespace syntaxnet

#endif  // $TARGETDIR_FML_PARSER_H_

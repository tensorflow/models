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

#ifndef NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_FML_PARSER_H_
#define NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_FML_PARSER_H_

#include <string>

#include "utils.h"
#include "feature_extractor.pb.h"

namespace neurosis {

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

}  // namespace neurosis

#endif  // NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_FML_PARSER_H_

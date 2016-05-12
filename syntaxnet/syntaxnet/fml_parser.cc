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

#include "syntaxnet/fml_parser.h"

#include <ctype.h>
#include <string>

#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

void FMLParser::Initialize(const string &source) {
  // Initialize parser state.
  source_ = source;
  current_ = source_.begin();
  item_start_ = line_start_ = current_;
  line_number_ = item_line_number_ = 1;

  // Read first input item.
  NextItem();
}

void FMLParser::Error(const string &error_message) {
  LOG(FATAL) << "Error in feature model, line " << item_line_number_
             << ", position " << (item_start_ - line_start_ + 1)
             << ": " << error_message
             << "\n    " << string(line_start_, current_) << " <--HERE";
}

void FMLParser::Next() {
  // Move to the next input character. If we are at a line break update line
  // number and line start position.
  if (*current_ == '\n') {
    ++line_number_;
    ++current_;
    line_start_ = current_;
  } else {
    ++current_;
  }
}

void FMLParser::NextItem() {
  // Skip white space and comments.
  while (!eos()) {
    if (*current_ == '#') {
      // Skip comment.
      while (!eos() && *current_ != '\n') Next();
    } else if (isspace(*current_)) {
      // Skip whitespace.
      while (!eos() && isspace(*current_)) Next();
    } else {
      break;
    }
  }

  // Record start position for next item.
  item_start_ = current_;
  item_line_number_ = line_number_;

  // Check for end of input.
  if (eos()) {
    item_type_ = END;
    return;
  }

  // Parse number.
  if (isdigit(*current_) || *current_ == '+' || *current_ == '-') {
    string::iterator start = current_;
    Next();
    while (isdigit(*current_) || *current_ == '.') Next();
    item_text_.assign(start, current_);
    item_type_ = NUMBER;
    return;
  }

  // Parse string.
  if (*current_ == '"') {
    Next();
    string::iterator start = current_;
    while (*current_ != '"') {
      if (eos()) Error("Unterminated string");
      Next();
    }
    item_text_.assign(start, current_);
    item_type_ = STRING;
    Next();
    return;
  }

  // Parse identifier name.
  if (isalpha(*current_) || *current_ == '_' || *current_ == '/') {
    string::iterator start = current_;
    while (isalnum(*current_) || *current_ == '_' || *current_ == '-' ||
           *current_ == '/') Next();
    item_text_.assign(start, current_);
    item_type_ = NAME;
    return;
  }

  // Single character item.
  item_type_ = *current_;
  Next();
}

void FMLParser::Parse(const string &source,
                      FeatureExtractorDescriptor *result) {
  // Initialize parser.
  Initialize(source);

  while (item_type_ != END) {
    // Parse either a parameter name or a feature.
    if (item_type_ != NAME) Error("Feature type name expected");
    string name = item_text_;
    NextItem();

    if (item_type_ == '=') {
      Error("Invalid syntax: feature expected");
    } else {
      // Parse feature.
      FeatureFunctionDescriptor *descriptor = result->add_feature();
      descriptor->set_type(name);
      ParseFeature(descriptor);
    }
  }
}

void FMLParser::ParseFeature(FeatureFunctionDescriptor *result) {
  // Parse argument and parameters.
  if (item_type_ == '(') {
    NextItem();
    ParseParameter(result);
    while (item_type_ == ',') {
      NextItem();
      ParseParameter(result);
    }

    if (item_type_ != ')') Error(") expected");
    NextItem();
  }

  // Parse feature name.
  if (item_type_ == ':') {
    NextItem();
    if (item_type_ != NAME && item_type_ != STRING) {
      Error("Feature name expected");
    }
    string name = item_text_;
    NextItem();

    // Set feature name.
    result->set_name(name);
  }

  // Parse sub-features.
  if (item_type_ == '.') {
    // Parse dotted sub-feature.
    NextItem();
    if (item_type_ != NAME) Error("Feature type name expected");
    string type = item_text_;
    NextItem();

    // Parse sub-feature.
    FeatureFunctionDescriptor *subfeature = result->add_feature();
    subfeature->set_type(type);
    ParseFeature(subfeature);
  } else if (item_type_ == '{') {
    // Parse sub-feature block.
    NextItem();
    while (item_type_ != '}') {
      if (item_type_ != NAME) Error("Feature type name expected");
      string type = item_text_;
      NextItem();

      // Parse sub-feature.
      FeatureFunctionDescriptor *subfeature = result->add_feature();
      subfeature->set_type(type);
      ParseFeature(subfeature);
    }
    NextItem();
  }
}

void FMLParser::ParseParameter(FeatureFunctionDescriptor *result) {
  if (item_type_ == NUMBER) {
    int argument =
        utils::ParseUsing<int>(item_text_, tensorflow::strings::safe_strto32);
    NextItem();

    // Set default argument for feature.
    result->set_argument(argument);
  } else if (item_type_ == NAME) {
     string name = item_text_;
     NextItem();
     if (item_type_ != '=') Error("= expected");
     NextItem();
     if (item_type_ >= END) Error("Parameter value expected");
     string value = item_text_;
     NextItem();

     // Add parameter to feature.
     Parameter *parameter;
     parameter = result->add_parameter();
     parameter->set_name(name);
     parameter->set_value(value);
  } else {
    Error("Syntax error in parameter list");
  }
}

void ToFMLFunction(const FeatureFunctionDescriptor &function, string *output) {
  output->append(function.type());
  if (function.argument() != 0 || function.parameter_size() > 0) {
    output->append("(");
    bool first = true;
    if (function.argument() != 0) {
      tensorflow::strings::StrAppend(output, function.argument());
      first = false;
    }
    for (int i = 0; i < function.parameter_size(); ++i) {
      if (!first) output->append(",");
      output->append(function.parameter(i).name());
      output->append("=");
      output->append("\"");
      output->append(function.parameter(i).value());
      output->append("\"");
      first = false;
    }
    output->append(")");
  }
}

void ToFML(const FeatureFunctionDescriptor &function, string *output) {
  ToFMLFunction(function, output);
  if (function.feature_size() == 1) {
    output->append(".");
    ToFML(function.feature(0), output);
  } else if (function.feature_size() > 1) {
    output->append(" { ");
    for (int i = 0; i < function.feature_size(); ++i) {
      if (i > 0) output->append(" ");
      ToFML(function.feature(i), output);
    }
    output->append(" } ");
  }
}

void ToFML(const FeatureExtractorDescriptor &extractor, string *output) {
  for (int i = 0; i < extractor.feature_size(); ++i) {
    ToFML(extractor.feature(i), output);
    output->append("\n");
  }
}

string AsFML(const FeatureFunctionDescriptor &function) {
  string str;
  ToFML(function, &str);
  return str;
}

string AsFML(const FeatureExtractorDescriptor &extractor) {
  string str;
  ToFML(extractor, &str);
  return str;
}

void StripFML(string *fml_string) {
  auto it = fml_string->begin();
  while (it != fml_string->end()) {
    if (*it == '"') {
      it = fml_string->erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace syntaxnet

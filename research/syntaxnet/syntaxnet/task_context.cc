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

#include "syntaxnet/task_context.h"

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace syntaxnet {
namespace {

const char *const kShardPrintFormat = "%05d";

}  // namespace

TaskInput *TaskContext::GetInput(const string &name) {
  // Return existing input if it exists.
  for (int i = 0; i < spec_.input_size(); ++i) {
    if (spec_.input(i).name() == name) return spec_.mutable_input(i);
  }

  // Create new input.
  TaskInput *input = spec_.add_input();
  input->set_name(name);
  return input;
}

TaskInput *TaskContext::GetInput(const string &name, const string &file_format,
                                 const string &record_format) {
  TaskInput *input = GetInput(name);
  if (!file_format.empty()) {
    bool found = false;
    for (int i = 0; i < input->file_format_size(); ++i) {
      if (input->file_format(i) == file_format) found = true;
    }
    if (!found) input->add_file_format(file_format);
  }
  if (!record_format.empty()) {
    bool found = false;
    for (int i = 0; i < input->record_format_size(); ++i) {
      if (input->record_format(i) == record_format) found = true;
    }
    if (!found) input->add_record_format(record_format);
  }
  return input;
}

void TaskContext::SetParameter(const string &name, const string &value) {
  // If the parameter already exists update the value.
  for (int i = 0; i < spec_.parameter_size(); ++i) {
    if (spec_.parameter(i).name() == name) {
      spec_.mutable_parameter(i)->set_value(value);
      return;
    }
  }

  // Add new parameter.
  TaskSpec::Parameter *param = spec_.add_parameter();
  param->set_name(name);
  param->set_value(value);
}

string TaskContext::GetParameter(const string &name) const {
  // First try to find parameter in task specification.
  for (int i = 0; i < spec_.parameter_size(); ++i) {
    if (spec_.parameter(i).name() == name) return spec_.parameter(i).value();
  }

  // Parameter not found, return empty string.
  return "";
}

int TaskContext::GetIntParameter(const string &name) const {
  string value = GetParameter(name);
  return utils::ParseUsing<int>(value, 0, utils::ParseInt32);
}

int64 TaskContext::GetInt64Parameter(const string &name) const {
  string value = GetParameter(name);
  return utils::ParseUsing<int64>(value, 0ll, utils::ParseInt64);
}

bool TaskContext::GetBoolParameter(const string &name) const {
  string value = GetParameter(name);
  return value == "true";
}

double TaskContext::GetFloatParameter(const string &name) const {
  string value = GetParameter(name);
  return utils::ParseUsing<double>(value, .0, utils::ParseDouble);
}

string TaskContext::Get(const string &name, const char *defval) const {
  // First try to find parameter in task specification.
  for (int i = 0; i < spec_.parameter_size(); ++i) {
    if (spec_.parameter(i).name() == name) return spec_.parameter(i).value();
  }

  // Parameter not found, return default value.
  return defval;
}

string TaskContext::Get(const string &name, const string &defval) const {
  return Get(name, defval.c_str());
}

int TaskContext::Get(const string &name, int defval) const {
  string value = Get(name, "");
  return utils::ParseUsing<int>(value, defval, utils::ParseInt32);
}

int64 TaskContext::Get(const string &name, int64 defval) const {
  string value = Get(name, "");
  return utils::ParseUsing<int64>(value, defval, utils::ParseInt64);
}

double TaskContext::Get(const string &name, double defval) const {
  string value = Get(name, "");
  return utils::ParseUsing<double>(value, defval, utils::ParseDouble);
}

bool TaskContext::Get(const string &name, bool defval) const {
  string value = Get(name, "");
  return value.empty() ? defval : value == "true";
}

string TaskContext::InputFile(const TaskInput &input) {
  CHECK_EQ(input.part_size(), 1) << input.name();
  return input.part(0).file_pattern();
}

bool TaskContext::Supports(const TaskInput &input, const string &file_format,
                           const string &record_format) {
  // Check file format.
  if (input.file_format_size() > 0) {
    bool found = false;
    for (int i = 0; i < input.file_format_size(); ++i) {
      if (input.file_format(i) == file_format) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }

  // Check record format.
  if (input.record_format_size() > 0) {
    bool found = false;
    for (int i = 0; i < input.record_format_size(); ++i) {
      if (input.record_format(i) == record_format) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }

  return true;
}

}  // namespace syntaxnet

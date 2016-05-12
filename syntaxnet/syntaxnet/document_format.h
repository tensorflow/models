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

// An interface for document formats.

#ifndef $TARGETDIR_DOCUMENT_FORMAT_H__
#define $TARGETDIR_DOCUMENT_FORMAT_H__

#include <string>
#include <vector>

#include "syntaxnet/utils.h"
#include "syntaxnet/registry.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "tensorflow/core/lib/io/inputbuffer.h"

namespace syntaxnet {

// A document format component converts a key/value pair from a record to one or
// more documents. The record format is used for selecting the document format
// component. A document format component can be registered with the
// REGISTER_DOCUMENT_FORMAT macro.
class DocumentFormat : public RegisterableClass<DocumentFormat> {
 public:
  DocumentFormat() {}
  virtual ~DocumentFormat() {}

  // Reads a record from the given input buffer with format specific logic.
  // Returns false if no record could be read because we reached end of file.
  virtual bool ReadRecord(tensorflow::io::InputBuffer *buffer,
                          string *record) = 0;

  // Converts a key/value pair to one or more documents.
  virtual void ConvertFromString(const string &key, const string &value,
                                 vector<Sentence *> *documents) = 0;

  // Converts a document to a key/value pair.
  virtual void ConvertToString(const Sentence &document,
                               string *key, string *value) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DocumentFormat);
};

#define REGISTER_DOCUMENT_FORMAT(type, component) \
  REGISTER_CLASS_COMPONENT(DocumentFormat, type, component)

}  // namespace syntaxnet

#endif  // $TARGETDIR_DOCUMENT_FORMAT_H__

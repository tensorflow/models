// An interface for document formats.

#ifndef NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_DOCUMENT_FORMAT_H__
#define NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_DOCUMENT_FORMAT_H__

#include <string>
#include <vector>

#include "utils.h"
#include "registry.h"
#include "sentence.pb.h"
#include "task_context.h"

namespace neurosis {

// A document format component converts a key/value pair from a record to one or
// more documents. The record format is used for selecting the document format
// component. A document format component can be registered with the
// REGISTER_DOCUMENT_FORMAT macro.
class DocumentFormat : public RegisterableClass<DocumentFormat> {
 public:
  DocumentFormat() {}
  virtual ~DocumentFormat() {}

  // Initializes formatter from task parameters.
  virtual void Init(TaskContext *context) {}

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

}  // namespace neurosis

#endif  // NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_DOCUMENT_FORMAT_H__

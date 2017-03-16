#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_INPUT_BATCH_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_INPUT_BATCH_H_

#include <string>
#include <vector>

#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {

// An InputBatch object converts strings into a given data type. It is used to
// abstract DRAGNN internal data typing. Each internal DRAGNN data type should
// subclass InputBatch, with a public accessor to the type in question.

class InputBatch {
 public:
  virtual ~InputBatch() {}

  // Set the data to translate to the subclass' data type.
  virtual void SetData(const std::vector<string> &data) = 0;

  // Translate the underlying data back to a vector of strings, as appropriate.
  virtual const std::vector<string> GetSerializedData() const = 0;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INTERFACES_INPUT_BATCH_H_

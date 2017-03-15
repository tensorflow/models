#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INDEX_TRANSLATOR_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INDEX_TRANSLATOR_H_

#include <memory>
#include <vector>

#include "dragnn/core/interfaces/component.h"
#include "dragnn/core/interfaces/transition_state.h"

namespace syntaxnet {
namespace dragnn {

// A IndexTranslator provides an interface into the data of another component.
// It allows one component to look up a translated array index from the history
// or state of another component.
//
// When it is created, it is passed a pointer to the source component (that is,
// the component whose data it will be accessing) and a string representing the
// type of data access it will perform. There are two universal data access
// methods - "identity" and "history" - and components can declare more via
// their GetStepLookupFunction function.

class IndexTranslator {
 public:
  // Index into a TensorArray. Provides a given step, and the beam index within
  // that step, for TensorArray access to data in the given batch.
  struct Index {
    int batch_index = -1;
    int beam_index = -1;
    int step_index = -1;
  };

  // Creates a new IndexTranslator with access method as determined by the
  // passed string. The Translator will walk the path "path" in order, and will
  // translate from the last Component in the path.
  IndexTranslator(const std::vector<Component *> &path, const string &method);

  // Returns an index in (step, beam, batch) index space as computed from the
  // given feature value.
  Index Translate(int batch_index, int beam_index, int feature_value);

  // Returns the path to be walked by this translator.
  const std::vector<Component *> &path() const { return path_; }

  // Returns the method to be used by this translator.
  const string &method() const { return method_; }

 private:
  // The ordered list of components that must be walked to get from the
  // requesting component to the source component. This vector has the
  // requesting component at index 0 and the source component at the end. If
  // the requesting component is the source component, this vector has only one
  // entry.
  const std::vector<Component *> path_;

  // The function this translator will use to look up the step in the source
  // component. The function is invoked as:
  // step_lookup_(batch_index, beam_index, feature).
  std::function<int(int, int, int)> step_lookup_;

  // This translator's method.
  string method_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_INDEX_TRANSLATOR_H_

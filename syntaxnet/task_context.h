#ifndef NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_TASK_CONTEXT_H_
#define NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_TASK_CONTEXT_H_

#include <string>
#include <vector>

#include "nlp/saft/components/dependencies/opensource/task_spec.pb.h"

namespace neurosis {

// A task context holds configuration information for a task. It is basically a
// wrapper around a TaskSpec protocol buffer.
class TaskContext {
 public:
  // Returns the underlying task specification protocol buffer for the context.
  const TaskSpec &spec() const { return spec_; }
  TaskSpec *mutable_spec() { return &spec_; }

  // Returns a named input descriptor for the task. A new input  is created if
  // the task context does not already have an input with that name.
  TaskInput *GetInput(const string &name);
  TaskInput *GetInput(const string &name, const string &file_format,
                      const string &record_format);

  // Sets task parameter.
  void SetParameter(const string &name, const string &value);

  // Returns task parameter. If the parameter is not in the task configuration
  // the (default) value of the corresponding command line flag is returned.
  string GetParameter(const string &name) const;
  int GetIntParameter(const string &name) const;
  int64 GetInt64Parameter(const string &name) const;
  bool GetBoolParameter(const string &name) const;
  double GetFloatParameter(const string &name) const;

  // Returns task parameter. If the parameter is not in the task configuration
  // the default value is returned. Parameters retrieved using these methods
  // don't need to be defined with a DEFINE_*() macro.
  string Get(const string &name, const string &defval) const;
  string Get(const string &name, const char *defval) const;
  int Get(const string &name, int defval) const;
  int64 Get(const string &name, int64 defval) const;
  double Get(const string &name, double defval) const;
  bool Get(const string &name, bool defval) const;

  // Returns input file name for a single-file task input.
  static string InputFile(const TaskInput &input);

  // Returns true if task input supports the file and record format.
  static bool Supports(const TaskInput &input, const string &file_format,
                       const string &record_format);

 private:
  // Underlying task specification protocol buffer.
  TaskSpec spec_;

  // Vector of parameters required by this task.  These must be specified in the
  // task rather than relying on default values.
  vector<string> required_parameters_;
};

}  // namespace neurosis

#endif  // NLP_SAFT_COMPONENTS_DEPENDENCIES_OPENSOURCE_TASK_CONTEXT_H_

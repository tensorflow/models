#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPUTE_SESSION_POOL_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPUTE_SESSION_POOL_H_

#include <memory>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "tensorflow/core/platform/mutex.h"

namespace syntaxnet {
namespace dragnn {

// This pool creates and manages the reuse of ComputeSession objects.

class ComputeSessionPool {
 public:
  // Create a ComputeSessionPool that creates ComputeSessions for the given
  // MasterSpec and hyperparameters.
  ComputeSessionPool(const MasterSpec &master_spec,
                     const GridPoint &hyperparams);

  virtual ~ComputeSessionPool();

  // Get a ComputeSession. This function will attempt to use an already-created
  // ComputeSession, but if none are available a new one will be created.
  std::unique_ptr<ComputeSession> GetSession();

  // Returns a ComputeSession to the backing pool.
  void ReturnSession(std::unique_ptr<ComputeSession> session);

  // Returns the count of outstanding unique sessions.
  int num_outstanding_sessions() {
    tensorflow::mutex_lock lock(lock_);
    return num_unique_sessions_ - sessions_.size();
  }

 private:
  friend class ComputeSessionImplTestPoolAccessor;
  friend class ComputeSessionPoolTestPoolAccessor;

  // This is a creational injection setter. It should be used for tests
  // where we want our ComputeSessionPool to prepare and return
  // MockComputeSessions instead of actual ComputeSessionImpls.
  void SetComputeSessionBuilder(
      std::function<std::unique_ptr<ComputeSession>()> session_builder);

  // This injector will cause ComputeSessions built in this pool to use the
  // passed function to create Components. This is useful when you want a
  // ComputeSession to create MockComponents instead of real ones.
  void SetComponentBuilder(
      std::function<std::unique_ptr<Component>(const string &component_name,
                                               const string &backend_type)>
          component_builder);

  // The MasterSpec that will be used to initialize ComputeSessions from this
  // pool.
  const MasterSpec master_spec_;

  // The hyperparameters that will be used to initialize ComputeSessions from
  // this pool.
  const GridPoint hyperparams_;

  // The function that is used to create ComputeSessions.
  std::function<std::unique_ptr<ComputeSession>()> session_builder_;

  // The function passed to ComputeSessions that will be used by that session
  // to create components.
  std::function<std::unique_ptr<Component>(const string &component_name,
                                           const string &backend_type)>
      component_builder_;

  // ComputeSessions that are not currently being used. These sessions are not
  // reset until they are requested by another thread.
  std::vector<std::unique_ptr<ComputeSession>> sessions_;

  // Count of the number of unique ComputeSession objects that have been
  // created. Used to assign IDs to new Sessions.
  int num_unique_sessions_;

  // Mutex that protects accesses to all members of this object.
  tensorflow::mutex lock_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPUTE_SESSION_POOL_H_

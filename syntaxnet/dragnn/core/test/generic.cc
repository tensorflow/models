#include "dragnn/core/test/generic.h"

#include "tensorflow/core/lib/io/path.h"

namespace syntaxnet {
namespace test {

string GetTestDataPrefix() {
  const char *env = getenv("TEST_SRCDIR");
  const char *workspace = getenv("TEST_WORKSPACE");
  if (!env || env[0] == '\0' || !workspace || workspace[0] == '\0') {
    LOG(FATAL) << "Test directories not set up";
  }
  return tensorflow::io::JoinPath(

      env, workspace
      );
}

}  // namespace test
}  // namespace syntaxnet

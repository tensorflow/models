#include "neurosis/parser_transitions.h"

#include "neurosis/parser_state.h"

namespace neurosis {

// Transition system registry.
REGISTER_CLASS_REGISTRY("transition system", ParserTransitionSystem);

void ParserTransitionSystem::PerformAction(ParserAction action,
                                           ParserState *state) const {
  PerformActionWithoutHistory(action, state);
}

}  // namespace neurosis

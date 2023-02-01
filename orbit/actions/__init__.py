# Copyright 2022 The Orbit Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines an "action" abstraction for use with `orbit.Controller`.

"Actions" are simply arbitrary callables that are applied by the `Controller`
to the output of train steps (after each inner loop of `steps_per_loop` steps)
or an evaluation. This provides a hook mechanism, enabling things like reporting
metrics to Vizier, model exporting, additional logging, etc.

The basic `Action` abstraction (just a type alias) is defined in the
`controller` module. This `actions` module adds a `ConditionalAction` utility
class to make it easy to trigger actions conditionally based on reusable
predicates, as well as a small handful of predefined conditions/actions (in
particular, a `NewBestMetric` condition and an `ExportSavedModel` action).

One example of using actions to do metric-conditional export:

    new_best_metric = orbit.actions.NewBestMetric('accuracy')
    export_action = orbit.actions.ConditionalAction(
        condition=lambda x: x['accuracy'] > 0.9 and new_best_metric(x),
        action=orbit.actions.ExportSavedModel(
            model,
            orbit.actions.ExportFileManager(
                base_name=f'{FLAGS.model_dir}/saved_model',
                next_id_fn=trainer.global_step.numpy),
            signatures=model.infer))

    controller = orbit.Controller(
        strategy=strategy,
        trainer=trainer,
        evaluator=evaluator,
        eval_actions=[export_action],
        global_step=trainer.global_step,
        steps_per_loop=FLAGS.steps_per_loop,
        checkpoint_manager=checkpoint_manager,
        summary_interval=1000)

Note: In multi-client settings where each client runs its own `Controller`
instance, some care should be taken in deciding which clients should run certain
actions. Isolating actions to an individual client (say client 0) can be
achieved using `ConditionalAction` as follows:

    client_0_actions = orbit.actions.ConditionalAction(
        condition=lambda _: client_id() == 0,
        action=[
            ...
        ])

In particular, the `NewBestMetric` condition may be used in multi-client
settings if all clients are guaranteed to compute the same metric (ensuring this
is up to client code, not Orbit). However, when saving metrics it may be helpful
to avoid unnecessary writes by setting the `write_value` parameter to `False`
for most clients.
"""

from orbit.actions.conditional_action import ConditionalAction

from orbit.actions.export_saved_model import ExportFileManager
from orbit.actions.export_saved_model import ExportSavedModel

from orbit.actions.new_best_metric import JSONPersistedValue
from orbit.actions.new_best_metric import NewBestMetric

from orbit.actions.save_checkpoint_if_preempted import SaveCheckpointIfPreempted

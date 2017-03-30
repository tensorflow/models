# dragnn_ops.bulk_advance_from_prediction(handle, scores, component=None, name=None)

### `dragnn_ops.bulk_advance_from_prediction(handle, scores, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_bulk_ops.py`.

Given a handle to a ComputeSession and a tensor of scores, advances the state
until

all scores are used up or all states are final.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`scores`</b>: A `Tensor`. tensor of scores with shape {batch_size *
    beam_size * num_steps, num_actions}.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. handle to the same ComputeSession after
advancement.

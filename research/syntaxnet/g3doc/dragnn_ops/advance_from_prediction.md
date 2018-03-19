# dragnn_ops.advance_from_prediction(handle, scores, component=None, name=None)

### `dragnn_ops.advance_from_prediction(handle, scores, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession and a tensor of scores, advances the state.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`scores`</b>: A `Tensor` of type `float32`. tensor of scores with shape
    {batch_size, beam_size, num_actions}.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. handle to the same ComputeSession after
advancement.

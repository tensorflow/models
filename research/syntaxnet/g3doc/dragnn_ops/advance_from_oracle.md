# dragnn_ops.advance_from_oracle(handle, component=None, name=None)

### `dragnn_ops.advance_from_oracle(handle, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, advances based on the next oracle action.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. handle to the same ComputeSession after
advancement.

# dragnn_ops.emit_oracle_labels(handle, component=None, name=None)

### `dragnn_ops.emit_oracle_labels(handle, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, emits a vector of gold labels.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `int32`. [batch_size * beam_size] vector of gold labels for
the current ComputeSession.

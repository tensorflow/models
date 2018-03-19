# dragnn_ops.emit_all_final(handle, component=None, name=None)

### `dragnn_ops.emit_all_final(handle, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, emits a single bool indicating whether all

elements in the batch contain beams containing all final states.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `bool`. whether every batch element has all final states.

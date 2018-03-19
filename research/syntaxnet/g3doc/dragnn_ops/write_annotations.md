# dragnn_ops.write_annotations(handle, component=None, name=None)

### `dragnn_ops.write_annotations(handle, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, has the given component write its

annotations to the underlying data.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. handle to the same ComputeSession after writing.

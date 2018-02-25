# dragnn_ops.batch_size(handle, component=None, name=None)

### `dragnn_ops.batch_size(handle, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, returns the batch size of the given
component.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `int32`. size of the given component's batch.

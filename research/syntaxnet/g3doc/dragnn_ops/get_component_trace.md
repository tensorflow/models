# dragnn_ops.get_component_trace(handle, component=None, name=None)

### `dragnn_ops.get_component_trace(handle, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Gets the raw MasterTrace proto for each batch, state, and beam slot.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. vector of MasterTrace protos.

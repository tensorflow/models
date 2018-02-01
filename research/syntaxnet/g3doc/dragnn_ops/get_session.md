# dragnn_ops.get_session(container, master_spec=None, grid_point=None, name=None)

### `dragnn_ops.get_session(container, master_spec=None, grid_point=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given MasterSpec and GridPoint protos, outputs a handle to a ComputeSession.

#### Args:

*   <b>`container`</b>: A `Tensor` of type `string`. unique identifier for the
    ComputeSessionPool from which a ComputeSession will be allocated.
*   <b>`master_spec`</b>: An optional `string`. Defaults to `""`. a serialized
    syntaxnet.dragnn.MasterSpec proto.
*   <b>`grid_point`</b>: An optional `string`. Defaults to `""`. a serialized
    syntaxnet.dragnn.GridPoint proto.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. string handle to the ComputeSession.

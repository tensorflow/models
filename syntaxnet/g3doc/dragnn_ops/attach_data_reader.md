# dragnn_ops.attach_data_reader(handle, input_spec, component=None, name=None)

### `dragnn_ops.attach_data_reader(handle, input_spec, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, attaches a data source.

This op is agnostic to the type of input data. The vector of input strings is
interpreted by the backend.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`input_spec`</b>: A `Tensor` of type `string`. string vector data.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. handle to the same ComputeSession after attachment.

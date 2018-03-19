# dragnn_ops.set_tracing(handle, tracing_on, component=None, name=None)

### `dragnn_ops.set_tracing(handle, tracing_on, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, turns on or off tracing.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`tracing_on`</b>: A `Tensor` of type `bool`. Whether or not to record
    traces.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. handle to the same ComputeSession after
advancement.

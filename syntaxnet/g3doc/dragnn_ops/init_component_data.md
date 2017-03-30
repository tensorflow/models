# dragnn_ops.init_component_data(handle, beam_size, component=None, name=None)

### `dragnn_ops.init_component_data(handle, beam_size, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession, initializes the given component with the

specified beam size.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`beam_size`</b>: A `Tensor` of type `int32`. size of the beam to use on
    the component.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `string`. handle to the same ComputeSession after component
initialization.

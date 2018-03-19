# dragnn_ops.extract_link_features(handle, channel_id, component=None, name=None)

### `dragnn_ops.extract_link_features(handle, channel_id, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession and a channel index, outputs link features.

Output indices have shape {batch_size * beam_size * channel_size}.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`channel_id`</b>: An `int`. feature channel to extract features for.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A tuple of `Tensor` objects (step_idx, idx). * <b>`step_idx`</b>: A `Tensor` of
type `int32`. step indices to read activations from. * <b>`idx`</b>: A `Tensor`
of type `int32`. indices within a step to read activations from.

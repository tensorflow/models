# dragnn_ops.bulk_fixed_feature_ids(handle, num_channels, component=None, num_ids_per_channel=None, name=None)

### `dragnn_ops.bulk_fixed_feature_ids(handle, num_channels, component=None, num_ids_per_channel=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_bulk_ops.py`.

This op is a variant of BulkFixedFeatures that only outputs fixed feature IDs.

Each fixed feature channel must produce exactly |num_ids_per_channel| feature
IDs per step, and feature weights are ignored.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`num_channels`</b>: An `int` that is `>= 1`. number of feature channels
    in the component.
*   <b>`component`</b>: An optional `string`. Defaults to `""`. name of the
    component to run.
*   <b>`num_ids_per_channel`</b>: An optional `int`. Defaults to `1`. number of
    feature IDs extracted per channel at each step.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A tuple of `Tensor` objects (output_handle, ids). * <b>`output_handle`</b>: A
`Tensor` of type `string`. handle to updated ComputeSession. * <b>`ids`</b>: A
list of `num_channels` `Tensor` objects of type `int64`. a list of num_channels
[batch_size, num_steps, num_ids_per_channel] tensors of feature IDs, one per
channel. The feature IDs for each batch item are padded with trailing -1s to the
maximum number of steps across the batch.

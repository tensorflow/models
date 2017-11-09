# dragnn_ops.bulk_fixed_features(handle, num_channels, component=None, name=None)

### `dragnn_ops.bulk_fixed_features(handle, num_channels, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_bulk_ops.py`.

Given a handle to a ComputeSession and a component name, outputs fixed features

for the entire oracle path of the component. Unlike ExtractFixedFeatures, this
op mutates the master state, advancing all of its states until they are final.
For every channel, indices[channel], ids[channel], and weights[channel] have the
same length, ie. the number of predicates, ordered by batch, beam, step.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`num_channels`</b>: An `int` that is `>= 1`.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A tuple of `Tensor` objects (output_handle, indices, ids, weights, num_steps). *
<b>`output_handle`</b>: A `Tensor` of type `string`. handle to the same
ComputeSession after advancement. indices (num_channels vectors of int32): if
indices[i] = j, then embedding_sum[j] += embedding_matrix[ids[i]] * weights[i].
ids (num_channels vectors of int64): ids to lookup in embedding matrices.
weights (num_channels vectors of float): weight for each embedding. num_steps
(int32 scalar): batch was unrolled for these many steps. * <b>`indices`</b>: A
list of `num_channels` `Tensor` objects of type `int32`. * <b>`ids`</b>: A list
of `num_channels` `Tensor` objects of type `int64`. * <b>`weights`</b>: A list
of `num_channels` `Tensor` objects of type `float32`. * <b>`num_steps`</b>: A
`Tensor` of type `int32`.

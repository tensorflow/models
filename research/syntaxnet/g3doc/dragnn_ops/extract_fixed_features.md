# dragnn_ops.extract_fixed_features(handle, channel_id, component=None, name=None)

### `dragnn_ops.extract_fixed_features(handle, channel_id, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Given a handle to a ComputeSession and a channel index, outputs fixed features.

Fixed features returned as 3 vectors, 'indices', 'ids', and 'weights' of equal
length. 'ids' specifies which rows should be looked up in the embedding matrix.
'weights' specifies a scale for each embedding vector. 'indices' is a sorted
vector that assigns the same index to embedding vectors that should be summed
together.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`channel_id`</b>: An `int`. feature channel to extract features for.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A tuple of `Tensor` objects (indices, ids, weights). * <b>`indices`</b>: A
`Tensor` of type `int32`. row to add embeddings to. * <b>`ids`</b>: A `Tensor`
of type `int64`. lookup indices into embedding matrices. * <b>`weights`</b>: A
`Tensor` of type `float32`. weight for each looked up embedding.

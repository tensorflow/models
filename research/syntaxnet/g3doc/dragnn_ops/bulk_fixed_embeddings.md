# dragnn_ops.bulk_fixed_embeddings(handle, embedding_matrix, component=None, pad_to_batch=None, pad_to_steps=None, name=None)

### `dragnn_ops.bulk_fixed_embeddings(handle, embedding_matrix, component=None, pad_to_batch=None, pad_to_steps=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_bulk_ops.py`.

This op is a more efficient version of BulkFixedFeatures to be run with large

batch sizes at inference time. The op takes a handle to ComputeSession and
embedding matrices as tensor inputs, and directly outputs concatenated embedding
vectors.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to ComputeSession.
    embedding_matrix (num_channels matrices of float): embedding matrices, each
    shaped as vocab_dim[channel] x embedding_dim[channel].
*   <b>`embedding_matrix`</b>: A list of at least 1 `Tensor` objects of the same
    type. embedding matrices.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`pad_to_batch`</b>: An optional `int`. Defaults to `-1`.
*   <b>`pad_to_steps`</b>: An optional `int`. Defaults to `-1`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A tuple of `Tensor` objects (output_handle, embedding_vectors, num_steps). *
<b>`output_handle`</b>: A `Tensor` of type `string`. handle to the same
ComputeSession after advancement. embedding_vectors (matrix of float): output
concatenated embeddings, shaped as (batch * beam * token) x
sum_channel(embedding_dim[channel]). num_steps (int32 scalar): batch was
unrolled for these many steps. * <b>`embedding_vectors`</b>: A `Tensor`. Has the
same type as `embedding_matrix`. * <b>`num_steps`</b>: A `Tensor` of type
`int32`.

# dragnn_ops.dragnn_embedding_initializer(sstable, vocab, embedding_init=None, name=None)

### `dragnn_ops.dragnn_embedding_initializer(sstable, vocab, embedding_init=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_ops.py`.

Reads embeddings from an sstable of dist_belief.TokenEmbedding protos for

every key specified in a text vocabulary file.

#### Args:

*   <b>`sstable`</b>: A `string`. path to sstable location with embedding
    vectors.
*   <b>`vocab`</b>: A `string`. path to list of keys corresponding to the
    sstable to extract.
*   <b>`embedding_init`</b>: An optional `float`. Defaults to `1`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A `Tensor` of type `float32`. a tensor containing embeddings from the specified
sstable.

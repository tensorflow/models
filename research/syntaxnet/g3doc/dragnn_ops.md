# Module: dragnn_ops

### Module `dragnn_ops`

Defined in
[`tensorflow/dragnn/python/dragnn_ops.py`](https://github.com/tensorflow/models/blob/master/syntaxnet/dragnn/python/dragnn_ops.py).

Groups the DRAGNN TensorFlow ops in one module.

## Members

[`advance_from_oracle(...)`](./dragnn_ops/advance_from_oracle.md): Given a
handle to a ComputeSession, advances based on the next oracle action.

[`advance_from_prediction(...)`](./dragnn_ops/advance_from_prediction.md): Given
a handle to a ComputeSession and a tensor of scores, advances the state.

[`attach_data_reader(...)`](./dragnn_ops/attach_data_reader.md): Given a handle
to a ComputeSession, attaches a data source.

[`batch_size(...)`](./dragnn_ops/batch_size.md): Given a handle to a
ComputeSession, returns the batch size of the given component.

[`bulk_advance_from_oracle(...)`](./dragnn_ops/bulk_advance_from_oracle.md):
Given a handle to a ComputeSession, advances until all states are final. Note

[`bulk_advance_from_prediction(...)`](./dragnn_ops/bulk_advance_from_prediction.md):
Given a handle to a ComputeSession and a tensor of scores, advances the state
until

[`bulk_fixed_embeddings(...)`](./dragnn_ops/bulk_fixed_embeddings.md): This op
is a more efficient version of BulkFixedFeatures to be run with large

[`bulk_fixed_feature_ids(...)`](./dragnn_ops/bulk_fixed_feature_ids.md): This op
is a variant of BulkFixedFeatures that only outputs fixed feature IDs.

[`bulk_fixed_features(...)`](./dragnn_ops/bulk_fixed_features.md): Given a
handle to a ComputeSession and a component name, outputs fixed features

[`dragnn_embedding_initializer(...)`](./dragnn_ops/dragnn_embedding_initializer.md):
Reads embeddings from an sstable of dist_belief.TokenEmbedding protos for

[`emit_all_final(...)`](./dragnn_ops/emit_all_final.md): Given a handle to a
ComputeSession, emits a single bool indicating whether all

[`emit_annotations(...)`](./dragnn_ops/emit_annotations.md): Given a handle to a
ComputeSession, emits a vector of strings corresponding to the

[`emit_oracle_labels(...)`](./dragnn_ops/emit_oracle_labels.md): Given a handle
to a ComputeSession, emits a vector of gold labels.

[`extract_fixed_features(...)`](./dragnn_ops/extract_fixed_features.md): Given a
handle to a ComputeSession and a channel index, outputs fixed features.

[`extract_link_features(...)`](./dragnn_ops/extract_link_features.md): Given a
handle to a ComputeSession and a channel index, outputs link features.

[`get_component_trace(...)`](./dragnn_ops/get_component_trace.md): Gets the raw
MasterTrace proto for each batch, state, and beam slot.

[`get_session(...)`](./dragnn_ops/get_session.md): Given MasterSpec and
GridPoint protos, outputs a handle to a ComputeSession.

[`init_component_data(...)`](./dragnn_ops/init_component_data.md): Given a
handle to a ComputeSession, initializes the given component with the

[`release_session(...)`](./dragnn_ops/release_session.md): Given a handle to the
ComputeSession, deletes it from memory.

[`set_tracing(...)`](./dragnn_ops/set_tracing.md): Given a handle to a
ComputeSession, turns on or off tracing.

[`write_annotations(...)`](./dragnn_ops/write_annotations.md): Given a handle to
a ComputeSession, has the given component write its

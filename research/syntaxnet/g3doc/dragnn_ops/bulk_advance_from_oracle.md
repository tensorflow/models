# dragnn_ops.bulk_advance_from_oracle(handle, component=None, name=None)

### `dragnn_ops.bulk_advance_from_oracle(handle, component=None, name=None)`

Defined in `tensorflow/dragnn/core/ops/gen_dragnn_bulk_ops.py`.

Given a handle to a ComputeSession, advances until all states are final. Note

that, unlike AdvanceFromOracle, this op does mutate the master state, by
advancing all of its states until they are final.

#### Args:

*   <b>`handle`</b>: A `Tensor` of type `string`. handle to a ComputeSession.
*   <b>`component`</b>: An optional `string`. Defaults to `""`.
*   <b>`name`</b>: A name for the operation (optional).

#### Returns:

A tuple of `Tensor` objects (output_handle, gold_labels). *
<b>`output_handle`</b>: A `Tensor` of type `string`. handle to updated
ComputeSession. * <b>`gold_labels`</b>: A `Tensor` of type `int32`. [batch_size
* beam_size * max_num_steps] vector of oracle actions, where max_num_steps is
the maximum number of steps in the oracle action sequences for every state in
the batch of beams. Each sub-segment of length max_num_steps provides the oracle
action sequence for the corresponding state in the batch of beams, padded with
trailing -1s.

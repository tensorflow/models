# NCF Recommendation Pipeline for MovieLens
## Background
The data pipeline for Neural Collaborative Filtering (NCF) requires particular care for several
reasons:
* False Negative Generation
  * During each epoch new data must be generated to sample the space of negative results. This
  sampling is non-trivial to compute such that it must be done in parallel to obtain any reasonable 
  throughput; otherwise an accelerator will be woefully underutilized. However, this task requires
  all of a user's positive results, and thus cannot be easily computed. (For instance by calling 
  `tf.data.Dataset.flat_map()` on each positive instance to generate corresponding negatives.)
  
* High Throughput
  * A single GPU can process well over a million data points per second, which poses a significant
  challenge for shuffling and batching. If batching is done first 
  (`.batch(batch_size).shuffle(buffer_size)`), then the shuffle is performed on the granularity of
  batches, which is inadequate. (Batches of 8192 or 16384 are common.) By contrast, if shuffling is 
  done first (`.shuffle(buffer_size).batch(batch_size)`) then shuffling bottlenecks the pipeline. 
  As a result, in order to achieve good performance the data pipeline must receive data from
  a producer which is capable of performing vectorized shuffling. (Note: this is not a shortcoming
  in `tf.data.Dataset.shuffle()`, which cannot play the same tricks used here for loss of 
  generality.) 
  
* Small Data Points
    * Each data point consists of a user id, and item id, and a boolean label. These can be 
    represented using an int32, a uint16, and an int8 respectively for a total of seven bytes per
    data point. Technically the label could even be stored in the sign bit of the user id and
    extracted on-the-fly. This would reduce the size to 6 bytes (a 15% reduction in bus I/O!), but
    is extremely cheeky and risks angering Poseidon. The practical implication is that any schema
    attached on a per-point basis explodes the size and severely reduces throughput. So 
    `{"user_id": user_id, "item_id": item_id}, label}` is significantly larger than 
    `user_id, item_id, label`. Ultimately this all means that having single points anywhere inside
    the `tf.data.Dataset` portion of the pipeline is likely to cause slowdowns.

## Overview
####1)  Sharding
The set of positive examples are sorted by user, and then sharded across pickle files such that 
each shard contains ~32,000 examples. Additionally, the shards are chosen such that all positive 
training examples for a user are guaranteed to be in a contiguous region of the same shard. This 
means that a mapping worker can perform false negative generation independently of all other 
workers.
  
####2) Negative Generation
Negatives are generated using a vectorized sampling method:
1) Generate a vector of random integers slightly longer than the expected number necessary.
2) Keep those which are not in the positive set.
3) Repeat until enough points have been generated, and then discard any extras generated.

Generating negatives in this way is slightly wasteful since extra points are generated and 
discarded; however the benefits of vectorization far outweigh this overhead.
  
####3) Shuffling and Rebatching
Results are produced by an iterative pool (`multiprocessing.Pool.imap()`). Periodically a buffer 
vector is overfilled to 2x the desired shuffle buffer size; this is because numpy does not support 
dynamic growth of arrays and thus refreshing this buffer is performed infrequently and amortized 
over a large number of batches per refill. This buffer is used to perform a vectorized 
Fischer-Yates subsample of the first shuffle_buffer_size elements; vectorization requires that the 
sample actually be performed over `shuffle_buffer_size + batch_size - 1` elements to correctly 
account for replacement and match the behavior of `tf.data.Dataset.shuffle()`; however in practice 
shuffle_buffer_size >> batch_size.

####4) GRPC Interface
One approach to packaging steps 1-3 is to use `tf.data.Dataset.from_generator()`. This works
quite well, but requires a very complicated generator and cannot be serialized and distributed 
which is problematic for TPUs. Luckily, there is a `tf.contrib.rpc.rpc` op that allows datasets
to receive input from arbitrary sources (including from another instance or producing cluster) over
GRPC. The overall approach is:

* Begin training (main thread)
* Load, preprocess, and shard data (main thread)
* Launch GRPC server in a subprocess
* Construct Estimator and input_fn (main thread)
* GRPC server asynchronously begins processing samples
* Training proceeds (main thread)
* GRPC server is spun down

The practical benefits of this approach are:
1) It can be easily extended to non-local workers such as TPUs
2) It isolates the Python runtimes of the Estimator and the data pipeline and prevents collision
between the TensorFlow and multiprocessing threadpools.
3) It makes asynchronous computation by the data producer (for instance while Estimator is 
initializing) straightforward.

The cons are:
1) It is not straightforward to pass argument tensors into the RPC call since the message is defined
in a Python class. This can be accomplished with `tf.py_func`, although in this case it is not 
necessary.
2) It imposes additional complexity as one must manage the GRPC server process and decode the
proto messages returned.

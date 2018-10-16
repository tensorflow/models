# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

MODELS:
High-Performance Models
This document and accompanying scripts detail how to build highly scalable models that target a variety of system types and network topologies. The techniques in this document utilize some low-level TensorFlow Python primitives. In the future, many of these techniques will be incorporated into high-level APIs.

Input Pipeline
The Performance Guide explains how to identify possible input pipeline issues and best practices. We found that using tf.FIFOQueue and tf.train.queue_runner could not saturate multiple current generation GPUs when using large inputs and processing with higher samples per second, such as training ImageNet with AlexNet. This is due to the use of Python threads as its underlying implementation. The overhead of Python threads is too large.

Another approach, which we have implemented in the scripts, is to build an input pipeline using the native parallelism in TensorFlow. Our implementation is made up of 3 stages:

I/O reads: Choose and read image files from disk.
Image Processing: Decode image records into images, preprocess, and organize into mini-batches.
CPU-to-GPU Data Transfer: Transfer images from CPU to GPU.
The dominant part of each stage is executed in parallel with the other stages using data_flow_ops.StagingArea. StagingArea is a queue-like operator similar to tf.FIFOQueue. The difference is that StagingArea does not guarantee FIFO ordering, but offers simpler functionality and can be executed on both CPU and GPU in parallel with other stages. Breaking the input pipeline into 3 stages that operate independently in parallel is scalable and takes full advantage of large multi-core environments. The rest of this section details the stages followed by details about using data_flow_ops.StagingArea.

Parallelize I/O Reads
data_flow_ops.RecordInput is used to parallelize reading from disk. Given a list of input files representing TFRecords, RecordInput continuously reads records using background threads. The records are placed into its own large internal pool and when it has loaded at least half of its capacity, it produces output tensors.

This op has its own internal threads that are dominated by I/O time that consume minimal CPU, which allows it to run smoothly in parallel with the rest of the model.

Parallelize Image Processing
After images are read from RecordInput they are passed as tensors to the image processing pipeline. To make the image processing pipeline easier to explain, assume that the input pipeline is targeting 8 GPUs with a batch size of 256 (32 per GPU).

256 records are read and processed individually in parallel. This starts with 256 independent RecordInput read ops in the graph. Each read op is followed by an identical set of ops for image preprocessing that are considered independent and executed in parallel. The image preprocessing ops include operations such as image decoding, distortion, and resizing.

Once the images are through preprocessing, they are concatenated together into 8 tensors each with a batch-size of 32. Rather than using tf.concat for this purpose, which is implemented as a single op that waits for all the inputs to be ready before concatenating them together, tf.parallel_stack is used. tf.parallel_stack allocates an uninitialized tensor as an output, and each input tensor is written to its designated portion of the output tensor as soon as the input is available.

When all the input tensors are finished, the output tensor is passed along in the graph. This effectively hides all the memory latency with the long tail of producing all the input tensors.

Parallelize CPU-to-GPU Data Transfer
Continuing with the assumption that the target is 8 GPUs with a batch size of 256 (32 per GPU). Once the input images are processed and concatenated together by the CPU, we have 8 tensors each with a batch-size of 32.

TensorFlow enables tensors from one device to be used on any other device directly. TensorFlow inserts implicit copies to make the tensors available on any devices where they are used. The runtime schedules the copy between devices to run before the tensors are actually used. However, if the copy cannot finish in time, the computation that needs those tensors will stall and result in decreased performance.

In this implementation, data_flow_ops.StagingArea is used to explicitly schedule the copy in parallel. The end result is that when computation starts on the GPU, all the tensors are already available.

Software Pipelining
With all the stages capable of being driven by different processors, data_flow_ops.StagingArea is used between them so they run in parallel. StagingArea is a queue-like operator similar to tf.FIFOQueue that offers simpler functionalities that can be executed on both CPU and GPU.

Before the model starts running all the stages, the input pipeline stages are warmed up to prime the staging buffers in between with one set of data. During each run step, one set of data is read from the staging buffers at the beginning of each stage, and one set is pushed at the end.

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)

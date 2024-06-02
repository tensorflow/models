# Read Custom Datasets



## Overview

[TFRecord](https://www.tensorflow.org/tutorials/load_data/tfrecord), a simple
format for storing a sequence of binary records, is the default and recommended
data format supported by TensorFlow Model Garden (TMG) for performance reasons.
The
[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
message (or protobuf) is a flexible message type that represents a `{"string":
value}` mapping. It is designed for use with TensorFlow and is used throughout
the higher-level APIs such as [TFX](https://www.tensorflow.org/tfx/).

If your dataset is already encoded as `tf.train.Example` and in TFRecord format,
please check the various
[dataloaders](https://github.com/tensorflow/models/tree/master/official/vision/dataloaders/)
we have created to handle standard input formats for classification, detection
and segmentation. If the dataset is not in the recommended format or not in
standard structure that can be handled by the provided
[dataloaders](https://github.com/tensorflow/models/tree/master/official/vision/dataloaders),
we have outlined the steps in the following sections to <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;&nbsp;   Encode the data using the
[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
message, and then serialize, write, and read
[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example)
messages to and from `.tfrecord` files.
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;&nbsp;  Customize the dataloader to reads, decodes and parses the input data.

## Convert the dataset into tf.train.Example and TFRecord

The primary reason for converting a dataset into TFRecord format in TensorFlow
is to improve input data reading performance during training. Reading data from
disk or over a network can be a bottleneck in the training process, and using
the TFRecord format can help to streamline this process and improve overall
training speed.

The TFRecord format is a binary format that stores data in a compressed,
serialized format. This makes it more efficient for reading, as the data can be
read quickly and without the need for decompression or deserialization.

Additionally, the TFRecord format is designed to be scalable and efficient for
large datasets. It can be split into multiple files and read from multiple
threads in parallel, improving overall input pipeline performance.

### Instructions

To convert a dataset into TFRecord format in TensorFlow, you need to<br>

*   first convert the data to TensorFlow's Feature format;<br>
*   then create a feature message using tf.train.Example;<br>
*   and lastly serialize the tf.train.Example message into a TFRecord file using
    tf.io.TFRecordWriter. The tf.train.Example holds the protobuf message (the
    data).

More concretely,:<br>

&nbsp;&nbsp; 1. Convert your data to TensorFlow's Feature format using `tf.train.Feature`:
<br>
<dl><dd>

A `tf.train.Feature` is a dictionary containing data types that can be
serialized to a TFRecord format. The `tf.train.Feature` message type can accept
one of the following three types:

*   tf.train.BytesList
*   tf.train.FloatList
*   tf.train.Int64List

Based on the type of values in the dataset, the user must first convert them
into above types. Below are the simple helper functions that help in the
conversion and return a `tf.train.Feature` object. Refer to the helper builder
[class](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/tf_example_builder.py#L100)
here.

**tf.train.Int64List:** This type is used to represent a list of 64-bit integer
values. Below is the example of how to put int data into an Int64List.

```python
def add_ints_feature(self, key: str,
                      value: Union[int, Sequence[int]]) -> TfExampleBuilder:
  ....
   return self.add_feature(key,tf.train.Feature(
           int64_list=tf.train.Int64List(value=_to_array(value))))
```

<br>

**tf.train.BytesList:** This type is used to represent a list of byte strings,
which can be used to store arbitrary data as a string of bytes.

```python
def add_bytes_feature(self, key: str,
                       value: BytesValueType) -> TfExampleBuilder:
  ....
   return self.add_feature(key, tf.train.Feature(
           bytes_list=tf.train.BytesList(value=_to_bytes_array(value))))
```

<br>

**tf.train.FloatList:** This type is used to represent a list of floating-point values. Below is a conversion example.

```python
def add_floats_feature(self, key: str,
                        value: Union[float, Sequence[float]]) -> TfExampleBuilder:
  ....
   return self.add_feature(key,tf.train.Feature(
           float_list=tf.train.FloatList(value=_to_array(value))))
```

Note: The exact steps for converting your data to TensorFlow's Feature format
will depend on the structure of your data. You may need to create multiple
Feature objects for each record, depending on the number of features in your
data. </dd></dl>

<br>

 &nbsp;&nbsp; 2. Map the features using `tf.train.Example`:
 
 <br>

<dd><dl> 

Fundamentally, a `tf.train.Example` is a {"string": tf.train.Feature}
mapping. From above we have `tf.train.Feature` values, we can now map them in a
`tf.train.Example`. The format for keys to features mapping of tf.train.Example
varies based on the use case.

For example,<br>

```python
feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
}
tf.train.Example(features=tf.train.Features(feature=feature))
```

<br>

The sample usage of helper builder [class](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/tf_example_builder.py#L100) is<br>

```python
  >>> example_builder = TfExampleBuilder()
  >>> example = (
          example_builder.add_bytes_feature('feature_a', 'foobarbaz')
          .add_ints_feature('feature_b', [1, 2, 3])
          .example
```

</dd></dl>
<br>
&nbsp;&nbsp; 3. Serialize the data: <br>

<dd><dl> 

Serialize the `tf.train.Example` message into a TFRecord file, use
TensorFlow API’s `tf.io.TFRecordWriter` and `SerializeToString()`to serialize
the data. Here is some code to iterate over annotations, process them and write
into TFRecords. Refer to the
[code](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/vision/data/tfrecord_lib.py#L118)
here.

```python
def write_tf_record_dataset(output_path, tf_example_iterator,num_shards):
  writers = [
     tf.io.TFRecordWriter(
      output_path + '-%05d-of-%05d.tfrecord' % (i, num_shards))
     for i in range(num_shards)
 ]
  ....

 for idx, record in enumerate(
     tf_example_iterator):
   if idx % LOG_EVERY == 0:
   tf_example = process_features(record)
   writers[idx % num_shards].write(tf_example.SerializeToString())
```

</dd></dl>

### Example

Here is an
[example](https://github.com/tensorflow/models/blob/master/official/vision/data/create_coco_tf_record.py)
of how to create a TFRecords file in TensorFlow. In this example, we Convert raw
COCO dataset to TFRecord format. The resulting TFRecords file can then be used
to train the model.
<br>
<br>

## Decoder

With a customized dataset in TFRecord, a customized
[Decoder](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/vision/examples/starter/example_input.py#L30)
is typically needed. The decoder decodes a TF Example record and returns a dictionary of decoded tensors. Below
are some essential steps to customize a decoder.

### Instructions

To create a custom data loader for new dataset , user need to follow the below
steps:

*   **Create a subclass Class**
    <br>

<dd><dl>

Create  `class CustomizeDecoder(decoder.Decoder)`.The CustomizeDecoder class should be a subclass of the [generic decoder interface](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/decoder.py) and must implement all the abstract methods. In particular, it should have the implementation of abstract method `decode`, to decode the serialized example into tensors.<br>

The constructor defines the mapping between the field name and the value from an input tf.Example. There is no limit on the number of fields to decode based on the usecase.<br>

Below is the tf.Example decoder for classification task and Object Detection.
Here we define two fields for image bytes and labels for classification tasks
whereas ten fields for Object Detection.

```python
class Decoder(decoder.Decoder):

 def __init__(self):
   self._keys_to_features = {

       'image/encoded':
           tf.io.FixedLenFeature((), tf.string, default_value=''),

       'image/class/label':
           tf.io.FixedLenFeature((), tf.int64, default_value=-1)
   }
    ....
```

<br>
Sample Constructor for Object Detection :

```python
class Decoder(decoder.Decoder):

 def __init__(self):
   self._keys_to_features = {

       'image/encoded': tf.io.FixedLenFeature((), tf.string),
       'image/height': tf.io.FixedLenFeature((), tf.int64, -1),
       'image/width': tf.io.FixedLenFeature((), tf.int64, -1),
       'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
       'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
       'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
       'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
       'image/object/class/label': tf.io.VarLenFeature(tf.int64),
       'image/object/area': tf.io.VarLenFeature(tf.float32),
       'image/object/is_crowd': tf.io.VarLenFeature(tf.int64),
   }
      ....
```

</dd></dl>

<br>

*   **Abstract Method Implementation and Return Type**<br>

<dd><dl>

 The implementation method `decode()` decodes the serialized example
into tensors. It takes in a serialized string tensor argument that encodes the
data. And returns decoded tensors i.e a dictionary of field key name and decoded
tensor mapping. The output will be consumed by methods in Parser.

```python
class Decoder(decoder.Decoder):

 def __init__(self):
  ....

 def decode(self,
     serialized_example: tf.train.Example) -> Mapping[str,tf.Tensor]:

     return tf.io.parse_single_example(
       serialized_example, self._keys_to_features)

```

</dd></dl>

### Example

Creating a Decoder is an optional step and it varies with the use case. Below
are some use cases where we have included the Decoder and Parser based on the
requirements.

 Use case|  Decoder/Parser |
-------------------------------------------------------------------------------------------------------------------------------------------------------- | ---
[Classification](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/classification_input.py) | Both Decoder and Parser
[Object Detection](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/tf_example_decoder.py) | Only Decoder

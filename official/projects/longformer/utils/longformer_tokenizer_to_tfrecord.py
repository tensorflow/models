# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Longformer training examples to Tfrecord."""
import collections
import os

import datasets
import tensorflow as tf
import transformers

pretrained_lm = "allenai/longformer-base-4096"
task_name = "mnli"
save_path = "./"

raw_datasets = datasets.load_dataset("glue", task_name, cache_dir=None)
label_list = raw_datasets["train"].features["label"].names
num_labels = len(label_list)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_lm,
    use_fast=True,
)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

sentence1_key, sentence2_key = task_to_keys[task_name]
padding = "max_length"

# make sure this is the same with model input size.
max_seq_length = 512


def preprocess_function(examples):
  # Tokenize the texts
  args = ((examples[sentence1_key],) if sentence2_key is None else
          (examples[sentence1_key], examples[sentence2_key]))
  result = tokenizer(
      *args, padding=padding, max_length=max_seq_length, truncation=True)
  return result


raw_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on dataset",
)

train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation_matched" if task_name ==
                            "mnli" else "validation"]

print("train_dataset", train_dataset[0])
print("eval_dataset", eval_dataset[0])


def file_based_convert_examples_to_features(examples, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""
  tf.io.gfile.makedirs(os.path.dirname(output_file))
  writer = tf.io.TFRecordWriter(output_file)

  for ex_index, example in enumerate(examples):
    if ex_index % 10000 == 0:
      print(f"Writing example {ex_index} of {len(examples)}")

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(example["input_ids"])
    features["input_mask"] = create_int_feature(example["attention_mask"])
    features["segment_ids"] = create_int_feature([0] *
                                                 len(example["attention_mask"]))
    features["label_ids"] = create_int_feature([example["label"]])
    features["is_real_example"] = create_int_feature([1])
    features["example_id"] = create_int_feature([example["idx"]])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


file_based_convert_examples_to_features(
    train_dataset,
    os.path.join(save_path,
                 f"{pretrained_lm.replace('/', '_')}_train.tf_record"))
file_based_convert_examples_to_features(
    eval_dataset,
    os.path.join(save_path,
                 f"{pretrained_lm.replace('/', '_')}_eval.tf_record"))

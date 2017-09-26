/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("GloveModel")
    .Output("vocab_words: string")
    .Output("indices: int64")
    .Output("values: float")
    .Output("words_per_epoch: int64")
    .Output("current_epoch: int32")
    .Output("total_words_processed: int64")
    .Output("examples: int32")
    .Output("labels: int32")
    .Output("ccounts: float")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Attr("min_count: int = 0")
    .Doc(R"doc(
Parses a text file and creates the coocurrence matrix and batches
of examples necessary to train the GloVe model.

vocab_words: A vector of words in the corpus.
indices: A vector of indices that contain a coocurrence for
    the corpus.
values: A vector of values for each index in indices, indicating the coocurrence
    value between two words.
words_per_epoch: Number of words per epoch in the data file.
current_epoch: The current epoch.
total_words_processed: The total number of words processed so far.
examples: A vector of word ids.
labels: A vector of word ids.
ccounts: A vector of co-ocurrence values between each word pair
    in examples and labels.
filename: The corpus's text file name.
batch_size: The size of produced batch.
window_size: The number of words to predict to the left and right of the target.
min_count: The minimum number of word occurrences for it to be included in the
    vocabulary.
)doc");

} // end namespace tensorflow

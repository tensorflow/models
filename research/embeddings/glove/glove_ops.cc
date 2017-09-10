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
    .Output("ccount: float")
    .SetIsStateful()
    .Attr("filename: string")
    .Attr("batch_size: int")
    .Attr("window_size: int = 5")
    .Attr("min_count: int = 0")
    .Doc(R"doc(
Parses a text file and creates the coocurrence matrix and batches
of examples necessary to train a GloVe model.

vocab_words: A vector of words in the corpus.
indices: A vector of non zero indices that contain a coocurrence for
the corpus.
values: A vector of values for each index in indices, indicating the coocurrence
value between two words.
)doc");

} // end namespace tensorflow

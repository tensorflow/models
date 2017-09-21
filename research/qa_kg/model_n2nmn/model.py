# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

import numpy as np
import tensorflow as tf
import tensorflow_fold as td
from model_n2nmn import netgen_att
from model_n2nmn import assembler
from model_n2nmn.modules import Modules


class Model:

  def __init__(self,
               config,
               kb,
               text_seq_batch,
               seq_length_batch,
               num_vocab_txt,
               num_vocab_nmn,
               EOS_idx,
               num_choices,
               decoder_sampling,
               use_gt_layout=None,
               gt_layout_batch=None,
               scope='neural_module_network',
               reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
      # Part 1: Seq2seq RNN to generate module layout tokens

      embedding_mat = tf.get_variable(
        'embedding_mat', [num_vocab_txt, config.embed_dim_txt],
        initializer=tf.contrib.layers.xavier_initializer())

      with tf.variable_scope('layout_generation'):
        att_seq2seq = netgen_att.AttentionSeq2Seq(
            config, text_seq_batch, seq_length_batch, num_vocab_txt,
            num_vocab_nmn, EOS_idx, decoder_sampling, embedding_mat,
            use_gt_layout, gt_layout_batch)
        self.att_seq2seq = att_seq2seq
        predicted_tokens = att_seq2seq.predicted_tokens
        token_probs = att_seq2seq.token_probs
        word_vecs = att_seq2seq.word_vecs
        neg_entropy = att_seq2seq.neg_entropy
        self.atts = att_seq2seq.atts

        self.predicted_tokens = predicted_tokens
        self.token_probs = token_probs
        self.word_vecs = word_vecs
        self.neg_entropy = neg_entropy

        # log probability of each generated sequence
        self.log_seq_prob = tf.reduce_sum(tf.log(token_probs), axis=0)

      # Part 2: Neural Module Network
      with tf.variable_scope('layout_execution'):
        modules = Modules(config, kb, word_vecs, num_choices, embedding_mat)
        self.modules = modules
        # Recursion of modules
        att_shape = [len(kb)]
        # Forward declaration of module recursion
        att_expr_decl = td.ForwardDeclaration(td.PyObjectType(),
                                              td.TensorType(att_shape))
        # _key_find
        case_key_find = td.Record([('time_idx', td.Scalar(dtype='int32')),
                                   ('batch_idx', td.Scalar(dtype='int32'))])
        case_key_find = case_key_find >> td.ScopedLayer(
            modules.KeyFindModule, name_or_scope='KeyFindModule')
        # _key_filter
        case_key_filter = td.Record([('input_0', att_expr_decl()),
                                     ('time_idx', td.Scalar('int32')),
                                     ('batch_idx', td.Scalar('int32'))])
        case_key_filter = case_key_filter >> td.ScopedLayer(
            modules.KeyFilterModule, name_or_scope='KeyFilterModule')
        recursion_cases = td.OneOf(
            td.GetItem('module'),
            {'_key_find': case_key_find,
             '_key_filter': case_key_filter})
        att_expr_decl.resolve_to(recursion_cases)
        # _val_desc: output scores for choice (for valid expressions)
        predicted_scores = td.Record([('input_0', recursion_cases),
                                      ('time_idx', td.Scalar('int32')),
                                      ('batch_idx', td.Scalar('int32'))])
        predicted_scores = predicted_scores >> td.ScopedLayer(
            modules.ValDescribeModule, name_or_scope='ValDescribeModule')

        # For invalid expressions, define a dummy answer
        # so that all answers have the same form
        INVALID = assembler.INVALID_EXPR
        dummy_scores = td.Void() >> td.FromTensor(
            np.zeros(num_choices, np.float32))
        output_scores = td.OneOf(
            td.GetItem('module'),
            {'_val_desc': predicted_scores,
             INVALID: dummy_scores})

        # compile and get the output scores
        self.compiler = td.Compiler.create(output_scores)
        self.scores = self.compiler.output_tensors[0]

      # Regularization: Entropy + L2
      self.entropy_reg = tf.reduce_mean(neg_entropy)
      module_weights = [
          v for v in tf.trainable_variables()
          if (scope in v.op.name and v.op.name.endswith('weights'))
      ]
      self.l2_reg = tf.add_n([tf.nn.l2_loss(v) for v in module_weights])

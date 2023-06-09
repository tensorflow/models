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

"""Convert pixel model from numpy weights to official.projects.pixel."""
import json
import sys

import numpy as np
import tensorflow as tf

from official.projects.pixel.tasks import classification


def convert(vit_encoder, hf_model_param_dict):
  """Convert pixel model from huggingface to official.projects.pixel."""
  num_layers = 12
  num_attention_heads = 12
  hidden_size = 768
  head_size = hidden_size // num_attention_heads
  assert head_size * num_attention_heads == hidden_size
  vit_encoder.encoder.patch_to_embed.set_weights([
      hf_model_param_dict[
          "vit.embeddings.patch_embeddings.projection.weight"
      ].transpose(2, 3, 1, 0),
      hf_model_param_dict["vit.embeddings.patch_embeddings.projection.bias"],
  ])
  # pylint: disable=protected-access
  vit_encoder.encoder.encoder._pos_embed.pos_embedding.assign(
      hf_model_param_dict["vit.embeddings.position_embeddings"][:, :257]
  )
  vit_encoder.encoder.encoder._norm.set_weights([
      hf_model_param_dict["vit.layernorm.weight"],
      hf_model_param_dict["vit.layernorm.bias"],
  ])
  vit_encoder.encoder.token_cls.cls.assign(
      hf_model_param_dict["vit.embeddings.cls_token"]
  )

  for layer_num in range(num_layers):
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._attention_layer._query_dense.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.attention.query.weight"
        ].T.reshape((hidden_size, num_attention_heads, head_size)),
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.attention.query.bias"
        ].reshape((num_attention_heads, head_size)),
    ])
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._attention_layer._key_dense.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.attention.key.weight"
        ].T.reshape((hidden_size, num_attention_heads, head_size)),
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.attention.key.bias"
        ].reshape((num_attention_heads, head_size)),
    ])
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._attention_layer._value_dense.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.attention.value.weight"
        ].T.reshape((hidden_size, num_attention_heads, head_size)),
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.attention.value.bias"
        ].reshape((num_attention_heads, head_size)),
    ])
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._attention_layer._output_dense.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.output.dense.weight"
        ].T.reshape((num_attention_heads, head_size, hidden_size)),
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.attention.output.dense.bias"
        ],
    ])
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._attention_layer_norm.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.layernorm_before.weight"
        ],
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.layernorm_before.bias"
        ],
    ])
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._intermediate_dense.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.intermediate.dense.weight"
        ].T,
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.intermediate.dense.bias"
        ],
    ])
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._output_dense.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.output.dense.weight"
        ].T,
        hf_model_param_dict[f"vit.encoder.layer.{layer_num}.output.dense.bias"],
    ])
    vit_encoder.encoder.encoder._encoder_layers[
        layer_num
    ]._output_layer_norm.set_weights([
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.layernorm_after.weight"
        ],
        hf_model_param_dict[
            f"vit.encoder.layer.{layer_num}.layernorm_after.bias"
        ],
    ])


if __name__ == "__main__":

  data_path = sys.argv[1]
  output_model_name = sys.argv[2] if len(sys.argv) > 2 else "pixel_encoder.ckpt"

  model_name = json.load(open(f"{data_path}/model_name.json"))
  model_params = np.load(
      open(f"{data_path}/model_param.npy", "rb"), allow_pickle=True
  )

  config = classification.PixelConfig()
  task = classification.PixelClassificationTask(config)
  model = task.build_model()

  convert(model, {k: v for k, v in zip(model_name, model_params)})
  tf.train.Checkpoint(encoder=model.encoder).write(output_model_name)

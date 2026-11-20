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

"""Implements Road Network Graph Detection by Transformer in Aerial Images.

Model paper: https://arxiv.org/abs/2202.07824
This module does not support Keras de/serialization. Please use
tf.train.Checkpoint for object based saving and loading and tf.saved_model.save
for graph serializaiton.
"""
import math
from typing import Any, List

import tensorflow as tf

from official.modeling import tf_utils
from official.projects.pix2seq.modeling import transformer
from official.vision.ops import spatial_transform_ops

from typing import Any, Mapping, List, Union

def position_embedding_sine(attention_mask,
                            num_pos_features=256,
                            temperature=10000.,
                            normalize=True,
                            scale=2 * math.pi):
  """Sine-based positional embeddings for 2D images.

  Args:
    attention_mask: a `bool` Tensor specifying the size of the input image to
      the Transformer and which elements are padded, of size [batch_size,
      height, width]
    num_pos_features: a `int` specifying the number of positional features,
      should be equal to the hidden size of the Transformer network
    temperature: a `float` specifying the temperature of the positional
      embedding. Any type that is converted to a `float` can also be accepted.
    normalize: a `bool` determining whether the positional embeddings should be
      normalized between [0, scale] before application of the sine and cos
      functions.
    scale: a `float` if normalize is True specifying the scale embeddings before
      application of the embedding function.

  Returns:
    embeddings: a `float` tensor of the same shape as input_tensor specifying
      the positional embeddings based on sine features.
  """
  if num_pos_features % 2 != 0:
    raise ValueError(
        "Number of embedding features (num_pos_features) must be even when "
        "column and row embeddings are concatenated.")
  num_pos_features = num_pos_features // 2

  # Produce row and column embeddings based on total size of the image
  # <tf.float>[batch_size, height, width]
  attention_mask = tf.cast(attention_mask, tf.float32)
  row_embedding = tf.cumsum(attention_mask, 1)
  col_embedding = tf.cumsum(attention_mask, 2)

  if normalize:
    eps = 1e-6
    row_embedding = row_embedding / (row_embedding[:, -1:, :] + eps) * scale
    col_embedding = col_embedding / (col_embedding[:, :, -1:] + eps) * scale

  dim_t = tf.range(num_pos_features, dtype=row_embedding.dtype)
  dim_t = tf.pow(temperature, 2 * (dim_t // 2) / num_pos_features)

  # Creates positional embeddings for each row and column position
  # <tf.float>[batch_size, height, width, num_pos_features]
  pos_row = tf.expand_dims(row_embedding, -1) / dim_t
  pos_col = tf.expand_dims(col_embedding, -1) / dim_t
  pos_row = tf.stack(
      [tf.sin(pos_row[:, :, :, 0::2]),
       tf.cos(pos_row[:, :, :, 1::2])], axis=4)
  pos_col = tf.stack(
      [tf.sin(pos_col[:, :, :, 0::2]),
       tf.cos(pos_col[:, :, :, 1::2])], axis=4)

  # final_shape = pos_row.shape.as_list()[:3] + [-1]
  final_shape = tf_utils.get_shape_list(pos_row)[:3] + [-1]
  pos_row = tf.reshape(pos_row, final_shape)
  pos_col = tf.reshape(pos_col, final_shape)
  output = tf.concat([pos_row, pos_col], -1)

  embeddings = output
  return embeddings

class MHAttentionMap(tf.keras.layers.Layer):
    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super(MHAttentionMap, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.q_linear = tf.keras.layers.Dense(hidden_dim, use_bias=bias)
        self.k_linear = tf.keras.layers.Conv2D(hidden_dim, kernel_size=1, use_bias=bias)
        
        #self.q_linear.build((None, query_dim))
        #self.k_linear.build((None, query_dim))

        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def call(self, q, k, mask=None):
        
        batch_size = tf.shape(q)[0]
        #print(k.shape)
        q = self.q_linear(q)  # pe [batch, Q, self.hidden_dim] 
        k = self.k_linear(k)  # memory [batch, n, n, self.hidden_dim] 

        qh = tf.reshape(q, (batch_size, tf.shape(q)[1], self.num_heads, self.hidden_dim // self.num_heads)) 
        kh = tf.reshape(k, (batch_size, self.num_heads, self.hidden_dim // self.num_heads, tf.shape(k)[-3], tf.shape(k)[-2]))
        
        weights = tf.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh) 
        w_shape = tf.shape(weights) # (b, q, n, h, w)

        if mask is not None:
            weights = tf.where(tf.expand_dims(tf.expand_dims(mask, 1), 1), float("-inf"), weights)
            
        weights = tf.reshape( tf.nn.softmax(tf.reshape(weights, (batch_size, -1)), axis=-1), w_shape )
        weights = self.dropout(weights)
        return weights


class MultiScale(tf.keras.Model): 
  
  def __init__(self, transformer, dim, nheads, fpn_dims, output_size, multi=True, **kwargs) :
    super().__init__(**kwargs)
    self.in_planes = 64 
    self.dim = dim
    self.fpn_dims = fpn_dims
    self.output_size = output_size
    self.multi = multi 
    #Top lyaer
    self.transformer = transformer
    sqrt_k = math.sqrt(1.0 / dim)
    self.input_proj_layer1 = InputProjection(dim)
    self.input_proj_layer2 = InputProjection(dim)
    self.input_proj_layer3 = InputProjection(dim)
    self.input_proj_layer4 = InputProjection(dim)

    self.bbox_attention = MHAttentionMap(dim, dim, nheads, dropout=0.0) 

    # padding = 1 -> padding = same
    self.lay1 = tf.keras.layers.Conv2D( dim+nheads, 3, padding='same')
    self.gn1 = tf.keras.layers.GroupNormalization(8) 
    self.lay2 = tf.keras.layers.Conv2D( dim//2+nheads, 3, padding='same')
    self.gn2 = tf.keras.layers.GroupNormalization(8 )
    self.lay3 = tf.keras.layers.Conv2D( dim//4+nheads, 3, padding='same')
    self.gn3 = tf.keras.layers.GroupNormalization(8 )
    self.lay4 = tf.keras.layers.Conv2D( dim//8+nheads, 3, padding='same')
    self.gn4 = tf.keras.layers.GroupNormalization(8 )
    self.lay5 = tf.keras.layers.Conv2D( dim//16+nheads, 3, padding='same')
    self.gn5 = tf.keras.layers.GroupNormalization(8 )
    self.out_lay = tf.keras.layers.Conv2D( 1, 3, padding='same')

    self.adapter1 = tf.keras.layers.Conv2D(dim//2+nheads, 1)
    self.adapter2 = tf.keras.layers.Conv2D(dim//4+nheads, 1)
    self.adapter3 = tf.keras.layers.Conv2D(dim//8+nheads, 1)
    self.relu = tf.keras.layers.ReLU()

  def _upsample(self, x, h, w):
      return tf.image.resize(x, size=(h, w), method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

  def _upsample_add(self, x, y):
      _, H, W, _ = y.shape
      return tf.image.resize(x, size=(H, W), method=tf.image.ResizeMethod.BILINEAR, align_corners=True) + y
  
  def _generate_image_mask(self, inputs: tf.Tensor,
                           target_shape: tf.Tensor) -> tf.Tensor:
    """Generates image mask from input image."""
    mask = tf.expand_dims(
        tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0), inputs.dtype),
        axis=-1)
    mask = tf.image.resize(
        mask, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return mask

  def call(self, inputs, features, features2, query_embed_weight, training): 

      # pytorch : (B, C, H, W) -> tensorflow : (B, H, W, C)
      c1 = features['2'] + features2['2'] #(b, 32, 32, 256)
      c2 = features['3'] + features2['3'] #(b, 16, 16, 512)
      c3 = features['4'] + features2['4'] #(b, 8, 8, 1024)
      c4 = features['5'] + features2['5'] #(b, 4, 4, 2048)
      
      batch_size = tf.shape(c4)[0]
      mask_4= self._generate_image_mask(inputs, tf.shape(features2['5'])[1:3])
      t4 = tf.reshape( self.input_proj_layer1(c4), [batch_size, -1, self.dim])   # (b, 4, 4, 2048) -> (b, 4, 4, 128)
      pos_embed_4 = position_embedding_sine(mask_4[:, :, :, 0], num_pos_features = self.dim)
      pos_embed_4 = tf.reshape(pos_embed_4, [batch_size, -1, self.dim])

      hs4, memory4  = self.transformer( { 
          "inputs" : tf.reshape( t4, [batch_size, -1, self.dim]),  
          "mask" : tf.reshape(mask_4, [batch_size, -1]), 
          "targets" : tf.tile( tf.expand_dims(query_embed_weight, axis=0), (batch_size, 1, 1)), 
          "pos_embed" : pos_embed_4,
          }, training=training )  

      if (self.multi == False) : 
        return hs4 

      else : 
        mask_3 = self._generate_image_mask(inputs, tf.shape(features2['4'])[1:3])
        
        t3 = self.input_proj_layer2(c3)
        pos_embed_3 = position_embedding_sine(mask_3[:, :, :, 0], num_pos_features = self.dim) 
        pos_embed_3 = tf.reshape(pos_embed_3, [batch_size, -1, self.dim])
        hs3, memory3  = self.transformer( { 
            "inputs" : tf.reshape( t3, [batch_size, -1, self.dim]),  
            "mask" : tf.reshape(mask_3, [batch_size, -1]), 
            "targets" : hs4, 
            "pos_embed" : pos_embed_3,
            }, training=training )
 
        mask_2 = self._generate_image_mask(inputs, tf.shape(features2['3'])[1:3])
        t2 = self.input_proj_layer3(c2)
        pos_embed_2 = position_embedding_sine(mask_2[:, :, :, 0], num_pos_features = self.dim) 
        pos_embed_2 = tf.reshape(pos_embed_2, [batch_size, -1, self.dim])
        hs2, memory2  = self.transformer( {
          "inputs" : tf.reshape( t2, [batch_size, -1, self.dim] ) , 
          "mask" : tf.reshape(mask_2, [batch_size, -1]), 
          "targets" : hs3, 
          "pos_embed" : pos_embed_2, 
          }, training=training )

        mask_1 = self._generate_image_mask(inputs, tf.shape(features2['2'])[1:3])
        t1 = self.input_proj_layer4(c1)
        pos_embed_1 = position_embedding_sine(mask_1[:, :, :, 0], num_pos_features = self.dim) 
        pos_embed_1 = tf.reshape(pos_embed_1, [batch_size, -1, self.dim])

        hs1, memory1 = self.transformer( {
          "inputs" : tf.reshape( t1, [batch_size, -1, self.dim]) ,
          "mask" :tf.reshape(mask_1, [batch_size, -1]), 
          "targets" : hs2, 
          "pos_embed" : pos_embed_1, 
          }, training=training)

        
        # memory input [B, f_size x f_size, hidden_dim] f_size = (4, 8, 16, 32)
        # hs input [B, query, dim]         
        # mask_output [B, q, n_head, f_size, f_size] 
        am4 = self.bbox_attention(hs4, tf.reshape( memory4, [batch_size, 4, 4, self.dim]))
        am3 = self.bbox_attention(hs3, tf.reshape( memory3, [batch_size, 8, 8, self.dim]))
        am2 = self.bbox_attention(hs2, tf.reshape( memory2, [batch_size, 16, 16, self.dim]))
        am1 = self.bbox_attention(hs1, tf.reshape( memory1, [batch_size, 32, 32, self.dim]))

        # c shape [B, f_size, f_size, channel ] -> expand [B, q, f_size, f_size, channel]
        # am shape [B, q, n_head, f_size, f_size] -> permute [B, q, f_size, f_size, n_head]

        am4_c4 = tf.concat([tf.transpose(am4, perm=[0, 1, 3, 4, 2]),  
                        tf.tile( tf.expand_dims(c4, axis=1) , [1, am4.shape[1], 1,1,1])],
                        -1) 
        am3_c3 = tf.concat([tf.transpose(am3, perm=[0, 1, 3, 4, 2]),  
                        tf.tile( tf.expand_dims(c3, axis=1) , [1, am3.shape[1], 1,1,1])],
                        -1) 
        am2_c2 = tf.concat([tf.transpose(am2, perm=[0, 1, 3, 4, 2]),  
                tf.tile( tf.expand_dims(c2, axis=1) , [1, am2.shape[1], 1,1,1])],
                -1) 
        am1_c1 = tf.concat([tf.transpose(am1, perm=[0, 1, 3, 4, 2]),  
                tf.tile( tf.expand_dims(c1, axis=1) , [1, am1.shape[1], 1,1,1])],
                -1) 
        
        num_q = am4_c4.shape[1]
        # CGR_4
        x = self.lay1(am4_c4)
        x = self.gn1(x) 
        x = self.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = self.relu(x)

        # CGR_3
        cur_fpn = self.adapter1(am3_c3)
        x = tf.reshape(x, [ batch_size*num_q, x.shape[2], x.shape[3], x.shape[4] ])  
        resize_x = tf.image.resize(x, cur_fpn.shape[2:4], method=tf.image.ResizeMethod.BILINEAR)
        x = cur_fpn + tf.reshape( resize_x , [ batch_size, num_q, resize_x.shape[1], resize_x.shape[2], resize_x.shape[3] ] ) 
        x = self.lay3(x)
        x = self.gn3(x)
        x = self.relu(x)

        # CGR_2
        cur_fpn = self.adapter2(am2_c2)
        x = tf.reshape(x, [ batch_size*num_q, x.shape[2], x.shape[3], x.shape[4] ])  
        resize_x = tf.image.resize(x, cur_fpn.shape[2:4], method=tf.image.ResizeMethod.BILINEAR)
        x = cur_fpn + tf.reshape( resize_x , [ batch_size, num_q, resize_x.shape[1], resize_x.shape[2], resize_x.shape[3] ] ) 
        x = self.lay4(x)
        x = self.gn4(x)
        x = self.relu(x)

        # CGR_1 
        cur_fpn = self.adapter3(am1_c1)
        x = tf.reshape(x, [ batch_size*num_q, x.shape[2], x.shape[3], x.shape[4] ])  
        resize_x = tf.image.resize(x, cur_fpn.shape[2:4], method=tf.image.ResizeMethod.BILINEAR)
        x = cur_fpn + tf.reshape( resize_x , [ batch_size, num_q, resize_x.shape[1], resize_x.shape[2], resize_x.shape[3] ] ) 
        x = self.lay5(x)
        x = self.gn5(x)
        x = self.relu(x) 

        x = self.out_lay(x)

        x = tf.reshape(x, [ batch_size*num_q, x.shape[2], x.shape[3], x.shape[4] ])
        x = tf.image.resize(x,  size=(self.output_size, self.output_size), method=tf.image.ResizeMethod.BILINEAR) 
        x = tf.reshape(x, [batch_size, num_q, x.shape[1], x.shape[2]])
        #vertex instance segmentation [B, q, H, W]
        return hs1, x

  
class RNGDet(tf.keras.Model):
  """RNGDet model with Keras.

  RNGDet consists of two backbones, two FPN decoders, query embedding,
  RNGDetTransformer, class and box heads.
  """

  def __init__(self,
               backbone,
               backbone_history,
               backbone_endpoint_name,
               segment_fpn,
               keypoint_fpn,
               transformer,
               multi_scale,
               num_queries,
               hidden_size,
               num_classes, 
               **kwargs):

    super().__init__(**kwargs)
    self._num_queries = num_queries
    self._hidden_size = hidden_size
    self._num_classes = num_classes
    if hidden_size % 2 != 0:
      raise ValueError("hidden_size must be a multiple of 2.")
    self._backbone = backbone
    self._backbone_history = backbone_history
    self._backbone_endpoint_name = backbone_endpoint_name
    self._segment_fpn = segment_fpn
    self._keypoint_fpn = keypoint_fpn
    self._transformer = transformer
    self._multi_scale = multi_scale

    
    self._query_embeddings = self.add_weight(
        name = "detr/query_embeddings",
        shape=[self._num_queries, self._hidden_size],
        initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
        dtype=tf.float32)
    sqrt_k = math.sqrt(1.0 / self._hidden_size)
    self._segment_head = tf.keras.layers.Conv2D(
        1, 1,
        kernel_initializer=tf.keras.initializers.RandomUniform(-sqrt_k, sqrt_k),
        name="detr/segment_dense")
    self._keypoint_head = tf.keras.layers.Conv2D(
        1, 1,
        kernel_initializer=tf.keras.initializers.RandomUniform(-sqrt_k, sqrt_k),
        name="detr/keypoint_dense")
    self._class_embed = tf.keras.layers.Dense(
        self._num_classes,
        kernel_initializer=tf.keras.initializers.RandomUniform(-sqrt_k, sqrt_k),
        name="detr/cls_dense")
    self._bbox_embed = tf.keras.Sequential([
        tf.keras.layers.Dense(
            self._hidden_size, activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(
                -sqrt_k, sqrt_k),
            name="detr/box_dense_0"),
        tf.keras.layers.Dense(
            self._hidden_size, activation="relu",
            kernel_initializer=tf.keras.initializers.RandomUniform(
                -sqrt_k, sqrt_k),
            name="detr/box_dense_1"),
        tf.keras.layers.Dense(
            2, kernel_initializer=tf.keras.initializers.RandomUniform(
                -sqrt_k, sqrt_k),
            name="detr/box_dense_2")])
    self._tanh = tf.keras.layers.Activation("tanh")

  @property 
  def multi_scale(self) -> tf.keras.Model:
    return self._multi_scale  

  @property
  def backbone(self) -> tf.keras.Model:
    return self._backbone
  
  @property
  def backbone_history(self) -> tf.keras.Model:
    return self._backbone_history
  
  @property
  def transformer(self) -> tf.keras.layers.Layer:
    return self._transformer


  @property
  def class_embed(self) -> tf.keras.layers.Layer:
    return self._class_embed
  
  @property
  def checkpoint_items(
      self) -> Mapping[str, Union[tf.keras.Model, tf.keras.layers.Layer]]:
    #"""Returns a dictionary of items to be additionally checkpointed."""
    items = dict(
        backbone=self.backbone,
        backbone_history=self.backbone_history,
        transformer=self.transformer,
        segment_fpn=self._segment_fpn,
        keypoint_fpn=self._keypoint_fpn,
        query_embeddings=self._query_embeddings,
        segment_head=self._segment_head,
        keypoint_head=self._keypoint_head,
        class_embed=self.class_embed,
        bbox_embed=self._bbox_embed,
        multi_scale=self._multi_scale,
        )

    return items
  
  def get_config(self):
    return {
        "backbone": self._backbone,
        "backbone_history": self._backbone_history,
        "backbone_endpoint_name": self._backbone_endpoint_name,
        "segment_fpn": self._segment_fpn,
        "keypoint_fpn": self._keypoint_fpn,
        "transformer": self.transformer,
        "multi_scale": self._multi_scale,
        "num_queries": self._num_queries,
        "hidden_size": self._hidden_size,
        "num_classes": self._num_classes
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def _generate_image_mask(self, inputs: tf.Tensor,
                           target_shape: tf.Tensor) -> tf.Tensor:
    """Generates image mask from input image."""
    mask = tf.expand_dims(
        tf.cast(tf.not_equal(tf.reduce_sum(inputs, axis=-1), 0), inputs.dtype),
        axis=-1)
    mask = tf.image.resize(
        mask, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return mask

  def _upsample_add(self, x, y):
        _,_,H,W = y.shape
        return tf.image.resize(x, size=(H,W), method=tf.image.ResizeMethod.BILINEAR, align_corners=True ) + y

  def call(self, inputs: tf.Tensor, history_samples: tf.Tensor,
           gt_labels: tf.Tensor = None, training: bool = None) -> List[Any]:
    # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    
    batch_size = tf.shape(inputs)[0] # bs = features[-1].tensor.shape[0]
    features = self._backbone(inputs) #features, pos = self.detr.backbone(samples)

    pred_segment = self._segment_fpn(features) # FPN
    pred_segment = spatial_transform_ops.nearest_upsampling(pred_segment['2'], 4, use_keras_layer=False)
    pred_segment = self._segment_head(pred_segment)
    pred_keypoint = self._keypoint_fpn(features) # FPN
    pred_keypoint = spatial_transform_ops.nearest_upsampling(pred_keypoint['2'], 4, use_keras_layer=False)
    pred_keypoint = self._keypoint_head(pred_keypoint)
    
    inputs_history = tf.concat([pred_segment, pred_keypoint], -1) #cat_tensor = torch.cat([pred_segment_mask, pred_keypoint_mask], dim=1)
    inputs_history = tf.stop_gradient(inputs_history)
    segmentation_map = tf.sigmoid((inputs_history))

    if gt_labels is not None:
      history_samples = tf.concat([history_samples,gt_labels],-1)
    else:
      history_samples = tf.concat([history_samples,segmentation_map],-1)
    
    history_outs = self._backbone_history(history_samples)

    decoded, seg_masks = self._multi_scale(inputs, features, history_outs, self._query_embeddings, training)     

    output_class = self._class_embed(decoded)
    box_out = decoded
    box_out = self._bbox_embed(box_out)
    output_coord = self._tanh(box_out)
    out = {"cls_outputs": output_class, "box_outputs": output_coord, "pred_instance_masks": seg_masks } 

    return out, pred_segment, pred_keypoint 


class DETRTransformer(tf.keras.layers.Layer):
  """Encoder and Decoder of DETR."""

  def __init__(self, hidden_size, num_encoder_layers=6, num_decoder_layers=6,
               drop_path=0.1, drop_units=0.1, drop_att=0.0, output_bias=True,
               num_heads=8, **kwargs):
    super().__init__(**kwargs)
    self._hidden_size = hidden_size
    self._num_encoder_layers = num_encoder_layers
    self._num_decoder_layers = num_decoder_layers
    self._drop_path = drop_path
    self._drop_units = drop_units
    self._drop_att = drop_att
    self._output_bias = output_bias
    self._num_heads = num_heads

  def build(self, input_shape=None):
    if self._num_encoder_layers > 0:
      self._encoder = transformer.TransformerEncoder(
          num_layers=self._num_encoder_layers,
          dim=self._hidden_size,
          mlp_ratio=4,
          num_heads=self._num_heads,
          drop_path=self._drop_path,
          drop_units=self._drop_units,
          drop_att=self._drop_att,
      )
    else:
      self._encoder = None

    self._output_ln_enc = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="output_ln_enc"
    )

    self._proj = tf.keras.layers.Dense(self._hidden_size, name="proj/linear")
    self._proj_ln = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="proj/ln"
    )
    self._proj_mlp = transformer.MLP(
        num_layers=1,
        dim=self._hidden_size,
        mlp_ratio=4,
        drop_path=self._drop_path,
        drop_units=self._drop_units,
        name="proj/mlp",
    )

    self._decoder = transformer.TransformerDecoder(
        num_layers=self._num_decoder_layers,
        dim=self._hidden_size,
        mlp_ratio=4,
        num_heads=self._num_heads,
        drop_path=self._drop_path,
        drop_units=self._drop_units,
        drop_att=self._drop_att,
    )
    self._output_ln_dec = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="output_ln_dec"
    )
    super().build(input_shape)

  def get_config(self):
    return {
        "num_encoder_layers": self._num_encoder_layers,
        "num_decoder_layers": self._num_decoder_layers,
        "dropout_rate": self._dropout_rate,
    }

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def call(self, inputs, training=None):
    sources = inputs["inputs"]
    targets = inputs["targets"]
    pos_embed = inputs["pos_embed"]
    mask = inputs["mask"]
    input_shape = tf_utils.get_shape_list(sources)
    source_attention_mask = tf.tile(
        tf.expand_dims(mask, axis=1), [1, input_shape[1], 1])

    sources = sources + pos_embed
    if self._encoder is not None:
      memory = self._encoder(
          sources, mask=source_attention_mask, training=training)
    else:
      memory = sources

    memory = memory + pos_embed

    target_shape = tf_utils.get_shape_list(targets)
    cross_attention_mask = tf.tile(
        tf.expand_dims(mask, axis=1), [1, target_shape[1], 1])
    self_attention_mask = tf.ones(
            [target_shape[0], target_shape[1], target_shape[1]])
    target_shape = tf.shape(targets)
    decoded, _ = self._decoder(
        tf.zeros_like(targets)+targets,
        memory,
        None,
        self_attention_mask,
        cross_attention_mask,
        training=training)
    return decoded, memory 



@tf.keras.utils.register_keras_serializable(package='Vision')
class InputProjection(tf.keras.layers.Layer):

  def __init__(self, hidden_size, **kwargs):
    super(InputProjection, self).__init__(**kwargs)
    self._hidden_size = hidden_size

  def build(self, input_shape=None):
    self._conv = tf.keras.layers.Conv2D(
        self._hidden_size, 1)
    super(InputProjection, self).build(input_shape)

  def call(self, inputs):
    out = self._conv(inputs)
    return out

  def get_config(self):
    return {
        "hidden_size": self._hidden_size,
    }
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)
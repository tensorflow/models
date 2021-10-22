"""Contains common building blocks for Mesh R-CNN."""
import tensorflow as tf
# @tf.keras.utils.register_keras_serializable(package='mesh_rcnn')
class GraphConv(tf.keras.layers.Layer):
  def __init__(
      self,
      input_dim: int,
      output_dim: int,
      init: str = "normal",
      directed: bool = False
    ) -> None:
    """
    Args:
      input_dim: Number of input features per vertex.
      output_dim: Number of output features per vertex.
      init: Weight initialization method. Can be one of ['zero', 'normal'].
      directed: Bool indicating if edges in the graph are directed.
    """
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.directed = directed

    if init == "normal":
      self.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)
      self.bias_initializer = tf.keras.initializers.Zeros()
    elif init == "zeros":
      self.kernel_initializer = tf.keras.initializers.Zeros()
      bias_initializer = tf.keras.initializers.GlorotUniform()
    else:
      raise ValueError('invalid GraphConv initialization "%s"' % init)

  def build(self, input_shape):
    self.w0 = tf.keras.layers.Dense(
        input_shape = (self.input_dim,),
        units = self.output_dim,
        kernel_initializer = self.kernel_initializer,
        bias_initializer = self.bias_initializer
    )
    self.w1 = tf.keras.layers.Dense(
        input_shape = (self.input_dim,),
        units = self.output_dim,
        kernel_initializer = self.kernel_initializer,
        bias_initializer = self.bias_initializer
    )
    self.w0 = tf.keras.layers.Dense(
        input_shape = (self.input_dim,),
        units = self.output_dim,
        kernel_initializer = tf.keras.initializers.Constant(value=1),
        bias_initializer = tf.keras.initializers.Zeros()
    )
    self.w1 = tf.keras.layers.Dense(
        input_shape = (self.input_dim,),
        units = self.output_dim,
        kernel_initializer = tf.keras.initializers.Constant(value=-1),
        bias_initializer = tf.keras.initializers.Zeros()
    )

  def call(self, verts, edges):
    """
    Args:
      verts: Float Tensor of shape (V, input_dim) where V is the number of
        vertices and input_dim is the number of input features
        per vertex. input_dim has to match the input_dim specified
        in __init__.
      edges: Long Tensor of shape (E, 2) where E is the number of edges
        where each edge has the indices of the two vertices which
        form the edge.

    Returns:
      out: Float Tensor of shape (V, output_dim) where output_dim is the
      number of output features per vertex.
    """

    # TODO: Tensors verts and edges of same device (is_cuda in pytorch)
    if verts.shape[0] == 0:
      # empty graph
      return tf.zeros((0,self.output_dim)) * tf.reduce_sum(verts)

    verts_w0 = self.w0(verts) # (V, output_dim)
    verts_w1 = self.w1(verts) # (V, output_dim)

    neighbor_sums = gather_scatter(verts_w1, edges, self.directed) # (V, output_dim)

    out = verts_w0 + neighbor_sums

    return out


  def get_config(self):
    layer_config = {

    }

    layer_config.update(super().get_config())
    return layer_config


def gather_scatter(verts, edges, directed: bool = False):
    if not (len(verts.shape) == 2):
        raise ValueError("input can only have 2 dimensions")
    if not (len(edges.shape) == 2):
        raise ValueError("edges can only have 2 dimensions")
    if not (edges.shape[1] == 2):
        raise ValueError("edges must be of shape (num_edges, 2).")

    output = tf.zeros_like(verts)
    gather_0 = tf.gather(verts, edges[:, 0])
    gather_1 = tf.gather(verts, edges[:, 1])
    new_zero = tf.reshape(edges[:, 0], (edges.shape[0], 1))
    output = tf.tensor_scatter_nd_add(output, new_zero, gather_1)
    if not directed:
        new_one = tf.reshape(edges[:, 1], (edges.shape[0], 1))
        output = tf.tensor_scatter_nd_add(output, new_one, gather_0)
    return output

@tf.keras.utils.register_keras_serializable(package='mesh_rcnn')
class ResGraphConv(tf.keras.layers.Layer):
  def __init__(self,):
    pass

  def build(self, input_shape):
    pass

  def call(self, x):
    pass

  def get_config(self):
    layer_config = {

    }

    layer_config.update(super().get_config())
    return layer_config
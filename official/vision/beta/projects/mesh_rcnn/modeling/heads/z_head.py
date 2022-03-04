import tensorflow as tf

class ZHead(tf.keras.layers.Layer):

    def __init__(self,
        num_fc: int,
        fc_dim: int,
        cls_agnostic: bool,
        num_classes: int,
        **kwargs):
        """
        Initialize Z-head
        Args:
            num_fc: number of fully connected layers
            fc_dim: dimension of fully connected layers
            cls_agnostic:
            num_classes: number of prediction classes
        """
        super().__init__(**kwargs)

        self._num_fc = num_fc
        self._fc_dim = fc_dim
        self._cls_agnostic = cls_agnostic
        self._num_classes = num_classes

    def build(self, input_shape: tf.TensorShape) -> None:
        # INPUT SHAPE: N x H x W x C
        self.flatten = tf.keras.layers.Flatten()

        self.fcs = []
        for k in range(self._num_fc):
            fc = tf.keras.layers.Dense(self._fc_dim)
            self.fcs.append(fc)
        
        num_z_reg_classes = 1 if self._cls_agnostic else self._num_classes
        self.z_pred = tf.keras.layers.Dense(num_z_reg_classes, activation='relu')


        # MAY HAVE TO DO WEIGHT INIT
        # for layer in self.fcs:
        #     weight_init.c2._xavier_fill(layer)

        # nn.init.normal_(self.z_pred.weight, std=0.001)
        # nn.init.constant_(self.z_pred.bias, 0)    

    def call(self, x):
        x = self.flatten(x)
        for layer in self.fcs:
            x = layer(x)
        x = self.z_pred(x)
        return x

    def get_config(self):
        """Get config dict of the ZHead layer."""
        config = dict(
            num_fc = self._num_fc,
            fc_dim = self._fc_dim,
            cls_agnostic = self._cls_agnostic,
            num_classes = self._num_classes
        )
        return config

    @classmethod
    def from_config(cls, config):
        return (cls(**config))

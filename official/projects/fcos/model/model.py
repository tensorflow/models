"""
implementation for the FCOS model, the coompiler 
and the training loop
"""

import tensorflow as tf

from official.projects.fcos.model.heads import head
from official.projects.fcos.model.backbone import backbone
from official.projects.fcos.model.pyramid import FPN


class FCOS(tf.keras.Model):
    """
    implementation for the FCOS model using the 
    ResNet50 as a backbone
    """
    def __init__(self):
        super(FCOS, self).__init__()
        self.backbone = backbone()
        self.pyramid = FPN()
        self.bias = tf.keras.initializers.Constant(-tf.math.log((1 - 0.01) / 0.01))
        self.class_num = 80
        self.classification_head = head(self.class_num, self.bias)
        self.centerness_head = head(1, self.bias)
        self.box_head = head(4, "zero")
        
    def call(self, images, training=False):
        """
        the full fcos model assembled parts 
        
        Args:
            images: input images tensor
            training: boolean, whether in training mode
            
        Returns:
            dict: dictionary of all the outputs of the model heads 
        """
        # Extract batch size from input to ensure consistency
        batch_size = tf.shape(images)[0]
        
        c3, c4, c5 = self.backbone(images, training=training)
        p3_out, p4_out, p5_out, p6_out, p7_out = self.pyramid(c3, c4, c5)
        
        classifier_out = []
        box_out = []
        centerness_out = []
        
        for layer in [p3_out, p4_out, p5_out, p6_out, p7_out]:
            # Classification
            cls_out = self.classification_head(layer)
            # Reshape (B, H, W, C) -> (B, H*W, C)
            cls_out = tf.reshape(cls_out, [batch_size, -1, self.class_num])
            classifier_out.append(cls_out)

            # Centerness
            ctr_out = self.centerness_head(layer)
            ctr_out = tf.reshape(ctr_out, [batch_size, -1, 1])
            centerness_out.append(ctr_out)

            # Box Regression
            box_out_layer = self.box_head(layer)
            box_out_layer = tf.reshape(box_out_layer, [batch_size, -1, 4])
            box_out_layer = tf.exp(box_out_layer) # FCOS uses exp to force positive distances
            box_out.append(box_out_layer)
        
        classifier_out = tf.concat(classifier_out, axis=1, name="classifier")
        centerness_out = tf.concat(centerness_out, axis=1, name="centerness")
        box_out = tf.concat(box_out, axis=1, name="box")
        
        return {"classifier": classifier_out, "centerness": centerness_out, "box": box_out}



# loss function
# loss function
# focal = tf.keras.losses.CategoricalFocalCrossentropy(
#     alpha = 0.25,
#     gamma = 2.0,
# )
# iouloss = IOULoss()
# bce = tf.keras.losses.BinaryCrossentropy()


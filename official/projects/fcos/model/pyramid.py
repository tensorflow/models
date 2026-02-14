"""
implementation of the feature pyramid network 
similar to the one used in the RetinaNet model
"""


import tensorflow as tf


class FPN(tf.keras.layers.Layer):
    
    def __init__(self):
        super(FPN, self).__init__()
        self.conv3_1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv4_1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv5_1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv3_3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv4_3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv5_3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv6_3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        self.conv7_3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample = tf.keras.layers.UpSampling2D(2)

    def call(self, c3, c4, c5):
        """
        the feature pyramid network implementation
            
        Args:
            c3 : the third resnet block output
            c4 : the fourth resnet block output
            c5 : the fifth resnet block outout 

        Returns:
            [p3_out, p4_out, p5_out, p6_out, p7_out] (list): list of the outputs of 
            the third forth fifth sixth seventh layers of the feature pyramid network
        """
        # there are changes to be done here, adding GN
        p3_out = self.conv3_1(c3)
        p4_out = self.conv4_1(c4)
        p5_out = self.conv5_1(c5)
        p4_out = p4_out + self.upsample(p5_out)
        p3_out = p3_out + self.upsample(p4_out)
        p3_out = self.conv3_3(p3_out)
        p4_out = self.conv4_3(p4_out)
        p5_out = self.conv5_3(p5_out)
        p6_out = self.conv6_3(p5_out)
        p7_out = self.conv7_3(tf.nn.relu(p6_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]

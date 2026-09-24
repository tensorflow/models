"""
the back bone used for the feature pyramid network implementation
which is the RsNet50 model pre-trained on the ImageNet dataset
"""


import tensorflow as tf

# model backbone
# FCOS uses ResNeXt not his one for better performance but this one to begin with.


def backbone():
    """
    Implementation of the ResNet backbone model pre-trained 
    on the ImageNet dataset.
    
    Returns:
         model: A tf.keras.Model that takes an input image and outputs [c3, c4, c5]
    """
    
    base_model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(None, None, 3),
            classes=80,
        )

    c3 = base_model.get_layer("conv3_block4_out").output
    c4 = base_model.get_layer("conv4_block6_out").output
    c5 = base_model.get_layer("conv5_block3_out").output
    
    return tf.keras.Model(inputs=base_model.input, outputs=[c3, c4, c5], name="res50_backbone")

# Volumetric Models

**DISCLAIMER**: This implementation is still under development. No support will
be provided during the development phase.

This folder contains implementation of volumetric models, i.e., UNet 3D model,
for 3D semantic segmentation.

## Modeling

Following the style of TF-Vision, a UNet 3D model is implemented as a backbone
and a decoder.

## Backbone

The backbone is the left U-shape of the complete UNet model. It takes batch of
images as input, and outputs a dictionary in a form of `{level: features}`.
`features` in the output is a tensor of feature maps.

## Decoder

The decoder is the right U-shape of the complete UNet model. It takes the output
dictionary from the backbone and connects the feature maps from each level to
the decoder's decoding branches. The final output is the raw segmentation
predictions.

An additional head is attached to the output of the decoder to optionally
perform more operations and then generate the prediction map of logits.

The `factory.py` file builds and connects the backbone, decoder and head
together to form the complete UNet model.

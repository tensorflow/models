from official.vision.beta.dataloaders import decoder


class ImageNetDecoder(decoder.Decoder):
  """Tensorflow Example proto decoder."""
  def decode(self, sample):
    """Decode the serialized example

    Args:
      sample: a single serialized tf.Example string.

    Returns:
       decoded_tensors: a dictionary of tensors with the following fields:
        - image/encoded: a string sclaer tensor
        - image/class/label: an integer tensor of shape [None]
    """
    decoded_tensors = {
        "image/encoded": sample["image"],
        "image/class/label": sample["label"],
    }
    return decoded_tensors

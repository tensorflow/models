import tensorflow_datasets as tfds
import tensorflow as tf
from official.vision.beta.dataloaders import decoder


class ImageNetDecoder(decoder.Decoder):
    """Tensorflow Example proto decoder."""
    def decode(self, sample):
        """Decode the serialized example"""
        decoded_tensors = {
            'image/encoded': sample['image'],
            'image/class/label': sample['label'],
        }
        return decoded_tensors

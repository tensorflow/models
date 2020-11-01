import tensorflow as tf
import tensorflow.keras as ks

@tf.keras.utils.register_keras_serializable(package='Text')
def mish(x):
    """Mish: A Self Regularized Non-Monotonic Activation Function
    
    This activation is far smoother than ReLU.
    Original paper: https://arxiv.org/abs/1908.08681

    Args:
        x: float Tensor to perform activation.
    
    Returns:
        `x` with the MISH activation applied.
    """
    return x * tf.math.tanh(ks.activations.softplus(x))
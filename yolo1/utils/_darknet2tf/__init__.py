import collections
import collections.abc
import io

from ..file_manager import PathABC

from typing import Union, Type, TypeVar

T = TypeVar('T', bound='DarkNetModel')


class _DarkNetSectionList(collections.abc.MutableSequence):
    __slots__ = ['data']

    def __init__(self, initlist=None):
        self.data = []
        if initlist is not None:
            self.data = list(initlist)

    @property
    def net(self):
        return self.data[0]

    # Overriding Python list operations
    def __len__(self):
        return max(0, len(self.data) - 1)

    def __getitem__(self, i):
        if i >= 0:
            i += 1
        if isinstance(i, slice):
            return self.__class__(self.data[i])
        else:
            return self.data[i]

    def __setitem__(self, i, item):
        if i >= 0:
            i += 1
        self.data[i] = item

    def __delitem__(self, i):
        if i >= 0:
            i += 1
        del self.data[i]

    def insert(self, i, item):
        if i >= 0:
            i += 1
        self.data.insert(i, item)


class DarkNetConverter(_DarkNetSectionList):
    """
    This is a special list-like object to handle the storage of layers in a
    model that is defined in the DarkNet format. Note that indexing layers in a
    DarkNet model can be unintuitive and doesn't follow the same conventions
    as a Python list.

    In DarkNet, a [net] section is at the top of every model definition. This
    section defines the input and training parameters for the entire model.
    As such, it is not a layer and cannot be referenced directly. For our
    convenience, we allowed relative references to [net] but disallowed absolute
    ones. Like the DarkNet implementation, our implementation numbers the first
    layer (after [net]) with a 0 and

    To use conventional list operations on the DarkNetConverter object, use the
    data property provided by this class.
    """
    @classmethod
    def read(
        clz: Type[T],
        config_file: Union[PathABC, io.TextIOBase],
        weights_file: Union[PathABC, io.RawIOBase,
                            io.BufferedIOBase] = None) -> T:
        """
        Parse the config and weights files and read the DarkNet layer's encoder,
        decoder, and output layers. The number of bytes in the file is also returned.

        Args:
            config_file: str, path to yolo config file from Darknet
            weights_file: str, path to yolo weights file from Darknet

        Returns:
            a DarkNetConverter object
        """
        from .read_weights import read_weights

        full_net = clz()
        read_weights(full_net, config_file, weights_file)
        return full_net

    def to_tf(self,
              thresh=0.45,
              class_thresh=0.45,
              max_boxes=200,
              use_mixed=True):
        import tensorflow as tf

        tensors = _DarkNetSectionList()
        layers = _DarkNetSectionList()
        yolo_tensors = []
        for i, cfg in enumerate(self.data):
            tensor = cfg.to_tf(tensors)

            # Handle weighted layers
            if type(tensor) is tuple:
                tensor, layer = tensor
            else:
                layer = None

            assert tensor.shape[1:] == cfg.shape, str(
                cfg
            ) + f" shape inconsistent\n\tExpected: {cfg.shape}\n\tGot: {tensor.shape[1:]}"
            if cfg._type == 'yolo':
                yolo_tensors.append((i, cfg, tensor))
            tensors.append(tensor)
            layers.append(layer)

        model = tf.keras.Model(inputs=tensors.net,
                               outputs=self._process_yolo_layer(
                                   yolo_tensors,
                                   thresh=thresh,
                                   class_thresh=class_thresh,
                                   max_boxes=max_boxes,
                                   use_mixed=use_mixed))
        model.build(self.net.shape)

        for cfg, layer in zip(self, layers):
            if layer is not None:
                layer.set_weights(cfg.get_weights())
        return model

    def _process_yolo_layer(self,
                            yolo_tensors,
                            thresh=0.45,
                            class_thresh=0.45,
                            max_boxes=200,
                            use_mixed=True):
        import tensorflow as tf
        from yolo.modeling.building_blocks import YoloLayer

        if use_mixed:
            from tensorflow.keras.mixed_precision import experimental as mixed_precision
            # using mixed type policy give better performance than strictly float32
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_policy(policy)
            dtype = policy.compute_dtype
        else:
            dtype = tf.float32

        outs = collections.OrderedDict()
        masks = {}
        anchors = None
        scale_x_y = 1
        path_scales = {}

        for i, yolo_cfg, yolo_tensor in yolo_tensors:
            masks[yolo_tensor.name] = yolo_cfg.mask

            if anchors is None:
                anchors = yolo_cfg.anchors
            elif anchors != yolo_cfg.anchors:
                raise ValueError('Anchors inconsistent in [yolo] layers')

            if scale_x_y is None:
                scale_x_y = yolo_cfg.scale_x_y
            elif scale_x_y != yolo_cfg.scale_x_y:
                raise ValueError('Scale inconsistent in [yolo] layers')

            outs[yolo_tensor.name] = yolo_tensor

            path_scales[yolo_tensor.name] = self.data[i - 1].c >> 5

        yolo_layer = YoloLayer(
            masks=masks,
            anchors=anchors,
            thresh=thresh,
            cls_thresh=class_thresh,
            max_boxes=max_boxes,
            dtype=dtype,
            #scale_boxes=self.net.w,
            scale_xy=scale_x_y,
            path_scale=path_scales)
        return yolo_layer(outs)

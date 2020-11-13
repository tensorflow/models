"""
This file contains the code to parse DarkNet weight files.
"""

import io
import numpy as np
import os

from typing import Union

from .config_classes import *
from .dn2dicts import convertConfigFile
from ..file_manager import PathABC, get_size, open_if_not_open


def build_layer(layer_dict, file, net):
    """consturct layer and load weights from file"""

    layer = layer_builder[layer_dict['_type']].from_dict(net, layer_dict)

    bytes_read = 0
    if file is not None:
        bytes_read = layer.load_weights(file)

    return layer, bytes_read


def read_file(full_net, config, weights=None):
    """read the file and construct weights net list"""
    bytes_read = 0

    if weights is not None:
        major, minor, revision = read_n_int(3, weights)
        bytes_read += 12

        if ((major * 10 + minor) >= 2):
            print("64 seen")
            iseen = read_n_long(1, weights, unsigned=True)[0]
            bytes_read += 8
        else:
            print("32 seen")
            iseen = read_n_int(1, weights, unsigned=True)[0]
            bytes_read += 4

        print(f"major: {major}")
        print(f"minor: {minor}")
        print(f"revision: {revision}")
        print(f"iseen: {iseen}")

    for i, layer_dict in enumerate(config):
        try:
            layer, num_read = build_layer(layer_dict, weights, full_net)
        except Exception as e:
            raise ValueError(f"Cannot read weights for layer [#{i}]") from e
        full_net.append(layer)
        bytes_read += num_read
        print(f"{bytes_read} {layer}")
    return bytes_read


def read_weights(full_net, config_file, weights_file):
    if weights_file is None:
        with open_if_not_open(config_file) as config:
            config = convertConfigFile(config)
            read_file(full_net, config)
        return full_net

    size = get_size(weights_file)
    with open_if_not_open(config_file) as config, \
        open_if_not_open(weights_file, "rb") as weights:
        config = convertConfigFile(config)
        bytes_read = read_file(full_net, config, weights)
        print('full net: ')
        for e in full_net:
            print(f"{e.w} {e.h} {e.c}\t{e}")
        print(
            f"bytes_read: {bytes_read}, original_size: {size}, final_position: {weights.tell()}"
        )
    """
    if (bytes_read != size):
        raise IOError('error reading weights file')
    """

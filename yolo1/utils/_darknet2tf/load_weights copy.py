"""
This file contains the code to load parsed weights that are in the DarkNet
format into TensorFlow layers
"""
import itertools
from tensorflow import keras as ks
from yolo.modeling.building_blocks import DarkConv


def split_converter(lst, i, j=None):
    if j is None:
        return lst.data[:i], lst.data[i:]
    return lst.data[:i], lst.data[i:j], lst.data[j:]


def interleve_weights(block):
    """merge weights to fit the DarkResnet block style"""
    if len(block) == 0:
        return []
    weights_temp = []
    for layer in block:
        weights = layer.get_weights()
        weights = [tuple(weights[0:3]), tuple(weights[3:])]
        weights_temp.append(weights)
    top, bottom = tuple(zip(*weights_temp))
    weights = list(itertools.chain.from_iterable(top)) + \
        list(itertools.chain.from_iterable(bottom))
    return weights


def get_darknet53_tf_format(net, only_weights=True):
    """convert weights from darknet sequntial to tensorflow weave, Darknet53 Backbone"""
    combo_blocks = []
    for i in range(2):
        layer = net.pop(0)
        combo_blocks.append(layer)
    # ugly code i will document, very tired
    encoder = []
    while len(net) != 0:
        blocks = []
        layer = net.pop(0)
        while layer._type != "shortcut":
            blocks.append(layer)
            layer = net.pop(0)
        encoder.append(blocks)
    new_net = combo_blocks + encoder
    weights = []
    if only_weights:
        for block in new_net:
            if type(block) != list:
                weights.append(block.get_weights())
            else:
                weights.append(interleve_weights(block))
    print("converted/interleved weights for tensorflow format")
    return new_net, weights


def get_tiny_tf_format(encoder):
    weights = []
    for layer in encoder:
        if layer._type != "maxpool":
            weights.append(layer.get_weights())
    return encoder, weights


def load_weights_dnBackbone(backbone, encoder, mtype="darknet53"):
    # get weights for backbone
    if mtype == "darknet53":
        encoder, weights_encoder = get_darknet53_tf_format(encoder[:])
    elif mtype == "darknet_tiny":
        encoder, weights_encoder = get_tiny_tf_format(encoder[:])

    # set backbone weights
    print(
        f"\nno. layers: {len(backbone.layers)}, no. weights: {len(weights_encoder)}"
    )
    set_darknet_weights(backbone, weights_encoder)

    backbone.trainable = False
    print(f"\nsetting backbone.trainable to: {backbone.trainable}\n")
    return


def load_weights_dnHead(head, decoder, v4=True):
    # get weights for head
    decoder, weights_decoder, head_layers, head_weights = get_decoder_weights(
        decoder)
    # set detection head weights
    print(
        f"\nno. layers: {len(head.layers)}, no. weights: {len(weights_decoder)}"
    )
    flat_full = list(flatten_model(head, r_list=False))
    flat_main = flat_full[:-3]
    flat_head = flat_full[-3:]

    # not the right way to do it
    if v4:
        flat_main.insert(1, flat_main[-1])

    print(len(flat_main), len(decoder))
    print(len(flat_head), len(head_layers))

    set_darknet_weights(head, weights_decoder, flat_model=flat_main)
    set_darknet_weights_head(flat_head, head_weights)

    head.trainable = False
    print(f"\nsetting head.trainable to: {head.trainable}\n")
    return


# DEBUGGING
def print_layer_shape(layer):
    try:
        weights = layer.get_weights()
    except:
        weights = layer
    for item in weights:
        print(item.shape)
    return


def flatten_model(model, r_list=True):
    for layer in model.layers:
        if r_list and isinstance(model, ks.Model):
            yield from model.layers
        else:
            yield layer


def set_darknet_weights_head(flat_head, weights_head):
    for layer in flat_head:
        weights = layer.get_weights()
        for weight in weights:
            print(weight.shape)
        weight_depth = weights[0].shape[-2]
        for weight in weights_head:
            if weight[0].shape[-2] == weight_depth:
                print(
                    f"loaded weights for layer: head layer with depth {weight_depth}  -> name: {layer.name}",
                    sep='      ',
                    end="\r")
                layer.set_weights(weight)
    return


def set_darknet_weights(model, weights_list, flat_model=None):
    if flat_model == None:
        zip_fill = flatten_model(model)
    else:
        zip_fill = flat_model
    for i, (layer, weights) in enumerate(zip(zip_fill, weights_list)):
        print(layer.name, len(weights))
        #layer.set_weights(weights)
    return


def split_decoder(lst):
    decoder = []
    outputs = []
    for layer in lst:
        if layer._type == 'yolo':
            outputs.append(decoder.pop())
            outputs.append(layer)
        else:
            decoder.append(layer)
    return decoder, outputs


def get_decoder_weights(decoder):
    layers = [[]]
    block = []
    weights = []

    decoder, head = split_decoder(decoder)

    # get decoder weights and group them together
    for i, layer in enumerate(decoder):
        if layer._type == "route" and len(
                layer.layers) >= 2 and decoder[i - 1]._type != 'maxpool':
            layers.append([])
            layers.append(block)
            block = []
        elif layer._type == "route" and decoder[i - 1]._type != 'maxpool':
            layers.append(block)
            block = []
        elif (layer._type == "route" and decoder[i - 1]._type
              == "maxpool") or layer._type == "maxpool":
            # made only for spp
            continue
        elif layer._type == "convolutional":
            block.append(layer)
        # else:
        #     # if you upsample
        #     layers.append([])

    if len(block) > 0:
        layers.append(block)

    # interleve weights for blocked layers
    for layer in layers:
        weights.append(interleve_weights(layer))

    # get weights for output detection heads
    head_weights = []
    head_layers = []
    for layer in (head):
        if layer != None and layer._type == "convolutional":
            head_weights.append(layer.get_weights())
            head_layers.append(layer)

    return layers, weights, head_layers, head_weights

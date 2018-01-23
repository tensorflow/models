# Architecture based on MobileNetV2 https://arxiv.org/pdf/1801.04381.pdf

import tensorflow as tf
import tensorflow.contrib.slim as slim


def mobilenet_v2_arg_scope(weight_decay, is_training=True, depth_multiplier=1.0, regularize_depthwise=False,
                           dropout_keep_prob=1.0):

    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'center': True, 'scale': True }):

        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):

            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=depth_multiplier):

                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob) as sc:

                    return sc


def block(net, input_filters, output_filters, expansion, stride):
    res_block = net
    res_block = slim.conv2d(inputs=res_block, num_outputs=input_filters * expansion, kernel_size=[1, 1])
    res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride)
    res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
    if stride == 2:
        return res_block
    else:
        if input_filters != output_filters:
            net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
        return tf.add(res_block, net)


def blocks(net, expansion, output_filters, repeat, stride):
    input_filters = net.shape[3].value

    # first layer should take stride into account
    net = block(net, input_filters, output_filters, expansion, stride)

    for _ in range(1, repeat):
        net = block(net, input_filters, output_filters, expansion, 1)

    return net


def mobilenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 depth_multiplier=1.0,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 scope='MobilenetV2'):

    endpoints = dict()

    expansion = 6

    with tf.variable_scope(scope):

        with slim.arg_scope(mobilenet_v2_arg_scope(0.0004, is_training=is_training, depth_multiplier=depth_multiplier,
                                                   dropout_keep_prob=dropout_keep_prob)):
            net = tf.identity(inputs)

            net = slim.conv2d(net, 32, [3, 3], scope='conv11', stride=2)

            net = blocks(net=net, expansion=1, output_filters=16, repeat=1, stride=1)

            net = blocks(net=net, expansion=expansion, output_filters=24, repeat=2, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=32, repeat=3, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=64, repeat=4, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=96, repeat=3, stride=1)

            net = blocks(net=net, expansion=expansion, output_filters=160, repeat=3, stride=2)

            net = blocks(net=net, expansion=expansion, output_filters=320, repeat=1, stride=1)

            net = slim.conv2d(net, 1280, [1, 1], scope='last_bottleneck')

            net = slim.avg_pool2d(net, [7, 7])

            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='features')

            if spatial_squeeze:
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            endpoints['Logits'] = logits

            if prediction_fn:
                endpoints['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, endpoints

mobilenet_v2.default_image_size = 224

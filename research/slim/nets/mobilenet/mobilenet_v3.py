# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mobilenet V3 conv defs and helper functions.

# pylint: disable=line-too-long

Model definitions and layer breakdowns:
==================
==== V3 LARGE ====
==================
    Conv2D MobilenetV3/Conv/Conv2D                                                              351.2 k      1x224x224x3            432.0           5.42 M     1x112x112x16
     Relu6 MobilenetV3/Conv/hard_swish/Relu6                                                          ?                -                ?                ?     1x112x112x16
 DepthConv MobilenetV3/expanded_conv/depthwise/depthwise                                        401.4 k                -            144.0           1.81 M     1x112x112x16
      Relu MobilenetV3/expanded_conv/depthwise/Relu                                                   ?                -                ?                ?     1x112x112x16
    Conv2D MobilenetV3/expanded_conv/project/Conv2D                                             401.4 k     1x112x112x16            256.0           3.21 M     1x112x112x16
    Conv2D MobilenetV3/expanded_conv_1/expand/Conv2D                                             1.00 M     1x112x112x16           1.02 k           12.8 M     1x112x112x64
      Relu MobilenetV3/expanded_conv_1/expand/Relu                                                    ?                -                ?                ?     1x112x112x64
 DepthConv MobilenetV3/expanded_conv_1/depthwise/depthwise                                       1.00 M                -            576.0           1.81 M       1x56x56x64
      Relu MobilenetV3/expanded_conv_1/depthwise/Relu                                                 ?                -                ?                ?       1x56x56x64
    Conv2D MobilenetV3/expanded_conv_1/project/Conv2D                                           276.0 k       1x56x56x64           1.54 k           4.82 M       1x56x56x24
    Conv2D MobilenetV3/expanded_conv_2/expand/Conv2D                                            301.1 k       1x56x56x24           1.73 k           5.42 M       1x56x56x72
      Relu MobilenetV3/expanded_conv_2/expand/Relu                                                    ?                -                ?                ?       1x56x56x72
 DepthConv MobilenetV3/expanded_conv_2/depthwise/depthwise                                      451.6 k                -            648.0           2.03 M       1x56x56x72
      Relu MobilenetV3/expanded_conv_2/depthwise/Relu                                                 ?                -                ?                ?       1x56x56x72
    Conv2D MobilenetV3/expanded_conv_2/project/Conv2D                                           301.1 k       1x56x56x72           1.73 k           5.42 M       1x56x56x24
    Conv2D MobilenetV3/expanded_conv_3/expand/Conv2D                                            301.1 k       1x56x56x24           1.73 k           5.42 M       1x56x56x72
      Relu MobilenetV3/expanded_conv_3/expand/Relu                                                    ?                -                ?                ?       1x56x56x72
 DepthConv MobilenetV3/expanded_conv_3/depthwise/depthwise                                      282.2 k                -           1.80 k           1.41 M       1x28x28x72
      Relu MobilenetV3/expanded_conv_3/depthwise/Relu                                                 ?                -                ?                ?       1x28x28x72
    Conv2D MobilenetV3/expanded_conv_3/squeeze_excite/Conv/Conv2D                                  96.0         1x1x1x72           1.73 k           1.73 k         1x1x1x24
      Relu MobilenetV3/expanded_conv_3/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x24
    Conv2D MobilenetV3/expanded_conv_3/squeeze_excite/Conv_1/Conv2D                                96.0         1x1x1x24           1.73 k           1.73 k         1x1x1x72
     Relu6 MobilenetV3/expanded_conv_3/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?         1x1x1x72
    Conv2D MobilenetV3/expanded_conv_3/project/Conv2D                                            87.8 k       1x28x28x72           2.88 k           2.26 M       1x28x28x40
    Conv2D MobilenetV3/expanded_conv_4/expand/Conv2D                                            125.4 k       1x28x28x40           4.80 k           3.76 M      1x28x28x120
      Relu MobilenetV3/expanded_conv_4/expand/Relu                                                    ?                -                ?                ?      1x28x28x120
 DepthConv MobilenetV3/expanded_conv_4/depthwise/depthwise                                      188.2 k                -           3.00 k           2.35 M      1x28x28x120
      Relu MobilenetV3/expanded_conv_4/depthwise/Relu                                                 ?                -                ?                ?      1x28x28x120
    Conv2D MobilenetV3/expanded_conv_4/squeeze_excite/Conv/Conv2D                                 152.0        1x1x1x120           3.84 k           3.84 k         1x1x1x32
      Relu MobilenetV3/expanded_conv_4/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x32
    Conv2D MobilenetV3/expanded_conv_4/squeeze_excite/Conv_1/Conv2D                               152.0         1x1x1x32           3.84 k           3.84 k        1x1x1x120
     Relu6 MobilenetV3/expanded_conv_4/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x120
    Conv2D MobilenetV3/expanded_conv_4/project/Conv2D                                           125.4 k      1x28x28x120           4.80 k           3.76 M       1x28x28x40
    Conv2D MobilenetV3/expanded_conv_5/expand/Conv2D                                            125.4 k       1x28x28x40           4.80 k           3.76 M      1x28x28x120
      Relu MobilenetV3/expanded_conv_5/expand/Relu                                                    ?                -                ?                ?      1x28x28x120
 DepthConv MobilenetV3/expanded_conv_5/depthwise/depthwise                                      188.2 k                -           3.00 k           2.35 M      1x28x28x120
      Relu MobilenetV3/expanded_conv_5/depthwise/Relu                                                 ?                -                ?                ?      1x28x28x120
    Conv2D MobilenetV3/expanded_conv_5/squeeze_excite/Conv/Conv2D                                 152.0        1x1x1x120           3.84 k           3.84 k         1x1x1x32
      Relu MobilenetV3/expanded_conv_5/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x32
    Conv2D MobilenetV3/expanded_conv_5/squeeze_excite/Conv_1/Conv2D                               152.0         1x1x1x32           3.84 k           3.84 k        1x1x1x120
     Relu6 MobilenetV3/expanded_conv_5/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x120
    Conv2D MobilenetV3/expanded_conv_5/project/Conv2D                                           125.4 k      1x28x28x120           4.80 k           3.76 M       1x28x28x40
    Conv2D MobilenetV3/expanded_conv_6/expand/Conv2D                                            219.5 k       1x28x28x40           9.60 k           7.53 M      1x28x28x240
     Relu6 MobilenetV3/expanded_conv_6/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x28x28x240
 DepthConv MobilenetV3/expanded_conv_6/depthwise/depthwise                                      235.2 k                -           2.16 k          423.4 k      1x14x14x240
     Relu6 MobilenetV3/expanded_conv_6/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x240
    Conv2D MobilenetV3/expanded_conv_6/project/Conv2D                                            62.7 k      1x14x14x240           19.2 k           3.76 M       1x14x14x80
    Conv2D MobilenetV3/expanded_conv_7/expand/Conv2D                                             54.9 k       1x14x14x80           16.0 k           3.14 M      1x14x14x200
     Relu6 MobilenetV3/expanded_conv_7/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x200
 DepthConv MobilenetV3/expanded_conv_7/depthwise/depthwise                                       78.4 k                -           1.80 k          352.8 k      1x14x14x200
     Relu6 MobilenetV3/expanded_conv_7/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x200
    Conv2D MobilenetV3/expanded_conv_7/project/Conv2D                                            54.9 k      1x14x14x200           16.0 k           3.14 M       1x14x14x80
    Conv2D MobilenetV3/expanded_conv_8/expand/Conv2D                                             51.7 k       1x14x14x80           14.7 k           2.89 M      1x14x14x184
     Relu6 MobilenetV3/expanded_conv_8/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x184
 DepthConv MobilenetV3/expanded_conv_8/depthwise/depthwise                                       72.1 k                -           1.66 k          324.6 k      1x14x14x184
     Relu6 MobilenetV3/expanded_conv_8/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x184
    Conv2D MobilenetV3/expanded_conv_8/project/Conv2D                                            51.7 k      1x14x14x184           14.7 k           2.89 M       1x14x14x80
    Conv2D MobilenetV3/expanded_conv_9/expand/Conv2D                                             51.7 k       1x14x14x80           14.7 k           2.89 M      1x14x14x184
     Relu6 MobilenetV3/expanded_conv_9/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x184
 DepthConv MobilenetV3/expanded_conv_9/depthwise/depthwise                                       72.1 k                -           1.66 k          324.6 k      1x14x14x184
     Relu6 MobilenetV3/expanded_conv_9/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x184
    Conv2D MobilenetV3/expanded_conv_9/project/Conv2D                                            51.7 k      1x14x14x184           14.7 k           2.89 M       1x14x14x80
    Conv2D MobilenetV3/expanded_conv_10/expand/Conv2D                                           109.8 k       1x14x14x80           38.4 k           7.53 M      1x14x14x480
     Relu6 MobilenetV3/expanded_conv_10/expand/hard_swish/Relu6                                       ?                -                ?                ?      1x14x14x480
 DepthConv MobilenetV3/expanded_conv_10/depthwise/depthwise                                     188.2 k                -           4.32 k          846.7 k      1x14x14x480
     Relu6 MobilenetV3/expanded_conv_10/depthwise/hard_swish/Relu6                                    ?                -                ?                ?      1x14x14x480
    Conv2D MobilenetV3/expanded_conv_10/squeeze_excite/Conv/Conv2D                                600.0        1x1x1x480           57.6 k           57.6 k        1x1x1x120
      Relu MobilenetV3/expanded_conv_10/squeeze_excite/Conv/Relu                                      ?                -                ?                ?        1x1x1x120
    Conv2D MobilenetV3/expanded_conv_10/squeeze_excite/Conv_1/Conv2D                              600.0        1x1x1x120           57.6 k           57.6 k        1x1x1x480
     Relu6 MobilenetV3/expanded_conv_10/squeeze_excite/Conv_1/Relu6                                   ?                -                ?                ?        1x1x1x480
    Conv2D MobilenetV3/expanded_conv_10/project/Conv2D                                          116.0 k      1x14x14x480           53.8 k           10.5 M      1x14x14x112
    Conv2D MobilenetV3/expanded_conv_11/expand/Conv2D                                           153.7 k      1x14x14x112           75.3 k           14.8 M      1x14x14x672
     Relu6 MobilenetV3/expanded_conv_11/expand/hard_swish/Relu6                                       ?                -                ?                ?      1x14x14x672
 DepthConv MobilenetV3/expanded_conv_11/depthwise/depthwise                                     263.4 k                -           6.05 k           1.19 M      1x14x14x672
     Relu6 MobilenetV3/expanded_conv_11/depthwise/hard_swish/Relu6                                    ?                -                ?                ?      1x14x14x672
    Conv2D MobilenetV3/expanded_conv_11/squeeze_excite/Conv/Conv2D                                840.0        1x1x1x672          112.9 k          112.9 k        1x1x1x168
      Relu MobilenetV3/expanded_conv_11/squeeze_excite/Conv/Relu                                      ?                -                ?                ?        1x1x1x168
    Conv2D MobilenetV3/expanded_conv_11/squeeze_excite/Conv_1/Conv2D                              840.0        1x1x1x168          112.9 k          112.9 k        1x1x1x672
     Relu6 MobilenetV3/expanded_conv_11/squeeze_excite/Conv_1/Relu6                                   ?                -                ?                ?        1x1x1x672
    Conv2D MobilenetV3/expanded_conv_11/project/Conv2D                                          153.7 k      1x14x14x672           75.3 k           14.8 M      1x14x14x112
    Conv2D MobilenetV3/expanded_conv_12/expand/Conv2D                                           153.7 k      1x14x14x112           75.3 k           14.8 M      1x14x14x672
     Relu6 MobilenetV3/expanded_conv_12/expand/hard_swish/Relu6                                       ?                -                ?                ?      1x14x14x672
 DepthConv MobilenetV3/expanded_conv_12/depthwise/depthwise                                     164.6 k                -           16.8 k          823.2 k        1x7x7x672
     Relu6 MobilenetV3/expanded_conv_12/depthwise/hard_swish/Relu6                                    ?                -                ?                ?        1x7x7x672
    Conv2D MobilenetV3/expanded_conv_12/squeeze_excite/Conv/Conv2D                                840.0        1x1x1x672          112.9 k          112.9 k        1x1x1x168
      Relu MobilenetV3/expanded_conv_12/squeeze_excite/Conv/Relu                                      ?                -                ?                ?        1x1x1x168
    Conv2D MobilenetV3/expanded_conv_12/squeeze_excite/Conv_1/Conv2D                              840.0        1x1x1x168          112.9 k          112.9 k        1x1x1x672
     Relu6 MobilenetV3/expanded_conv_12/squeeze_excite/Conv_1/Relu6                                   ?                -                ?                ?        1x1x1x672
    Conv2D MobilenetV3/expanded_conv_12/project/Conv2D                                           40.8 k        1x7x7x672          107.5 k           5.27 M        1x7x7x160
    Conv2D MobilenetV3/expanded_conv_13/expand/Conv2D                                            54.9 k        1x7x7x160          153.6 k           7.53 M        1x7x7x960
     Relu6 MobilenetV3/expanded_conv_13/expand/hard_swish/Relu6                                       ?                -                ?                ?        1x7x7x960
 DepthConv MobilenetV3/expanded_conv_13/depthwise/depthwise                                      94.1 k                -           24.0 k           1.18 M        1x7x7x960
     Relu6 MobilenetV3/expanded_conv_13/depthwise/hard_swish/Relu6                                    ?                -                ?                ?        1x7x7x960
    Conv2D MobilenetV3/expanded_conv_13/squeeze_excite/Conv/Conv2D                               1.20 k        1x1x1x960          230.4 k          230.4 k        1x1x1x240
      Relu MobilenetV3/expanded_conv_13/squeeze_excite/Conv/Relu                                      ?                -                ?                ?        1x1x1x240
    Conv2D MobilenetV3/expanded_conv_13/squeeze_excite/Conv_1/Conv2D                             1.20 k        1x1x1x240          230.4 k          230.4 k        1x1x1x960
     Relu6 MobilenetV3/expanded_conv_13/squeeze_excite/Conv_1/Relu6                                   ?                -                ?                ?        1x1x1x960
    Conv2D MobilenetV3/expanded_conv_13/project/Conv2D                                           54.9 k        1x7x7x960          153.6 k           7.53 M        1x7x7x160
    Conv2D MobilenetV3/expanded_conv_14/expand/Conv2D                                            54.9 k        1x7x7x160          153.6 k           7.53 M        1x7x7x960
     Relu6 MobilenetV3/expanded_conv_14/expand/hard_swish/Relu6                                       ?                -                ?                ?        1x7x7x960
 DepthConv MobilenetV3/expanded_conv_14/depthwise/depthwise                                      94.1 k                -           24.0 k           1.18 M        1x7x7x960
     Relu6 MobilenetV3/expanded_conv_14/depthwise/hard_swish/Relu6                                    ?                -                ?                ?        1x7x7x960
    Conv2D MobilenetV3/expanded_conv_14/squeeze_excite/Conv/Conv2D                               1.20 k        1x1x1x960          230.4 k          230.4 k        1x1x1x240
      Relu MobilenetV3/expanded_conv_14/squeeze_excite/Conv/Relu                                      ?                -                ?                ?        1x1x1x240
    Conv2D MobilenetV3/expanded_conv_14/squeeze_excite/Conv_1/Conv2D                             1.20 k        1x1x1x240          230.4 k          230.4 k        1x1x1x960
     Relu6 MobilenetV3/expanded_conv_14/squeeze_excite/Conv_1/Relu6                                   ?                -                ?                ?        1x1x1x960
    Conv2D MobilenetV3/expanded_conv_14/project/Conv2D                                           54.9 k        1x7x7x960          153.6 k           7.53 M        1x7x7x160
    Conv2D MobilenetV3/Conv_1/Conv2D                                                             54.9 k        1x7x7x160          153.6 k           7.53 M        1x7x7x960
     Relu6 MobilenetV3/Conv_1/hard_swish/Relu6                                                        ?                -                ?                ?        1x7x7x960
   AvgPool MobilenetV3/AvgPool2D/AvgPool                                                              ?        1x7x7x960                ?           47.0 k        1x1x1x960
    Conv2D MobilenetV3/Conv_2/Conv2D                                                             2.24 k        1x1x1x960           1.23 M           1.23 M       1x1x1x1280
     Relu6 MobilenetV3/Conv_2/hard_swish/Relu6                                                        ?                -                ?                ?       1x1x1x1280
    Conv2D MobilenetV3/Logits/Conv2d_1c_1x1/Conv2D                                               2.28 k       1x1x1x1280           1.28 M           1.28 M       1x1x1x1001
-----


==================
==== V3 SMALL ====
==================
      op name                                                                                  ActMem        ConvInput   ConvParameters            Madds     OutputTensor
    Conv2D MobilenetV3/Conv/Conv2D                                                              351.2 k      1x224x224x3            432.0           5.42 M     1x112x112x16
     Relu6 MobilenetV3/Conv/hard_swish/Relu6                                                          ?                -                ?                ?     1x112x112x16
 DepthConv MobilenetV3/expanded_conv/depthwise/depthwise                                        250.9 k                -            144.0          451.6 k       1x56x56x16
      Relu MobilenetV3/expanded_conv/depthwise/Relu                                                   ?                -                ?                ?       1x56x56x16
    Conv2D MobilenetV3/expanded_conv/squeeze_excite/Conv/Conv2D                                    24.0         1x1x1x16            128.0            128.0          1x1x1x8
      Relu MobilenetV3/expanded_conv/squeeze_excite/Conv/Relu                                         ?                -                ?                ?          1x1x1x8
    Conv2D MobilenetV3/expanded_conv/squeeze_excite/Conv_1/Conv2D                                  24.0          1x1x1x8            128.0            128.0         1x1x1x16
     Relu6 MobilenetV3/expanded_conv/squeeze_excite/Conv_1/Relu6                                      ?                -                ?                ?         1x1x1x16
    Conv2D MobilenetV3/expanded_conv/project/Conv2D                                             100.4 k       1x56x56x16            256.0          802.8 k       1x56x56x16
    Conv2D MobilenetV3/expanded_conv_1/expand/Conv2D                                            276.0 k       1x56x56x16           1.15 k           3.61 M       1x56x56x72
      Relu MobilenetV3/expanded_conv_1/expand/Relu                                                    ?                -                ?                ?       1x56x56x72
 DepthConv MobilenetV3/expanded_conv_1/depthwise/depthwise                                      282.2 k                -            648.0          508.0 k       1x28x28x72
      Relu MobilenetV3/expanded_conv_1/depthwise/Relu                                                 ?                -                ?                ?       1x28x28x72
    Conv2D MobilenetV3/expanded_conv_1/project/Conv2D                                            75.3 k       1x28x28x72           1.73 k           1.35 M       1x28x28x24
    Conv2D MobilenetV3/expanded_conv_2/expand/Conv2D                                             87.8 k       1x28x28x24           2.11 k           1.66 M       1x28x28x88
      Relu MobilenetV3/expanded_conv_2/expand/Relu                                                    ?                -                ?                ?       1x28x28x88
 DepthConv MobilenetV3/expanded_conv_2/depthwise/depthwise                                      138.0 k                -            792.0          620.9 k       1x28x28x88
      Relu MobilenetV3/expanded_conv_2/depthwise/Relu                                                 ?                -                ?                ?       1x28x28x88
    Conv2D MobilenetV3/expanded_conv_2/project/Conv2D                                            87.8 k       1x28x28x88           2.11 k           1.66 M       1x28x28x24
    Conv2D MobilenetV3/expanded_conv_3/expand/Conv2D                                             94.1 k       1x28x28x24           2.30 k           1.81 M       1x28x28x96
     Relu6 MobilenetV3/expanded_conv_3/expand/hard_swish/Relu6                                        ?                -                ?                ?       1x28x28x96
 DepthConv MobilenetV3/expanded_conv_3/depthwise/depthwise                                       94.1 k                -           2.40 k          470.4 k       1x14x14x96
     Relu6 MobilenetV3/expanded_conv_3/depthwise/hard_swish/Relu6                                     ?                -                ?                ?       1x14x14x96
    Conv2D MobilenetV3/expanded_conv_3/squeeze_excite/Conv/Conv2D                                 120.0         1x1x1x96           2.30 k           2.30 k         1x1x1x24
      Relu MobilenetV3/expanded_conv_3/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x24
    Conv2D MobilenetV3/expanded_conv_3/squeeze_excite/Conv_1/Conv2D                               120.0         1x1x1x24           2.30 k           2.30 k         1x1x1x96
     Relu6 MobilenetV3/expanded_conv_3/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?         1x1x1x96
    Conv2D MobilenetV3/expanded_conv_3/project/Conv2D                                            26.7 k       1x14x14x96           3.84 k          752.6 k       1x14x14x40
    Conv2D MobilenetV3/expanded_conv_4/expand/Conv2D                                             54.9 k       1x14x14x40           9.60 k           1.88 M      1x14x14x240
     Relu6 MobilenetV3/expanded_conv_4/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x240
 DepthConv MobilenetV3/expanded_conv_4/depthwise/depthwise                                       94.1 k                -           6.00 k           1.18 M      1x14x14x240
     Relu6 MobilenetV3/expanded_conv_4/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x240
    Conv2D MobilenetV3/expanded_conv_4/squeeze_excite/Conv/Conv2D                                 304.0        1x1x1x240           15.4 k           15.4 k         1x1x1x64
      Relu MobilenetV3/expanded_conv_4/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x64
    Conv2D MobilenetV3/expanded_conv_4/squeeze_excite/Conv_1/Conv2D                               304.0         1x1x1x64           15.4 k           15.4 k        1x1x1x240
     Relu6 MobilenetV3/expanded_conv_4/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x240
    Conv2D MobilenetV3/expanded_conv_4/project/Conv2D                                            54.9 k      1x14x14x240           9.60 k           1.88 M       1x14x14x40
    Conv2D MobilenetV3/expanded_conv_5/expand/Conv2D                                             54.9 k       1x14x14x40           9.60 k           1.88 M      1x14x14x240
     Relu6 MobilenetV3/expanded_conv_5/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x240
 DepthConv MobilenetV3/expanded_conv_5/depthwise/depthwise                                       94.1 k                -           6.00 k           1.18 M      1x14x14x240
     Relu6 MobilenetV3/expanded_conv_5/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x240
    Conv2D MobilenetV3/expanded_conv_5/squeeze_excite/Conv/Conv2D                                 304.0        1x1x1x240           15.4 k           15.4 k         1x1x1x64
      Relu MobilenetV3/expanded_conv_5/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x64
    Conv2D MobilenetV3/expanded_conv_5/squeeze_excite/Conv_1/Conv2D                               304.0         1x1x1x64           15.4 k           15.4 k        1x1x1x240
     Relu6 MobilenetV3/expanded_conv_5/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x240
    Conv2D MobilenetV3/expanded_conv_5/project/Conv2D                                            54.9 k      1x14x14x240           9.60 k           1.88 M       1x14x14x40
    Conv2D MobilenetV3/expanded_conv_6/expand/Conv2D                                             31.4 k       1x14x14x40           4.80 k          940.8 k      1x14x14x120
     Relu6 MobilenetV3/expanded_conv_6/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x120
 DepthConv MobilenetV3/expanded_conv_6/depthwise/depthwise                                       47.0 k                -           3.00 k          588.0 k      1x14x14x120
     Relu6 MobilenetV3/expanded_conv_6/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x120
    Conv2D MobilenetV3/expanded_conv_6/squeeze_excite/Conv/Conv2D                                 152.0        1x1x1x120           3.84 k           3.84 k         1x1x1x32
      Relu MobilenetV3/expanded_conv_6/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x32
    Conv2D MobilenetV3/expanded_conv_6/squeeze_excite/Conv_1/Conv2D                               152.0         1x1x1x32           3.84 k           3.84 k        1x1x1x120
     Relu6 MobilenetV3/expanded_conv_6/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x120
    Conv2D MobilenetV3/expanded_conv_6/project/Conv2D                                            32.9 k      1x14x14x120           5.76 k           1.13 M       1x14x14x48
    Conv2D MobilenetV3/expanded_conv_7/expand/Conv2D                                             37.6 k       1x14x14x48           6.91 k           1.35 M      1x14x14x144
     Relu6 MobilenetV3/expanded_conv_7/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x144
 DepthConv MobilenetV3/expanded_conv_7/depthwise/depthwise                                       56.4 k                -           3.60 k          705.6 k      1x14x14x144
     Relu6 MobilenetV3/expanded_conv_7/depthwise/hard_swish/Relu6                                     ?                -                ?                ?      1x14x14x144
    Conv2D MobilenetV3/expanded_conv_7/squeeze_excite/Conv/Conv2D                                 184.0        1x1x1x144           5.76 k           5.76 k         1x1x1x40
      Relu MobilenetV3/expanded_conv_7/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x40
    Conv2D MobilenetV3/expanded_conv_7/squeeze_excite/Conv_1/Conv2D                               184.0         1x1x1x40           5.76 k           5.76 k        1x1x1x144
     Relu6 MobilenetV3/expanded_conv_7/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x144
    Conv2D MobilenetV3/expanded_conv_7/project/Conv2D                                            37.6 k      1x14x14x144           6.91 k           1.35 M       1x14x14x48
    Conv2D MobilenetV3/expanded_conv_8/expand/Conv2D                                             65.9 k       1x14x14x48           13.8 k           2.71 M      1x14x14x288
     Relu6 MobilenetV3/expanded_conv_8/expand/hard_swish/Relu6                                        ?                -                ?                ?      1x14x14x288
 DepthConv MobilenetV3/expanded_conv_8/depthwise/depthwise                                       70.6 k                -           7.20 k          352.8 k        1x7x7x288
     Relu6 MobilenetV3/expanded_conv_8/depthwise/hard_swish/Relu6                                     ?                -                ?                ?        1x7x7x288
    Conv2D MobilenetV3/expanded_conv_8/squeeze_excite/Conv/Conv2D                                 360.0        1x1x1x288           20.7 k           20.7 k         1x1x1x72
      Relu MobilenetV3/expanded_conv_8/squeeze_excite/Conv/Relu                                       ?                -                ?                ?         1x1x1x72
    Conv2D MobilenetV3/expanded_conv_8/squeeze_excite/Conv_1/Conv2D                               360.0         1x1x1x72           20.7 k           20.7 k        1x1x1x288
     Relu6 MobilenetV3/expanded_conv_8/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x288
    Conv2D MobilenetV3/expanded_conv_8/project/Conv2D                                            18.8 k        1x7x7x288           27.6 k           1.35 M         1x7x7x96
    Conv2D MobilenetV3/expanded_conv_9/expand/Conv2D                                             32.9 k         1x7x7x96           55.3 k           2.71 M        1x7x7x576
     Relu6 MobilenetV3/expanded_conv_9/expand/hard_swish/Relu6                                        ?                -                ?                ?        1x7x7x576
 DepthConv MobilenetV3/expanded_conv_9/depthwise/depthwise                                       56.4 k                -           14.4 k          705.6 k        1x7x7x576
     Relu6 MobilenetV3/expanded_conv_9/depthwise/hard_swish/Relu6                                     ?                -                ?                ?        1x7x7x576
    Conv2D MobilenetV3/expanded_conv_9/squeeze_excite/Conv/Conv2D                                 720.0        1x1x1x576           82.9 k           82.9 k        1x1x1x144
      Relu MobilenetV3/expanded_conv_9/squeeze_excite/Conv/Relu                                       ?                -                ?                ?        1x1x1x144
    Conv2D MobilenetV3/expanded_conv_9/squeeze_excite/Conv_1/Conv2D                               720.0        1x1x1x144           82.9 k           82.9 k        1x1x1x576
     Relu6 MobilenetV3/expanded_conv_9/squeeze_excite/Conv_1/Relu6                                    ?                -                ?                ?        1x1x1x576
    Conv2D MobilenetV3/expanded_conv_9/project/Conv2D                                            32.9 k        1x7x7x576           55.3 k           2.71 M         1x7x7x96
    Conv2D MobilenetV3/expanded_conv_10/expand/Conv2D                                            32.9 k         1x7x7x96           55.3 k           2.71 M        1x7x7x576
     Relu6 MobilenetV3/expanded_conv_10/expand/hard_swish/Relu6                                       ?                -                ?                ?        1x7x7x576
 DepthConv MobilenetV3/expanded_conv_10/depthwise/depthwise                                      56.4 k                -           14.4 k          705.6 k        1x7x7x576
     Relu6 MobilenetV3/expanded_conv_10/depthwise/hard_swish/Relu6                                    ?                -                ?                ?        1x7x7x576
    Conv2D MobilenetV3/expanded_conv_10/squeeze_excite/Conv/Conv2D                                720.0        1x1x1x576           82.9 k           82.9 k        1x1x1x144
      Relu MobilenetV3/expanded_conv_10/squeeze_excite/Conv/Relu                                      ?                -                ?                ?        1x1x1x144
    Conv2D MobilenetV3/expanded_conv_10/squeeze_excite/Conv_1/Conv2D                              720.0        1x1x1x144           82.9 k           82.9 k        1x1x1x576
     Relu6 MobilenetV3/expanded_conv_10/squeeze_excite/Conv_1/Relu6                                   ?                -                ?                ?        1x1x1x576
    Conv2D MobilenetV3/expanded_conv_10/project/Conv2D                                           32.9 k        1x7x7x576           55.3 k           2.71 M         1x7x7x96
    Conv2D MobilenetV3/Conv_1/Conv2D                                                             32.9 k         1x7x7x96           55.3 k           2.71 M        1x7x7x576
     Relu6 MobilenetV3/Conv_1/hard_swish/Relu6                                                        ?                -                ?                ?        1x7x7x576
   AvgPool MobilenetV3/AvgPool2D/AvgPool                                                              ?        1x7x7x576                ?           28.2 k        1x1x1x576
    Conv2D MobilenetV3/Conv_2/Conv2D                                                             1.60 k        1x1x1x576          589.8 k          589.8 k       1x1x1x1024
     Relu6 MobilenetV3/Conv_2/hard_swish/Relu6                                                        ?                -                ?                ?       1x1x1x1024
    Conv2D MobilenetV3/Logits/Conv2d_1c_1x1/Conv2D                                               2.02 k       1x1x1x1024           1.03 M           1.03 M       1x1x1x1001
-----
     Total Total                                                                                 2.96 M                -           2.53 M           56.5 M                -


====================
==== V3 EDGETPU ====
====================
        op name                                                                                  ActMem        ConvInput   ConvParameters            Madds     OutputTensor
    Conv2D MobilenetEdgeTPU/Conv/Conv2D                                                         551.9 k      1x224x224x3            864.0           10.8 M     1x112x112x32
      Relu MobilenetEdgeTPU/Conv/Relu                                                                 ?                -                ?                ?     1x112x112x32
    Conv2D MobilenetEdgeTPU/expanded_conv/project/Conv2D                                        602.1 k     1x112x112x32            512.0           6.42 M     1x112x112x16
    Conv2D MobilenetEdgeTPU/expanded_conv_1/expand/Conv2D                                       602.1 k     1x112x112x16           18.4 k           57.8 M      1x56x56x128
      Relu MobilenetEdgeTPU/expanded_conv_1/expand/Relu                                               ?                -                ?                ?      1x56x56x128
    Conv2D MobilenetEdgeTPU/expanded_conv_1/project/Conv2D                                      501.8 k      1x56x56x128           4.10 k           12.8 M       1x56x56x32
    Conv2D MobilenetEdgeTPU/expanded_conv_2/expand/Conv2D                                       501.8 k       1x56x56x32           36.9 k          115.6 M      1x56x56x128
      Relu MobilenetEdgeTPU/expanded_conv_2/expand/Relu                                               ?                -                ?                ?      1x56x56x128
    Conv2D MobilenetEdgeTPU/expanded_conv_2/project/Conv2D                                      501.8 k      1x56x56x128           4.10 k           12.8 M       1x56x56x32
    Conv2D MobilenetEdgeTPU/expanded_conv_3/expand/Conv2D                                       501.8 k       1x56x56x32           36.9 k          115.6 M      1x56x56x128
      Relu MobilenetEdgeTPU/expanded_conv_3/expand/Relu                                               ?                -                ?                ?      1x56x56x128
    Conv2D MobilenetEdgeTPU/expanded_conv_3/project/Conv2D                                      501.8 k      1x56x56x128           4.10 k           12.8 M       1x56x56x32
    Conv2D MobilenetEdgeTPU/expanded_conv_4/expand/Conv2D                                       501.8 k       1x56x56x32           36.9 k          115.6 M      1x56x56x128
      Relu MobilenetEdgeTPU/expanded_conv_4/expand/Relu                                               ?                -                ?                ?      1x56x56x128
    Conv2D MobilenetEdgeTPU/expanded_conv_4/project/Conv2D                                      501.8 k      1x56x56x128           4.10 k           12.8 M       1x56x56x32
    Conv2D MobilenetEdgeTPU/expanded_conv_5/expand/Conv2D                                       301.1 k       1x56x56x32           73.7 k           57.8 M      1x28x28x256
      Relu MobilenetEdgeTPU/expanded_conv_5/expand/Relu                                               ?                -                ?                ?      1x28x28x256
    Conv2D MobilenetEdgeTPU/expanded_conv_5/project/Conv2D                                      238.3 k      1x28x28x256           12.3 k           9.63 M       1x28x28x48
    Conv2D MobilenetEdgeTPU/expanded_conv_6/expand/Conv2D                                       188.2 k       1x28x28x48           82.9 k           65.0 M      1x28x28x192
      Relu MobilenetEdgeTPU/expanded_conv_6/expand/Relu                                               ?                -                ?                ?      1x28x28x192
    Conv2D MobilenetEdgeTPU/expanded_conv_6/project/Conv2D                                      188.2 k      1x28x28x192           9.22 k           7.23 M       1x28x28x48
    Conv2D MobilenetEdgeTPU/expanded_conv_7/expand/Conv2D                                       188.2 k       1x28x28x48           82.9 k           65.0 M      1x28x28x192
      Relu MobilenetEdgeTPU/expanded_conv_7/expand/Relu                                               ?                -                ?                ?      1x28x28x192
    Conv2D MobilenetEdgeTPU/expanded_conv_7/project/Conv2D                                      188.2 k      1x28x28x192           9.22 k           7.23 M       1x28x28x48
    Conv2D MobilenetEdgeTPU/expanded_conv_8/expand/Conv2D                                       188.2 k       1x28x28x48           82.9 k           65.0 M      1x28x28x192
      Relu MobilenetEdgeTPU/expanded_conv_8/expand/Relu                                               ?                -                ?                ?      1x28x28x192
    Conv2D MobilenetEdgeTPU/expanded_conv_8/project/Conv2D                                      188.2 k      1x28x28x192           9.22 k           7.23 M       1x28x28x48
    Conv2D MobilenetEdgeTPU/expanded_conv_9/expand/Conv2D                                       338.7 k       1x28x28x48           18.4 k           14.5 M      1x28x28x384
      Relu MobilenetEdgeTPU/expanded_conv_9/expand/Relu                                               ?                -                ?                ?      1x28x28x384
 DepthConv MobilenetEdgeTPU/expanded_conv_9/depthwise/depthwise                                 376.3 k                -           3.46 k          677.4 k      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_9/depthwise/Relu                                            ?                -                ?                ?      1x14x14x384
    Conv2D MobilenetEdgeTPU/expanded_conv_9/project/Conv2D                                       94.1 k      1x14x14x384           36.9 k           7.23 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_10/expand/Conv2D                                       94.1 k       1x14x14x96           36.9 k           7.23 M      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_10/expand/Relu                                              ?                -                ?                ?      1x14x14x384
 DepthConv MobilenetEdgeTPU/expanded_conv_10/depthwise/depthwise                                150.5 k                -           3.46 k          677.4 k      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_10/depthwise/Relu                                           ?                -                ?                ?      1x14x14x384
    Conv2D MobilenetEdgeTPU/expanded_conv_10/project/Conv2D                                      94.1 k      1x14x14x384           36.9 k           7.23 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_11/expand/Conv2D                                       94.1 k       1x14x14x96           36.9 k           7.23 M      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_11/expand/Relu                                              ?                -                ?                ?      1x14x14x384
 DepthConv MobilenetEdgeTPU/expanded_conv_11/depthwise/depthwise                                150.5 k                -           3.46 k          677.4 k      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_11/depthwise/Relu                                           ?                -                ?                ?      1x14x14x384
    Conv2D MobilenetEdgeTPU/expanded_conv_11/project/Conv2D                                      94.1 k      1x14x14x384           36.9 k           7.23 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_12/expand/Conv2D                                       94.1 k       1x14x14x96           36.9 k           7.23 M      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_12/expand/Relu                                              ?                -                ?                ?      1x14x14x384
 DepthConv MobilenetEdgeTPU/expanded_conv_12/depthwise/depthwise                                150.5 k                -           3.46 k          677.4 k      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_12/depthwise/Relu                                           ?                -                ?                ?      1x14x14x384
    Conv2D MobilenetEdgeTPU/expanded_conv_12/project/Conv2D                                      94.1 k      1x14x14x384           36.9 k           7.23 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_13/expand/Conv2D                                      169.3 k       1x14x14x96           73.7 k           14.5 M      1x14x14x768
      Relu MobilenetEdgeTPU/expanded_conv_13/expand/Relu                                              ?                -                ?                ?      1x14x14x768
 DepthConv MobilenetEdgeTPU/expanded_conv_13/depthwise/depthwise                                301.1 k                -           6.91 k           1.35 M      1x14x14x768
      Relu MobilenetEdgeTPU/expanded_conv_13/depthwise/Relu                                           ?                -                ?                ?      1x14x14x768
    Conv2D MobilenetEdgeTPU/expanded_conv_13/project/Conv2D                                     169.3 k      1x14x14x768           73.7 k           14.5 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_14/expand/Conv2D                                       94.1 k       1x14x14x96           36.9 k           7.23 M      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_14/expand/Relu                                              ?                -                ?                ?      1x14x14x384
 DepthConv MobilenetEdgeTPU/expanded_conv_14/depthwise/depthwise                                150.5 k                -           3.46 k          677.4 k      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_14/depthwise/Relu                                           ?                -                ?                ?      1x14x14x384
    Conv2D MobilenetEdgeTPU/expanded_conv_14/project/Conv2D                                      94.1 k      1x14x14x384           36.9 k           7.23 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_15/expand/Conv2D                                       94.1 k       1x14x14x96           36.9 k           7.23 M      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_15/expand/Relu                                              ?                -                ?                ?      1x14x14x384
 DepthConv MobilenetEdgeTPU/expanded_conv_15/depthwise/depthwise                                150.5 k                -           3.46 k          677.4 k      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_15/depthwise/Relu                                           ?                -                ?                ?      1x14x14x384
    Conv2D MobilenetEdgeTPU/expanded_conv_15/project/Conv2D                                      94.1 k      1x14x14x384           36.9 k           7.23 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_16/expand/Conv2D                                       94.1 k       1x14x14x96           36.9 k           7.23 M      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_16/expand/Relu                                              ?                -                ?                ?      1x14x14x384
 DepthConv MobilenetEdgeTPU/expanded_conv_16/depthwise/depthwise                                150.5 k                -           3.46 k          677.4 k      1x14x14x384
      Relu MobilenetEdgeTPU/expanded_conv_16/depthwise/Relu                                           ?                -                ?                ?      1x14x14x384
    Conv2D MobilenetEdgeTPU/expanded_conv_16/project/Conv2D                                      94.1 k      1x14x14x384           36.9 k           7.23 M       1x14x14x96
    Conv2D MobilenetEdgeTPU/expanded_conv_17/expand/Conv2D                                      169.3 k       1x14x14x96           73.7 k           14.5 M      1x14x14x768
      Relu MobilenetEdgeTPU/expanded_conv_17/expand/Relu                                              ?                -                ?                ?      1x14x14x768
 DepthConv MobilenetEdgeTPU/expanded_conv_17/depthwise/depthwise                                188.2 k                -           19.2 k          940.8 k        1x7x7x768
      Relu MobilenetEdgeTPU/expanded_conv_17/depthwise/Relu                                           ?                -                ?                ?        1x7x7x768
    Conv2D MobilenetEdgeTPU/expanded_conv_17/project/Conv2D                                      45.5 k        1x7x7x768          122.9 k           6.02 M        1x7x7x160
    Conv2D MobilenetEdgeTPU/expanded_conv_18/expand/Conv2D                                       39.2 k        1x7x7x160          102.4 k           5.02 M        1x7x7x640
      Relu MobilenetEdgeTPU/expanded_conv_18/expand/Relu                                              ?                -                ?                ?        1x7x7x640
 DepthConv MobilenetEdgeTPU/expanded_conv_18/depthwise/depthwise                                 62.7 k                -           16.0 k          784.0 k        1x7x7x640
      Relu MobilenetEdgeTPU/expanded_conv_18/depthwise/Relu                                           ?                -                ?                ?        1x7x7x640
    Conv2D MobilenetEdgeTPU/expanded_conv_18/project/Conv2D                                      39.2 k        1x7x7x640          102.4 k           5.02 M        1x7x7x160
    Conv2D MobilenetEdgeTPU/expanded_conv_19/expand/Conv2D                                       39.2 k        1x7x7x160          102.4 k           5.02 M        1x7x7x640
      Relu MobilenetEdgeTPU/expanded_conv_19/expand/Relu                                              ?                -                ?                ?        1x7x7x640
 DepthConv MobilenetEdgeTPU/expanded_conv_19/depthwise/depthwise                                 62.7 k                -           16.0 k          784.0 k        1x7x7x640
      Relu MobilenetEdgeTPU/expanded_conv_19/depthwise/Relu                                           ?                -                ?                ?        1x7x7x640
    Conv2D MobilenetEdgeTPU/expanded_conv_19/project/Conv2D                                      39.2 k        1x7x7x640          102.4 k           5.02 M        1x7x7x160
    Conv2D MobilenetEdgeTPU/expanded_conv_20/expand/Conv2D                                       39.2 k        1x7x7x160          102.4 k           5.02 M        1x7x7x640
      Relu MobilenetEdgeTPU/expanded_conv_20/expand/Relu                                              ?                -                ?                ?        1x7x7x640
 DepthConv MobilenetEdgeTPU/expanded_conv_20/depthwise/depthwise                                 62.7 k                -           16.0 k          784.0 k        1x7x7x640
      Relu MobilenetEdgeTPU/expanded_conv_20/depthwise/Relu                                           ?                -                ?                ?        1x7x7x640
    Conv2D MobilenetEdgeTPU/expanded_conv_20/project/Conv2D                                      39.2 k        1x7x7x640          102.4 k           5.02 M        1x7x7x160
    Conv2D MobilenetEdgeTPU/expanded_conv_21/expand/Conv2D                                       70.6 k        1x7x7x160          204.8 k           10.0 M       1x7x7x1280
      Relu MobilenetEdgeTPU/expanded_conv_21/expand/Relu                                              ?                -                ?                ?       1x7x7x1280
 DepthConv MobilenetEdgeTPU/expanded_conv_21/depthwise/depthwise                                125.4 k                -           11.5 k          564.5 k       1x7x7x1280
      Relu MobilenetEdgeTPU/expanded_conv_21/depthwise/Relu                                           ?                -                ?                ?       1x7x7x1280
    Conv2D MobilenetEdgeTPU/expanded_conv_21/project/Conv2D                                      72.1 k       1x7x7x1280          245.8 k           12.0 M        1x7x7x192
    Conv2D MobilenetEdgeTPU/Conv_1/Conv2D                                                        72.1 k        1x7x7x192          245.8 k           12.0 M       1x7x7x1280
      Relu MobilenetEdgeTPU/Conv_1/Relu                                                               ?                -                ?                ?       1x7x7x1280
   AvgPool MobilenetEdgeTPU/Logits/AvgPool2D                                                          ?       1x7x7x1280                ?           62.7 k       1x1x1x1280
    Conv2D MobilenetEdgeTPU/Logits/Conv2d_1c_1x1/Conv2D                                          2.28 k       1x1x1x1280           1.28 M           1.28 M       1x1x1x1001
-----
     Total Total                                                                                 11.6 M                -           4.05 M          990.7 M                -


# pylint: enable=line-too-long
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import numpy as np

import tensorflow.compat.v1 as tf
import tf_slim as slim

from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib

op = lib.op
expand_input = ops.expand_input_by_factor

# Squeeze Excite with all parameters filled-in, we use hard-sigmoid
# for gating function and relu for inner activation function.
squeeze_excite = functools.partial(
    ops.squeeze_excite, squeeze_factor=4,
    inner_activation_fn=tf.nn.relu,
    gating_fn=lambda x: tf.nn.relu6(x+3)*0.16667)

# Wrap squeeze excite op as expansion_transform that takes
# both expansion and input tensor.
_se4 = lambda expansion_tensor, input_tensor: squeeze_excite(expansion_tensor)


def hard_swish(x):
  with tf.name_scope('hard_swish'):
    return x * tf.nn.relu6(x + np.float32(3)) * np.float32(1. / 6.)


def reduce_to_1x1(input_tensor, default_size=7, **kwargs):
  h, w = input_tensor.shape.as_list()[1:3]
  if h is not None and w == h:
    k = [h, h]
  else:
    k = [default_size, default_size]
  return slim.avg_pool2d(input_tensor, kernel_size=k, **kwargs)


def mbv3_op(ef, n, k, s=1, act=tf.nn.relu, se=None, **kwargs):
  """Defines a single Mobilenet V3 convolution block.

  Args:
    ef: expansion factor
    n: number of output channels
    k: stride of depthwise
    s: stride
    act: activation function in inner layers
    se: squeeze excite function.
    **kwargs: passed to expanded_conv

  Returns:
    An object (lib._Op) for inserting in conv_def, representing this operation.
  """
  return op(
      ops.expanded_conv,
      expansion_size=expand_input(ef),
      kernel_size=(k, k),
      stride=s,
      num_outputs=n,
      inner_activation_fn=act,
      expansion_transform=se,
      **kwargs)


def mbv3_fused(ef, n, k, s=1, **kwargs):
  """Defines a single Mobilenet V3 convolution block.

  Args:
    ef: expansion factor
    n: number of output channels
    k: stride of depthwise
    s: stride
    **kwargs: will be passed to mbv3_op

  Returns:
    An object (lib._Op) for inserting in conv_def, representing this operation.
  """
  expansion_fn = functools.partial(slim.conv2d, kernel_size=k, stride=s)
  return mbv3_op(
      ef,
      n,
      k=1,
      s=s,
      depthwise_location=None,
      expansion_fn=expansion_fn,
      **kwargs)


mbv3_op_se = functools.partial(mbv3_op, se=_se4)


DEFAULTS = {
    (ops.expanded_conv,):
        dict(
            normalizer_fn=slim.batch_norm,
            residual=True),
    (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
        'normalizer_fn': slim.batch_norm,
        'activation_fn': tf.nn.relu,
    },
    (slim.batch_norm,): {
        'center': True,
        'scale': True
    },
}

DEFAULTS_GROUP_NORM = {
    (ops.expanded_conv,): dict(normalizer_fn=slim.group_norm, residual=True),
    (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
        'normalizer_fn': slim.group_norm,
        'activation_fn': tf.nn.relu,
    },
    (slim.group_norm,): {
        'groups': 8
    },
}
# Compatible checkpoint: http://mldash/5511169891790690458#scalars
V3_LARGE = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3),
           activation_fn=hard_swish),
        mbv3_op(ef=1, n=16, k=3),
        mbv3_op(ef=4, n=24, k=3, s=2),
        mbv3_op(ef=3, n=24, k=3, s=1),
        mbv3_op_se(ef=3, n=40, k=5, s=2),
        mbv3_op_se(ef=3, n=40, k=5, s=1),
        mbv3_op_se(ef=3, n=40, k=5, s=1),
        mbv3_op(ef=6, n=80, k=3, s=2, act=hard_swish),
        mbv3_op(ef=2.5, n=80, k=3, s=1, act=hard_swish),
        mbv3_op(ef=184/80., n=80, k=3, s=1, act=hard_swish),
        mbv3_op(ef=184/80., n=80, k=3, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=112, k=3, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=160, k=5, s=2, act=hard_swish),
        mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=160, k=5, s=1, act=hard_swish),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=960,
           activation_fn=hard_swish),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280,
           normalizer_fn=None, activation_fn=hard_swish)
    ]))

# 72.2% accuracy.
V3_LARGE_MINIMALISTIC = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3)),
        mbv3_op(ef=1, n=16, k=3),
        mbv3_op(ef=4, n=24, k=3, s=2),
        mbv3_op(ef=3, n=24, k=3, s=1),
        mbv3_op(ef=3, n=40, k=3, s=2),
        mbv3_op(ef=3, n=40, k=3, s=1),
        mbv3_op(ef=3, n=40, k=3, s=1),
        mbv3_op(ef=6, n=80, k=3, s=2),
        mbv3_op(ef=2.5, n=80, k=3, s=1),
        mbv3_op(ef=184 / 80., n=80, k=3, s=1),
        mbv3_op(ef=184 / 80., n=80, k=3, s=1),
        mbv3_op(ef=6, n=112, k=3, s=1),
        mbv3_op(ef=6, n=112, k=3, s=1),
        mbv3_op(ef=6, n=160, k=3, s=2),
        mbv3_op(ef=6, n=160, k=3, s=1),
        mbv3_op(ef=6, n=160, k=3, s=1),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=960),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d,
           stride=1,
           kernel_size=[1, 1],
           num_outputs=1280,
           normalizer_fn=None)
    ]))

# Compatible run: http://mldash/2023283040014348118#scalars
V3_SMALL = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3),
           activation_fn=hard_swish),
        mbv3_op_se(ef=1, n=16, k=3, s=2),
        mbv3_op(ef=72./16, n=24, k=3, s=2),
        mbv3_op(ef=(88./24), n=24, k=3, s=1),
        mbv3_op_se(ef=4, n=40, k=5, s=2, act=hard_swish),
        mbv3_op_se(ef=6, n=40, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=40, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=3, n=48, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=3, n=48, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=96, k=5, s=2, act=hard_swish),
        mbv3_op_se(ef=6, n=96, k=5, s=1, act=hard_swish),
        mbv3_op_se(ef=6, n=96, k=5, s=1, act=hard_swish),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=576,
           activation_fn=hard_swish),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1024,
           normalizer_fn=None, activation_fn=hard_swish)
    ]))

# 62% accuracy.
V3_SMALL_MINIMALISTIC = dict(
    defaults=dict(DEFAULTS),
    spec=([
        # stage 1
        op(slim.conv2d, stride=2, num_outputs=16, kernel_size=(3, 3)),
        mbv3_op(ef=1, n=16, k=3, s=2),
        mbv3_op(ef=72. / 16, n=24, k=3, s=2),
        mbv3_op(ef=(88. / 24), n=24, k=3, s=1),
        mbv3_op(ef=4, n=40, k=3, s=2),
        mbv3_op(ef=6, n=40, k=3, s=1),
        mbv3_op(ef=6, n=40, k=3, s=1),
        mbv3_op(ef=3, n=48, k=3, s=1),
        mbv3_op(ef=3, n=48, k=3, s=1),
        mbv3_op(ef=6, n=96, k=3, s=2),
        mbv3_op(ef=6, n=96, k=3, s=1),
        mbv3_op(ef=6, n=96, k=3, s=1),
        op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=576),
        op(reduce_to_1x1, default_size=7, stride=1, padding='VALID'),
        op(slim.conv2d,
           stride=1,
           kernel_size=[1, 1],
           num_outputs=1024,
           normalizer_fn=None)
    ]))


# EdgeTPU friendly variant of MobilenetV3 that uses fused convolutions
# instead of depthwise in the early layers.
V3_EDGETPU = dict(
    defaults=dict(DEFAULTS),
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=(3, 3)),
        mbv3_fused(k=3, s=1, ef=1, n=16),
        mbv3_fused(k=3, s=2, ef=8, n=32),
        mbv3_fused(k=3, s=1, ef=4, n=32),
        mbv3_fused(k=3, s=1, ef=4, n=32),
        mbv3_fused(k=3, s=1, ef=4, n=32),
        mbv3_fused(k=3, s=2, ef=8, n=48),
        mbv3_fused(k=3, s=1, ef=4, n=48),
        mbv3_fused(k=3, s=1, ef=4, n=48),
        mbv3_fused(k=3, s=1, ef=4, n=48),
        mbv3_op(k=3, s=2, ef=8, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=8, n=96, residual=False),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=3, s=1, ef=4, n=96),
        mbv3_op(k=5, s=2, ef=8, n=160),
        mbv3_op(k=5, s=1, ef=4, n=160),
        mbv3_op(k=5, s=1, ef=4, n=160),
        mbv3_op(k=5, s=1, ef=4, n=160),
        mbv3_op(k=3, s=1, ef=8, n=192),
        op(slim.conv2d, stride=1, num_outputs=1280, kernel_size=(1, 1)),
    ])


@slim.add_arg_scope
def mobilenet(input_tensor,
              num_classes=1001,
              depth_multiplier=1.0,
              scope='MobilenetV3',
              conv_defs=None,
              finegrain_classification_mode=False,
              use_groupnorm=False,
              **kwargs):
  """Creates mobilenet V3 network.

  Inference mode is created by default. To create training use training_scope
  below.

  with slim.arg_scope(mobilenet_v3.training_scope()):
     logits, endpoints = mobilenet_v3.mobilenet(input_tensor)

  Args:
    input_tensor: The input tensor
    num_classes: number of classes
    depth_multiplier: The multiplier applied to scale number of
    channels in each layer.
    scope: Scope of the operator
    conv_defs: Which version to create. Could be large/small or
    any conv_def (see mobilenet_v3.py for examples).
    finegrain_classification_mode: When set to True, the model
    will keep the last layer large even for small multipliers. Following
    https://arxiv.org/abs/1801.04381
    it improves performance for ImageNet-type of problems.
      *Note* ignored if final_endpoint makes the builder exit earlier.
    use_groupnorm: When set to True, use group_norm as normalizer_fn.

    **kwargs: passed directly to mobilenet.mobilenet:
      prediction_fn- what prediction function to use.
      reuse-: whether to reuse variables (if reuse set to true, scope
      must be given).
  Returns:
    logits/endpoints pair

  Raises:
    ValueError: On invalid arguments
  """
  if conv_defs is None:
    conv_defs = V3_LARGE
  if 'multiplier' in kwargs:
    raise ValueError('mobilenetv2 doesn\'t support generic '
                     'multiplier parameter use "depth_multiplier" instead.')

  if use_groupnorm:
    conv_defs = copy.deepcopy(conv_defs)
    conv_defs['defaults'] = dict(DEFAULTS_GROUP_NORM)
    conv_defs['defaults'].update({
        (slim.group_norm,): {
            'groups': kwargs.pop('groups', 8)
        }
    })

  if finegrain_classification_mode:
    conv_defs = copy.deepcopy(conv_defs)
    conv_defs['spec'][-1] = conv_defs['spec'][-1]._replace(
        multiplier_func=lambda params, multiplier: params)
  depth_args = {}
  with slim.arg_scope((lib.depth_multiplier,), **depth_args):
    return lib.mobilenet(
        input_tensor,
        num_classes=num_classes,
        conv_defs=conv_defs,
        scope=scope,
        multiplier=depth_multiplier,
        **kwargs)

mobilenet.default_image_size = 224
training_scope = lib.training_scope


@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
  """Creates base of the mobilenet (no pooling and no logits) ."""
  return mobilenet(
      input_tensor, depth_multiplier=depth_multiplier, base_only=True, **kwargs)


def wrapped_partial(func, new_defaults=None,
                    **kwargs):
  """Partial function with new default parameters and updated docstring."""
  if not new_defaults:
    new_defaults = {}
  def func_wrapper(*f_args, **f_kwargs):
    new_kwargs = dict(new_defaults)
    new_kwargs.update(f_kwargs)
    return func(*f_args, **new_kwargs)
  functools.update_wrapper(func_wrapper, func)
  partial_func = functools.partial(func_wrapper, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


large = wrapped_partial(mobilenet, conv_defs=V3_LARGE)
small = wrapped_partial(mobilenet, conv_defs=V3_SMALL)
edge_tpu = wrapped_partial(mobilenet,
                           new_defaults={'scope': 'MobilenetEdgeTPU'},
                           conv_defs=V3_EDGETPU)
edge_tpu_075 = wrapped_partial(
    mobilenet,
    new_defaults={'scope': 'MobilenetEdgeTPU'},
    conv_defs=V3_EDGETPU,
    depth_multiplier=0.75,
    finegrain_classification_mode=True)

# Minimalistic model that does not have Squeeze Excite blocks,
# Hardswish, or 5x5 depthwise convolution.
# This makes the model very friendly for a wide range of hardware
large_minimalistic = wrapped_partial(mobilenet, conv_defs=V3_LARGE_MINIMALISTIC)
small_minimalistic = wrapped_partial(mobilenet, conv_defs=V3_SMALL_MINIMALISTIC)


def _reduce_consecutive_layers(conv_defs, start_id, end_id, multiplier=0.5):
  """Reduce the outputs of consecutive layers with multiplier.

  Args:
    conv_defs: Mobilenet conv_defs.
    start_id: 0-based index of the starting conv_def to be reduced.
    end_id: 0-based index of the last conv_def to be reduced.
    multiplier: The multiplier by which to reduce the conv_defs.

  Returns:
    Mobilenet conv_defs where the output sizes from layers [start_id, end_id],
    inclusive, are reduced by multiplier.

  Raises:
    ValueError if any layer to be reduced does not have the 'num_outputs'
    attribute.
  """
  defs = copy.deepcopy(conv_defs)
  for d in defs['spec'][start_id:end_id+1]:
    d.params.update({
        'num_outputs': np.int(np.round(d.params['num_outputs'] * multiplier))
    })
  return defs


V3_LARGE_DETECTION = _reduce_consecutive_layers(V3_LARGE, 13, 16)
V3_SMALL_DETECTION = _reduce_consecutive_layers(V3_SMALL, 9, 12)


__all__ = ['training_scope', 'mobilenet', 'V3_LARGE', 'V3_SMALL', 'large',
           'small', 'V3_LARGE_DETECTION', 'V3_SMALL_DETECTION']

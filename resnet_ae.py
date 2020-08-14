# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow.compat.v1 as tf

from tensorflow.python.tpu import tpu_function  # pylint: disable=g-direct-tensorflow-import

from resnet_encoder import resnet_encoder_v1
from resnet_decoder import resnet_decoder_v1

FLAGS = flags.FLAGS
BATCH_NORM_EPSILON = 1e-5



def resnet_autoencoder_v1_generator(encoder, decoder, data_format='channels_last'):

  def model(inputs, is_training):
    """Creation of the model graph."""
    # if isinstance(inputs, tuple):
    if FLAGS.use_td_loss and isinstance(inputs, tuple):
      # print('#'*80)
      # print(inputs)
      inputs, augs = inputs
      with tf.variable_scope('encoder'): # variable_scope name_scope
        features = encoder(inputs, is_training)
      
      if data_format == 'channels_last':
        outputs = tf.reduce_mean(features, [1, 2])
      else:
        outputs = tf.reduce_mean(features, [2, 3])
      outputs = tf.identity(outputs, 'final_avg_pool')
      
      h_w = features.get_shape().as_list()[1]
      # print(h_w)

      augs = tf.tile(augs[:,None,None,:], tf.constant([1,h_w,h_w,1]))
      
      features = tf.concat([features, augs], axis=-1)
    
      with tf.variable_scope('decoder'):
        images = decoder(features, is_training)
      
      return outputs, images

    else:
      # augs = None
    
      with tf.variable_scope('encoder'): # variable_scope name_scope
        features = encoder(inputs, is_training)
      
      if data_format == 'channels_last':
        outputs = tf.reduce_mean(features, [1, 2])
      else:
        outputs = tf.reduce_mean(features, [2, 3])
      outputs = tf.identity(outputs, 'final_avg_pool')
      
      # filter_trainable_variables(trainable_variables, after_block=5)
      # add_to_collection(trainable_variables, 'trainable_variables_inblock_')

      return outputs

  return model


def resnet_autoencoder_v1(resnet_depth, width_multiplier,
              cifar_stem=False, data_format='channels_last',
              dropblock_keep_probs=None, dropblock_size=None):
  """Returns the ResNet model for a given size and number of output classes."""

  encoder = resnet_encoder_v1(resnet_depth, 
                              width_multiplier,
                              cifar_stem=cifar_stem, 
                              data_format=data_format,
                              dropblock_keep_probs=dropblock_keep_probs, 
                              dropblock_size=dropblock_size)

  decoder = resnet_decoder_v1(resnet_depth, 
                              width_multiplier,
                              cifar_stem=cifar_stem, 
                              data_format=data_format,
                              dropblock_keep_probs=dropblock_keep_probs, 
                              dropblock_size=dropblock_size)
  
  return resnet_autoencoder_v1_generator(encoder, decoder, data_format=data_format)


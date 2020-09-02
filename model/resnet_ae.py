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

from model.resnet_encoder import resnet_encoder_v1
from model.resnet_decoder import resnet_decoder_v1, learned_metric_v1

FLAGS = flags.FLAGS
BATCH_NORM_EPSILON = 1e-5


def resnet_autoencoder_v1_generator(encoder, decoder, metric, skip, mask_augs=0., greyscale_viz=False, data_format='channels_last'):
  def model(inputs, target_images, is_training):
    """Creation of the model graph."""
    # if isinstance(inputs, tuple):
    assert mask_augs >= 0. and mask_augs <= 1., "mask_augs must be in [0, 1]"
    if FLAGS.use_td_loss and isinstance(inputs, tuple):
      # print('#'*80)
      # print(inputs)
      assert metric is not None, "Metric function is None"
      inputs, augs = inputs
      B = inputs.get_shape().as_list()[0]
      A = augs.get_shape().as_list()[1]
      if mask_augs > 0:
        mask = tf.cast(tf.greater(tf.random.uniform(shape=[B, A], minval=0., maxval=1.), 0.5), augs.dtype)  # noqa
        bias = mask * -1
        augs = (augs * mask) + bias  # Randomly mask out augs for difficulty and code those dims as -1
      with tf.variable_scope('encoder'): # variable_scope name_scope
        features, block_activities = encoder(inputs, is_training=is_training)
      print("Features: ")
      print(features)
      print("---")
      # Global average pool of B 7 7 2048 -> B 2048
      if data_format == 'channels_last':
        outputs = tf.reduce_mean(features, [1, 2])
      else:
        outputs = tf.reduce_mean(features, [2, 3])
      outputs = tf.identity(outputs, 'final_avg_pool')
      print("Outputs: ")
      print(outputs)
      print("---")
      # B 2048

      h_w = features.get_shape().as_list()[1]
      # print(h_w)

      augs = tf.tile(augs[:,None,None,:], tf.constant([1,h_w,h_w,1]))
      print("Augs: ")
      print(augs)
      print("---")
      features = tf.concat([features, augs], axis=-1)
    
      with tf.variable_scope('decoder'):
        recon_images = decoder(
          features,
          block_activities,
          is_training=is_training,
          skip=skip)
      print("Reconstructed images and target images: ")
      print(recon_images)
      print(target_images)
      print("---")
      with tf.variable_scope('metric'):
        # Squash both recon and target images
        recon_images_squash = tf.tanh(recon_images)
        target_images = (target_images * 2) - 1
        Bt = target_images.get_shape().as_list()[0]
        Br = recon_images_squash.get_shape().as_list()[0]
        if Bt == Br:
          # Attractive + repulsive loss
          pass
        elif Bt * 2 == Br:
          # Attractive-only loss
          target_images = tf.concat([target_images, target_images], 0)

        # Differentiable perceptual metric. First reconstruction.
        # both_images = tf.concat([recon_images, target_images], -1)  # B H W 6
        all_images = tf.concat([recon_images_squash, target_images], 0)  # Stack these in batch dim
        metric_all_images = metric(all_images, is_training=is_training)
        # B = metric_all_images.get_shape().as_list()[0]
        metric_all_images = tf.reshape(metric_all_images, [B, -1])
        metric_hidden_r, metric_hidden_t = tf.split(metric_all_images, 2, 0)  # Split these in batch dim

        # Prep recon_images for visualization
        # recon_images = tf.clip_by_value(recon_images, clip_value_min=-5, clip_value_max=5)
        # recon_images = (recon_images + 5) / 10

        recon_mean, recon_std = tf.nn.moments(recon_images, axes=[1, 2], keep_dims=True)
        recon_images = (recon_images - recon_mean) / recon_std
        recon_images = tf.clip_by_value(recon_images, clip_value_min=-5, clip_value_max=5)
        recon_images = (recon_images + 5) / 10
        if greyscale_viz:
          recon_images = tf.image.rgb_to_grayscale(recon_images)
          recon_images = tf.concat([recon_images, recon_images, recon_images], -1)
      print("Embedding output: ")
      print(metric_hidden_t)
      print("---")
      return outputs, recon_images, metric_hidden_r, metric_hidden_t

    else:
      # augs = None
    
      with tf.variable_scope('encoder'): # variable_scope name_scope
        features = encoder(inputs, is_training)
      
      if data_format == 'channels_last':
        print("Features:")
        print(features)
        outputs = tf.reduce_mean(features, [1, 2])
      else:
        outputs = tf.reduce_mean(features, [2, 3])
      outputs = tf.identity(outputs, 'final_avg_pool')
      
      # filter_trainable_variables(trainable_variables, after_block=5)
      # add_to_collection(trainable_variables, 'trainable_variables_inblock_')

      return outputs

  return model


def resnet_autoencoder_v1(encoder_depth, decoder_depth, width_multiplier, metric_channels,  # noqa
              cifar_stem=False, data_format='channels_last',
              dropblock_keep_probs=None, dropblock_size=None,
              mask_augs=0., greyscale_viz=False, skip=True):
  """Returns the ResNet model for a given size and number of output classes."""
  encoder = resnet_encoder_v1(encoder_depth, 
                              width_multiplier,
                              cifar_stem=cifar_stem, 
                              data_format=data_format,
                              dropblock_keep_probs=dropblock_keep_probs, 
                              dropblock_size=dropblock_size)

  decoder = resnet_decoder_v1(decoder_depth=decoder_depth,
                              encoder_depth=encoder_depth,
                              width_multiplier=width_multiplier,
                              cifar_stem=cifar_stem, 
                              data_format=data_format,
                              dropblock_keep_probs=dropblock_keep_probs, 
                              dropblock_size=dropblock_size)

  metric = learned_metric_v1(data_format=data_format, metric_channels=metric_channels) 
  
  return resnet_autoencoder_v1_generator(
    encoder=encoder,
    decoder=decoder,
    metric=metric,
    skip=skip,
    mask_augs=mask_augs,
    greyscale_viz=greyscale_viz,
    data_format=data_format)


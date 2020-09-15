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
"""Contrastive loss functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow.compat.v1 as tf

from tensorflow.compiler.tf2xla.python import xla  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

LARGE_NUM = 1e9


def add_supervised_loss(labels, logits, weights, **kwargs):
  """Compute loss for model and add it to loss collection."""
  return tf.losses.softmax_cross_entropy(labels, logits, weights, **kwargs)

####################
# DON'T USE THIS !
####################
def add_moco_contrastive_loss_2(
                         query,
                         key_pos,
                         key_neg,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0):
  """Compute loss for model.

  Args:
    query: (`Tensor`) of shape (bsz, dim).
    key_pos: positive keys (`Tensor`) of shape (bsz, dim).
    key_neg: queue of negative keys(`Tensor`) of shape (queue_size, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    query = tf.math.l2_normalize(query, -1)
    key_pos = tf.math.l2_normalize(key_pos, -1)
    key_neg = tf.math.l2_normalize(key_neg, -1)
  
  batch_size = tf.shape(query)[0]
  key_pos_batch_size = tf.shape(key_pos)[0]
  queue_size = tf.shape(key_neg)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  if tpu_context is not None:
    # hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
    
    replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels = tf.one_hot(labels_idx, key_pos_batch_size + queue_size)
  
  else:
    labels = tf.one_hot(tf.range(batch_size), key_pos_batch_size + queue_size)
  
  # logits_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))
  # logits_neg = tf.einsum('nc,kc->nk', q_feat, queue)  # nxK

  logits_pos = tf.matmul(query, key_pos, transpose_b=True) / temperature
  logits_neg = tf.matmul(query, key_neg, transpose_b=True) / temperature
  
  logits = tf.concat([logits_pos, logits_neg], 1)

  loss = tf.losses.softmax_cross_entropy(
      labels, logits, weights=weights)
  
  # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  # loss = tf.reduce_mean(loss, name='xentropy-loss')

  return loss, logits, labels


def add_moco_contrastive_loss(
                         query,
                         key_pos,
                         key_neg,
                         hidden_norm=True,
                         temperature=1.0,
                         tpu_context=None,
                         weights=1.0):
  """Compute loss for model.

  Args:
    query: (`Tensor`) of shape (bsz, dim).
    key_pos: positive keys (`Tensor`) of shape (bsz, dim).
    key_neg: queue of negative keys(`Tensor`) of shape (queue_size, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    tpu_context: context information for tpu.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    query = tf.math.l2_normalize(query, -1)
    key_pos = tf.math.l2_normalize(key_pos, -1)
    key_neg = tf.math.l2_normalize(key_neg, -1)
  
  batch_size = tf.shape(query)[0]
  queue_size = tf.shape(key_neg)[0]

  logits_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))
  logits_neg = tf.einsum('nc,kc->nk', q_feat, queue)  # nxK

  logits = tf.concat([logits_pos, logits_neg], 1)

  labels = tf.zeros(batch_size, dtype=tf.int64)  # n
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
  loss = tf.reduce_mean(loss, name='xentropy-loss')

  return loss, logits, labels

def tpu_cross_replica_concat(tensor, tpu_context=None):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    tpu_context: A `TPUContext`. If not set, CPU execution is assumed.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  if tpu_context is None or tpu_context.num_replicas <= 1:
    return tensor

  num_replicas = tpu_context.num_replicas

  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[xla.replica_id()]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])

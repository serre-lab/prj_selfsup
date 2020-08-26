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
"""Model specification for SimCLR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import data.default.data_util as data_util
import model.model_util as model_util
import model.objective as obj_lib

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
import tensorflow

FLAGS = flags.FLAGS


def build_model_fn(model, num_classes, num_train_examples):
  """Build model function."""
  tf.disable_eager_execution()
  def model_fn(features, labels, mode, params=None):
    """Build model and optimizer."""
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Check training mode.
    if FLAGS.train_mode == 'pretrain':
      num_transforms = 1
      if FLAGS.use_td_loss:
        num_transforms += 1
      if FLAGS.use_bu_loss:
        num_transforms += 1

      if FLAGS.fine_tune_after_block > -1:
        raise ValueError('Does not support layer freezing during pretraining,'
                         'should set fine_tune_after_block<=-1 for safety.')
    elif FLAGS.train_mode == 'finetune':
      num_transforms = 1
    else:
      raise ValueError('Unknown train_mode {}'.format(FLAGS.train_mode))
    
    # Split channels, and optionally apply extra batched augmentation.
    features_list = tf.split(
        features, num_or_size_splits=num_transforms, axis=-1)
    
    if FLAGS.use_td_loss:
      target_images = features_list[-1]
      features_list = features_list[:-1]
      # transforms
      thetas_list = tf.split(
        labels['thetas'], num_or_size_splits=num_transforms, axis=-1)
      thetas = tf.concat(thetas_list[:-1], 0)
    else:
      target_images = features_list
    

    if FLAGS.use_blur and is_training and FLAGS.train_mode == 'pretrain':
      features_list, sigmas = data_util.batch_random_blur(
          features_list, FLAGS.image_size, FLAGS.image_size)
      if FLAGS.use_td_loss: 
        sigmas = tf.concat(sigmas, 0)
        thetas = tf.concat([thetas, sigmas[:,None]], 1) 
    else:
      if FLAGS.use_td_loss: 
        sigmas = tf.zeros_like(thetas[:,0])
        thetas = tf.concat([thetas, sigmas[:,None]], 1) 
    
    features = tf.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)
    
    # Base network forward pass.
    with tf.variable_scope('base_model'):
      if FLAGS.train_mode == 'finetune':
        if FLAGS.fine_tune_after_block >= 4:
          # Finetune just supervised (linear) head will not update BN stats.
          model_train_mode = False
      else:
        if FLAGS.use_td_loss:
          viz_features = features
          features = (features, thetas)
        else:
          viz_features = features

        # Pretrain or finetune anything else will update BN stats.
        model_train_mode = is_training

      outputs = model(features, target_images, is_training=model_train_mode)
    
    # Add head and loss.
    if FLAGS.train_mode == 'pretrain':
      tpu_context = params['context'] if 'context' in params else None
      
      if FLAGS.use_td_loss and isinstance(outputs, tuple):
        hiddens, reconstruction, metric_hidden_r, metric_hidden_t = outputs
      else:
        hiddens = outputs
        # reconstruction = tf.zeros_like(target_images)
      if FLAGS.use_td_loss:
        with tf.name_scope('td_loss'):
          if FLAGS.td_loss=='attractive':
            td_loss = obj_lib.add_dot_product_td_attractive_loss(
              metric_hidden_r,
              power=FLAGS.rec_loss_exponent)
            logits_td_con = tf.zeros([params['batch_size'], params['batch_size']])
            labels_td_con = tf.zeros([params['batch_size'], params['batch_size']])

          elif FLAGS.td_loss=='attractive_repulsive':
            td_loss, logits_td_con, labels_td_con = obj_lib.add_dot_product_td_attractive_repulsive_loss(
              metric_hidden_r,
              metric_hidden_t,
              power=FLAGS.rec_loss_exponent,
              temperature=FLAGS.temperature,
              tpu_context=tpu_context if is_training else None)
          else:
            raise 'Unknown loss'
      else:
        logits_td_con = tf.zeros([params['batch_size'], params['batch_size']])
        labels_td_con = tf.zeros([params['batch_size'], params['batch_size']])
        td_loss = tf.zeros([])
      hiddens_proj = model_util.projection_head(hiddens, is_training)

      if FLAGS.use_bu_loss:
        with tf.name_scope('bu_loss'):
          if FLAGS.bu_loss=='attractive':
            bu_loss = obj_lib.add_bu_attractive_loss(
              hiddens_proj,
              hidden_norm=FLAGS.hidden_norm)
            logits_bu_con = tf.zeros([params['batch_size'], params['batch_size']])
            labels_bu_con = tf.zeros([params['batch_size'], params['batch_size']])

          elif FLAGS.bu_loss=='attractive_repulsive':
            bu_loss, logits_bu_con, labels_bu_con = obj_lib.add_bu_attractive_repulsive_loss(
              hiddens_proj,
              hidden_norm=FLAGS.hidden_norm,
              temperature=FLAGS.temperature,
              tpu_context=tpu_context if is_training else None)  
          else:
            raise 'Unknown loss'
      else:
        logits_bu_con = tf.zeros([params['batch_size'], params['batch_size']])
        labels_bu_con = tf.zeros([params['batch_size'], params['batch_size']])
        bu_loss = tf.zeros([])
      logits_sup = tf.zeros([params['batch_size'], num_classes])

    else:
      # contrast_loss = tf.zeros([])
      td_loss = tf.zeros([])
      bu_loss = tf.zeros([])
      logits_td_con = tf.zeros([params['batch_size'], 10])
      labels_td_con = tf.zeros([params['batch_size'], 10])
      logits_bu_con = tf.zeros([params['batch_size'], 10])
      labels_bu_con = tf.zeros([params['batch_size'], 10])
      hiddens = outputs
      hiddens = model_util.projection_head(hiddens, is_training)
      logits_sup = model_util.supervised_head(
          hiddens, num_classes, is_training)
      sup_loss = obj_lib.add_supervised_loss(
          labels=labels['labels'],
          logits=logits_sup,
          weights=labels['mask'])

    # Add weight decay to loss, for non-LARS optimizers.
    model_util.add_weight_decay(adjust_per_optimizer=True)
    
    # reg_loss = tf.losses.get_regularization_losses()

    
    if FLAGS.train_mode == 'pretrain':
      loss =  tf.add_n([td_loss * FLAGS.td_loss_weight, bu_loss * FLAGS.bu_loss_weight] + tf.losses.get_regularization_losses())
    else:
      loss =  tf.add_n([sup_loss] + tf.losses.get_regularization_losses())
           
    # loss = tf.losses.get_total_loss()

    if FLAGS.train_mode == 'pretrain':
      variables_to_train = tf.trainable_variables()
    else:
      collection_prefix = 'trainable_variables_inblock_'
      variables_to_train = []
      for j in range(FLAGS.fine_tune_after_block + 1, 6):
        variables_to_train += tf.get_collection(collection_prefix + str(j))
      assert variables_to_train, 'variables_to_train shouldn\'t be empty!'

    tf.logging.info('===============Variables to train (begin)===============')
    tf.logging.info(variables_to_train)
    tf.logging.info('================Variables to train (end)================')

    learning_rate = model_util.learning_rate_schedule(
        FLAGS.learning_rate, num_train_examples)

    if is_training:
      optimizer = model_util.get_optimizer(learning_rate)
      control_deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      if FLAGS.train_summary_steps > 0:
        control_deps.extend(tf.summary.all_v2_summary_ops())
      with tf.control_dependencies(control_deps):
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_or_create_global_step(),
            var_list=variables_to_train)

      if FLAGS.train_summary_steps > 0:
        # Compute stats for the summary.
        def host_call_fn(
            gs,
            prob_bu_con_t,
            entropy_bu_con_t,
            prob_td_con_t,
            entropy_td_con_t,
            contrast_bu_acc_t,
            contrast_td_acc_t):
          """Training host call. Creates scalar summaries for training metrics.
          This function is executed on the CPU and should not directly reference
          any Tensors in the rest of the `model_fn`. To pass Tensors from the
          model to the `metric_fn`, provide as part of the `host_call`. See
          https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
          for more information.
          Arguments should match the list of `Tensor` objects passed as the second
          element in the tuple passed to `host_call`.
          Args:
            gs: `Tensor with shape `[batch]` for the global_step
            loss: `Tensor` with shape `[batch]` for the training loss.
            lr: `Tensor` with shape `[batch]` for the learning_rate.
            ce: `Tensor` with shape `[batch]` for the current_epoch.
          Returns:
            List of summary ops to run on the CPU host.
          """
          gs = gs[0]
          # Host call fns are executed params['iterations_per_loop'] times after
          # one TPU loop is finished, setting max_queue value to the same as
          # number of iterations will make the summary writer only flush the data
          # to storage once per loop.
          with tf2.summary.create_file_writer(
              FLAGS.model_dir,
              max_queue=params['iterations_per_loop']).as_default():
            with tf2.summary.record_if(True):

              tf2.summary.scalar(
                  'train_bottomup_loss',
                  bu_loss[0],
                  step=gs)
              tf2.summary.scalar(
                  'train_topdown_loss',
                  td_loss[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'train_bottomup_acc',
                  contrast_bu_acc[0],
                  step=gs)
              tf2.summary.scalar(
                  'train_topdown_acc',
                  contrast_td_acc[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'train_label_accuracy',
                  label_acc[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'contrast_bu_entropy',
                  entropy_bu_con[0],
                  step=gs)
              tf2.summary.scalar(
                  'contrast_td_entropy',
                  entropy_td_con[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'learning_rate', learning_rate[0],
                  step=gs)

              # # Images
              # print("Images")
              # print(target_images)
              # print("Features")
              # print(viz_features)
              # print("Reconstruction")
              # print(reconstruction)
              # tf2.summary.image(
              #     'Images',
              #     tf.cast(target_images, tf.float32),
              #     step=step)
              # tf2.summary.image(
              #     'Transformed_images',
              #     tf.cast(viz_features, tf.float32),
              #     step=step)
              # tf2.summary.image(
              #     'Reconstructed_images',
              #     tf.cast(reconstruction, tf.float32),
              #     step=step)
            return tf.summary.all_v2_summary_ops()

        # To log the loss, current learning rate, and epoch for Tensorboard, the
        # summary op needs to be run on the host CPU via host_call. host_call
        # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
        # dimension. These Tensors are implicitly concatenated to
        # [params['batch_size']].
        prob_bu_con = tf.nn.softmax(logits_bu_con)
        entropy_bu_con = - tf.reduce_mean(
            tf.reduce_sum(prob_bu_con * tf.math.log(prob_bu_con + 1e-8), -1))
        prob_td_con = tf.nn.softmax(logits_td_con)
        entropy_td_con = - tf.reduce_mean(
            tf.reduce_sum(prob_td_con * tf.math.log(prob_td_con + 1e-8), -1))
        contrast_bu_acc = tf.equal(
            tf.argmax(labels_bu_con, 1), tf.argmax(logits_bu_con, axis=1))
        contrast_bu_acc = tf.reduce_mean(tf.cast(contrast_bu_acc, tf.float32))
        contrast_td_acc = tf.equal(
            tf.argmax(labels_td_con, 1), tf.argmax(logits_td_con, axis=1))
        contrast_td_acc = tf.reduce_mean(tf.cast(contrast_td_acc, tf.float32))

        summary_writer = tf2.summary.create_file_writer(FLAGS.model_dir)

        label_acc = tf.equal(
            tf.argmax(labels['labels'], 1), tf.argmax(logits_sup, axis=1))
        label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
        step = tf.train.get_or_create_global_step()

        gs_t = tf.reshape(step, [1])
        prob_bu_con_t = tf.reshape(prob_bu_con, [1])
        entropy_bu_con_t = tf.reshape(entropy_bu_con, [1])
        prob_td_con_t = tf.reshape(prob_td_con, [1])
        entropy_td_con_t = tf.reshape(entropy_td_con, [1])
        contrast_bu_acc_t = tf.reshape(contrast_bu_acc, [1])
        contrast_td_acc_t = tf.reshape(contrast_td_acc, [1])

        host_call = (
          host_call_fn,
          [
            gs_t,
            prob_bu_con_t,
            entropy_bu_con_t,
            prob_td_con_t,
            entropy_td_con_t,
            contrast_bu_acc_t,
            contrast_td_acc_t,])
      
      if FLAGS.checkpoint:
        def scaffold_fn():
          """Scaffold function to restore non-logits vars from checkpoint."""
          tf.logging.info('*'*180)
          tf.logging.info('Initializing from checkpoint %s'%FLAGS.checkpoint)
          tf.logging.info('*'*180)

          tf.train.init_from_checkpoint(
              FLAGS.checkpoint,
              {v.op.name: v.op.name
               for v in tf.global_variables(FLAGS.variable_schema)})

          if FLAGS.zero_init_logits_layer:
            # Init op that initializes output layer parameters to zeros.
            output_layer_parameters = [
                var for var in tf.trainable_variables() if var.name.startswith(
                    'head_supervised')]
            tf.logging.info('Initializing output layer parameters %s to zero',
                            [x.op.name for x in output_layer_parameters])
            with tf.control_dependencies([tf.global_variables_initializer()]):
              init_op = tf.group([
                  tf.assign(x, tf.zeros_like(x))
                  for x in output_layer_parameters])
            return tf.train.Scaffold(init_op=init_op)
          else:
            return tf.train.Scaffold()
      else:
        scaffold_fn = None

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          train_op=train_op,
          loss=loss,
          host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:

      def metric_fn(logits_sup, labels_sup, logits_bu_con, labels_bu_con, 
                    logits_td_con, labels_td_con, mask,
                    **kws):
        """Inner metric function."""
        metrics = {k: tf.metrics.mean(v, weights=mask)
                   for k, v in kws.items()}
        metrics['label_top_1_accuracy'] = tf.metrics.accuracy(
            tf.argmax(labels_sup, 1), tf.argmax(logits_sup, axis=1),
            weights=mask)
        metrics['label_top_5_accuracy'] = tf.metrics.recall_at_k(
            tf.argmax(labels_sup, 1), logits_sup, k=5, weights=mask)
        
        metrics['bottomup_top_1_accuracy'] = tf.metrics.accuracy(
            tf.argmax(labels_bu_con, 1), tf.argmax(logits_bu_con, axis=1),
            weights=mask)
        metrics['bottomup_top_5_accuracy'] = tf.metrics.recall_at_k(
            tf.argmax(labels_bu_con, 1), logits_bu_con, k=5, weights=mask)

        metrics['topdown_top_1_accuracy'] = tf.metrics.accuracy(
            tf.argmax(labels_td_con, 1), tf.argmax(logits_td_con, axis=1),
            weights=mask)
        metrics['topdown_top_5_accuracy'] = tf.metrics.recall_at_k(
            tf.argmax(labels_td_con, 1), logits_td_con, k=5, weights=mask)
        return metrics

      metrics = {
          'logits_sup': logits_sup,
          'labels_sup': labels['labels'],
          'logits_bu_con': logits_bu_con,
          'logits_td_con': logits_td_con,
          'labels_bu_con': labels_bu_con,
          'labels_td_con': labels_td_con,
          'mask': labels['mask'],
          'td_loss': tf.fill((params['batch_size'],), bu_loss),
          'bu_loss': tf.fill((params['batch_size'],), td_loss),
          'regularization_loss': tf.fill((params['batch_size'],),
                                         tf.losses.get_regularization_loss()),
      }

      return tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, metrics),
          scaffold_fn=None)

  return model_fn
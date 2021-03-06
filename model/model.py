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

FLAGS = flags.FLAGS


def build_model_fn(model, num_classes, num_train_examples):
  """Build model function."""
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
      if FLAGS.train_mode == 'pretrain':  # Fix for fine-tuning/eval
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
        # thetas = tf.zeros([target_images.get_shape().as_list()[0], 11]) 

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
        reconstruction = features

      if FLAGS.use_td_loss:
        with tf.name_scope('td_loss'):
          if FLAGS.td_loss=='attractive':
            td_loss, logits_td_con, labels_td_con = obj_lib.td_attractive_loss(
              reconstruction=metric_hidden_r,
              target=metric_hidden_t,
              temperature=FLAGS.temperature,
              tpu_context=tpu_context if is_training else None)
            logits_td_con = tf.zeros([params['batch_size'], params['batch_size']])
            labels_td_con = tf.zeros([params['batch_size'], params['batch_size']])
          elif FLAGS.td_loss=='attractive_repulsive':
            td_loss, logits_td_con, labels_td_con = obj_lib.td_attractive_repulsive_loss(
              reconstruction=metric_hidden_r,
              target=metric_hidden_t,
              temperature=FLAGS.temperature,
              tpu_context=tpu_context if is_training else None)
          else:
            raise NotImplementedError("Error at TD loss {}".format(FLAGS.td_loss))
      else:
        # No TD loss
        logits_td_con = tf.zeros([params['batch_size'], params['batch_size']])
        labels_td_con = tf.zeros([params['batch_size'], params['batch_size']])
        td_loss = 0.
      hiddens_proj = model_util.projection_head(hiddens, is_training)

      if FLAGS.use_bu_loss:
        with tf.name_scope('bu_loss'):
          if FLAGS.bu_loss=='attractive':
            bu_loss, logits_bu_con, labels_bu_con = obj_lib.attractive_loss(
              hiddens_proj,
              temperature=FLAGS.temperature,
              hidden_norm=FLAGS.hidden_norm)
            logits_bu_con = tf.zeros([params['batch_size'], params['batch_size']])
            labels_bu_con = tf.zeros([params['batch_size'], params['batch_size']])

          elif FLAGS.bu_loss=='attractive_repulsive':
            bu_loss, logits_bu_con, labels_bu_con = obj_lib.attractive_repulsive_loss(
              hiddens_proj,
              hidden_norm=FLAGS.hidden_norm,
              temperature=FLAGS.temperature,
              tpu_context=tpu_context if is_training else None)  
          else:
            raise NotImplementedError('Unknown loss')
      else:
        # No BU loss
        logits_bu_con = tf.zeros([params['batch_size'], params['batch_size']])
        labels_bu_con = tf.zeros([params['batch_size'], params['batch_size']])
        bu_loss = 0.
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
      sup_loss = obj_lib.supervised_loss(
          labels=labels['labels'],
          logits=logits_sup,
          weights=labels['mask'])

    # Add weight decay to loss, for non-LARS optimizers.
    model_util.add_weight_decay(adjust_per_optimizer=True)
    
    # reg_loss = tf.losses.get_regularization_losses()

    
    if FLAGS.train_mode == 'pretrain':
      print(bu_loss)
      print(td_loss)
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
      
      if FLAGS.train_summary_steps > 0:
        # Compute stats for the summary.
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
        
        label_acc = tf.equal(
            tf.argmax(labels['labels'], 1), tf.argmax(logits_sup, axis=1))
        label_acc = tf.reduce_mean(tf.cast(label_acc, tf.float32))
        

        def host_call_fn(gs, g_l, bu_l, td_l, c_bu_a, c_td_a, l_a, c_e_bu, c_e_td, lr, tar_im, viz_f, rec_im):
          gs = gs[0]
          with tf2.summary.create_file_writer(
              FLAGS.model_dir,
              max_queue=FLAGS.checkpoint_steps).as_default():
            with tf2.summary.record_if(True):
              tf2.summary.scalar(
                  'total_loss',
                  g_l[0],
                  step=gs)
                  
              tf2.summary.scalar(
                  'train_bottomup_loss',
                  bu_l[0],
                  step=gs)

              tf2.summary.scalar(
                  'train_topdown_loss',
                  td_l[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'train_bottomup_acc',
                  c_bu_a[0],
                  step=gs)
              tf2.summary.scalar(
                  'train_topdown_acc',
                  c_td_a[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'train_label_accuracy',
                  l_a[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'contrast_bu_entropy',
                  c_e_bu[0],
                  step=gs)
              tf2.summary.scalar(
                  'contrast_td_entropy',
                  c_e_td[0],
                  step=gs)
              
              tf2.summary.scalar(
                  'learning_rate', lr[0],
                  step=gs)

              # print("Images")
              # print(target_images)
              # print("Features")
              # print(viz_features)
              # print("Reconstruction")
              # print(reconstruction)
              tf2.summary.image(
                  'Images',
                  tar_im[0],
                  step=gs)
              tf2.summary.image(
                  'Transformed images',
                  viz_f[0],
                  step=gs)
              tf2.summary.image(
                  'Reconstructed images',
                  rec_im[0],
                  step=gs)

            return tf.summary.all_v2_summary_ops()


        n_images = 4
        if isinstance(target_images, list):
          target_images = target_images[0]
        image_shape = target_images.get_shape().as_list()

        tar_im = tf.reshape(tf.cast(target_images[:n_images], tf.float32), [1, n_images] + image_shape[1:])
        viz_f = tf.reshape(tf.cast(viz_features[:n_images], tf.float32), [1, n_images] + image_shape[1:])
        rec_im = tf.reshape(tf.cast(reconstruction[:n_images], tf.float32), [1, n_images] + image_shape[1:])
        
        gs = tf.reshape(tf.train.get_global_step(), [1])
        
        g_l = tf.reshape(loss, [1])

        bu_l = tf.reshape(bu_loss, [1])
        td_l = tf.reshape(td_loss, [1])

        c_bu_a = tf.reshape(contrast_bu_acc, [1])
        c_td_a = tf.reshape(contrast_td_acc, [1])
        
        l_a = tf.reshape(label_acc, [1])
        c_e_bu = tf.reshape(entropy_bu_con, [1])
        c_e_td = tf.reshape(entropy_td_con, [1])
        
        lr = tf.reshape(learning_rate, [1])
        
        host_call = (host_call_fn, [gs, g_l, bu_l, td_l, c_bu_a, c_td_a, l_a, c_e_bu, c_e_td, lr, tar_im, viz_f, rec_im])
        
      else:
        host_call=None

      optimizer = model_util.get_optimizer(learning_rate)
      control_deps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      # if FLAGS.train_summary_steps > 0:
      #   control_deps.extend(tf.summary.all_v2_summary_ops())
      with tf.control_dependencies(control_deps):
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_or_create_global_step(),
            var_list=variables_to_train)
      
      
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
          scaffold_fn=scaffold_fn, 
          host_call=host_call
          )

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
        # metrics['bottomup_top_5_accuracy'] = tf.metrics.recall_at_k(
        #     tf.argmax(labels_bu_con, 1), logits_bu_con, k=5, weights=mask)

        metrics['topdown_top_1_accuracy'] = tf.metrics.accuracy(
            tf.argmax(labels_td_con, 1), tf.argmax(logits_td_con, axis=1),
            weights=mask)
        # metrics['topdown_top_5_accuracy'] = tf.metrics.recall_at_k(
        #     tf.argmax(labels_td_con, 1), logits_td_con, k=5, weights=mask)
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
          host_call=None,
          scaffold_fn=None)

  return model_fn

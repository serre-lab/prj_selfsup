# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Train a ResNet-50 model on ImageNet on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from common import inference_warmup
from common import tpu_profiler_hook
from hyperparameters import common_hparams_flags
from hyperparameters import common_tpu_flags
from hyperparameters import flags_to_params
from hyperparameters import params_dict
from official.resnet import imagenet_input
from official.resnet import lars_util
from official.resnet import resnet_model
from official.resnet.configs import resnet_config
from tensorflow.core.protobuf import rewriter_config_pb2  # pylint: disable=g-direct-tensorflow-import

common_tpu_flags.define_common_tpu_flags()
common_hparams_flags.define_common_hparams_flags()

FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

flags.DEFINE_string(
    'bigtable_project', None,
    'The Cloud Bigtable project.  If None, --gcp_project will be used.')
flags.DEFINE_string(
    'bigtable_instance', None,
    'The Cloud Bigtable instance to load data from.')
flags.DEFINE_string(
    'bigtable_table', 'imagenet',
    'The Cloud Bigtable table to load data from.')
flags.DEFINE_string(
    'bigtable_train_prefix', 'train_',
    'The prefix identifying training rows.')
flags.DEFINE_string(
    'bigtable_eval_prefix', 'validation_',
    'The prefix identifying evaluation rows.')
flags.DEFINE_string(
    'bigtable_column_family', 'tfexample',
    'The column family storing TFExamples.')
flags.DEFINE_string(
    'bigtable_column_qualifier', 'example',
    'The column name storing TFExamples.')

def _verify_non_empty_string(value, field_name):
  """Ensures that a given proposed field value is a non-empty string.
  Args:
    value:  proposed value for the field.
    field_name:  string name of the field, e.g. `project`.
  Returns:
    The given value, provided that it passed the checks.
  Raises:
    ValueError:  the value is not a string, or is a blank string.
  """
  if not isinstance(value, str):
    raise ValueError(
        'Bigtable parameter "%s" must be a string.' % field_name)
  if not value:
    raise ValueError(
        'Bigtable parameter "%s" must be non-empty.' % field_name)
  return value


def _select_tables_from_flags():
  """Construct training and evaluation Bigtable selections from flags.
  Returns:
    [training_selection, evaluation_selection]
  """
  project = _verify_non_empty_string(
      FLAGS.bigtable_project or FLAGS.gcp_project,
      'project')
  instance = _verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
  table = _verify_non_empty_string(FLAGS.bigtable_table, 'table')
  train_prefix = _verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                          'train_prefix')
  eval_prefix = _verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                         'eval_prefix')
  column_family = _verify_non_empty_string(FLAGS.bigtable_column_family,
                                           'column_family')
  column_qualifier = _verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                              'column_qualifier')
  return [  # pylint: disable=g-complex-comprehension
      imagenet_input.BigtableSelection(
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


def main(unused_argv):

  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu if (FLAGS.tpu or params.use_tpu) else '',
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project)

  if params.use_async_checkpointing:
    save_checkpoints_steps = None
  else:
    save_checkpoints_steps = max(5000, params.iterations_per_loop)

  assert (params.precision == 'bfloat16' or
          params.precision == 'float32'), (
              'Invalid value for precision parameter; '
              'must be bfloat16 or float32.')
  tf.logging.info('Precision: %s', params.precision)
  
  
  use_bfloat16 = params.precision == 'bfloat16'

  # Input pipelines are slightly different (with regards to shuffling and
  # preprocessing) between training and evaluation.
  if FLAGS.bigtable_instance:
    tf.logging.info('Using Bigtable dataset, table %s', FLAGS.bigtable_table)
    select_train, select_eval = _select_tables_from_flags()
    imagenet_train, imagenet_eval = [
        imagenet_input.ImageNetBigtableInput(  # pylint: disable=g-complex-comprehension
            is_training=is_training,
            use_bfloat16=use_bfloat16,
            transpose_input=params.transpose_input,
            selection=selection,
            augment_name=FLAGS.augment_name,
            randaug_num_layers=FLAGS.randaug_num_layers,
            randaug_magnitude=FLAGS.randaug_magnitude)
        for (is_training, selection) in [(True,
                                          select_train), (False, select_eval)]
    ]
  else:
    if FLAGS.data_dir == FAKE_DATA_DIR:
      tf.logging.info('Using fake dataset.')
    else:
      tf.logging.info('Using dataset: %s', FLAGS.data_dir)
    imagenet_train, imagenet_eval = [
        imagenet_input.ImageNetInput(  # pylint: disable=g-complex-comprehension
            is_training=is_training,
            data_dir=FLAGS.data_dir,
            transpose_input=params.transpose_input,
            cache=params.use_cache and is_training,
            image_size=params.image_size,
            num_parallel_calls=params.num_parallel_calls,
            include_background_label=(params.num_label_classes == 1001),
            use_bfloat16=use_bfloat16,
            augment_name=FLAGS.augment_name,
            randaug_num_layers=FLAGS.randaug_num_layers,
            randaug_magnitude=FLAGS.randaug_magnitude)
        for is_training in [True, False]
    ]

  steps_per_epoch = params.num_train_images // params.train_batch_size
  eval_steps = params.num_eval_images // params.eval_batch_size

  if FLAGS.mode == 'eval':

    # Run evaluation when there's a new checkpoint
    for ckpt in tf.train.checkpoints_iterator(
        FLAGS.model_dir, timeout=FLAGS.eval_timeout):
      tf.logging.info('Starting to evaluate.')
      try:
        start_timestamp = time.time()  # This time will include compilation time
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=eval_steps,
            checkpoint_path=ckpt)
        elapsed_time = int(time.time() - start_timestamp)
        tf.logging.info('Eval results: %s. Elapsed seconds: %d',
                        eval_results, elapsed_time)

        # Terminate eval job when final checkpoint is reached
        current_step = int(os.path.basename(ckpt).split('-')[1])
        if current_step >= params.train_steps:
          tf.logging.info(
              'Evaluation finished after training step %d', current_step)
          break

      except tf.errors.NotFoundError:
        # Since the coordinator is on a different job than the TPU worker,
        # sometimes the TPU worker does not finish initializing until long after
        # the CPU job tells it to start evaluating. In this case, the checkpoint
        # file could have been deleted already.
        tf.logging.info(
            'Checkpoint %s no longer exists, skipping checkpoint', ckpt)

  else:   # FLAGS.mode == 'train' or FLAGS.mode == 'train_and_eval'
    try:
      current_step = tf.train.load_variable(FLAGS.model_dir,
                                            tf.GraphKeys.GLOBAL_STEP)
    except (TypeError, ValueError, tf.errors.NotFoundError):
      current_step = 0
    steps_per_epoch = params.num_train_images // params.train_batch_size
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current'
                    ' step %d.',
                    params.train_steps,
                    params.train_steps / steps_per_epoch,
                    current_step)

    start_timestamp = time.time()  # This time will include compilation time

    if FLAGS.mode == 'train':
      hooks = []
      if params.use_async_checkpointing:
        try:
          from tensorflow.contrib.tpu.python.tpu import async_checkpoint  # pylint: disable=g-import-not-at-top
        except ImportError as e:
          logging.exception(
              'Async checkpointing is not supported in TensorFlow 2.x')
          raise e

        hooks.append(
            async_checkpoint.AsyncCheckpointSaverHook(
                checkpoint_dir=FLAGS.model_dir,
                save_steps=max(5000, params.iterations_per_loop)))
      if FLAGS.profile_every_n_steps > 0:
        hooks.append(
            tpu_profiler_hook.TPUProfilerHook(
                save_steps=FLAGS.profile_every_n_steps,
                output_dir=FLAGS.model_dir, tpu=FLAGS.tpu)
            )
      resnet_classifier.train(
          input_fn=imagenet_train.input_fn,
          max_steps=params.train_steps,
          hooks=hooks)

    else:
      assert FLAGS.mode == 'train_and_eval'
      while current_step < params.train_steps:
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              params.train_steps)
        resnet_classifier.train(
            input_fn=imagenet_train.input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                        next_checkpoint, int(time.time() - start_timestamp))

        # Evaluate the model on the most recent model in --model_dir.
        # Since evaluation happens in batches of --eval_batch_size, some images
        # may be excluded modulo the batch size. As long as the batch size is
        # consistent, the evaluated images are also consistent.
        tf.logging.info('Starting to evaluate.')
        eval_results = resnet_classifier.evaluate(
            input_fn=imagenet_eval.input_fn,
            steps=params.num_eval_images // params.eval_batch_size)
        tf.logging.info('Eval results at step %d: %s',
                        next_checkpoint, eval_results)

      elapsed_time = int(time.time() - start_timestamp)
      tf.logging.info('Finished training up to step %d. Elapsed seconds %d.',
                      params.train_steps, elapsed_time)



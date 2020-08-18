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

import collections
# from absl import app
from absl import flags
# from absl import logging
# import numpy as np
# import tensorflow.compat.v1 as tf
# import tensorflow.compat.v2 as tf2

# from common import inference_warmup
# from common import tpu_profiler_hook
# from hyperparameters import common_hparams_flags
# from hyperparameters import common_tpu_flags
# from hyperparameters import flags_to_params
# from hyperparameters import params_dict
# from official.resnet import imagenet_input
# from official.resnet import lars_util
# from official.resnet import resnet_model
# from official.resnet.configs import resnet_config
# from tensorflow.core.protobuf import rewriter_config_pb2 

# common_tpu_flags.define_common_tpu_flags()
# common_hparams_flags.define_common_hparams_flags()

FLAGS = flags.FLAGS

FAKE_DATA_DIR = 'gs://cloud-tpu-test-datasets/fake_imagenet'

# flags.DEFINE_string(
#     'bigtable_project', None,
#     'The Cloud Bigtable project.  If None, --gcp_project will be used.')
# flags.DEFINE_string(
#     'bigtable_instance', None,
#     'The Cloud Bigtable instance to load data from.')
# flags.DEFINE_string(
#     'bigtable_table', 'imagenet',
#     'The Cloud Bigtable table to load data from.')
# flags.DEFINE_string(
#     'bigtable_train_prefix', 'train_',
#     'The prefix identifying training rows.')
# flags.DEFINE_string(
#     'bigtable_eval_prefix', 'validation_',
#     'The prefix identifying evaluation rows.')
# flags.DEFINE_string(
#     'bigtable_column_family', 'tfexample',
#     'The column family storing TFExamples.')
# flags.DEFINE_string(
#     'bigtable_column_qualifier', 'example',
#     'The column name storing TFExamples.')

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

# Defines a selection of data from a Cloud Bigtable.
BigtableSelection = collections.namedtuple('BigtableSelection', [
    'project',
    'instance',
    'table',
    'prefix',
    'column_family',
    'column_qualifier',
])


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
  return [
      # imagenet_input.BigtableSelection(
      BigtableSelection(
          project=project,
          instance=instance,
          table=table,
          prefix=p,
          column_family=column_family,
          column_qualifier=column_qualifier)
      for p in (train_prefix, eval_prefix)
  ]


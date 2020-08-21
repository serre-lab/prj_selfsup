"""
Testing the brand new datasets from tensorflow community for experimenting on
ImageNet2012 dataset.
We identify several problems while working with ImageNet dataset:
1. The dataset is not easy to download. Credentials (email) of some well known
organization/university is required to get the dowanload link.
2. The huge size if dataset, namely "ILSVRC2012_img_train.tar" -> 138Gb
and "ILSVRC2012_img_val.tar" -> 7Gb
3. Dowanloading and preparing the dataset for some ML algorithm takes a good
chunck of time.
4. No easy way to parallelize the consumption of data across GPU for model
training
--------------------------------------------------------------------------------
In this script, we show that tensorflow dataset library tries to solve most of
the above mentioned problems.
"""

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import numpy as np

from absl import app
from absl import flags

import scipy
import scipy.misc
from scipy.misc import imsave

import data.imagenet_simclr.data as imagenet_input

from data.imagenet_simclr import data_util
tf.enable_eager_execution()


FLAGS = flags.FLAGS


flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')


def main(argv):
    print("Demonstration for using Imagenet2012 dataset with tensorflow datset")
    
    buffer_size=8*1024*1024
    dataset = tf.data.TFRecordDataset(filenames=['gs://imagenet_data/train/train-00995-of-01024'])
    print(dataset)
    
    
    def data_parser(value):

        # parsed = tf.parse_single_example(value, keys_to_features)
        # image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

        # image = tf.image.decode_jpeg(image_bytes, channels=3)
        # image = tf.image.resize_bicubic([image], [224, 224])[0]

        # return image

        keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
        }

        parsed = tf.parse_single_example(value, keys_to_features)
        image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
        label = tf.cast(
            tf.reshape(parsed['image/class/label'], shape=[]), dtype=tf.int32)

        preprocess_fn_pretrain = data_util.get_preprocess_fn(True, is_pretrain=True)
        preprocess_fn_finetune = data_util.get_preprocess_fn(True, is_pretrain=False) 
        # preprocess_fn_target = data_util.get_preprocess_target_fn() 
        num_classes = 1000 # builder.info.features['label'].num_classes
        
        if FLAGS.train_mode == 'pretrain':
            xs = []
            for _ in range(2):  # Two transformations
                xs.append(preprocess_fn_pretrain(image_bytes)[0])
            image = tf.concat(xs, -1)
            label = tf.zeros([num_classes])
        else:
            image = preprocess_fn_finetune(image_bytes)[0]
            label = tf.one_hot(label, num_classes)
        
        return image, {'labels': label, 'mask': 1.0} # label, thetas, 1.0
        
    # dataset = dataset.apply(
    #     tf.data.experimental.map_and_batch(
    #         dataset_parser,
    #         batch_size=1,
    #         num_parallel_batches=1,
    #         drop_remainder=True))
    
    dataset = dataset.map(data_parser)
    
    # im = dataset.take(1)
    im, _ = next(iter(dataset))
    im = im.numpy()

    #iterator = dataset.make_one_shot_iterator()
    #res = iterator.get_next()

    print("Image_shape", im.shape)

    imsave("imagenet_examples/image1.png", im[:,:,:3])
    imsave("imagenet_examples/image2.png", im[:,:,3:])
    
if __name__ == "__main__":
    # main()
    app.run(main)

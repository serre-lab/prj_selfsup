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

import scipy
import scipy.misc
from scipy.misc import imsave

import data.imagenet_simclr.data as imagenet_input
tf.enable_eager_execution()

def main():
    print("Demonstration for using Imagenet2012 dataset with tensorflow datset")
    
    buffer_size=8*1024*1024
    dataset = tf.data.TFRecordDataset(filenames=['gs://imagenet_data/train/train-00995-of-01024'])
    print(dataset)
    
    
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, ''),
        'image/class/label': tf.FixedLenFeature([], tf.int64, -1),
    }

    def data_parser(value):

        parsed = tf.parse_single_example(value, keys_to_features)
        image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.resize_bicubic([image], [224, 224])[0]

        return image

    # dataset = dataset.apply(
    #     tf.data.experimental.map_and_batch(
    #         dataset_parser,
    #         batch_size=1,
    #         num_parallel_batches=1,
    #         drop_remainder=True))
    
    dataset = dataset.map(data_parser)
    
    # im = dataset.take(1)
    im = next(iter(dataset))
    im = im.numpy()

    #iterator = dataset.make_one_shot_iterator()
    #res = iterator.get_next()

    print("Image_shape", im.shape)

    imsave("image1.png", im[:,:,:3])
    
if __name__ == "__main__":
    main()

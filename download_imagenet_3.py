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

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from scipy.misc import imsave

tf.enable_eager_execution()

def main():
    print("Demonstration for using Imagenet2012 dataset with tensorflow datset")
    # List all the datasets provided in the tensorflow_datasets
    # print(tfds.list_builders())
    # Step 1: get a dataset builder for the required dataset
    dataset_name = "imagenet2012"
    if dataset_name in tfds.list_builders():
        imagenet_dataset_builder = tfds.builder(dataset_name)
        print("retrived " + dataset_name + " builder")
    else:
        return
    # get all the information regarding dataset
    print(imagenet_dataset_builder.info)
    print("Image shape", imagenet_dataset_builder.info.features['image'].shape)
    print("class",imagenet_dataset_builder.info.features['label'].num_classes)
    print("classname",imagenet_dataset_builder.info.features['label'].names)
    print("NrTrain",imagenet_dataset_builder.info.splits['train'].num_examples)
    print("Val",imagenet_dataset_builder.info.splits['validation'].num_examples)
    # Download and prepare the dataset internally
    # The dataset should be downloaded to ~/tensorflow-datasets/download
    # but for Imagenet case, we need to manually download the dataset and
    # specify the manual_dir where the downloaded files are kept.
    manual_dataset_dir = "/data/datasets"
    # The download_and_prepare function will assume that two files namely
    # ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar are present in
    # directory manual_dataset_dir + "/manual/imagenet2012"
    imagenet_download_config = tfds.download.DownloadConfig(
                                                manual_dir = manual_dataset_dir)
    # Conditionally, download config can be passed as second argument.
    imagenet_dataset_builder.download_and_prepare(
                                    download_dir = manual_dataset_dir)
    # Once this is complete (that just pre-process without downloading anything)
    # it will create a director "~/tensorflow_datasets/imagenet2012/2.0.0"
    # having 1000 train tfrecords and 5 validation tfrecords in addition to some
    # bookkeeping json and label txt files.

    # now, we get the tf.data.Dataset structure which tensorflow data-pipeline
    # understands and process in tf graph.
    imagenet_train = imagenet_dataset_builder.as_dataset(split=tfds.Split.TRAIN)
    assert isinstance(imagenet_train, tf.data.Dataset)
    imagenet_validation = imagenet_dataset_builder.as_dataset(
                                                    split=tfds.Split.VALIDATION)
    assert isinstance(imagenet_validation, tf.data.Dataset)

    # Now we can peek into the sample images present in the dataset with take
    (imagenet_example,) = imagenet_train.take(1) # returns a dictionary
    img, label = imagenet_example["image"], imagenet_example["label"]
    # img and label are constant tensors, with numpy field containing numpy arry
    print("Image_shape", img.numpy().shape)
    print("Label_shape", label.numpy().shape)
    # print out the image file on the disk, and print the corresponding label
    imsave("image.png", img.numpy())
    print("label", label.numpy())

    # From the tf.data.Datasets imagenet_train and imagenet_validation,
    # the input pipeline can be created for tf training and serving.

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
# file: write_resized_image_back.py
# author: JinTian
# time: 01/03/2017 10:56 AM
# Copyright 2017 JinTian. All Rights Reserved.
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
import tensorflow as tf
import numpy as np
import os
from PIL import Image

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("image_number", 300, "Number of images in your tfrecord, default is 300.")
flags.DEFINE_integer("class_number", 5, "Number of class in your dataset/label.txt, default is 3.")
flags.DEFINE_integer("image_height", 299, "Height of the output image after crop and resize. Default is 299.")
flags.DEFINE_integer("image_width", 299, "Width of the output image after crop and resize. Default is 299.")


def tf_records_walker(tf_records_dir):
    all_files = [os.path.abspath(os.path.join(tf_records_dir, i_)) for i_ in os.listdir(tf_records_dir)]
    if all_files:
        print("[INFO] %s files were found under current folder. " % len(all_files))
        print("[INFO] Please be noted that only files end with '*.tf record' will be load!")
        tf_records_files = [i for i in all_files if os.path.basename(i).endswith('tfrecord')]
        if tf_records_files:
            for i_ in tf_records_files:
                print("[INFO] Find tf records file: {}".format(i_))
            return tf_records_files
        else:
            raise FileNotFoundError("Can not find any records file.")
    else:
        raise Exception("Cannot find any file under this path.")


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class ImageObject:
    def __init__(self):
        self.image = tf.Variable([], dtype=tf.string)
        self.height = tf.Variable([], dtype=tf.int64)
        self.width = tf.Variable([], dtype=tf.int64)
        self.filename = tf.Variable([], dtype=tf.string)
        self.label = tf.Variable([], dtype=tf.int32)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "image/encoded": tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/filename": tf.FixedLenFeature([], tf.string),
        "image/class/label": tf.FixedLenFeature([], tf.int64), })

    image_encoded = features["image/encoded"]
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    current_image_object_ = ImageObject()

    current_image_object_.image = tf.image.resize_image_with_crop_or_pad(image_raw, FLAGS.image_height,
                                                                         FLAGS.image_width)
    current_image_object_.height = features["image/height"]
    current_image_object_.width = features["image/width"]
    current_image_object_.filename = features["image/filename"]
    current_image_object_.label = tf.cast(features["image/class/label"], tf.int32)

    return current_image_object_

if __name__ == '__main__':
    filename_queue = tf.train.string_input_producer(
        tf_records_walker(tf_records_dir='./tiny_5_tfrecords/'),
        shuffle=True)

    current_image_object = read_and_decode(filename_queue)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        print("Write cropped and resized image to the folder './resized_image'")
        for i in range(FLAGS.image_number):  # number of examples in your tfrecord
            pre_image, pre_label = sess.run([current_image_object.image, current_image_object.label])
            img = Image.fromarray(pre_image, "RGB")
            if not os.path.isdir("./resized_image/"):
                os.mkdir("./resized_image")
            img.save(os.path.join("./resized_image/class_" + str(pre_label) + "_Index_" + str(i) + ".jpeg"))
            if i % 10 == 0:
                print("%d images in %d has finished!" % (i, FLAGS.image_number))
        print("Complete!!")
        coord.request_stop()
        coord.join(threads)
        sess.close()

    print("cd to current directory, the folder 'resized_image' should contains %d images with %dx%d size." % (
        FLAGS.image_number, FLAGS.image_height, FLAGS.image_width))


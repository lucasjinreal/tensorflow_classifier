# -*- coding: utf-8 -*-
"""
file: read_tfrecords.py
author: JinTian
time: 28/02/2017 9:20 AM
--------------------
Copyright 2017 JInTian. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow as tf
import numpy as np
import os
import skimage.io as io
import matplotlib.pyplot as plt


tf.app.flags.DEFINE_string('record_file', os.path.abspath('./tiny_5_tfrecords/validation-00000-of-00002'),
                           'Record File')

FLAGS = tf.app.flags.FLAGS


def read_tfrecords():
    print('reading from tfrecords file {}'.format(FLAGS.record_file))
    record_iterator = tf.python_io.tf_record_iterator(path=FLAGS.record_file)

    with tf.Session() as sess:
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            height_ = int(example.features.feature['image/height'].int64_list.value[0])
            width_ = int(example.features.feature['image/width'].int64_list.value[0])
            channels_ = int(example.features.feature['image/channels'].int64_list.value[0])

            image_bytes_ = example.features.feature['image/encoded'].bytes_list.value[0]
            label_ = int(example.features.feature['image/class/label'].int64_list.value[0])
            text_bytes_ = example.features.feature['image/class/text'].bytes_list.value[0]

            # image_array_ = np.fromstring(image_bytes_, dtype=np.uint8).reshape((height_, width_, 3))
            image_ = tf.image.decode_jpeg(image_bytes_)
            image_ = sess.run(image_)
            text_ = text_bytes_.decode('utf-8')

            print('tfrecords height {0}, width {1}, channels {2}: '.format(height_, width_, channels_))
            print('decode image shape: ', image_.shape)
            print('label text: ', text_)
            print('label: ', label_)
            # io.imshow(image_)
            # plt.show()


def main(_):
    read_tfrecords()

if __name__ == '__main__':
    tf.app.run()
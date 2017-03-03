# -*- coding: utf-8 -*-
# file: evaluate_tiny5.py
# author: JinTian
# time: 02/03/2017 5:26 PM
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
# ------------------------------------------------------------------------
import os.path
import sys
import time
import tensorflow as tf
import inference
import cv2
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import alexnet


tf.app.flags.DEFINE_integer('target_image_height', 150, 'train input image height')
tf.app.flags.DEFINE_integer('target_image_width', 150, 'train input image width')
tf.app.flags.DEFINE_integer('num_classes', 5, 'all categories in data set.')

tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size of training.')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'epochs of training.')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'learning rate of training.')

tf.app.flags.DEFINE_string('tf_record_dir', './data/tiny_5_tfrecords/', 'tf record file dir.')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoints', 'slim train log dir.')
tf.app.flags.DEFINE_boolean('restore', True, 'if restore or not.')

tf.app.flags.DEFINE_string('label_file', './data/labels.txt', 'label txt file path.')

FLAGS = tf.app.flags.FLAGS


def tf_records_walker(tf_records_dir, is_train):
    all_files = [os.path.abspath(os.path.join(tf_records_dir, i_)) for i_ in os.listdir(tf_records_dir)]
    if all_files:
        print("[INFO] %s files were found under current folder. " % len(all_files))
        print("[INFO] Please be noted that only files end with '*.tf record' will be load!")
        tf_records_files_train = [i for i in all_files if os.path.basename(i).endswith('tfrecord') and
                                  os.path.basename(i).startswith('train')]
        tf_records_files_valid = [i for i in all_files if os.path.basename(i).endswith('tfrecord') and
                                  os.path.basename(i).startswith('validation')]
        if is_train and tf_records_files_train:
            for i_ in tf_records_files_train:
                print('[INFO] loaded train tf_records file at: {}'.format(i_))
            return tf_records_files_train
        elif not is_train and tf_records_files_valid:
            for i_ in tf_records_files_valid:
                print('[INFO] loaded evaluate tf_records file at: {}'.format(i_))
            return tf_records_files_valid
        else:
            raise FileNotFoundError("Can not find any records file.")
    else:
        raise Exception("Cannot find any file under this path.")


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
        })
    height = tf.cast(features['image/height'], dtype=tf.int32)
    width = tf.cast(features['image/width'], dtype=tf.int32)
    channels = tf.cast(features['image/channels'], dtype=tf.int32)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.resize_image_with_crop_or_pad(
        image=image,
        target_height=FLAGS.target_image_height,
        target_width=FLAGS.target_image_width,
    )
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label


def inputs(is_train, batch_size, one_hot_labels):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            tf_records_walker(tf_records_dir=FLAGS.tf_record_dir, is_train=is_train),
            num_epochs=None,
            shuffle=True)
        image, label = read_and_decode(filename_queue)

        if one_hot_labels:
            print('[INFO] One hot label used 0 as reserve value for None, so total dim is num_classes+1')
            label = tf.one_hot(indices=label, depth=FLAGS.num_classes+1, dtype=tf.int32)
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=2,
            capacity=10 + 3 * batch_size,
            min_after_dequeue=10)

        return images, sparse_labels


def evaluate():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    images, labels = inputs(is_train=False, batch_size=FLAGS.batch_size, one_hot_labels=True)

    endpoints = inference.network(images, labels, num_classes=FLAGS.num_classes + 1)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)

        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        print('[INFO] evaluating start...')
        try:
            while not coord.should_stop():
                start_time = time.time()
                loss_val, accuracy, step = sess.run([endpoints['loss'], endpoints['accuracy'], endpoints[
                    'global_step']])
                end_time = time.time()
                print("[INFO] Epoch %d:  cost %.3fs,  validation_loss %.7f,  validation_accuracy %.3f" %
                      (step, end_time - start_time, loss_val, accuracy))

        except tf.errors.OutOfRangeError:
            print('[INFO] evaluation finished.')
        except KeyboardInterrupt:
            print('[INFO] aborted evaluation.')
        finally:
            coord.request_stop()
            coord.join(threads)


def predict_single_image(image_path):
    image = cv2.imread(image_path, cv2.CAP_MODE_RGB)
    print('[INFO] Predict image from {}'.format(os.path.abspath(image_path)))
    print(image)
    if image is not None:
        image = tf.image.resize_image_with_crop_or_pad(
            image=image,
            target_height=FLAGS.target_image_height,
            target_width=FLAGS.target_image_width,
        )
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        image = tf.reshape(image, [-1, FLAGS.target_image_height, FLAGS.target_image_width, 3])

        images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.target_image_height, FLAGS.target_image_width, 3])
        endpoints = inference.network(images, num_classes=FLAGS.num_classes+1)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            image = image.eval()
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print('[INFO] loaded model weights...')

                start_time = time.time()
                predict, predict_top3, predict_index_top3 = sess.run([endpoints['output_score'], endpoints[
                    'predict_val_top3'], endpoints['predict_index_top3']], feed_dict={images: image})
                end_time = time.time()
                print("[INFO] Successfully predicted, cost %.4f s" % (
                    end_time-start_time))
                label_names = get_label_text()
                for i in range(len(predict_index_top3[0])):
                    print('[INFO] predict as: %s,  probabilities: %.4f' %
                          (label_names[predict_index_top3[0][i]], predict_top3[0][i]))
                image = tf.cast((image + 0.5)*255, tf.int32)
                image = image.eval()
                print(image)
                cv2.imshow('image', image[0])
                cv2.waitKey(0)
            else:
                print('[INFO] Can not find checkpoints.')
    else:
        print('[INFO] Cannot open image or not find it.')


def get_label_text():
    label_dict = {}
    with open(FLAGS.label_file, 'r+') as f:
        i = 1
        for l in f.readlines():
            label_dict[i] = l.strip()
            i = (i + 1)
    return label_dict


def main(_):
    # evaluate()
    predict_single_image('test_1.jpeg')


if __name__ == '__main__':
    tf.app.run()
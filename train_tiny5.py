# -*- coding: utf-8 -*-
"""
file: train_tiny5.py
author: JinTian
time: 28/02/2017 6:54 PM
--------------------
Copyright 2017 JinTian. All Rights Reserved.

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
import os.path
import sys
import time
import tensorflow as tf
import inference
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

tf.app.flags.DEFINE_integer('max_steps', 1000, 'training max steps.')
tf.app.flags.DEFINE_integer('save_steps', 30, 'training save steps.')
tf.app.flags.DEFINE_integer('eval_steps', 1, 'training eval steps.')

tf.app.flags.DEFINE_string('checkpoints_prefix', 'tiny5', 'prefix for checkpoints file.')
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


def run_training():
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.mkdir(FLAGS.checkpoint_dir)
    images, labels = inputs(is_train=True, batch_size=FLAGS.batch_size, one_hot_labels=True)

    endpoints = inference.network(images, labels, num_classes=FLAGS.num_classes + 1)
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run(init_op)

        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("[INFO] restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])
        print('[INFO] training start...')
        try:
            while not coord.should_stop():
                start_time = time.time()
                _, loss_val, train_summary, step, accuracy = sess.run(
                    [endpoints['train_op'], endpoints['loss'], endpoints['merged_summary_op'],
                     endpoints['global_step'], endpoints['accuracy']])
                end_time = time.time()
                print("[INFO] Epoch %d:  cost %.3fs,  loss %.7f,  accuracy %.3f" % (step, end_time - start_time,
                                                                                    loss_val, accuracy))

                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    accuracy_val, test_summary, step = sess.run(
                        [endpoints['accuracy'], endpoints['merged_summary_op'], endpoints['global_step']])
                    print('[INFO] Eval a batch in train data.')
                    print('[INFO] Epoch {0}:  accuracy {1}'.format(step, accuracy_val))
                if step % FLAGS.save_steps == 1:
                    print('[INFO] Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoints_prefix),
                               global_step=endpoints['global_step'])
        except tf.errors.OutOfRangeError:
            print('[INFO] train finished.')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoints_prefix), global_step=endpoints[
                'global_step'])
        except KeyboardInterrupt:
            print('[INFO] Interrupt manually, try saving checkpoint for now...')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.checkpoints_prefix), global_step=endpoints[
                'global_step'])
            print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(step))
        finally:
            coord.request_stop()
            coord.join(threads)


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
# -*- coding: utf-8 -*-
# file: inference.py
# author: JinTian
# time: 28/02/2017 7:29 PM
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
import tensorflow as tf
import tensorflow.contrib.slim as slim


def lenet(images, num_classes, activation_fn):
    net = slim.layers.conv2d(images, 20, [5, 5], scope='conv1')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.layers.conv2d(net, 50, [5, 5], scope='conv2')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.layers.flatten(net, scope='flatten3')
    net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
    net = slim.layers.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='fully_connected5')
    return net


def simple_cnn(images, num_classes):
    net = slim.conv2d(images, 32, [3, 3], 1, padding='SAME', scope='conv1')
    net = slim.max_pool2d(net, [2, 2], [2, 2], padding='SAME', scope='pool1')
    net = slim.conv2d(net, 64, [3, 3], padding='SAME', scope='conv2')
    net = slim.max_pool2d(net, [2, 2], [2, 2], padding='SAME', scope='conv2')
    net = slim.flatten(net)
    net = slim.fully_connected(net, num_classes, activation_fn=None)
    return net


def network(images, labels=None, num_classes=10):
    endpoints = {}

    net = slim.conv2d(images, 32, [3, 3], 1, padding='SAME')
    net = slim.max_pool2d(net, [2, 2], [2, 2], padding='SAME')
    net = slim.conv2d(net, 64, [3, 3], padding='SAME')
    net = slim.max_pool2d(net, [2, 2], [2, 2], padding='SAME')
    net = slim.flatten(net)
    net = slim.fully_connected(net, num_classes, activation_fn=None)

    global_step = tf.Variable(initial_value=0)

    if labels is not None:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=labels))
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(labels, 1)), tf.float32))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged_summary_op = tf.summary.merge_all()
    else:
        output_score = tf.nn.softmax(net)
        predict_val_top3, predict_index_top3 = tf.nn.top_k(output_score, k=3)

    if labels is not None:
        endpoints['labels'] = labels
        endpoints['train_op'] = train_op
        endpoints['loss'] = loss
        endpoints['accuracy'] = accuracy
        endpoints['merged_summary_op'] = merged_summary_op
    else:
        endpoints['global_step'] = global_step
        endpoints['output_score'] = output_score
        endpoints['predict_val_top3'] = predict_val_top3
        endpoints['predict_index_top3'] = predict_index_top3
    return endpoints

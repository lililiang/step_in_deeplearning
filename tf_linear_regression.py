#!/usr/bin/env python 
# -*- coding:utf-8 -*- 
'''
 * @Author: liliangasd@gmail.com 
 * @Date:   2018-11-08 21:47:46 
 * @Last Modified by:   liliangasd@gmail.com 
 * @Last Modified time: 2018-11-08 21:47:46 
 * @Desc: 
'''

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import random
import sys

def weight_variable(shape):
    initial = tf.truncated_normal(shape, dtype = tf.float32, stddev = 1e-1)
    return tf.Variable(initial, trainable = True, name = 'weights')

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape, dtype = tf.float32)
    return tf.Variable(initial, trainable = True, name = 'biases')

def display_result(los_list):
    # display loss
    plt.figure(1)
    plt.plot(los_list, 'r')
    plt.grid(True)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()

def generate_samples(true_w, true_b, num_examples = 1000):
    true_w = np.array([true_w])
    row, col = true_w.shape
    true_w = true_w.transpose()

    features = np.random.normal(scale=1, size = num_examples * row * col)
    features = features.reshape([num_examples, col])

    labels = np.dot(features, true_w) + true_b

    labels = labels + np.random.normal(scale=0.01, size=labels.shape)
    labels = labels.reshape([num_examples, 1])

    return features, labels

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) # 样本的读取顺序是随机的。
    for i in range(0, num_examples, batch_size):
        j = np.array(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels[j] 

true_w = [2, -3.4, 7]
true_b = [4.2]

# 模型定义

learning_rate = 0.1
batch_size = 20
num_examples = 1000

features, labels = generate_samples(true_w, true_b, num_examples)

x_input = tf.placeholder(tf.float32, [None, 3])
y_input = tf.placeholder(tf.float32, [None, 1])

para_W = weight_variable([3, 1])
para_b = bias_variable([1])

y_predict = tf.matmul(x_input, para_W) + para_b

loss = tf.nn.l2_loss(y_input - y_predict)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)      

# 训练过程

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    los_list = []
    item_step = 0
    for batch_xs, batch_ys in data_iter(batch_size, features, labels):
        _, y_pre, los = sess.run([train_step, y_predict, loss], feed_dict={x_input: batch_xs, y_input: batch_ys})
        item_step += 1
        print('[Step %d]: loss: %f' % (item_step, los))

        los_list.append(los)

    pre_w, pre_b = sess.run([para_W, para_b])
    print(pre_w)
    print(pre_b)
    
    display_result(los_list)
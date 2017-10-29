# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/28
# Brief: 

import tensorflow as tf


class Network(object):
    def __init__(self, batch_size, words, input_data_placeholder):
        self.batch_size = batch_size
        self.words = words
        self.input_data_placeholder = input_data_placeholder

    def model(self, model_type='lstm', rnn_size=128, num_layers=2):
        if model_type == 'rnn':
            cell_fun = tf.nn.rnn_cell.BasicRNNCell
        elif model_type == 'gru':
            cell_fun = tf.nn.rnn_cell.GRUCell
        elif model_type == 'lstm':
            cell_fun = tf.nn.rnn_cell.BasicLSTMCell

        cell = cell_fun(rnn_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

        initial_state = cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [rnn_size, len(self.words) + 1])
            softmax_b = tf.get_variable("softmax_b", [len(self.words) + 1])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [len(self.words) + 1, rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, self.input_data_placeholder)

        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
        output = tf.reshape(outputs, [-1, rnn_size])

        logits = tf.matmul(output, softmax_w) + softmax_b
        probs = tf.nn.softmax(logits)
        return logits, last_state, probs, cell, initial_state

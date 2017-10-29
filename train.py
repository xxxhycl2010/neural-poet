# -*- coding: utf-8 -*-
# Author: XuMing <shibing624@126.com>
# Data: 17/10/28
# Brief: 
import numpy as np
import tensorflow as tf

import config
from utils import logger, word_count, get_train_data
from network import Network


class Train(object):
    def __init__(self, train_data=None,
                 test_data=None,
                 model_type='lstm',
                 dim_nn=128,
                 num_layers=2,
                 batch_size=64,
                 num_passes=10,
                 num_epoch_to_save_model=7,
                 save_model_path="./",
                 use_gpu=False):
        self.train_data = train_data
        self.test_data = test_data
        self.model_type = model_type
        self.dim_nn = dim_nn
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_passes = num_passes
        self.num_epoch_to_save_model = num_epoch_to_save_model
        self.save_model_path = save_model_path
        self.use_gpu = use_gpu

    def get_words(self):
        words = word_count(self.train_data)
        # 取常用字
        return words[:len(words)] + (' ',)

    def get_train_vector(self):
        words = self.get_words()
        # 每个字映射为一个数字ID
        word_num_map = dict(zip(words, range(len(words))))
        # 把诗转换为向量形式
        to_num = lambda word: word_num_map.get(word, len(words))
        train_vector = [list(map(to_num, poetry)) for poetry in self.train_data]
        # [[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
        # [339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
        # ....]
        n_chunk = len(train_vector) // self.batch_size
        return word_num_map, train_vector, n_chunk

    def get_batches(self, word_num_map, train_vector, n_chunk):
        # 每次取64(batch_size)首诗进行训练
        x_batches = []
        y_batches = []
        for i in range(n_chunk):
            start_index = i * self.batch_size
            end_index = start_index + self.batch_size

            batches = train_vector[start_index:end_index]
            length = max(map(len, batches))
            xdata = np.full((self.batch_size, length), word_num_map[' '], np.int32)
            for row in range(self.batch_size):
                xdata[row, :len(batches[row])] = batches[row]
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]
            """
            xdata             ydata
            [6,2,4,6,9]       [2,4,6,9,9]
            [1,4,2,8,5]       [4,2,8,5,5]
            """
            x_batches.append(xdata)
            y_batches.append(ydata)
        return x_batches, y_batches

    # 训练
    def run(self):
        words = self.get_words()
        input_data = tf.placeholder(tf.int32, [self.batch_size, None])
        output_data = tf.placeholder(tf.int32, [self.batch_size, None])
        network = Network(batch_size=self.batch_size, words=words, input_data_placeholder=input_data)
        logits, last_state, _, _, _ = network.model(model_type=self.model_type, rnn_size=self.dim_nn,
                                                    num_layers=self.num_layers)
        targets = tf.reshape(output_data, [-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets],
                                                                  [tf.ones_like(targets, dtype=tf.float32)],
                                                                  len(words))
        cost = tf.reduce_mean(loss)
        learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        # get parameters
        word_num_map, train_vector, n_chunk = self.get_train_vector()
        x_batches, y_batches = self.get_batches(word_num_map, train_vector, n_chunk)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables())

            for epoch in range(self.num_passes):
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
                n = 0
                for batch in range(n_chunk):
                    train_loss, _, _ = sess.run([cost, last_state, train_op],
                                                feed_dict={input_data: x_batches[n], output_data: y_batches[n]})
                    n += 1
                    logger.info('epoch:%d batch:%d train_loss:%s' % (epoch, batch, str(train_loss)))
                    if epoch % self.num_epoch_to_save_model == 0:
                        saver.save(sess, self.save_model_path, global_step=epoch)


if __name__ == "__main__":
    config = config.config
    # 数据预处理
    train_data = get_train_data(config['train_data_paths'])
    # train
    trainer = Train(train_data=train_data,
                    model_type=config['model_type'],
                    dim_nn=config['dim_nn'],
                    num_layers=config['num_layers'],
                    batch_size=config['batch_size'],
                    num_passes=config['num_passes'],
                    num_epoch_to_save_model=config['num_epoch_to_save_model'],
                    save_model_path=config['save_model_path'])
    trainer.run()

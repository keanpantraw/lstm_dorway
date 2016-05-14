
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import itertools
import math
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
import collections
from six.moves import range
from six.moves.urllib.request import urlretrieve

import bigram_batch


class Lstm(object):

    def __init__(self, batch_size, embeddings, num_nodes, num_unrollings,
                 temperature=0.2, graph=None):
        self._graph = graph or tf.get_default_graph()
        self._batch_size = batch_size
        self._num_nodes = num_nodes
        self._num_unrollings = num_unrollings
        self._embeddings = embeddings.embeddings_tensor()
        self._embedding_size = embeddings.embedding_size
        self._temperature = temperature
        self._scope_name = "lstm_" + str(id(self))
        with tf.variable_scope(self._scope_name):
            self._define_lstm()

    def _define_parameters(self):
        self._embeddings = tf.Variable(self._embeddings, trainable=False)

        self._saved_output = tf.Variable(
            tf.zeros([self._batch_size, self._num_nodes]), trainable=False)
        self._saved_state = tf.Variable(
            tf.zeros([self._batch_size, self._num_nodes]), trainable=False)

        self._w = tf.Variable(tf.truncated_normal([self._num_nodes,
                                                   self._embedding_size], -0.1, 0.1))
        self._b = tf.Variable(tf.zeros([self._embedding_size]))

        self._iW = tf.Variable(tf.truncated_normal(
            [self._embedding_size, self._num_nodes * 4], -0.1, 0.1))
        self._oW = tf.Variable(tf.truncated_normal(
            [self._num_nodes, self._num_nodes * 4], -0.1, 0.1))
        self._B = tf.Variable(tf.zeros([1, self._num_nodes * 4]))

    def _define_inputs(self):
        self._train_data = []
        self._embedded_train_data = []
        for _ in range(self._num_unrollings + 1):
            batch = tf.placeholder(tf.int32, shape=[self._batch_size])
            self._train_data.append(batch)
            embedded_batch = tf.nn.embedding_lookup(self._embeddings, batch)
            self._embedded_train_data.append(embedded_batch)

        self._train_inputs = self._embedded_train_data[:self._num_unrollings]
        self._train_labels = self._embedded_train_data[1:]

    def _define_lstm_chain(self):
        self._outputs = []
        output = self._saved_output
        state = self._saved_state
        for i in self._train_inputs:
            output, state = self._lstm_cell(i, output, state)
            self._outputs.append(output)

        with tf.control_dependencies([self._saved_output.assign(output),
                                      self._saved_state.assign(state)]):
            self._logits = tf.nn.xw_plus_b(
                tf.concat(0, self._outputs), self._w, self._b)
            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                self._logits, tf.concat(0, self._train_labels)))

    def _define_sample_output(self):
        self._sample_input = tf.placeholder(tf.int32, shape=[1])
        self._embedded_sample_input = tf.nn.embedding_lookup(
            self._embeddings, self._sample_input)
        saved_sample_output = tf.Variable(tf.zeros([1, self._num_nodes]))
        saved_sample_state = tf.Variable(tf.zeros([1, self._num_nodes]))
        reset_sample_state = tf.group(
            saved_sample_output.assign(tf.zeros([1, self._num_nodes])),
            saved_sample_state.assign(tf.zeros([1, self._num_nodes])))
        sample_output, sample_state = self._lstm_cell(
            self._embedded_sample_input, saved_sample_output, saved_sample_state)
        with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                      saved_sample_state.assign(sample_state)]):
            self._sample_embedded_prediction = tf.nn.softmax(
                tf.nn.xw_plus_b(sample_output, self._w, self._b))
            diff = self._embeddings - self._sample_embedded_prediction
            distance = tf.sqrt(tf.reduce_sum(diff ** 2, 1))
            inverse = (tf.reduce_max(distance) - distance) / self._temperature
            prediction = tf.nn.softmax(tf.expand_dims(inverse, 0))
            self._sample_prediction = tf.squeeze(prediction)

    def _define_lstm(self):
        with self._graph.as_default():
            self._define_parameters()

            self._define_inputs()

            self._define_lstm_chain()

            global_step = tf.Variable(0)
            self._learning_rate = tf.train.exponential_decay(
                10.0, global_step, 5000, 0.1, staircase=True)
            self._optimizer = tf.train.GradientDescentOptimizer(
                self._learning_rate)
            gradients, v = zip(*self._optimizer.compute_gradients(self._loss))
            gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
            self._optimizer = self._optimizer.apply_gradients(
                zip(gradients, v), global_step=global_step)

            self._train_prediction = tf.nn.softmax(self._logits)

            self._define_sample_output()

    def _lstm_cell(self, i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        gate = tf.matmul(i, self._iW) + tf.matmul(o, self._oW) + self._B
        input_gate = tf.sigmoid(gate[:, 0:self._num_nodes])
        forget_gate = tf.sigmoid(gate[:, self._num_nodes:2 * self._num_nodes])
        update = tf.tanh(gate[:, 2 * self._num_nodes:3 * self._num_nodes])
        output_gate = tf.sigmoid(gate[:, 3 * self._num_nodes:])
        state = forget_gate * state + input_gate * update
        return output_gate * tf.tanh(state), state

    def train(self, session, text, num_steps):
        summary_frequency = 100
        is_own = lambda x: x.name.startswith(self._scope_name)
        tf.initialize_variables(filter(is_own, tf.all_variables())).run()
        print('Initialized')
        mean_loss = 0
        generator = bigram_batch.BigramGenerator(text, self._batch_size,
                                                 num_unrollings=self._num_unrollings)
        for step in range(num_steps):
            batches = generator.next()
            feed_dict = dict()
            for i in range(self._num_unrollings + 1):
                feed_dict[self._train_data[i]] = batches[i]
            _, l, predictions, lr = session.run(
                [self._optimizer, self._loss,
                 self._train_prediction, self._learning_rate], feed_dict=feed_dict)
            mean_loss += l
            if step % summary_frequency == 0:
                if step > 0:
                    mean_loss = mean_loss / summary_frequency
                # The mean loss is an estimate of the loss over the last few
                # batches.
                print(
                    'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))

    def say(self, length, start_from=None):
        bigram_id = start_from or random.randint(
            0, bigram_batch.vocabulary_size - 1)
        text = bigram_batch.id2bigram(bigram_id)

        def sample(distribution):
            """Sample one element from a distribution assumed
            to be an array of normalized probabilities.
            """
            r = random.uniform(0, 1)
            s = 0
            for i in range(len(distribution)):
                s += distribution[i]
                if s >= r:
                    return i
            return len(distribution) - 1

        for _ in range(length):
            prediction = self._sample_prediction.eval(
                {self._sample_input: [bigram_id]})
            bigram_id = sample(prediction)
            text += bigram_batch.id2bigram(bigram_id)
        return text

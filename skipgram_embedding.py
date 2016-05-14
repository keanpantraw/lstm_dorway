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


class SkipgramEmbeddings(object):
    """ Defines trainable graph of skipgram word embeddings"""

    def __init__(self, batch_size, embedding_size, num_sampled, num_skips,
                 graph=None):
        self._graph = graph or tf.get_default_graph()
        self._batch_size = batch_size
        self._num_skips = num_skips
        self.embedding_size = embedding_size
        self._scope_name = "skipgram_embedding_" + str(id(self))
        with tf.variable_scope(self._scope_name):
            self._define_embeddings(bigram_batch.vocabulary_size,
                                    embedding_size, num_sampled)

    def embeddings_value(self):
        """Returns evaluated embeddings"""
        return self._normalized_embeddings.eval()

    def embeddings_tensor(self):
        """Returns embeddings tensor.
        Useful to wire this graph into another one"""
        return self._normalized_embeddings

    def train(self, session, text, num_steps):
        """ Train embeddings on given text"""
        generator = bigram_batch.SkipgramGenerator(
            text, self._batch_size, self._num_skips)

        is_own = lambda x: x.name.startswith(self._scope_name)
        tf.initialize_variables(filter(is_own, tf.all_variables())).run()
        print('Initialized')
        average_loss = 0
        step = 0
        while step < num_steps:
            batches_labels = zip(*generator.next())
            for step, (batch, label) in enumerate(batches_labels, step):
                feed_dict = {self._train_dataset: batch,
                             self._train_labels: label.reshape(label.shape[0], 1)}

                _, l = session.run(
                    [self._optimizer, self._loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last
                    # 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0

    def _define_embeddings(self, vocabulary_size, embedding_size, num_sampled):
        """ Defines graph of word embeddings training"""
        with self._graph.as_default(), tf.device('/cpu:0'):
            self._train_dataset = tf.placeholder(
                tf.int32, shape=[self._batch_size])
            self._train_labels = tf.placeholder(
                tf.int32, shape=[self._batch_size, 1])

            embeddings = tf.Variable(tf.random_uniform([vocabulary_size,
                                                        embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size,
                                                               embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            embed = tf.nn.embedding_lookup(embeddings, self._train_dataset)

            self._loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(softmax_weights,
                                                                   softmax_biases,
                                                                   embed,
                                                                   self._train_labels, num_sampled,
                                                                   vocabulary_size))

            self._optimizer = tf.train.AdagradOptimizer(
                1.0).minimize(self._loss)

            self._normalized_embeddings = tf.nn.softmax(embeddings)

import unittest


import bigram_batch
import lstm
import skipgram_embedding
import tensorflow as tf


class EmbeddingsTest(unittest.TestCase):

    def text_bigramming(self):
        text = 'test1 test2 test3 bro!'
        bigrams = bigram_batch.BigramGenerator(text, 4)
        assert bigrams.next() != None

    def test_skipgramming(self):
        text = 'test1 test2 test3 bro! these skipgrams are working!'
        skipgrams = bigram_batch.SkipgramGenerator(text, 3, 2)
        batches, labels = skipgrams.next()
        assert batches != None
        assert labels != None
        assert len(batches) == 4
        assert len(labels) == 4

    def test_embeddings(self):
        skipgram_embeddings = skipgram_embedding.SkipgramEmbeddings(
            2, 128, 64, 2)
        with tf.Session() as session:
            skipgram_embeddings.train(session, "huj huj huj huj huj", 100)
            embeddings = skipgram_embeddings.embeddings_value()

    def lstm(self):
        skipgram_embeddings = skipgram_embedding.SkipgramEmbeddings(
            2, 128, 64, 2)
        lstm_network = lstm.Lstm(4, skipgram_embeddings, 64, 4)
        text = "huj huj huj huj huj"
        with tf.Session() as session:
            skipgram_embeddings.train(session, text, 2000)
            lstm_network.train(session, text, 1000)
            print(lstm_network.say(10))
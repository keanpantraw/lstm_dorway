import itertools
import numpy as np
import string

__all__ = ['BigramGenerator', 'SkipgramGenerator',
           'id2bigram', 'vocabulary_size', 'all_bigrams']

letters = sorted(set((string.ascii_letters + string.digits + " ").lower()))

all_bigrams = {x[0] + x[1]: i for i, x in
               enumerate(itertools.product(letters, letters))}
inversed_bigrams = {i: x for x, i in all_bigrams.items()}

vocabulary_size = len(all_bigrams)


def id2bigram(i):
    return inversed_bigrams[i]


def text_to_bigram_sequence(text):
    text = ''.join([c for c in text if c in letters])
    if len(text) % 2 != 0:
        text += " "
    sequence = [text[i:i + 2] for i in range(0, len(text), 2)]
    return np.array([all_bigrams[x] for x in sequence], dtype=np.int16)


class BatchGenerator(object):

    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(
            shape=(self._batch_size), dtype=np.int16)
        for b in range(self._batch_size):
            batch[b] = self._text[self._cursor[b]]
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def to_skipgrams(batches):
    """ This converts given number of batches to skipgrams
        returns skipgram_batches, skipgram_labels
    """
    assert len(batches) % 2 != 0

    skip_window = len(batches) // 2

    return ([batches[skip_window]] * (len(batches) - 1),
            [b for i, b in enumerate(batches) if i != skip_window])


class BigramGenerator(object):
    """Generates batches of bigrams for given text"""

    def __init__(self, text, batch_size, num_unrollings=0):
        self._bigrams = text_to_bigram_sequence(text)
        self._generator = BatchGenerator(
            self._bigrams, batch_size, num_unrollings)

    def next(self):
        return self._generator.next()


class SkipgramGenerator(object):
    """Generates batches/labels of skipgrams for given text"""

    def __init__(self, text, batch_size, num_skips):
        self._bigrams = text_to_bigram_sequence(text)
        self._generator = BatchGenerator(
            self._bigrams, batch_size, num_skips * 2)

    def next(self):
        return to_skipgrams(self._generator.next())

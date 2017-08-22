from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Categorical, Multinomial, Normal


class test_evaluate_class(tf.test.TestCase):

  def test_n_samples(self):
    with self.test_session():
    #   x = Multinomial(total_count=1.0, probs=tf.constant([0.48, 0.51, 0.01]),
    #                   sample_shape=5)
    #   x_data = tf.constant(
    #       [[0, 1, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]],
    #       dtype=x.dtype.as_numpy_dtype)
    #   self.assertAllClose(
    #       0.6,
    #       ed.evaluate('categorical_accuracy', {x: x_data}, n_samples=3))

      x = Multinomial(total_count=5.0, probs=tf.constant([0.48, 0.51, 0.01]))
      x_data = tf.constant([2, 3, 0], dtype=x.dtype.as_numpy_dtype)
      self.assertAllClose(
          1.0,
          ed.evaluate('categorical_accuracy', {x: x_data}, n_samples=1))


if __name__ == '__main__':
  tf.test.main()

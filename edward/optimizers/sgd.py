from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class KucukelbirOptimizer:

  """
  # TODO: add me
  """

  def __init__(self, t, delta, eta):
    self.t = t
    self.delta = delta
    self.eta = eta

  def apply_gradients(self, n, s_n, grads_and_vars):
    ops = []
    for i, (grad, var) in enumerate(grads_and_vars):
      updated_s_n = s_n[i].assign( (self.t * grad**2) + (1 - self.t) * s_n[i] )

      p_n_first = self.eta * n**(-.5 + self.delta)
      p_n_second = (1 + tf.sqrt(updated_s_n[i]))**(-1)
      p_n = p_n_first * p_n_second

      updated_var = var.assign_add(-p_n * grad)
      ops.append(updated_var)
    return ops

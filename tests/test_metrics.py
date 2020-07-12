import sys
sys.path.append('.')

import tensorflow as tf

from utils.metrics import MAPEavg


def test_MAPEavg_1d_ident():

    x = tf.Variable([1, 2, 3])
    y = tf.Variable([1, 2, 3])

    metric = MAPEavg([1, 3])
    metric.update_state(x, y)
    res = metric.result()

    tf.assert_equal(res, 0.0)

def test_MAPEavg_1d_pos():

    x = tf.Variable([1, 1, 1])
    y = tf.Variable([2, 2, 2])

    metric = MAPEavg([1, 3])
    metric.update_state(x, y)
    res = metric.result()

    tf.assert_equal(res, 1.0)

def test_MAPEavg_1d_neg():

    x = tf.Variable([2, 2, 2])
    y = tf.Variable([1, 1, 1])

    metric = MAPEavg([1, 3])
    metric.update_state(x, y)
    res = metric.result()

    tf.assert_equal(res, 0.5)

def test_MAPEavg_2d_ident():

    x = tf.Variable([[1, 2, 3], [1, 2, 3]])
    y = tf.Variable([[1, 2, 3], [1, 2, 3]])

    metric = MAPEavg([2, 3])
    metric.update_state(x, y)
    res = metric.result()

    tf.assert_equal(res, 0.0)

def test_MAPEavg_2d_pos():

    x = tf.Variable([[1, 1, 1], [1, 1, 1]])
    y = tf.Variable([[2, 2, 2], [2, 2, 2]])

    metric = MAPEavg([2, 3])
    metric.update_state(x, y)
    res = metric.result()

    tf.assert_equal(res, 1.0)

def test_MAPEavg_2d_neg():

    x = tf.Variable([[2, 2, 2], [2, 2, 2]])
    y = tf.Variable([[1, 1, 1], [1, 1, 1]])

    metric = MAPEavg([2, 3])
    metric.update_state(x, y)
    res = metric.result()

    tf.assert_equal(res, 0.5)

def test_MAPEavg_2d_pos_neg():

    x = tf.Variable([[1, 1, 1], [2, 2, 2]])
    y = tf.Variable([[2, 2, 2], [1, 1, 1]])

    metric = MAPEavg([2, 3])
    metric.update_state(x, y)
    res = metric.result()

    tf.assert_equal(res, 0.75)
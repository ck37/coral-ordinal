"""Tests for coral metric function."""
import numpy as np
import tensorflow as tf

from coral_ordinal.metrics import MeanAbsoluteErrorLabels


def test_config():
    mael_obj = MeanAbsoluteErrorLabels()
    assert mael_obj.name == "mean_absolute_error_labels"
    assert mael_obj.dtype == tf.float32


def get_data():
    actuals = [[[7], [2], [1]],
               [0, 0, 0],
               [0, 0, 0]]
    preds = [[[10.9, 6.3, 4.7, 3.4, 2.5, 1.8, 0.8, -0.4, -2.2], 
              [5.9, 1.3, -0.2, -1.4, -2.3, -3.1, -4.1, -5.3, -7.1], 
              [2.9, -1.6, -3.2, -4.5, -5.4, -6.1, -7.1, -8.4, -10.2]],
             [[10.9, 6.3, 4.7, 3.4, 2.5, 1.8, 0.8, -0.4, -2.2], 
              [5.9, 1.3, -0.2, -1.4, -2.3, -3.1, -4.1, -5.3, -7.1], 
              [2.9, -1.6, -3.2, -4.5, -5.4, -6.1, -7.1, -8.4, -10.2]],
             [[-1., -2., -3., -4., -5., -6., -7., -8., -9], \
              [-1., -2., -3., -4., -5., -6., -7., -8., -9], \
              [-1., -2., -3., -4., -5., -6., -7., -8., -9]]]
    return actuals, preds


def test_mae_labels_score():
    actuals, preds = get_data()

    mael_obj1 = MeanAbsoluteErrorLabels()
    mael_obj1.update_state(
        tf.constant(actuals[0], dtype=tf.int32),
        tf.constant(preds[0], dtype=tf.float32))
    # [7, 2, 1] - [7, 2, 1] = 0
    np.testing.assert_allclose(0., mael_obj1.result())

    mael_obj2 = MeanAbsoluteErrorLabels()
    mael_obj2.update_state(
        tf.constant(actuals[1], dtype=tf.int32),
        tf.constant(preds[1], dtype=tf.float32))
    # [7, 2, 1] - [0, 0, 0] = (7 + 2 + 1) / 3 = 3.3333333333
    np.testing.assert_allclose(3.3333333333, mael_obj2.result())

    mael_obj2 = MeanAbsoluteErrorLabels()
    mael_obj2.update_state(
        tf.constant(actuals[2], dtype=tf.int32),
        tf.constant(preds[2], dtype=tf.float32))
    # [0, 0, 0] - [0, 0, 0] = 0
    np.testing.assert_allclose(0., mael_obj2.result())


def test_mae_labels_running_score():
    mael_obj = MeanAbsoluteErrorLabels()
    actuals, preds = get_data()
    for a, p in zip(actuals, preds):
        mael_obj.update_state(
            tf.constant(a, dtype=tf.int32),
            tf.constant(p, dtype=tf.float32))
    np.testing.assert_allclose(1.1111111111, mael_obj.result())

    mael_obj.reset_states()
    np.testing.assert_allclose(0., mael_obj.result())


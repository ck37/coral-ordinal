"""Module for testing loss."""


import tensorflow as tf
import numpy as np
import pytest

from coral_ordinal import loss
from coral_ordinal import layer


def _create_test_data():
    # Test data from example in
    # https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    np.random.seed(10)

    X = np.random.normal(size=(8, 99))
    y = np.array([0, 1, 2, 2, 2, 3, 4, 4])
    sample_weights = np.array([0, 1, 1, 1, 1, 1, 1, 1])
    return X, y, sample_weights


def test_corn_loss():
    X, y, _ = _create_test_data()
    corn_loss = loss.CornOrdinalCrossEntropy()
    num_classes = len(np.unique(y))
    tf.random.set_seed(1)
    corn_net = layer.CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)
    assert logits.shape == (8, num_classes - 1)

    loss_val = corn_loss(y_true=y, y_pred=logits)
    # see https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    # for approximately same value for pytorch immplementation.
    # Divide by sample size = 8 here since TF defaults to sum over batch size, not sum.
    assert loss_val.numpy() == pytest.approx(3.54 / 8.0, 0.01)


@pytest.mark.parametrize(
    "reduction,expected_len",
    [("auto", 1), ("none", 8), ("sum", 1), ("sum_over_batch_size", 1)],
)
def test_coral_loss_reduction(reduction, expected_len):
    X, y, _ = _create_test_data()
    coral_loss = loss.OrdinalCrossEntropy(reduction=reduction)
    num_classes = len(np.unique(y))

    tf.random.set_seed(1)
    coral_net = layer.CoralOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = coral_net(X)
    loss_val = coral_loss(y_true=y, y_pred=logits)
    print(loss_val)
    if expected_len == 1:
        assert loss_val.numpy() > 0
    else:
        assert loss_val.shape[0] == expected_len


@pytest.mark.parametrize(
    "reduction,expected_len",
    [("auto", 1), ("none", 8), ("sum", 1), ("sum_over_batch_size", 1)],
)
def test_corn_loss_reduction(reduction, expected_len):
    X, y, _ = _create_test_data()
    corn_loss = loss.CornOrdinalCrossEntropy(reduction=reduction)
    num_classes = len(np.unique(y))

    tf.random.set_seed(1)
    corn_net = layer.CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)
    loss_val = corn_loss(y_true=y, y_pred=logits)
    print(loss_val)
    if expected_len == 1:
        assert loss_val.numpy() > 0
    else:
        assert loss_val.shape[0] == expected_len


def test_sample_weights_loss():
    X, y, sample_weights = _create_test_data()
    corn_loss = loss.CornOrdinalCrossEntropy(reduction="none")
    num_classes = len(np.unique(y))

    tf.random.set_seed(1)
    corn_net = layer.CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)

    loss_val = corn_loss(y_true=y, y_pred=logits).numpy()
    loss_val_weighted = corn_loss(
        y_true=y, y_pred=logits, sample_weight=sample_weights
    ).numpy()

    np.testing.assert_allclose(loss_val * sample_weights, loss_val_weighted)


def test_sample_weight_in_fit():
    X, y, _ = _create_test_data()
    w = np.zeros_like(y)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5, input_dim=X.shape[1]))
    model.add(layer.CornOrdinal(num_classes=4))
    model.compile(loss=loss.OrdinalCrossEntropy())

    history = model.fit(X, y, sample_weight=w, epochs=2)
    np.testing.assert_allclose(np.array(history.history["loss"]), np.array([0.0, 0.0]))

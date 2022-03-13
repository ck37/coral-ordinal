import tempfile
import tensorflow as tf
import numpy as np
import pytest

from coral_ordinal import layer


def _create_test_data():
    # Test data from example in
    # https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    np.random.seed(10)

    X = np.random.normal(size=(8, 99))
    y = np.array([0, 1, 2, 2, 2, 3, 4, 4])
    return X, y


def test_corn_layer():
    corn_layer = layer.CornOrdinal(num_classes=4, kernel_initializer="uniform")
    corn_layer_config = corn_layer.get_config()

    corn_layer2 = layer.CornOrdinal(**corn_layer_config)

    assert isinstance(corn_layer2, layer.CornOrdinal)


@pytest.mark.parametrize(
    "constructor",
    [(layer.CornOrdinal), (layer.CoralOrdinal)],
)
def test_serializing_layers(constructor):
    X, _ = _create_test_data()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(5, input_dim=X.shape[1]))
    model.add(constructor(num_classes=4))
    model.compile(loss="mse")

    preds = model.predict(X)
    with tempfile.TemporaryDirectory() as d:
        tf.keras.models.save_model(model, d)

        model_tmp = tf.keras.models.load_model(d)
        assert isinstance(model_tmp.layers[-1], constructor)
    preds_tmp = model_tmp.predict(X)
    np.testing.assert_allclose(preds, preds_tmp)

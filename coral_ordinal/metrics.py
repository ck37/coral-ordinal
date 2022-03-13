import tensorflow as tf
from tensorflow.keras import backend as K

from . import activations


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class MeanAbsoluteErrorLabels(tf.keras.metrics.Metric):
    """Computes mean absolute error for ordinal labels."""

    def __init__(
        self,
        corn_logits: bool = False,
        threshold: float = 0.5,
        name="mean_absolute_error_labels",
        **kwargs
    ):
        """Creates a `MeanAbsoluteErrorLabels` instance.

        Args:
          corn_logits: if True, inteprets y_pred as CORN logits; otherwise (default)
            as CORAL logits.
          threshold: which threshold should be used to determine the label from
            the cumulative probabilities. Defaults to 0.5.
          name: name of metric.
          **kwargs: keyword arguments passed to parent Metric().
        """
        super().__init__(name=name, **kwargs)
        self._corn_logits = corn_logits
        self._threshold = threshold
        self.maes = self.add_weight(name="maes", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Labels (int).
          y_pred: Cumulative logits from CoralOrdinal layer.
          sample_weight (optional): sample weights to weight absolute error.
        """

        # Predict the label as in Cao et al. - using cumulative probabilities.
        if self._corn_logits:
            cumprobs = activations.corn_cumprobs(y_pred)
        else:
            cumprobs = activations.coral_cumprobs(y_pred)

        # Threshold cumulative probabilities at predefined cutoff (user set).
        label_pred = tf.cast(
            activations.cumprobs_to_label(cumprobs, threshold=self._threshold),
            dtype=tf.float32,
        )
        y_true = tf.cast(y_true, label_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)
        label_pred = tf.squeeze(label_pred)
        label_abs_err = tf.abs(y_true - label_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, y_true.dtype)
            sample_weight = tf.broadcast_to(sample_weight, label_abs_err.shape)
            label_abs_err = tf.multiply(label_abs_err, sample_weight)

        self.maes.assign_add(tf.reduce_mean(label_abs_err))
        self.count.assign_add(tf.constant(1.0))

    def result(self):
        return tf.math.divide_no_nan(self.maes, self.count)

    def reset_state(self):
        """Resets all of the metric state variables at the start of each epoch."""
        K.batch_set_value([(v, 0) for v in self.variables])

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = {"threshold": self._threshold, "corn_logits": self._corn_logits}
        base_config = super().get_config()
        return {**base_config, **config}


"""
# WIP
def MeanAbsoluteErrorLabels_v2(y_true, y_pred):
  # There will be num_classes - 1 cumulative logits as columns of the tensor.
  num_classes = y_pred.shape[1] + 1
  
  probs = logits_to_probs(y_pred, num_classes)

# RootMeanSquaredErrorLabels
"""

import tensorflow as tf
from tensorflow.keras import backend as K


class MeanAbsoluteErrorLabels(tf.keras.metrics.Metric):
  """Computes mean absolute error for ordinal labels."""

  def __init__(self, name="mean_absolute_error_labels", **kwargs):
    """Creates a `MeanAbsoluteErrorLabels` instance."""
    super(MeanAbsoluteErrorLabels, self).__init__(name=name, **kwargs)
    self.maes = self.add_weight(name='maes', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Computes mean absolute error for ordinal labels.

    Args:
      y_true: Cumulatiuve logits from CoralOrdinal layer.
      y_pred: Labels.
      sample_weight (optional): Not implemented.
    """

    if sample_weight:
      raise NotImplementedError

    # Predict the label as in Cao et al. - using cumulative probabilities
    #cum_probs = tf.map_fn(tf.math.sigmoid, y_pred)

    # Calculate the labels using the style of Cao et al.
    # above_thresh = tf.map_fn(lambda x: tf.cast(x > 0.5, tf.float32), cum_probs)

    # Skip sigmoid and just operate on logit scale, since logit > 0 is
    # equivalent to prob > 0.5.
    above_thresh = tf.map_fn(lambda x: tf.cast(x > 0., tf.float32), y_pred)

    # Sum across columns to estimate how many cumulative thresholds are passed.
    labels_v2 = tf.reduce_sum(above_thresh, axis = 1)

    y_true = tf.cast(y_true, y_pred.dtype)

    # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
    y_true = tf.squeeze(y_true)

    self.maes.assign_add(tf.reduce_mean(tf.abs(y_true - labels_v2)))
    self.count.assign_add(tf.constant(1.))

  def result(self):
    return tf.math.divide_no_nan(self.maes, self.count)

  def reset_states(self):
    """Resets all of the metric state variables at the start of each epoch."""
    K.batch_set_value([(v, 0) for v in self.variables])

  def get_config(self):
    """Returns the serializable config of the metric."""
    config = {}
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

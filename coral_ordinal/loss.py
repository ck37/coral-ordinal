import tensorflow as tf
import tensorflow.python.ops as ops


# The outer function is a constructor to create a loss function using a certain number of classes.
class CoralOrdinalLoss(tf.keras.losses.Loss):
  def __init__(self, num_classes, importance = None, name = "coral_ordinal_loss", **kwargs):
    super().__init__(name = name, **kwargs)
    self.num_classes = num_classes
    
    if importance is None:
      self.importance = tf.ones(num_classes - 1, dtype = tf.float32)
    else:
      self.importance_weights = importance

  @tf.function
  def label_to_levels(self, label):
    # Original code that we are trying to replicate:
    # levels = [1] * label + [0] * (self.num_classes - 1 - label)
    label_vec = tf.repeat(1, tf.cast(tf.squeeze(label), tf.int32))
    num_zeros = self.num_classes - 1 - tf.cast(tf.squeeze(label), tf.int32)
    zero_vec = tf.zeros(shape = (num_zeros), dtype = tf.int32)
    levels = tf.concat([label_vec, zero_vec], axis = 0)

    return tf.cast(levels, tf.float32)

  # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
  def call(self, y_true, y_pred):

    # Ensure that y_true is the same type as y_pred (presumably a float).
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Convert each label to a vector of level indicators.
    tf_levels = tf.map_fn(self.label_to_levels, y_true)

    # Now call the original loss function.
    return ordinal_loss(y_pred, tf_levels, self.importance_weights)


def ordinal_loss(logits, levels, imp):
    levels = tf.cast(levels, tf.float32)
    val = (-tf.reduce_sum((tf.math.log_sigmoid(logits)*levels
                      + (tf.math.log_sigmoid(logits) - logits)*(1-levels))*imp,
           axis = 1))
    return tf.reduce_mean(val)

import tensorflow as tf

# TODO: this seems to be broken, compared to the Colab version.
def MeanAbsoluteErrorLabels(y_true, y_pred):
  # Assume that y_pred is cumulative logits from our CoralOrdinal layer.
  
  # Predict the label as in Cao et al. - using cumulative probabilities
  cum_probs = tf.map_fn(tf.math.sigmoid, y_pred)
  
  # Calculate the labels using the style of Cao et al.
  above_thresh = tf.map_fn(lambda x: tf.cast(x > 0.5, tf.float32), cum_probs)
  
  # Sum across columns so that we estimate how many cumulative thresholds are passed.
  labels_v2 = tf.reduce_sum(above_thresh, axis = 1)
  
  # This can convert to an integer, which will mess with the calculations.
  # labels_v2 = tf.cast(labels_v2, y_true.dtype)
  
  return tf.reduce_mean(tf.abs(y_true - labels_v2), axis = -1)


# WIP
def MeanAbsoluteErrorLabels_v2(y_true, y_pred):
  # There will be num_classes - 1 cumulative logits as columns of the tensor.
  num_classes = y_pred.shape[1] + 1
  
  probs = logits_to_probs(y_pred, num_classes)

# RootMeanSquaredErrorLabels

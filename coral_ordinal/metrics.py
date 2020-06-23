import tensorflow as tf

def MeanAbsoluteErrorLabels(y_true, y_pred):
  # Assume that y_pred is cumulative logits from our CoralOrdinal layer.
  
  # Predict the label as in Cao et al. - using cumulative probabilities
  cum_probs = tf.map_fn(tf.math.sigmoid, y_pred)
  
  # Calculate the labels using the style of Cao et al.
  labels_v2 = tf.reduce_sum(tf.map_fn(lambda x: tf.cast(x > 0.5, tf.int16), cum_probs), axis = 1)
  
  labels_v2 = tf.cast(labels_v2, y_true.dtype)
  
  return tf.reduce_mean(tf.abs(y_true - labels_v2))


# WIP
def MeanAbsoluteErrorLabels_v2(y_true, y_pred):
  # There will be num_classes - 1 cumulative logits as columns of the tensor.
  num_classes = y_pred.shape[1] + 1
  
  probs = logits_to_probs(y_pred, num_classes)

# RootMeanSquaredErrorLabels

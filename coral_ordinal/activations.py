import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.activations.ordinal_softmax')
def ordinal_softmax(x, axis = -1):
  """ Convert the ordinal logit output of CoralOrdinal() to label probabilities.
  
  Args:
    x: Logit output of the CoralOrdinal() layer.
    axis: Not yet supported.
  """
  
  # Convert the ordinal logits into cumulative probabilities.
  cum_probs = tf.map_fn(tf.math.sigmoid, x)
  
  # Create a list of tensors.
  probs = []
 
  # First, get probability predictions out of the cumulative logits.
  # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
  # Pr(Y = 0) = 1 - s(logit for column 0)
  probs.append(1. - cum_probs[:, 0])


  # For the other columns, the probability is:
  # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
  if num_classes > 2:
    for val in range(1, num_classes - 1):
      probs.append(cum_probs[:, val - 1) - cum_probs[:, val])
      
      
  # Special handling of the maximum label value.
  probs.append(cum_probs[:, num_classes - 2))
  
  # Column as columns into a new tensor.
  probs_tensor = tf.concat(probs, axis = 1)
  
  return probs_tensor

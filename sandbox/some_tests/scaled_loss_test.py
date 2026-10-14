import tensorflow as tf
from inr_apodizations.modeling.losses import ScaledLoss

base = tf.keras.losses.MeanAbsoluteError()
loss = ScaledLoss(base)

y_true = tf.constant([[1.0, 2.0], [3.0, 4.0]])
y_pred = tf.constant([[1.1, 10.9], [2.9, 4.1]])

_ = loss(y_true, y_pred)          # prints initial loss
print("Scaled loss (division only):", loss(y_true, y_pred).numpy())
import tensorflow as tf
import numpy as np

model_path = r"mse_1_2000_best_model"
model = tf.keras.models.load_model(model_path)
model.summary()
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
model.evaluate(x_test,y_test,verbose=2)

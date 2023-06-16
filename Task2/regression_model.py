import tensorflow as tf
import numpy as np
import argparse

def build_model(
        num_hidden=1, hidden_size=1000, input_shape=2048, output_shape=1,
        activation='relu', output_activation=None,
        dropout=0.2, learning_rate=0.001, loss="mse"
):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for _ in range(num_hidden):
        model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_shape, activation=output_activation))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss=loss
                  , metrics=["mae", "mse"])
    return model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a template prediction network')
    parser.add_argument('--num-hidden', dest='num_hidden', type=int, default=1)
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1000)
    parser.add_argument('--loss-function', dest='loss-function', type=str, default="mse")
    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_arguments()
    x_data = np.load("x_train.npy")
    y_data = np.load("y_train.npy")
    hidden_size = arg.hidden_size
    num_hidden = arg.num_hidden
    loss = arg.loss_fuction
    model = build_model(loss = loss,hidden_size=hidden_size,num_hidden=num_hidden)
    model.summary()
    filepath = loss + "_" + str(num_hidden) + "_" + str(hidden_size) + "_"
    callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=filepath + 'logs',update_freq="epoche"),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0005, mode='min', verbose=1),
            tf.keras.callbacks.ModelCheckpoint(filepath=filepath+"best_model", monitor='val_loss',
                                                   save_best_only=True, verbose=1)
    ]
    model.fit(x=x_data, y=y_data, epochs=50, callbacks=callbacks, validation_split=0.2)

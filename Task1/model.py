import tensorflow as tf
import argparse
import numpy as np
import os
def build_model(
         output_shape, num_hidden, hidden_size,input_shape=2048,
        activation='relu', output_activation=tf.keras.activations.softmax,
        dropout=0.2,learning_rate=0.001
):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for _ in range(num_hidden):
        model.add(tf.keras.layers.Dense(hidden_size, activation=activation))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_shape, activation=output_activation))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),loss=tf.keras.losses.categorical_crossentropy,
                  metrics=tf.keras.metrics.categorical_accuracy)
    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a template prediction network')
    parser.add_argument('--num-hidden', dest='num_hidden', type=int, default=1)
    parser.add_argument('--hidden-size', dest='hidden_size', type=int, default=1000)
    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_arguments()
    x_train = np.load("train_data_fingerprint.npy")
    y_train = np.load("train_data_onehot.npy")
    x_val = np.load("val_data_fingerprint.npy")
    y_val = np.load("val_data_onehot.npy")
    hidden_size =arg.hidden_size
    num_hidden = arg.num_hidden
    filepath= str(num_hidden)+"_"+str(hidden_size) +"_" + "best_model"
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=str(num_hidden)+"_"+str(hidden_size) +"_"+'logs',update_freq="epoche"),
        tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', min_delta=0.004, patience=3, mode='max',
                                             verbose=2),
        tf.keras.callbacks.ModelCheckpoint(filepath=filepath,monitor='val_categorical_accuracy',
                                               save_best_only=True,verbose=0)
    ]
    model = build_model(input_shape=x_train.shape[1],output_shape=y_train.shape[1],num_hidden = num_hidden,
                            hidden_size =hidden_size)
    model.summary()
    hist = model.fit(x=x_train,y=y_train,epochs=50,validation_data=(x_val,y_val),callbacks=callbacks)


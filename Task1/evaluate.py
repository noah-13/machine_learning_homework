import tensorflow as tf
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a template prediction network')
    parser.add_argument('--test_model_path', dest='test_model_path', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    arg = parse_arguments()
    # model = tf.keras.models.load_model(arg.test_model_path)
    x_test = np.load("test_data_fingerprint.npy")
    y_test = np.load("test_data_onehot.npy")
    # model.evaluate(x_test,y_test,verbose=2)
import pandas as pd
import numpy as np

data_path = 'test.pkl'
data=pd.read_pickle(data_path)
x_data = np.array([np.unpackbits(fingerprint)for fingerprint in data["packed_fp"]])
y_data = np.array(data["values"])
np.save("x_test.npy",x_data)
np.save("y_test.npy",y_data)
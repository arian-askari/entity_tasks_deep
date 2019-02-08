from sklearn.preprocessing import StandardScaler
import numpy as np
def get_normalize2D(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return np.array(scaler.transform(data))
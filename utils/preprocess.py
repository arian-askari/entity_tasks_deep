from sklearn.preprocessing import StandardScaler

def get_normalize2D(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)
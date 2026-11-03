import numpy as np

class KalmanFilter:
    def __init__(self, dim_x=4, dim_z=2):
        self.x = np.array([[300.0], [200.0], [0.0], [0.0]])
        
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        self.P = np.eye(dim_x) * 500.0
        self.Q = np.eye(dim_x) * 0.1
        self.R = np.eye(dim_z) * 10.0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x
    
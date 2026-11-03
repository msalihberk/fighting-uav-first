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
        
        self.P = np.eye(dim_x) * 100.0
        
        self.Q = np.eye(dim_x) * 1.0
        self.Q[2, 2] = 5.0
        self.Q[3, 3] = 5.0
        
        self.R = np.eye(dim_z) * 25.0

        self.lost_frames = 0
        self.max_lost_frames = 10

    def predict(self):
        if self.lost_frames > self.max_lost_frames:
            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        self.lost_frames += 1
        return self.x

    def update(self, z):
        self.lost_frames = 0

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        
        MAX_VELOCITY = 640.0
        self.x[2, 0] = np.clip(self.x[2, 0], -MAX_VELOCITY, MAX_VELOCITY)
        self.x[3, 0] = np.clip(self.x[3, 0], -MAX_VELOCITY, MAX_VELOCITY)

        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x
    
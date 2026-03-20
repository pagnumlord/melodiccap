"""
Simple 1D Kalman filter for landmark smoothing.
Identical to MelodicCapFresh implementation.
"""


class SimpleKalman:
    """1D Kalman filter for smoothing"""

    def __init__(self, q=1e-4, r=1e-2):
        self.q = q  # Process noise
        self.r = r  # Measurement noise
        self.x = 0.0  # State estimate
        self.p = 1.0  # Error covariance
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # Predict
        self.p += self.q

        # Update
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)

        return self.x

    def reset(self):
        self.initialized = False
        self.x = 0.0
        self.p = 1.0

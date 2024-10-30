#!/usr/bin/env python3
import numpy as np

class KalmanFilterTracker:
    def __init__(self, dt=0.1, process_noise=1e-5, measurement_noise=1e-1, state_dim=6, measure_dim=3):
        """
        Initialize the Kalman Filter Tracker.

        Parameters:
        dt (float): Time step between measurements.
        process_noise (float): Process noise covariance scalar.
        measurement_noise (float): Measurement noise covariance scalar.
        state_dim (int): Dimension of the state vector.
        measure_dim (int): Dimension of the measurement vector.
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state_dim = state_dim
        self.measure_dim = measure_dim

        # State transition matrix
        self.A = np.eye(state_dim)
        for i in range(measure_dim):
            if i + measure_dim < state_dim:
                self.A[i, i + measure_dim] = dt

        # Measurement matrix
        self.H = np.eye(measure_dim, state_dim)

        # Process noise covariance
        self.Q = process_noise * np.eye(state_dim)

        # Measurement noise covariance
        self.R = measurement_noise * np.eye(measure_dim)

        # Estimation error covariance
        self.P = np.eye(state_dim)

        # Initial state
        self.x = np.zeros((state_dim, 1))

    def predict(self):
        """
        Predict the state and estimation error covariance.
        
        Returns:
        numpy.ndarray: The predicted state vector (measurement components).
        """
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:self.measure_dim].flatten()

    def update(self, z):
        """
        Update the state and estimation error covariance using the measurement.

        Parameters:
        z (numpy.ndarray): Measurement vector.
        """
        z = np.reshape(z, (self.measure_dim, 1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

    def set_initial_state(self, state):
        """
        Set the initial state vector.

        Parameters:
        state (numpy.ndarray): Initial state vector.
        """
        self.x = np.reshape(state, (self.state_dim, 1))
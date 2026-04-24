import numpy as np
import torch
from torch import nn


class LSTM(nn.Module):
    """LSTM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PhysicsInformedLSTM(nn.Module):
    """Physics-Informed LSTM"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, omega_init=2*np.pi/365):
        super(PhysicsInformedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.omega = nn.Parameter(torch.tensor(omega_init))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

    def physics_loss(self, y_pred, t):
        dt = t[1] - t[0]
        y_dot = (y_pred[:, 1:] - y_pred[:, :-1]) / dt
        y_double_dot = (y_dot[:, 1:] - y_dot[:, :-1]) / dt
        y_mid = y_pred[:, 1:-1]
        physics_residual = y_double_dot + (self.omega ** 2) * y_mid

        sequence_length = y_pred.size(1)
        return torch.mean(physics_residual ** 2) / sequence_length**2

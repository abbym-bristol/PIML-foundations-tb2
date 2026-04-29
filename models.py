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


class RNN(nn.Module):
  """RNN"""
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(RNN, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.rnn(x)  # RNN output for all time steps
    out = out[:, -1, :]  # Take output from the last time step
    return self.fc(out)  # Pass through linear layer


class GRU(nn.Module):
    """GRU"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class PhysicsInformedRNN(nn.Module):
    """Physics-Informed RNN"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, omega_init=2*np.pi/365):
        super(PhysicsInformedRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.omega = nn.Parameter(torch.tensor(omega_init))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

    def physics_loss(self, y_pred, t):
        dt = t[1] - t[0]
        y_dot = (y_pred[:, 1:] - y_pred[:, :-1]) / dt
        y_double_dot = (y_dot[:, 1:] - y_dot[:, :-1]) / dt
        y_mid = y_pred[:, 1:-1]
        physics_residual = y_double_dot + (self.omega ** 2) * y_mid

        sequence_length = y_pred.size(1)
        return torch.mean(physics_residual ** 2) / sequence_length**2


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


class PhysicsInformedGRU(nn.Module):
    """Physics-Informed GRU"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, omega_init=2*np.pi/365):
        super(PhysicsInformedGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.omega = nn.Parameter(torch.tensor(omega_init))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

    def physics_loss(self, y_pred, t):
        dt = t[1] - t[0]
        y_dot = (y_pred[:, 1:] - y_pred[:, :-1]) / dt
        y_double_dot = (y_dot[:, 1:] - y_dot[:, :-1]) / dt
        y_mid = y_pred[:, 1:-1]
        physics_residual = y_double_dot + (self.omega ** 2) * y_mid
        
        # Normalize by sequence length
        sequence_length = y_pred.size(1)
        return torch.mean(physics_residual ** 2) / sequence_length**2

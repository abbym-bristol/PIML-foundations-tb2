import math
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn  
import torch.optim as optim


def set_seed(seed):
    random.seed(seed)  # Python
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


def create_sequences(data, seq_length, pred_length=1):
  """Create sequences and labels for data

  Args:
    data (array): 1D array of data to turn into sequences (X) and labels (y)
    seq_length (int): length of sequences (X)
    pred_length (int, optional): length of labels (y) (i.e. number of items to predict for).
        Defaults to 1.
  """
  xs, ys = [], []
  for i in range(len(data) - seq_length - pred_length + 1):
    xs.append(data[i:i+seq_length]) # Sequence of length `seq_length`
    ys.append(data[i+seq_length:i+seq_length+pred_length]) # Next sequence of length `pred_length`
  return np.array(xs), np.array(ys)


def plot_loss(training_loss, val_loss, num_epochs):
    """Plot the loss at each epoch to visualize model convergence

    Args:
        training_loss (array): training loss per epoch
        val_loss (array): validation loss per epoch
        num_epochs (int): number of epochs trained over
    """
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, num_epochs + 1), training_loss, label='Training Loss', color='purple')
    plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss', color='cyan')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_models(models, X_train, y_train, X_val, y_val,
                          num_epochs=1000, lr=0.001, physics_loss_weight=0.001,
                          omega=2 * np.pi / 365):
    """Train models
    
    Args:
        models (dict): dictionary containing keys: model name (string) and values: pytorch model object
        X_train (Tensor): containing training sequences
        y_train (Tensor): containing corresponding training labels
        X_val (Tensor): containing validation sequences
        y_val (Tensor): containing corresponding validation labels
        num_epochs (int, optional): number of epochs to train for. Default 1000.
        lr (float, optional): learning rate, default 0.001.
        physics_loss_weight (float, optional): amount to weight physics loss term in PINN models.
            Defaults to 0.001.
        omega (float, optional): starting value for omega in physics loss, defaults to 2 * np.pi / 365.

    Returns:
        None
    """
    for name, model in models.items():
        # Check if the model is physics-informed
        is_physics_informed = hasattr(model, "physics_loss")
        if is_physics_informed:
            print(f"Training Physics-Informed Model: {name}")
            # Set omega for the physics-informed model
            if hasattr(model, "omega"):
                model.omega = nn.Parameter(torch.tensor(omega, dtype=torch.float32, device=model.omega.device))
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            total_losses = []
            val_total_losses = []

            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()

                # Forward pass
                output = model(X_train)
                data_loss = criterion(output.squeeze(), y_train)  # Compute loss

                # Physics loss
                physics_loss = model.physics_loss(
                    output,
                    torch.linspace(0, 1, steps=output.size(1), device=output.device)
                )
                total_loss = data_loss + physics_loss_weight * physics_loss

                # Backward pass and optimization
                total_loss.backward()
                optimizer.step()

                # Logging
                total_losses.append(total_loss.item())
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.4f}")

                # Validation
                model.eval()
                val_preds = model(X_val)
                val_data_loss = criterion(val_preds.squeeze(), y_val)
                val_physics_loss = model.physics_loss(
                    val_preds,
                    torch.linspace(0, 1, steps=output.size(1), device=output.device)
                )
                val_total_loss = val_data_loss + physics_loss_weight * val_physics_loss
                val_total_losses.append(val_total_loss.item())

            plot_loss(total_losses, val_total_losses, num_epochs)

        else:
            print(f"Training Standard Model: {name}")
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            losses = []
            val_losses = []
            for epoch in range(num_epochs):
                model.train()
                output = model(X_train)
                loss = criterion(output.squeeze(), y_train)  # Compute loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

                # Validation
                model.eval()
                val_preds = model(X_val)
                val_loss = criterion(val_preds.squeeze(), y_val)
                val_losses.append(val_loss.item())

            plot_loss(losses, val_losses, num_epochs)


def forecast_with_model(model, start_data, forecast_length, pred_length=1):
    """ Autoregressively forecast using model over forecast_length

    Args:
        model (pytorch model): trained model
        start_data (pytorch Tensor): tensor containing the starting data sequence to forecast from
        forecast_length (int): length of data to forecast
        pred_length (int, optional): length of returned model predictions (defaults to 1)
    """
    X_forecast = start_data
    y_forecast = []
    for i in range(math.ceil(forecast_length/pred_length)):
        # Make predictions
            # Predict chunk of pred_length days enough times to cover whole test set
            # Will generate extra predictions beyond length of desired forecast
        model.eval()
        preds = model(X_forecast).squeeze().detach().numpy()
        y_forecast.append(preds)

        # Make new window
        X_old = X_forecast.squeeze().numpy()[pred_length:]  # Convert to numpy and drop length of prediction
        X_new = np.array([np.append(X_old, preds)])  # Add prediction on end of window
        X_forecast = torch.Tensor(X_new).unsqueeze(-1)  # Convert new window to pytorch tensor

    # Sort out formatting of results into one long numpy array of right length (trim extra generated predictions)
    if pred_length > 1:
        y_forecast = np.array(y_forecast).reshape(-1,1)[:forecast_length]
    else:
        y_forecast = np.array(y_forecast)
    return y_forecast

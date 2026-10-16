import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for AquaSense Anomaly Detection.

    Symmetrical encoder compression layers and decoder reconstruction layers.
    Trained strictly on normal, clean seabed imagery.
    High pixel-wise reconstruction residual error indicates an unknown anomaly
    or unexpected submerged object.
    """

    def __init__(self):
        super().__init__()

        # Encoder: Symmetrical compression layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Decoder: Symmetrical reconstruction layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

    def reconstruction_error(self, x):
        """
        Calculates mean squared reconstruction error.
        Higher value = more likely anomaly.
        """
        reconstructed = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2)
        return error, reconstructed


# Backward and forward compatibility alias
AquaSenseAutoencoder = ConvAutoencoder

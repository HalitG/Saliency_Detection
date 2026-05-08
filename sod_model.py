import torch.nn as nn

class SaliencyBaseline(nn.Module):
    def __init__(self):
        super(SaliencyBaseline, self).__init__()
        
        # Input: RGB image (3 x 128 x 128)
        # Encoder: 4 Conv2D layers with ReLU and MaxPooling
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        # Decoder: 4 ConvTranspose2D layers with ReLU
        # Output: 1-channel Sigmoid mask
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
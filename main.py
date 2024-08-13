import torch
from ddpm.unet import UNet
from ddpm import DiffusionModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'running on {device}')

unet = UNet()
unet.to(device)

# Random Forward pass
x = torch.randn(32, 3, 32, 32).to(device)
t = torch.randint(0, 1000, (32,)).to(device)

with torch.no_grad():
    unet(x, t)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from tqdm import tqdm
import matplotlib.pyplot as plt

from ddpm import UNet, Scheduler, EMA, DataLoaderLite
torch.set_float32_matmul_precision('high')

## Model Parameters
layers = 3
n_labels = 10
channels = 64
n_heads = 4
n_embd = 256
#---------------------------------------------

## Training Parameters
time_steps = 1000
epochs = 200
lr = 3e-4
batch_size = 32
warmup_epochs = 50
device = 'cuda'
sample_step = 20
#---------------------------------------------

# training setup
loader = DataLoaderLite()
scheduler = Scheduler(noise_steps=time_steps)
m = UNet(layers=layers, n_labels=n_labels, channels=channels, n_heads=n_heads, n_embd=n_embd, time_steps=time_steps).to(device)
m = torch.compile(m)
opt = torch.optim.AdamW(m.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup_epochs, eta_min=3e-6)
mse = nn.MSELoss()
#---------------------------------------------

def sample(
    m:torch.nn.Module,
    scheduler:Scheduler,
    n:int, 
    labels:Union[List[int], None] = None, 
    cfg_scale:Union[int, float] = 7.5, 
    img_size=32,
) -> torch.Tensor:
    m.eval()
    with torch.no_grad():
        x = torch.rand((n, 3, img_size, img_size)).to(device)
        labels = torch.tensor(labels).to(dtype=torch.int, device=device)
        pb = tqdm(reversed(range(1, scheduler.noise_steps)), position=0, total=scheduler.noise_steps, initial=1)
        
        for i in pb:
            t = (torch.ones(n) * i).int().to(device)
            predicted_noise = m(x, t, labels)
            if cfg_scale > 0:
                uncond_predicted_noise = m(x, t, None)
                predicted_noise = cfg_scale * (predicted_noise - uncond_predicted_noise) + uncond_predicted_noise
            
            alpha = scheduler.alpha[t][:, None, None, None]
            alpha_hat = scheduler.alpha_hat[t][:, None, None, None]
            beta = scheduler.beta[t][:, None, None, None]

            noise = torch.randn_like(x) if i>1 else torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
    m.train()
    x = x.clamp(0, 1) * 255
    x = x.type(torch.int)
    x = x.permute(0, 2, 3, 1).cpu().numpy()
    return x

#---------------------------------------------

for epoch in tqdm(range(epochs)):
    if epoch > warmup_epochs:
        lr_scheduler.step()
    x, y = loader.get_batch(batch_size=4, overfit=True)
    x = x / 255.
    x, y = x.to(device), y.to(device)
    t = scheduler.sample_timesteps(batch_size).to(device)
    x_t = scheduler.noise_images(x, t)
    if torch.rand(1).item() < 0.1:
        y = None
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        y_pred = m(x_t, t, y)
        loss = mse(y_pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

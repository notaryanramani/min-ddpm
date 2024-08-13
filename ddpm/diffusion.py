import torch
import torch.nn as nn
from torch.optim import AdamW
from .unet import UNet
from .scheduler import Scheduler
from .utils import get_data, CustomCosineAnnealingLR
from tqdm import tqdm
from typing import List, Union


class DiffusionModel:
    def __init__(self, scheduler:Union[Scheduler, None]=None, layers:int=3, n_labels:int=10, 
                 channels:int=64, n_heads:int=4, n_embd:int=256, time_steps:int=1000) -> None:

        self.scheduler = scheduler if scheduler else Scheduler(noise_steps=time_steps)
        self.m = UNet(layers=layers, n_labels=n_labels, channels=channels, n_heads=n_heads, n_embd=n_embd, time_steps=time_steps)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, 
              data_path:str,
              epochs:int=200, 
              lr:float= 3e-4, 
              batch_size:int=32, 
              eta_min = 3e-6,
              warmup_epcohs = 50, # try to keep this 25% of epochs
              return_unet_model=False) -> Union[UNet, None]:
        dataloader = get_data(data_path, batch_size)
        opt = AdamW(self.m.parameters(), lr=lr)
        opt_sch = CustomCosineAnnealingLR(opt, epochs, eta_min=eta_min, warmup_epochs=warmup_epcohs)
        mse = nn.MSELoss()

        for epoch in range(epochs):
            pb = tqdm(dataloader)
            pb.set_description(f'Train Epoch [{epoch+1}/{epochs}]: ')
            for i, (images, labels) in enumerate(pb):
                images, labels = images.to(self.device), labels.to(self.device)
                t = self.scheduler.sample_timesteps(batch_size).to(self.device)
                x_t, noise = self.scheduler.noise_images(images, t)
                if torch.rand(1).item() < 0.1:
                    labels = None               # 10% chance of training without labels
                predicted_noise = self.m(x_t, t, labels)
                loss = mse(noise, predicted_noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                opt_sch.step()
                pb.set_postfix({'Loss':loss.item()})

        if return_unet_model:
            return self.m
    
    def from_pretrained(self, state_dict:dict) -> None:
        self.m.load_state_dict(state_dict)

    def save(self, PATH:str) -> None:
        torch.save(self.m.state_dict(), PATH)

    def sample(self, n:int, labels:Union[List[int]], cfg_scale:Union[int, float] = 7.5, img_size=64) -> torch.Tensor:
        self.m.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, img_size, img_size)).to(self.device)
            labels = torch.tensor(labels).to(dtype=torch.uint8, device=self.device)
            pb = tqdm(reversed(range(1, self.scheduler.noise_steps)), position=0)
            for i in pb:
                t = (torch.ones(n) * i).int().to(self.device)
                predicted_noise = self.m(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_nosie = self.m(x, t, labels)
                    predicted_noise = cfg_scale * (predicted_noise - uncond_predicted_nosie) + uncond_predicted_nosie
                alpha = self.scheduler.alpha[t][:, None, None, None]
                alpha_hat = self.scheduler.alpha_hat[t][:, None, None, None]
                beta = self.scheduler.beta[t][:, None, None, None]

                noise = torch.randn_like(x) if i>1 else torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        self.m.train()
        x = x.clamp(1, -1) + 1
        x = x * 255 / 2
        x = x.type(torch.uint8)
        return x


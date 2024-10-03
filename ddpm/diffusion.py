import torch
import torch.nn as nn
from torch.optim import AdamW
from .unet import UNet
from .scheduler import Scheduler
from .utils import get_dataloader, CustomCosineAnnealingLR, EMA
from tqdm import tqdm
from typing import List, Union
import os
import copy
import matplotlib.pyplot as plt



class DiffusionModel:
    def __init__(self, scheduler:Union[Scheduler, None]=None, layers:int=3, n_labels:int=10, 
                 channels:int=64, n_heads:int=4, n_embd:int=256, time_steps:int=1000) -> None:

        self.scheduler = scheduler if scheduler else Scheduler(noise_steps=time_steps)
        self.m = UNet(layers=layers, n_labels=n_labels, channels=channels, n_heads=n_heads, n_embd=n_embd, time_steps=time_steps)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.m.to(self.device)
        self.ema_model = None

        # observed mean and std for data
        self.mean = torch.tensor([0.4530, 0.4449, 0.4098], device=self.device)
        self.std = torch.tensor([0.2457, 0.2433, 0.2603], device=self.device)

    def train(self, 
              baseFolder:str = 'cifar10/',
              epochs:int=200, 
              lr:float= 3e-4, 
              batch_size:int=32, 
              eta_min = 3e-6,
              warmup_epcohs = 50, # try to keep this 25% of epochs
              return_unet_model=False,
              checkpoint = True,
              checkpoint_step = 20,
              useAutocast = False,
              useEMA = False, 
              generateSamples=True) -> Union[UNet, None]:
        
        if useAutocast:
            torch.set_float32_matmul_precision('high')
        if useEMA:
            ema = EMA(beta=0.95)
            self.ema_model = copy.deepcopy(self.m).eval().requires_grad_(False)

        trainFolder = baseFolder+'train'
        valFolder = baseFolder+'test'
        train_dataloader = get_dataloader(trainFolder, batch_size)
        opt = AdamW(self.m.parameters(), lr=lr)
        opt_sch = CustomCosineAnnealingLR(opt, epochs, eta_min=eta_min, warmup_epochs=warmup_epcohs)
        mse = nn.MSELoss()
        if checkpoint:
            train_loss = []
            val_loss = []
        if generateSamples:
            all_samples = []

        for epoch in range(epochs):
            pb = tqdm(train_dataloader, position=0, leave=False)
            pb.set_description(f'Train Epoch [{epoch+1}/{epochs}]: ')
            for i, (images, labels) in enumerate(pb):
                images, labels = images.to(self.device), labels.to(self.device)
                t = self.scheduler.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.scheduler.noise_images(images, t)
                if torch.rand(1).item() < 0.1:
                    labels = None               # 10% chance of training without labels
                if useAutocast:
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        predicted_noise = self.m(x_t, t, labels)
                        loss = mse(noise, predicted_noise)
                else:
                    predicted_noise = self.m(x_t, t, labels)
                    loss = mse(noise, predicted_noise)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if useEMA:
                    ema.step(self.ema_model, self.m)
                pb.set_postfix({'Loss':loss.item()})
                if checkpoint:
                    train_loss.append(loss.item())
            opt_sch.step()
            if generateSamples and (epoch+1) % 20 == 0:
                n = 10
                labels = torch.arange(0, 10)
                samples = self.sample(n, labels)
                all_samples.append(samples)

            if epoch % checkpoint_step == 0 and checkpoint:
                tl = torch.tensor(train_loss)
                vl = torch.tensor(train_loss)

                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints', exist_ok=True)

                PATH = 'checkpoints/'
                torch.save({
                    'tl' : tl,
                    'vl' : vl
                }, PATH + 'metrics.pth')

                torch.save({
                    'model' : self.m.state_dict()
                })
        
        if generateSamples:
            os.makedirs('samples', exist_ok=True)
            samplePATH = 'samples/training_samples.png'
            all_samples = torch.stack(all_samples, dim=0)
            E, L, H, W, C = all_samples.shape
            all_samples = all_samples.view(E*L, H, W, C)
            fig, axes = plt.subplots(E, L, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
                ax.axis('off')  
            plt.subplots_adjust(wspace=0.05, hspace=0)
            plt.savefig(samplePATH)

        if return_unet_model:
            return self.m
    
    def from_pretrained(self, state_dict:dict) -> None:
        self.m.load_state_dict(state_dict)

    def save(self, PATH:str) -> None:
        torch.save(self.m.state_dict(), PATH)

    def sample(self, 
               n:int, 
               labels:Union[List[int], None] = None, 
               cfg_scale:Union[int, float] = 7.5, 
               img_size=32,
               permute=True, 
               useEMA = False) -> torch.Tensor:
        
        if useEMA is True:
            if self.ema_model is None:
                raise NotImplementedError('EMA was not used during training')
            else:
                m = self.ema_model
        else:
            m = self.m

        m.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, img_size, img_size)).to(self.device)
            labels = torch.tensor(labels).to(dtype=torch.int, device=self.device)
            pb = tqdm(reversed(range(1, self.scheduler.noise_steps)), position=0, total=self.scheduler.noise_steps, initial=1)
            
            for i in pb:
                t = (torch.ones(n) * i).int().to(self.device)
                predicted_noise = m(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = m(x, t, None)
                    predicted_noise = cfg_scale * (predicted_noise - uncond_predicted_noise) + uncond_predicted_noise
                
                alpha = self.scheduler.alpha[t][:, None, None, None]
                alpha_hat = self.scheduler.alpha_hat[t][:, None, None, None]
                beta = self.scheduler.beta[t][:, None, None, None]

                noise = torch.randn_like(x) if i>1 else torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        m.train()
        x = x * self.std + self.mean
        x = x.clamp(-1, 1) + 1
        x = x * 255 / 2
        x = x.type(torch.int)
        if permute:
            x = x.permute(0, 2, 3, 1)
        return x


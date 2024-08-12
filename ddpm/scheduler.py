import torch


device = "cuda" if torch.cuda.is_available() else 'cpu'

class Scheduler:
    def __init__(self, 
                 noise_steps:int = 1000,
                 beta_start:float = 1e-4,
                 beta_end:float = 0.02) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        ah_sqrt = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        ah_om_sqrt = torch.sqrt(1. - self.alpha_hat[t]).view(-1, 1, 1, 1)
        epsilon = torch.rand_like(x)
        x = ah_sqrt * x + ah_om_sqrt * epsilon
        return x, epsilon

    def sample(self, model, n, labels):
        pass
        
        # TODO after developing UNet - model
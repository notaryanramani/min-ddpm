from torch.utils.data import DataLoader
import torchvision as vision
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

def get_data(PATH:str, batch_size:int) -> DataLoader:
    transform = vision.transforms.Compose([
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.5, 0.5, 0,5), (0.5, 0.5, 0.5)),
    ])
    dataset = vision.datasets.ImageFolder(PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


class CustomCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=3, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        self.cosine_scheduler.step()
        return self.cosine_scheduler.get_lr()
    

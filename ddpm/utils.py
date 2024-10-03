import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
import torchvision as vision
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

def get_dataloader(
        imageFolder:str, 
        batch_size:int = 32) -> DataLoader:
    transform = vision.transforms.Compose([
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.4530, 0.4449, 0.4098), (0.2457, 0.2433, 0.2603))
    ])
    dataset = vision.datasets.ImageFolder(imageFolder, transform=transform)
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
    
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
    
class DataLoaderLite:
    def __init__(self):
        def get_batch(filename):
            with open(filename, 'rb') as f:
                _dict = pickle.load(f, encoding='bytes')
            return _dict[b'data'], _dict[b'labels']
        
        data = []
        labels = []
        for i in range(1, 6):
            filename = f'cifar-10-batches-py/data_batch_{i}'
            _data, _labels = get_batch(filename)
            data.append(_data)
            labels.extend(_labels)

        data = np.concatenate(data)
        data = data.reshape(-1, 3, 32, 32)
        self.data = torch.from_numpy(data).float()
        self.labels = torch.tensor(labels)

        test_filename = 'cifar-10-batches-py/test_batch'
        self.test_data, self.test_labels = get_batch(test_filename)
    
    def get_batch(self, batch_size:int = 8, overfit:bool = False):
        if overfit:
            idx = torch.arange(batch_size)
            return self.data[idx], self.labels[idx]
        idx = torch.randint(0, len(self.data), (batch_size,))
        return self.data[idx], self.labels[idx]
    
    def get_test_batch(self, batch_size:int = 8):
        idx = torch.randint(0, len(self.test_data), (batch_size,))
        return self.test_data[idx], self.test_labels[idx]
    
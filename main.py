from ddpm import DiffusionModel

PATH = 'cifar10-64/train'

ddpm = DiffusionModel()
ddpm.train(data_path=PATH)

# To save a trained model
# ddpm.save('PATH_TO_DIRECTORY') 

# To load a pretrained model
# ddpm.from_pretrained(state_dict)
# ddpm.train(data_path='PATH_T0_DIRECTORY')
# NOTE: try to keep a very low learning rate while fine-tuning

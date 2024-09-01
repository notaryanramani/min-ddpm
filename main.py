from ddpm import DiffusionModel


ddpm = DiffusionModel()
ddpm.train()

# for ampere archiecture
# ddpm.train(useAutocast=True)

# To save a trained model
# ddpm.save('PATH_TO_DIRECTORY') 

# To load a pretrained model
# ddpm.from_pretrained(state_dict)
# ddpm.train(data_path='PATH_T0_DIRECTORY')
# NOTE: try to keep a very low learning rate while fine-tuning

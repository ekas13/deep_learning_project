import torch
from models.ddpm_cifar import DDPM
from networks.score_network import CIFAR10ScoreNetwork
import numpy as np
import datetime as datetime

NUM_SAMPLES =  10000
EPOCHS = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ddpm = DDPM(device=device, network=CIFAR10ScoreNetwork())
ddpm.load(f"jan_experiments/CIFAR-10-model-1000-epochs/CIFAR_epoch_{EPOCHS}")
samples = []	

for i in range(NUM_SAMPLES):
    sample = ddpm.sample(shape=(1, 3, 32, 32)).squeeze(0).reshape(1, 3, 32, 32) # remove batch dimension if present
    samples.append(sample)
    print(f"Sample {i+1} generated")

samples = torch.cat(samples, dim=0)
samples = samples.cpu().numpy()
sampled_data = np.asarray(samples)

print("Sampled data shape:")
print(sampled_data.shape)

# wandb.finish()
np.save(f"experiments_cifar/CIFAR-10-model-500-epochs/samples/epoch-wise-16-samples/{NUM_SAMPLES}_samples_epoch_{EPOCHS}.npy", sampled_data)
print(f"Generated all {NUM_SAMPLES} samples")



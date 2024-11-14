import torch
from models.ddpm import DDPM
import numpy as np
from networks.score_network import score_network_0

NUM_SAMPLES = 10000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ddpm = DDPM(device=device, network=score_network_0())
ddpm.load("results/hpc_2000/testing")

samples = []	

for i in range(NUM_SAMPLES):
    sample = ddpm.sample().view(1, 28, 28)
    samples.append(sample)
    print(f"Sample {i+1} generated")

samples = torch.cat(samples, dim=0)

samples = samples.cpu().numpy()

sampled_data = np.asarray(samples)
np.save("results/hpc_2000/samples_1.npy", sampled_data)
print(f"Generated all {NUM_SAMPLES} samples")
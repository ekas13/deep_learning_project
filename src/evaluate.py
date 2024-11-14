import torch
from data.data_loader import data_loader
from models.ddpm import DDPM
import numpy as np
from networks.score_network import score_network_0
from metrics import fid_score

NUM_SAMPLES = 50000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ddpm = DDPM(device=device, network=score_network_0())
ddpm.load("results/hpc_2000/testing")

samples = []	

for i in range(NUM_SAMPLES):
    sample = ddpm.sample().view(1, 28, 28).cpu().numpy()
    samples.append(sample)
    print(f"Sample {i+1} generated")

sampled_data = np.asarray(samples)
np.save("results/hpc_2000/samples.npy", sampled_data)

train_data_loader = data_loader("MNIST", 64, device)
train_data = train_data_loader.x_train.view(-1, 1, 28, 28).cpu().numpy()[:NUM_SAMPLES, :]

score = fid_score(sampled_data, train_data, batch_size=64)

with open("results/hpc_2000/fid_score.txt", "w") as f:
    f.write(f"FID score for {NUM_SAMPLES} samples: {score}")
    f.write("\n")
    f.close()

print(f"FID score for {NUM_SAMPLES} samples: {score}")
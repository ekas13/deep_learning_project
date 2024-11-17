import torch
from data.data_loader import data_loader
import numpy as np
from metrics import fid_score

NUM_SAMPLES = 30000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

samples_1 = np.load("results/hpc_2000/samples_1.npy")
samples_2 = np.load("results/hpc_2000/samples_2.npy")

sampled_data = np.concatenate((samples_1, samples_2), axis=0).reshape(NUM_SAMPLES, 1, 28, 28)[20000:30000, :]

train_data_loader = data_loader("MNIST", 64, device)
# train_data = train_data_loader.x_train.view(-1, 1, 28, 28).cpu().numpy()[:NUM_SAMPLES, :]
train_data = train_data_loader.x_test.view(-1, 1, 28, 28).cpu().numpy()[:10000, :]

score = fid_score(sampled_data, train_data, batch_size=50)

with open("results/hpc_2000/fid_score.txt", "w") as f:
    f.write(f"FID score for {NUM_SAMPLES} samples: {score}")
    f.write("\n")
    f.close()

print(f"FID score for {NUM_SAMPLES} samples: {score}")
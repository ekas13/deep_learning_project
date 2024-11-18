from models.ddpm import DDPM
from networks.score_network import score_network_0
from visualizer import plot_loss, plot_samples
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ddpm = DDPM(device=device, network=score_network_0())

opt = torch.optim.AdamW(ddpm.network.parameters(), lr=2e-4)

losses = ddpm.train(dataset_name="MNIST", num_epochs=150, batch_size=128, opt=opt)

ddpm.save("results/testing")
# plot_loss(losses)

# For nice visualization make it num_samples have a sqrt
num_samples = 16
sampled_data = [ddpm.sample() for _ in range(num_samples)]

plot_samples(sampled_data, "results/samples.png")
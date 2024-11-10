from models.ddpm import DDPM
from networks.score_network import score_network_0
from visualizer import plot_loss
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ddpm = DDPM(device=device, network=score_network_0())

opt = torch.optim.AdamW(ddpm.network.parameters(), lr=2e-4)

losses = ddpm.train(dataset_name="MNIST", num_epochs=10, batch_size=64, opt=opt)

plot_loss(losses)
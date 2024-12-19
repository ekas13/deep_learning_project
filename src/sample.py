import torch
from models.ddpm import DDPM
import numpy as np
from networks.score_network import score_network_0, score_network_1
import matplotlib.pyplot as plt
from visualizer import plot_samples

def plot_timeline(sample_steps, path):
    # Number of images
    num_images = len(sample_steps)

    # Create a figure
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))  # Adjust figsize as needed

    # Plot each image
    for i, ax in enumerate(axes):
        ax.imshow(sample_steps[i].cpu().view(28, 28), cmap='gray')
        ax.axis('off')  # Turn off the axis for a cleaner look

    # Display the plot
    plt.tight_layout()
    plt.savefig(path)

NUM_SAMPLES = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results_path = "results/cfg_500/CFG-MNIST-model-500-epochs"

ddpm = DDPM(device=device, network=score_network_1(), p_unconditional=1, scheduler="cosine")
ddpm.load(f"{results_path}")

samples = []	

for i in range(NUM_SAMPLES):
    sample, sample_steps = ddpm.sample(label=4, w=3)
    samples.append(sample.view(1, 28, 28))
    print(f"Sample {i+1} generated")

samples = torch.cat(samples, dim=0)

samples = samples.cpu().numpy()

sampled_data = np.asarray(samples)
# np.save(f"{results_path}samples_1.npy", sampled_data)
print(len(sample_steps))
plot_samples([sample], f"{results_path}sample.png")
plot_timeline(sample_steps, f"{results_path}timeline.png")

print(f"Generated all {NUM_SAMPLES} samples")
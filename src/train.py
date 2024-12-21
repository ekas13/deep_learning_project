from models.ddpm import DDPM
from networks.score_network import score_network_0, score_network_1, CIFAR10ScoreNetwork
from visualizer import plot_loss, plot_samples
import torch
import datetime
import wandb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_ddpm():
    ddpm = DDPM(device=device, network=score_network_0())

    opt = torch.optim.AdamW(ddpm.network.parameters(), lr=2e-4)

    losses = ddpm.train(dataset_name="MNIST", num_epochs=150, batch_size=128, opt=opt)

    ddpm.save("results/testing")
    # plot_loss(losses)

    # For nice visualization make it num_samples have a sqrt
    num_samples = 16
    sampled_data = [ddpm.sample() for _ in range(num_samples)]

    plot_samples(sampled_data, "results/samples.png")


def train_cfg():
    ddpm = DDPM(device=device, network=score_network_1(), cfg=True, p_unconditional=0.1)
    opt = torch.optim.AdamW(ddpm.network.parameters(), lr=2e-4)
    losses = ddpm.train(dataset_name="MNIST", num_epochs=500, batch_size=64, opt=opt)
    ddpm.save("results/testing")
    # ddpm.load("results/testing", True)

    print("Sampling!")
    num_samples = 16
    sampled_data = [ddpm.sample(label=i % 10, w=3) for i in range(num_samples)]
    plot_samples(sampled_data, "results/samples.png")

def train_ddpm_cifar10():
    datetime_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    wandb.init(project="ddpm-training", name=f"train_ddpm_cifar10_{datetime_suffix}", 
               config={"learning_rate": 2e-4, "batch_size": 128, "epochs": 1000})
    config = wandb.config

    ddpm = DDPM(device=device, network=CIFAR10ScoreNetwork())
    opt = torch.optim.AdamW(ddpm.network.parameters(), lr=config.learning_rate)

    losses = ddpm.train(dataset_name="CIFAR-10", num_epochs=config.epochs, batch_size=config.batch_size, opt=opt)

    for epoch, loss in enumerate(losses, 1):
        wandb.log({"epoch": epoch, "avg_loss": loss})

    ddpm.save(f"experiments_cifar/CIFAR-10-model-{config.epochs}-epochs")
    wandb.finish()


if __name__ == "__main__":
    train_ddpm_cifar10()

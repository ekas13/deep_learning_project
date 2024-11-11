from torchvision.datasets import MNIST, CIFAR10
import torch

class data_loader():
    def __init__(self, dataset_name, batch_size, device):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.device = device

        self.x_train = None
        self.x_test = None

        if self.dataset_name == "MNIST":
            train_set = MNIST("./temp/", train=True, download=True)
            test_set = MNIST("./temp/", train=False, download=True)
            self.x_train = train_set.data.view(-1, 784).float().div_(255).to(device)
            self.x_test = test_set.data.view(-1, 784).float().div_(255).to(device)
        else:
            raise NotImplementedError(f"{dataset_name} not implemented")
        
        self.current_batch = 0
        self.dataset_size = self.x_train.shape[0]

    def has_next_batch(self):
        return self.current_batch < self.dataset_size // self.batch_size
    
    def get_batch(self):
        slce = range(self.current_batch * self.batch_size, (self.current_batch + 1) * self.batch_size)
        self.current_batch += 1
        return self.x_train[slce]
    
    def reset(self):
        self.current_batch = 0
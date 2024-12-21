import torch
import numpy as np
from data.data_loader import data_loader
import math

class DDPM():
    def __init__(self, device, network: torch.nn.Module, T: int = 1000, cfg = False, num_classes: int = 10, p_unconditional = 0.1, scheduler="cosine"):
        self.device = device
        self.network = network.to(device)
        self.T = T 
        self.betas = torch.linspace(0.0001, 0.02, T)
        self.alphas = 1 - self.betas
        self.loss = torch.nn.MSELoss()
        self.alphas_cumulative = self.alpha_calculate_cumulative()
        self.times = None
        self.embedding_dim = 128
        self.label_embedding = torch.nn.Embedding(num_embeddings=num_classes, embedding_dim=self.embedding_dim).to(device)
        self.cfg = cfg
        self.p_unconditional = p_unconditional
        self.scheduler = scheduler


    def alpha_calculate_cumulative(self):
        alphas_cum = [self.alphas[0]]
        for i in range(1, len(self.alphas)):
            alphas_cum.append(alphas_cum[i-1] * self.alphas[i])

        return torch.tensor(alphas_cum).to(self.device)
 
    
    def reset(self):
        self.network.reset()


    def train(self, dataset_name: str, num_epochs: int, batch_size: int, opt: torch.optim.Optimizer):
        train_data_loader = data_loader(dataset_name, batch_size, self.device)
        losses = []

        if self.times is None:
            if self.scheduler == "cosine":
                self.calculate_times_cosine()
            else:
                self.calculate_times_linear()

        for i in range(num_epochs):
            current_loss = []
            while train_data_loader.has_next_batch():
                #sample training params 
                x_0, y_0 = train_data_loader.get_batch()
                labels_embedded = self.label_embedding(y_0)
                
                t = torch.randint(1, self.T, (batch_size, 1), dtype=torch.int64, device=x_0.device)
                # print(f"Shape of t passed to network: {t.shape}")

                epsilon = torch.randn_like(x_0, device=self.device)   #N(0,1)
                # print(epsilon.shape)
                # print(x_0.shape)
                
                #create noisy observation
                alpha_ = [self.alphas_cumulative[time] for time in t]
                # MNIST version:
                # alpha_ = torch.tensor(alpha_, device=self.device).view(-1, 1)

                # CIFAR-10 version:
                alpha_ = torch.tensor(alpha_, device=self.device).view(-1, 1, 1, 1)
                # print(alpha_.shape)
                x_noisy = torch.sqrt(alpha_) * x_0 + torch.sqrt(1 - alpha_) * epsilon
                # print(x_noisy.shape)

                
                # train
                # time = self.times[t].view(1, 1)
                # time = self.times[t]  # MNIST version
                
                time = self.times[t].squeeze(1)   # CIFAR10 version -> otherwise the dimension is [batch_size, 1, embedding_dim], instead of just [batch_size, embedding_dim]
                # print(f"Shape of t passed to network: {time.shape}")

                # with a given probability, train condionally (use the classifier-free guidance)
                if self.cfg and torch.rand(1) > self.p_unconditional:
                    epsilon_hat = self.network(x_noisy, time, labels_embedded)
                else:
                    epsilon_hat = self.network(x_noisy, time)      #forward pass
                    # print(f"epsilon_hat.shape: {epsilon_hat.shape}, epsilon.shape: {epsilon.shape}")
                
                batch_loss = self.loss(epsilon, epsilon_hat)    #calculate loss
                opt.zero_grad()                                 #reset grad
                batch_loss.backward()                           #backprop
                opt.step()                                      #train params
                current_loss.append(batch_loss.item())
            
            # Log and reset data loader
            losses.append(np.mean(current_loss))
            train_data_loader.reset() # IMPORTANT!!!!

            print(f"Epoch {i+1} | Loss {losses[-1]}", flush=True)

            # Save checkpoint every 50 epochs
            if (i + 1) % 50 == 0:
                checkpoint_filename = f"jan_experiments/CIFAR10-model-500-epochs/CIFAR_epoch_{i+1}"
                self.save(checkpoint_filename)
                print(f"Checkpoint saved at epoch {i+1}: {checkpoint_filename}")

        return losses


    def sample(self, shape: tuple=(1, 3, 32, 32), label=None, w=-1):
        """Sample from the DDPM.

        Args:
            shape (tuple, optional): Shape of the output. Defaults to (1, 3, 32, 32).
            label (_type_, optional): Label to generate. Defaults to None.
            w (int, optional): Guidance strength. Defaults to -1, which means to use an unconditional DDPM.

        Returns:
            torch.Tensor: The generated image
        """
        if label is not None:
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long, device=self.device)
            label = self.label_embedding(label)
            label = label.view(1, -1)
          
        with torch.no_grad(): #turn of grad 
            self.network.eval() #turn of dropout and similar

            #generate noise sample
            x_T = x_previous_t = torch.randn(shape, device=self.device) * np.sqrt(self.alphas_cumulative[-1].item())
            x_0 = None

            if self.times is None:
                if self.scheduler == "cosine":
                    self.calculate_times_cosine()
                else:
                    self.calculate_times_linear()

            #removing the noise for each transition
            for t in range(self.T-1, 0, -1):

                #set noise 
                z =  torch.zeros_like(x_T)      #special case when it's the last transition 1->0
                if t > 1:
                    z = torch.randn_like(x_T)   #otherwise, N(0,1) 

                #remove noise for this timestep transition
                alpha_t = self.alphas[t]
                beta_t = self.betas[t]
                alpha_cum_t = self.alphas_cumulative[t].item()
                variance_t = beta_t             #variance of p_theta we have to choose based on x_0 - for now this since x_0 ~ N(0,I) 
                
                #ALTERNATIVE VARIANCE
                #alpha_cum_t_minus_1 = self.alphas_cumulative[t - 1].item()  # Cumulative product up to t-1
                #variance_t = beta_t * (1 - alpha_cum_t_minus_1) / (1 - alpha_cum_t)

                
                # TIME EMBEDDING

                # CIFAR-10 version
                time = self.times[t].unsqueeze(0)   # some random bugfix, from default 1, 728 to 1, 3, 32, 32
                # depending on w, use the conditional or unconditional
                if w == -1:             # unconditional only (REGULAR DDPM)
                    epsilon_hat = self.network(x_previous_t, time)
                elif w == 0:            # conditional only
                    epsilon_hat = self.network(x_previous_t, time, label)
                else:                   # classifier-free by interpolation between cond. and uncond.
                    epsilon_guided = self.network(x_previous_t, time, label)
                    epsilon_unguided = self.network(x_previous_t, time)
                    epsilon_hat = torch.lerp(epsilon_unguided, epsilon_guided, w)
                    # epsilon_hat = (1 + w) * epsilon_guided - w * epsilon_unguided
                # x_previous_t = (1/np.sqrt(alpha_t))*(x_previous_t - ((1-alpha_t)/np.sqrt(1-alpha_cum_t))*epsilon_hat) + (variance_t*z)
                x_previous_t = (1 / np.sqrt(alpha_t)) * (x_previous_t - ((1 - alpha_t) / np.sqrt(1 - alpha_cum_t)) * epsilon_hat) + (np.sqrt(variance_t) * z)

                x_0 = x_previous_t #remember last for return
                
                # print(f"timestep: {t}, alpha_t: {alpha_t}, alpha_cum_t: {alpha_cum_t}, variance_t: {variance_t}")


        #return final calculated x_0

        # TODO: check if this deormalization is ok
        # Denormalize: Reverse the normalization applied during training
        # x_0 = (x_0 + 1) / 2  # Scale from [-1, 1] to [0, 1] # -> this assumes [-1, 1]
        x_0 = (x_0 - x_0.min().item()) / (x_0.max().item() - x_0.min().item())  # this is general scaling to [0, 1]
        
        # print(f"x_0 min: {x_0.min().item()}, x_0 max: {x_0.max().item()}")

        return x_0


    def calculate_times_cosine(self):
        # Create linearly spaced time steps [0, T)
        t = torch.arange(self.T, dtype=torch.float32, device=self.device).view(-1, 1)

        # Calculate sinusoidal embeddings for each time step
        half_dim = self.embedding_dim // 2
        freqs = torch.exp(-torch.arange(half_dim, dtype=torch.float32, device=self.device) * math.log(10000) / half_dim)
        sinusoidal_emb = t * freqs[None, :]  # Broadcast frequencies across time steps

        # Combine sine and cosine embeddings
        self.times = torch.cat([sinusoidal_emb.sin(), sinusoidal_emb.cos()], dim=-1)
        print(self.times.shape)


    def calculate_times_linear(self):
        t = torch.arange(self.T, dtype=torch.int64, device=self.device).view(-1, 1)
        self.times = torch.tensor(np.tile(t.cpu(), (1, 28))).to(self.device)

  
    def save(self, path):
        torch.save(self.network.state_dict(), path + "_network.pth")
        np.save(path + "_T.npy", np.asarray(self.T))
        torch.save(self.label_embedding.state_dict(), f"{path}_label_embedding.pth")


    def load(self, path, load_T=False):
        self.network.load_state_dict(torch.load(f"{path}_network.pth", weights_only=False))
        try:
            self.label_embedding.load_state_dict(torch.load(f"{path}_label_embedding.pth", weights_only=False))
        except:
            print("No label embedding found")
        if load_T:
            self.T = np.load(f"{path}_T.npy").item()


          
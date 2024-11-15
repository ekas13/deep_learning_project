import torch
import numpy as np
from data.data_loader import data_loader

class DDPM():
    def __init__(self, device, network: torch.nn.Module, T: int = 1000):
        self.device = device
        self.network = network.to(device)
        self.T = T 
        self.betas = torch.linspace(0.0001, 0.02, T)
        self.alphas = 1 - self.betas
        self.loss = torch.nn.MSELoss()
        self.alphas_cumulative = self.alpha_calculate_cumulative()
        self.times = None

    def alpha_calculate_cumulative(self):
        # TODO: Check index
        alphas_cum = [self.alphas[0]]
        for i in range(1, len(self.alphas)):
            alphas_cum.append(alphas_cum[i-1] * self.alphas[i])
        
        return torch.tensor(alphas_cum).to(self.device)
    
    def reset(self):
        self.network.reset()
    
    def train(self, dataset_name: str, num_epochs: int, batch_size: int, opt: torch.optim.Optimizer):
        train_data_loader = data_loader(dataset_name, batch_size, self.device)
        losses = []

        previous_loss = 0
        for i in range(num_epochs):
            current_loss = []
            while train_data_loader.has_next_batch():
                #sample training params 
                batch = train_data_loader.get_batch()
                x_0 = batch.view(batch_size, 28*28)

                t = torch.randint(1, self.T, (batch_size, 1), dtype=torch.int64, device=x_0.device)

                epsilon = torch.randn_like(x_0, device=self.device)   #N(0,1)
                
                #create noisy observation
                alpha_ = [self.alphas_cumulative[time] for time in t]
                alpha_ = torch.tensor(alpha_, device=self.device).view(-1, 1)  
                x_noisy = torch.sqrt(alpha_)*x_0 + torch.sqrt(1-alpha_)*epsilon
                
                #train
                epsilon_hat = self.network(x_noisy, t)          #forward pass
                batch_loss = self.loss(epsilon, epsilon_hat)    #calculate loss
                opt.zero_grad()                                 #reset grad
                batch_loss.backward()                           #backprop
                opt.step()                                      #train params
                current_loss.append(batch_loss.item())
            losses.append(np.mean(current_loss))
            train_data_loader.reset()

            print(f"Epoch {i+1} | Loss {losses[-1]}")
            # if abs(previous_loss - current_loss) < convergence_threshold:
            #     break
            # previous_loss = current_loss

        return losses

    def sample(self):
        with torch.no_grad(): #turn of grad 
            self.network.eval() #turn of dropout and similar

            #generate noise sample
            x_T = x_previous_t = torch.randn(1, 784, device=self.device)
            x_0 = None

            if self.times is None:
                self.calculate_times()

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

                time = self.times[t].view(1, 1)
                epsilon_hat = self.network(x_previous_t, time)
                print(epsilon_hat)
                # x_previous_t = (1/np.sqrt(alpha_t))*(x_previous_t - ((1-alpha_t)/np.sqrt(1-alpha_cum_t))*epsilon_hat) + (variance_t*z)
                x_previous_t = (1 / np.sqrt(alpha_t)) * (x_previous_t - ((1 - alpha_t) / np.sqrt(1 - alpha_cum_t)) * epsilon_hat) + (np.sqrt(variance_t) * z)
                
                x_0 = x_previous_t #remember last for return

        #return final calculated x_0
        return x_0
    
    def calculate_times(self):
        self.times = torch.arange(self.T, dtype=torch.int64, device=self.device).view(-1, 1)

    
    def save(self, path):
        torch.save(self.network.state_dict(), path + "_network.pth")
        np.save(path + "_T.npy", np.asarray(self.T))

    def load(self, path, load_T=False):
        self.network.load_state_dict(torch.load(f"{path}_network.pth"))
        
        if load_T:
            self.T = np.load(f"{path}_T.npy").item()
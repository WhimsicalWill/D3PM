import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import numpy as np
from plotting import plot_swiss_roll, plot_noising_process
from utils import get_diffusion_betas


class NoisySwissRollDataset(data.Dataset):
    def __init__(self, betas, n_schedule_steps, n_states):
        self.n_samples = 10000
        self.betas = betas
        self.n_schedule_steps = n_schedule_steps
        self.n_states = n_states
        self.absorbing_idx = n_states
        self.data = self.create_quantized_swiss_roll()

    def create_quantized_swiss_roll(self, noise=0.2, random_state=None):
        data, _ = make_swiss_roll(self.n_samples, noise=noise, random_state=random_state)
        data = data[:, [0, 2]]  # we only take the first and last column.
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)

        # Normalize and quantize the data
        data = (data - min_val) / (max_val - min_val)
        data = np.floor(data * self.n_states)  # Quantize to 256 bins
        return torch.Tensor(data).long()

    def apply_noising(self, data_point, t):
        noise_level = self.betas[t]
        noisy_data = data_point.clone().detach()
        noisy_data += (torch.rand(2) < noise_level).long() * (self.absorbing_idx - noisy_data)
        return noisy_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t = torch.randint(0, self.n_schedule_steps, (1,)).item()
        data_point = self.data[idx]
        return self.apply_noising(data_point, t), t


class DiffusionModel(nn.Module):
    def __init__(self, n_features, n_states, n_schedule_steps):
        super(DiffusionModel, self).__init__()
        # A simple network with two hidden layers and ReLU activations
        self.n_hidden = 128
        self.n_schedule_steps = n_schedule_steps
        self.network = nn.Sequential(
            nn.Linear(3, self.n_hidden),  # 2 features + 1 for time embedding
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, n_features * (n_states + 1))  # Output logits for all features
        )
    
    def forward(self, x, t):
        t = t.float() / self.n_schedule_steps  # Normalize timesteps
        t_embed = t.view(-1, 1)  # Reshape t to have the correct shape (batch_size, 1)
        x = torch.cat([x.float(), t_embed], dim=1)
        return self.network(x)


class SwissRollDiffusion():
    def __init__(self):
        self.n_samples = 500  # number of samples for swiss roll dataset
        self.n_schedule_steps = 100  # number of steps for the diffusion process
        self.n_epochs = 20  # number of epochs for training
        self.n_states = 256  # number of states for the swiss roll data
        self.n_features = 2  # number of features (e.g. x and y coordinate)
        self.absorbing_idx = self.n_states  # index of the absorbing state
        self.betas = get_diffusion_betas({'type': 'jsd', 'num_timesteps': self.n_schedule_steps})
        self.eps = 1e-6  # small value to avoid division by zero
        self.q_one_step_mats = self.init_transition_matrices()
        self.noisy_dataset = NoisySwissRollDataset(self.betas, self.n_schedule_steps, self.n_states)
        self.dataset_loader = torch.utils.data.DataLoader(self.noisy_dataset, batch_size=128, shuffle=True)

    def train(self):
        model = DiffusionModel(self.n_features, self.n_states, self.n_schedule_steps)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()  # Assumes classification task for states
        model.train()
        
        for epoch in range(self.n_epochs):
            for i, (noisy_data, timesteps) in enumerate(self.dataset_loader):
                optimizer.zero_grad()
                predictions = model(noisy_data, timesteps)
                
                # Split predictions into separate parts for x and y
                predictions_x = predictions[:, :(self.n_states + 1)]
                predictions_y = predictions[:, (self.n_states + 1):]
                
                # Assume the targets are the original x and y indices
                loss_x = criterion(predictions_x, noisy_data[:, 0].long())  # Target for x
                loss_y = criterion(predictions_y, noisy_data[:, 1].long())  # Target for y
                loss = loss_x + loss_y  # Total loss is the sum of both losses
                
                loss.backward()
                optimizer.step()
                
                print(f"Epoch [{epoch+1}/{self.n_epochs}], Step [{i+1}/{len(self.dataset_loader)}], Loss: {loss.item():.4f}")

    def init_transition_matrices(self):
        """Initialize transition matrices with an absorbing state."""
        # Set values directly to avoid numerical error with betas
        transition_matrices = []
        for beta in self.betas:
            matrix = torch.zeros((self.n_states + 1, self.n_states + 1))  # +1 for absorbing state
            matrix.fill_diagonal_(1 - beta)
            matrix[:, self.absorbing_idx] = beta
            matrix[self.absorbing_idx, self.absorbing_idx] = 1
            transition_matrices.append(matrix)
        return torch.stack(transition_matrices)

    def compute_kl_divergence(self, model, data, timesteps):
        model_output = model(data, timesteps)
        # Assume model_output is logits of the reverse transition probabilities
        # Calculate the forward transition probabilities based on your transition matrices
        forward_probs = ...  # Extracted from q_one_step_mats based on current state and timesteps
        model_probs = torch.softmax(model_output, dim=-1)
        kl_div = torch.sum(model_probs * (torch.log(model_probs + self.eps) - torch.log(forward_probs + self.eps)), dim=-1)
        return kl_div.mean()

    def q_posterior_logits(self, transpose_q_mats, q_mats, x_t, t, x_start):
        """Compute the logits of q(x_{t-1} | x_t, x_start)."""
        # transpose_q_mats and q_mats are precomputed transition matrices
        fact1 = transpose_q_mats[t][x_t]  # Transition probabilities from x_t
        fact2 = q_mats[t-1][x_start] if t > 0 else torch.zeros_like(fact1)
        logits = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        return logits

    def visualize_noising_process(self):
        dataset = NoisySwissRollDataset(self.betas, self.n_schedule_steps, self.n_states)
        vis_steps = 5  # Number of visualization steps
        colors = np.arctan2(data[:, 0], data[:, 1]) / self.n_states  # use original data for coloring
        plot_noising_process(dataset, vis_steps, colors, self.n_states, self.n_schedule_steps)


if __name__ == "__main__":
    swiss_roll_diffusion = SwissRollDiffusion()
    swiss_roll_diffusion.train()
    # swiss_roll_diffusion.visualize_noising_process()
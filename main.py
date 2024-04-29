import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import numpy as np
from plotting import plot_swiss_roll, plot_noising_process
import utils


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

    # def apply_noising(self, data_point, t):
    #     noise_level = self.betas[t]
    #     noisy_data = data_point.clone().detach()
    #     noisy_data += (torch.rand(2) < noise_level).long() * (self.absorbing_idx - noisy_data)
    #     return noisy_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # t = torch.randint(0, self.n_schedule_steps, (1,)).item()
        # data_point = self.data[idx]
        # return self.apply_noising(data_point, t), t
        return self.data[idx]


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
        self.betas = utils.get_diffusion_betas({'type': 'jsd', 'num_timesteps': self.n_schedule_steps})
        self.eps = 1e-6  # small value to avoid division by zero
        self.noisy_dataset = NoisySwissRollDataset(self.betas, self.n_schedule_steps, self.n_states)
        self.dataset_loader = torch.utils.data.DataLoader(self.noisy_dataset, batch_size=128, shuffle=True)

        # Initialize the one-step and cumulative transition matrices
        self.q_one_step_mats = self.init_transition_matrices()
        self.q_cum_mats = self.init_cum_transition_matrices()

    def train(self):
        model = DiffusionModel(self.n_features, self.n_states, self.n_schedule_steps)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()  # Assumes classification task for states
        model.train()
        
        for epoch in range(self.n_epochs):
            for i, (x_start, t) in enumerate(self.dataset_loader):
                optimizer.zero_grad()
                predictions = model(noisy_data, timesteps)
                
                # Split predictions into separate parts for x and y
                predictions_x = predictions[:, :(self.n_states + 1)]
                predictions_y = predictions[:, (self.n_states + 1):]
                
                # Assume the targets are the original x and y indices
                loss_x = criterion(predictions_x, noisy_data[:, 0].long())  # Target for x
                loss_y = criterion(predictions_y, noisy_data[:, 1].long())  # Target for y

                true_logits_x = self.q_posterior_logits(..., noisy_data[:, 0].long, timesteps, x_start)  # Need to pass correct params
                true_logits_y = self.q_posterior_logits(..., noisy_data[:, 1].long, timesteps, x_start)  # Need to pass correct params
                loss_x = compute_kl_divergence(predictions_x, true_logits_x)
                loss_y = compute_kl_divergence(predictions_y, true_logits_y)
                loss = loss_x + loss_y

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

    def init_cum_transition_matrices(self):
        q_cum_mats = [self.q_one_step_mats[0]]
        for t in range(1, len(self.q_one_step_mats)):
            cumulative_mat = torch.matmul(q_cum_mats[-1], self.q_one_step_mats[t])
            q_cum_mats.append(cumulative_mat)
        return torch.stack(q_cum_mats)

    def q_posterior_logits(self, x_t, t, x_start, x_start_logits=False):
        """
        TODO: fix indexing of x_t (needs to be int or long)
        Compute the logits of q(x_{t-1} | x_t, x_start).

        Regarding x_start, we can either be passed the true onehot vector or the logits predicted from the model.
        fact1 is q(x_t | x_{t-1}), the probability of transitioning from x_{t-1} to x_t.
        fact2 is q(x_{t-1} | x_0), the probability of transitioning from x_0 to x_{t-1}.
        fact1 is generally harder because we are conditioning on a non-observed variable
        see: https://beckham.nz/2022/07/11/d3pms.html#outline-container-org36077cc

        Args:
            x_t: Tensor of shape (batch_size,) representing the noisy data at time t.
            t: Tensor of shape (batch_size,) representing the timestep.
            x_start: Tensor of shape (batch_size, n_states + 1) or (batch_size,) representing the starting distribution.
            x_start_logits: Boolean indicating whether x_start is the predicted logits or the indices.
        Returns:
            logits: Tensor of shape (batch_size, n_states + 1) representing the logits of the posterior distribution.
        """
        assert x_t.shape == (x_t.shape[0], self.n_states + 1)
        assert t.shape == (x_t.shape[0],)
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.n_states + 1,)
            # x_start is logits, convert to probabilities for computation
            x_start_probs = F.softmax(x_start, dim=-1)  # (B, D + 1), D = n_states
            # self.q_cum_mats[t-1]  (D + 1, D + 1)
            # x @ Q = (Q @ x^T)^T  (D + 1, B) ^ T = (B, D + 1)
            fact2 = torch.matmul(self.q_cum_mats[t-1], x_start_probs.transpose(0, 1)).transpose(0, 1)
            tzero_logits = x_start  # Directly use logits if x_start is logits
        else:
            assert x_start.shape == x_t.shape
            # x_start is indices, convert to one-hot for computation
            x_start_one_hot = F.one_hot(x_start, num_classes=self.n_states + 1)  # (B, D + 1)
            fact2 = torch.matmul(self.q_cum_mats[t-1], x_start_one_hot.float().transpose(0, 1)).transpose(0, 1)
            tzero_logits = torch.log(x_start_one_hot.float() + self.eps)  # Convert one-hot to log probabilities

        # Extract probabilities for x_t from transpose of one-step matrices
        # q_one_step_mats (T, D, D)
        # q_one_step_mats[t] (D, D)
        # x_t (B, D) where D is n_states + 1
        # fact1 is complicated because we are conditioning on x_
        # Use advanced indexing to pull out the timestep t and 
        fact1 = self.q_one_step_mats.transpose(0, 2, 1)[t, x_t]  # (B, D + 1)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)
        t_broadcast = t.unsqueeze(1).expand_as(out)
        return torch.where(t_broadcast == 0, tzero_logits, out)

    def compute_loss(self, x_start, x_t, t):
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        pred_logits, pred_x_start_logits = self.p_logits(x=x_t, t=t)

        kl = utils.categorical_kl_logits(logits1=true_logits, logits2=pred_logits)
        assert kl.shape == x_start.shape
        kl = utils.meanflat(kl) / np.log(2.)  # (B,)

        decoder_nll = -utils.categorical_log_likelihood(x_start, pred_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = utils.meanflat(decoder_nll) / np.log(2.)  # (B,)

        loss = torch.where(t == 0, decoder_nll, kl)  # (B,)
        return loss.mean()  # scalar

    def visualize_noising_process(self):
        dataset = NoisySwissRollDataset(self.betas, self.n_schedule_steps, self.n_states)
        vis_steps = 5  # Number of visualization steps
        colors = np.arctan2(data[:, 0], data[:, 1]) / self.n_states  # use original data for coloring
        plot_noising_process(dataset, vis_steps, colors, self.n_states, self.n_schedule_steps)


if __name__ == "__main__":
    swiss_roll_diffusion = SwissRollDiffusion()
    swiss_roll_diffusion.train()
    # swiss_roll_diffusion.visualize_noising_process()
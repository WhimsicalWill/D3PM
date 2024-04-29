import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import numpy as np
from plotting import visualize_reverse_noising_process
import utils


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SwissRollDataset(data.Dataset):
    def __init__(self, betas, n_schedule_steps, n_states):
        self.n_samples = 1_000_000
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DiffusionModel(nn.Module):
    def __init__(self, n_features, n_states, n_schedule_steps):
        super(DiffusionModel, self).__init__()
        # A simple network with two hidden layers and ReLU activations
        self.n_hidden = 128
        self.n_features = n_features
        self.n_states = n_states
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
        logits = self.network(x)
        # Reshape logits to have separate outputs for each feature
        return logits.view(-1, self.n_features, self.n_states + 1)  # (B, F, D + 1)


class SwissRollDiffusion():
    def __init__(self):
        self.n_schedule_steps = 100  # number of steps for the diffusion process
        self.n_epochs = 2  # number of epochs for training
        self.n_states = 256  # number of states for the swiss roll data
        self.n_features = 2  # number of features (e.g. x and y coordinate)
        self.absorbing_idx = self.n_states  # index of the absorbing state
        self.betas = utils.get_diffusion_betas({'type': 'jsd', 'num_timesteps': self.n_schedule_steps})
        self.eps = 1e-6  # small value to avoid division by zero
        self.noisy_dataset = SwissRollDataset(self.betas, self.n_schedule_steps, self.n_states)
        self.dataset_loader = torch.utils.data.DataLoader(self.noisy_dataset, batch_size=512, shuffle=True)

        # Initialize the one-step and cumulative transition matrices
        self.q_one_step_mats = self.init_transition_matrices()
        self.q_cum_mats = self.init_cum_transition_matrices()
        self.model = DiffusionModel(self.n_features, self.n_states, self.n_schedule_steps).to(device)

    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()

        # Initialize lists or logging mechanism for loss components
        log_kl = []
        log_nll = []
        
        for epoch in range(self.n_epochs):
            for i, x_start in enumerate(self.dataset_loader):
                # x_start has shape (B, F)

                x_start = x_start.to(device)
                noise = torch.rand(x_start.shape + (self.n_states + 1,)).to(device)  # (B, F, D + 1)
                t = torch.randint(0, self.n_schedule_steps, (x_start.shape[0],)).to(device)  # (B,)
                x_t = self.q_sample(x_start, t, noise).to(device) # (B, F)

                optimizer.zero_grad()
                total_loss, kl_loss, nll_loss = self.compute_loss(x_start, x_t, t)
                total_loss.backward()
                optimizer.step()

                # Log loss components for each step or periodically
                log_kl.append(kl_loss.item())
                log_nll.append(nll_loss.item())
                
                # Print detailed loss information periodically
                if i % 50 == 0:  # Adjust the frequency as needed
                    print(f"Epoch {epoch+1}, Step {i+1}, Total Loss: {total_loss.item():.4f}, KL: {kl_loss.item():.4f}, NLL: {nll_loss.item():.4f}")

    def init_transition_matrices(self):
        """Initialize transition matrices with an absorbing state."""
        # Set values directly to avoid numerical error with betas
        transition_matrices = []
        for beta in self.betas:
            matrix = torch.zeros((self.n_states + 1, self.n_states + 1), device=device)
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

    def q_sample(self, x_start, t, noise):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).

        Args:
            x_start: jnp.ndarray: original clean data, in integer form (not onehot).
                shape = (bs, ...).
            t: :jnp.ndarray: timestep of the diffusion process, shape (bs,).
            noise: jnp.ndarray: uniform noise on [0, 1) used to sample noisy data.
                Should be of shape (*x_start.shape, num_pixel_vals).

        Returns:
            sample: jnp.ndarray: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.n_states + 1,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues clip the noise to a minimum value
        noise = torch.clamp(noise, min=1e-6, max=1.)
        gumbel_noise = - torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, axis=-1)

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
            x_start: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
                Should not be of one hot representation, but have integer values
                representing the class values.
            t: jnp.ndarray: jax array of shape (bs,).

        Returns:
            probs: jnp.ndarray: jax array, shape (bs, x_start.shape[1:],
                                                    num_pixel_vals).
        """
        return self._at(self.q_cum_mats, t, x_start)

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """
        Compute the logits of q(x_{t-1} | x_t, x_start).

        Regarding x_start, we can either be passed the true onehot vector or the logits predicted from the model.
        fact1 is q(x_t | x_{t-1}), the probability of transitioning from x_{t-1} to x_t.
        fact2 is q(x_{t-1} | x_0), the probability of transitioning from x_0 to x_{t-1}.
        fact1 is generally harder because we are conditioning on a non-observed variable
        see: https://beckham.nz/2022/07/11/d3pms.html#outline-container-org36077cc

        Args:
            x_start: Tensor of shape (batch_size, n_states + 1) or (batch_size,) representing the starting distribution.
            x_t: Tensor of shape (batch_size,) representing the noisy data at time t.
            t: Tensor of shape (batch_size,) representing the timestep.
            x_start_logits: Boolean indicating whether x_start is the predicted logits or the indices.
        Returns:
            logits: Tensor of shape (batch_size, n_states + 1) representing the logits of the posterior distribution.
        """
        # if x_start_logits:
        #     assert x_start.shape == x_t.shape + (self.n_states + 1,)
        #     # x_start is logits, convert to probabilities for computation
        #     x_start_probs = F.softmax(x_start, dim=-1)  # (B, D + 1), D = n_states
        #     # self.q_cum_mats[t-1]  (D + 1, D + 1)
        #     # x @ Q = (Q @ x^T)^T  (D + 1, B) ^ T = (B, D + 1)
        #     # fact2 = torch.matmul(self.q_cum_mats[t-1], x_start_probs.transpose(0, 1)).transpose(0, 1)
        #     fact2 = torch.matmul(x_start_probs, self.q_cum_mats[t-1])
        #     tzero_logits = x_start  # Directly use logits if x_start is logits
        # else:
        #     assert x_start.shape == x_t.shape, f"Shapes do not match: {x_start.shape}, {x_t.shape}"
        #     # x_start is indices, convert to one-hot for computation
        #     x_start_one_hot = F.one_hot(x_start, num_classes=self.n_states + 1)  # (B, D + 1)
        #     # fact2 = torch.matmul(self.q_cum_mats[t-1], x_start_one_hot.float().transpose(0, 1)).transpose(0, 1)
        #     fact2 = torch.matmul(x_start_one_hot.float(), self.q_cum_mats[t-1])  # (128, 2, 257) @ (128, 257, 257) = (128, 2, 257)
        #     tzero_logits = torch.log(x_start_one_hot.float() + self.eps)  # Convert one-hot to log probabilities

        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.n_states + 1,)
            fact2 = self._at_onehot(self.q_cum_mats, t-1, F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            assert x_start.shape == x_t.shape, f"Shapes do not match: {x_start.shape}, {x_t.shape}"
            # print(f"Shapes: x_start={x_start.shape}, x_t={x_t.shape}")
            fact2 = self._at(self.q_cum_mats, t-1, x_start)
            tzero_logits = torch.log(F.one_hot(x_start, num_classes=self.n_states + 1) + self.eps)

        fact1 = self._at(torch.transpose(self.q_one_step_mats, 2, 1), t, x_t)  # (B, F, D + 1)
        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        # print(f"x_start_logits: {x_start_logits}")
        # print(f"fact1: {fact1[0][0][:]}")
        # print(f"fact2: {fact2[0][0][:]}")
        # print(f"out: {out[0][0][:]}")

        t_broadcast = t.unsqueeze(1).unsqueeze(2).expand(-1, *out.shape[1:])
        return torch.where(t_broadcast == 0, tzero_logits, out)

    def p_logits(self, x, t):
        """Compute logits of p(x_{t-1} | x_t)."""
        assert t.shape == (x.shape[0],)
        pred_logits = self.model(x, t)  # (B, F, D + 1)

        # print(f"Before prediction: x0 = {x[0][0]}")
        # print(f"After1 prediction: x0 = {pred_logits[0][0][:5]}")

        # Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
        # as ~ sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t)
        # TODO: clean up t_broadcast operations through a nicer operation like np.expand_dims()
        t_broadcast = t.unsqueeze(1).unsqueeze(2).expand(-1, *pred_logits.shape[1:])
        pred_logits = torch.where(t_broadcast == 0,
                                pred_logits,
                                self.q_posterior_logits(pred_logits, x,
                                                        t, x_start_logits=True)
                                )

        # print(f"After2 prediction: x0 = {pred_logits[0][0][:5]}")

        assert pred_logits.shape == x.shape + (self.n_states + 1,)
        return pred_logits

    def compute_loss(self, x_start, x_t, t):
        """Computes the loss using KL divergence between the true and predicted distributions.
        
        Args:
            x_start: Tensor of shape (batch_size,) representing the clean data.
            x_t: Tensor of shape (batch_size,) representing the noisy data.
            t: Tensor of shape (batch_size,) representing the timestep.
        """
        assert x_start.shape == x_t.shape, f"Shapes do not match: {x_start.shape}, {x_t.shape}"
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        pred_logits = self.p_logits(x_t, t)

        kl = utils.categorical_kl_logits(logits1=true_logits, logits2=pred_logits)
        assert kl.shape == x_start.shape
        kl = utils.meanflat(kl) / torch.log(torch.tensor(2.0))  # (B,)

        decoder_nll = -utils.categorical_log_likelihood(x_start, pred_logits)
        assert decoder_nll.shape == x_start.shape
        decoder_nll = utils.meanflat(decoder_nll) / torch.log(torch.tensor(2.0))  # (B,)

        total_loss = torch.where(t == 0, decoder_nll, kl).mean()
        return total_loss, kl.mean(), decoder_nll.mean()

    def _at(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
        a: np.ndarray: plain NumPy float64 array of constants indexed by time.
        t: jnp.ndarray: Jax array of time indices, shape = (batch_size,).
        x: jnp.ndarray: jax array of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one hot representation, but have integer
            values representing the class values.

        Returns:
        a[t, x]: jnp.ndarray: Jax array.
        """
        # t_broadcast = np.expand_dims(t, tuple(range(1, x.ndim)))
        t_broadcast = t.unsqueeze(1).expand(-1, *x.shape[1:])
        return a[t_broadcast, x]

    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
        a: np.ndarray: plain NumPy float64 array of constants indexed by time.
        t: jnp.ndarray: Jax array of time indices, shape = (bs,).
        x: jnp.ndarray: jax array, shape (bs, ..., num_pixel_vals), float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
        out: jnp.ndarray: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, ..., num_pixel_vals)
        """
        return torch.matmul(x, a[t])

    def p_sample(self, x, t, tau=0.1, tau_nonzero=1.0):
        """Sample one timestep from the model p(x_{t-1} | x_t) using PyTorch."""
        # Compute the logits for the current state x at time t
        pred_logits = self.p_logits(x, t)  # Assuming p_logits returns logits of shape (batch_size, num_features, num_states + 1)

        # print the last 5 logits of the x coordinate for x0
        # print(pred_logits[0][0][-5:])

        # create a mask to handle the no-noise condition when t == 0
        nonzero_mask = (t > 0).float().unsqueeze(-1).unsqueeze(-1)  # reshape to (batch_size, 1, 1) for broadcasting

        # Apply softmax with a low temperature for t=0, and a higher temperature for t>0
        temperatures = torch.full_like(t, tau_nonzero, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        temperatures[nonzero_mask == 0] = tau  # Set a lower temperature for t=0

        # Calculate the softmax over the logits, scaled by the temperature
        # pred_logits / temperatures adjusts logits according to the temperature
        probs = F.softmax(pred_logits / temperatures, dim=-1)

        # Sample from the categorical distribution based on the computed probabilities
        # torch.multinomial to sample from the categorical distribution described by probs
        sample = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1])

        return sample

    def sample_reverse(self, num_samples=128, vis_steps=5):
        """Perform reverse sampling to generate data from the model."""
        device = next(self.model.parameters()).device  # Get the device model is on

        # Initialize x with the absorbing state for all features
        x = torch.full((num_samples, self.n_features), self.absorbing_idx, dtype=torch.long, device=device)

        snapshots = []
        step_indices = np.linspace(0, self.n_schedule_steps, num=vis_steps, endpoint=False, dtype=int)

        # Iterate over timesteps backwards from T-1 to 0
        for t in reversed(range(self.n_schedule_steps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)

            # Sample the previous state given the current state x at time t
            x = self.p_sample(x, t_tensor)  # Update x for each timestep
            # print(f"timestep={t}, x0 = {x[0]}")

            if t in step_indices:
                snapshots.append(x.clone().detach())  # Clone and detach to avoid modifications

        return snapshots

    def visualize_samples(self, samples):
        """Visualize generated samples."""
        x, y = samples[:, 0].numpy(), samples[:, 1].numpy()
        plt.scatter(x, y, alpha=0.5)
        plt.title('Reverse Diffusion Generated Swiss Roll')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.show()


if __name__ == "__main__":
    swiss_roll_diffusion = SwissRollDiffusion()
    swiss_roll_diffusion.train()
    # swiss_roll_diffusion.visualize_noising_process()

    # Sample new data points
    snapshots = swiss_roll_diffusion.sample_reverse(num_samples=500, vis_steps=5)
    visualize_reverse_noising_process(snapshots, swiss_roll_diffusion.n_states, swiss_roll_diffusion.n_schedule_steps)
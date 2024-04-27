import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt


N_SAMPLES = 500  # number of samples for swiss roll dataset
N_SCHEDULE_STEPS = 100  # number of steps for the diffusion process
N_HIDDEN = 128  # number of hidden units in the neural network
N_STATES = 256  # number of states for the swiss roll data
ABSORBING_STATE = N_STATES  # index of the absorbing state

def create_quantized_swiss_roll(n_samples=1000, noise=0.2, random_state=None):
    data, _ = make_swiss_roll(n_samples, noise=noise, random_state=random_state)
    data = data[:, [0, 2]]  # we only take the first and last column.
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # Normalize and quantize the data
    data = (data - min_val) / (max_val - min_val)
    data = np.floor(data * N_STATES)  # Quantize to 256 bins

    # Add an absorbing state column initialized to zero (no points absorbed yet)
    absorbing_state_column = np.zeros((n_samples, 1), dtype=int)
    data_with_absorbing_state = np.hstack((data, absorbing_state_column))
    return torch.Tensor(data_with_absorbing_state).long()

# Assume a linear noise schedule for simplicity
def noise_schedule(t):
    return t / N_SCHEDULE_STEPS

class NoisySwissRollDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def apply_noising(self, data_point, t):
        noise_level = noise_schedule(t)
        noisy_data = data_point.clone().detach()
        noisy_data[:-1] += (torch.rand(2) < noise_level).long() * (ABSORBING_STATE - noisy_data[:-1])
        if (noisy_data[:-1] >= ABSORBING_STATE).any():
            noisy_data[-1] = 1  # Mark the point as absorbed
        return noisy_data

    def __getitem__(self, idx):
        t = torch.randint(0, N_SCHEDULE_STEPS, (1,)).item()
        data_point = self.data[idx]
        return self.apply_noising(data_point, t), t

class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        # A simple network with two hidden layers and ReLU activations
        self.network = nn.Sequential(
            nn.Linear(2, N_HIDDEN),  # 2 input features, since it's 2D data
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, 2)  # 2 output features (denoised x, y)
        )
    
    def forward(self, x, t):
        # You could include the time information t in your network input if desired
        # For now, we just ignore t
        return self.network(x)

def train(model, dataset_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model.train()  # set the model to training mode
    
    for epoch in range(epochs):
        for i, (noisy_data, _) in enumerate(dataset_loader):
            optimizer.zero_grad()  # clear previous gradients
            prediction = model(noisy_data.float(), t=None)  # forward pass without time embedding for simplicity
            loss = criterion(prediction, noisy_data.float())  # compute loss
            loss.backward()  # backward pass to calculate gradients
            optimizer.step()  # update model parameters
            
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataset_loader)}], Loss: {loss.item():.4f}")

def sample(model, num_samples=100, num_steps=1000):
    samples = torch.zeros(num_samples, 2).long()  # Replace with actual sampling code
    # Implement the reverse diffusion process here
    return samples

def plot_swiss_roll(data, save_path="plots/swiss_roll.png"):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=np.arctan2(data[:, 0], data[:, 1]), cmap=plt.cm.Spectral)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Quantized Swiss Roll')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def visualize_data():
    # Create a quantized swiss roll dataset
    quantized_data = create_quantized_swiss_roll(n_samples=N_SAMPLES)

    # Convert to a suitable range for visualization
    quantized_data = quantized_data / 256.0
    quantized_data = quantized_data.numpy()

    # Plot and visualize the swiss roll
    plot_swiss_roll(quantized_data)

def visualize_noising_process():
    original_data = create_quantized_swiss_roll(n_samples=N_SAMPLES)
    dataset = NoisySwissRollDataset(original_data)

    vis_steps = 5  # Number of visualization steps

    # Calculate original colors based on initial positions
    original_colors = np.arctan2(original_data[:, 0], original_data[:, 1]) / N_STATES

    _, axes = plt.subplots(1, vis_steps, figsize=(15, 3))
    for i in range(vis_steps):
        time_step = int(i * (N_SCHEDULE_STEPS / (vis_steps - 1)))
        if time_step > N_SCHEDULE_STEPS:
            time_step = N_SCHEDULE_STEPS
        
        data_noised = np.array([
            dataset.apply_noising(dataset.data[j], time_step)[:2].numpy() / N_STATES
            for j in range(len(dataset))
        ])

        axes[i].scatter(data_noised[:, 0], data_noised[:, 1], c=original_colors, cmap=plt.cm.Spectral)
        axes[i].set_title(f'Time step: {time_step}')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("plots/noising_process.png")
    plt.show()

def main():
    # Generating the dataset
    dataset = create_quantized_swiss_roll()
    noisy_dataset = NoisySwissRollDataset(dataset)
    dataset_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=128, shuffle=True)

    # Instantiate the model
    model = DiffusionModel()

    # Training the model
    train(model, dataset_loader)

    # Generate new data points
    # generated_samples = sample(model)


if __name__ == "__main__":
    # visualize_data()
    # main()
    visualize_noising_process()
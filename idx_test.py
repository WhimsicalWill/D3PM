import torch

def main():
    # Parameters for the test
    num_timesteps = 10
    num_states = 256
    batch_size = 5

    # Initialize a tensor for q_cum_mats with random probabilities
    q_cum_mats = torch.rand(num_timesteps, num_states + 1, num_states + 1)

    # Normalize to make it a valid probability distribution
    q_cum_mats = q_cum_mats / q_cum_mats.sum(dim=2, keepdim=True)

    # Simulate some random data for x_start
    x_start = torch.randint(0, num_states, (batch_size,))

    # Simulate random timesteps for each sample
    t = torch.randint(0, num_timesteps, (batch_size,))

    # Perform advanced indexing
    probs = q_cum_mats[t, x_start]

    # Print the shapes to confirm correctness
    print("Shape of q_cum_mats:", q_cum_mats.shape)
    print("Shape of x_start:", x_start.shape)
    print("Shape of t:", t.shape)
    print("Shape of output probs:", probs.shape)

    # Output some probabilities to visually inspect them
    print("Sample probabilities:\n", probs[:2])  # Show probabilities for first two samples

if __name__ == "__main__":
    main()

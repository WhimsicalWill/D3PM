import numpy as np
import matplotlib.pyplot as plt


def plot_swiss_roll(data, save_path="plots/swiss_roll.png"):
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=np.arctan2(data[:, 0], data[:, 1]), cmap=plt.cm.Spectral)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Quantized Swiss Roll')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

# def plot_noising_process(dataset, vis_steps, colors, n_states, n_schedule_steps):
#     _, axes = plt.subplots(1, vis_steps, figsize=(15, 3))
#     for i in range(vis_steps):
#         time_step = int(i * (n_schedule_steps / (vis_steps - 1)))
#         if time_step > n_schedule_steps:
#             time_step = n_schedule_steps
        
#         data_noised = np.array([
#             dataset.apply_noising(dataset.data[j], time_step)[:2].numpy() / n_states
#             for j in range(len(dataset))
#         ])

#         axes[i].scatter(data_noised[:, 0], data_noised[:, 1], c=colors, cmap=plt.cm.Spectral)
#         axes[i].set_title(f'Time step: {time_step}')
#         axes[i].set_xlim(0, 1)
#         axes[i].set_ylim(0, 1)

#     plt.tight_layout()
#     plt.savefig("plots/noising_process.png")
#     plt.show()

def visualize_reverse_noising_process(samples_snapshots, n_states, n_schedule_steps):
    vis_steps = len(samples_snapshots)
    _, axes = plt.subplots(1, vis_steps, figsize=(15, 3))
    
    for i, samples in enumerate(reversed(samples_snapshots)):
        # Assume samples are normalized between 0 and n_states
        data = samples.cpu().numpy() / n_states
        
        axes[i].scatter(data[:, 0], data[:, 1], cmap=plt.cm.Spectral)
        axes[i].set_title(f'Time step: {i * (n_schedule_steps // (vis_steps - 1))}')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("plots/denoising_process.png")
    plt.show()

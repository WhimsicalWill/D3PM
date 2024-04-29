# Some parts taken directly from the official d3pm code base
# https://github.com/google-research/google-research/tree/master/d3pm
import numpy as np


def get_diffusion_betas(spec):
  """Get betas from the hyperparameters."""
  if spec['type'] == 'linear':
    # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
    # To be used with Gaussian diffusion models in continuous and discrete
    # state spaces.
    # To be used with transition_mat_type = 'gaussian'
    return np.linspace(spec['start'], spec['stop'], spec['num_timesteps'])
  elif spec['type'] == 'cosine':
    # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
    # To be used with transition_mat_type = 'uniform'.
    steps = (
        np.arange(spec['num_timesteps'] + 1, dtype=np.float64) /
        spec['num_timesteps'])
    alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
    betas = np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999)
    return betas
  elif spec['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
    # To be used with absorbing state models.
    # ensures that the probability of decaying to the absorbing state
    # increases linearly over time, and is 1 for t = T-1 (the final time).
    # To be used with transition_mat_type = 'absorbing'
    return 1. / np.linspace(spec['num_timesteps'], 1., spec['num_timesteps'])
  else:
    raise NotImplementedError(spec['type'])

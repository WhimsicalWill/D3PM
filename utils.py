# Some parts taken directly from the official d3pm code base
# https://github.com/google-research/google-research/tree/master/d3pm
import numpy as np
import torch
import torch.nn.functional as F


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

def categorical_kl_probs(probs1, probs2, eps=1e-6):
    """KL divergence between categorical distributions.

    Distributions parameterized by logits.

    Args:
        probs1: probs of the first distribution. Last dim is class dim.
        probs2: probs of the second distribution. Last dim is class dim.
        eps: float small number to avoid numerical issues.

    Returns:
        KL(C(probs) || C(logits2)): shape: logits1.shape[:-1]
    """
    out = probs1 * (np.log(probs1 + eps) - np.log(probs2 + eps))
    return np.sum(out, axis=-1)

def categorical_log_likelihood(x, logits):
    """Log likelihood of a discretized Gaussian specialized for image data.

    Assumes data `x` consists of integers [0, num_classes-1].

    Args:
        x: where to evaluate the distribution. shape = (bs, ...), dtype=int32/int64
        logits: logits, shape = (bs, ..., num_classes)

    Returns:
        log likelihoods
    """
    log_probs = F.log_softmax(logits)
    x_onehot = F.one_hot(x, logits.shape[-1])
    return np.sum(log_probs * x_onehot, axis=-1)

def meanflat(x):
    """Take the mean over all axes except the first batch dimension."""
    return x.mean(axis=tuple(range(1, len(x.shape))))
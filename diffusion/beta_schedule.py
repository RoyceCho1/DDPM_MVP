import torch

def linear_beta_schedule(timesteps):
    """
    Standard linear beta schedule from DDPM paper.
    beta ranges from 0.0001 to 0.02.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
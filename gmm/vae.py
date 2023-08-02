import math

import torch

_C = math.log(2 * math.pi)


def sample(mu, logvar):
    eps = torch.randn_like(mu)
    return mu + (logvar / 2).exp() * eps


def loss_function(x, mu_e, logvar_e, mu_d, logvar_d):
    reconstruction_loss = 0.5 * (_C + logvar_d + 1 / logvar_d.exp() * (x - mu_d) ** 2)
    reconstruction_loss = torch.mean(reconstruction_loss)
    kl_loss = -0.5 * (1 + logvar_e - mu_e**2 - logvar_e.exp())
    kl_loss = torch.mean(kl_loss)
    return reconstruction_loss + kl_loss

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseModel
from .vae import loss_function, sample


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, c_dim):
        super().__init__()

        self.fc1 = nn.Linear(x_dim + c_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, c_dim):
        super().__init__()

        self.fc1 = nn.Linear(x_dim + c_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, z, c):
        h = torch.cat([z, c], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class GGMM(BaseModel):
    def __init__(self, x_dim, h_dim, c_dim, lr, step_size=None, gamma=None):
        super().__init__(lr, step_size, gamma)
        self.save_hyperparameters()

        self.encoder = Encoder(x_dim, h_dim, c_dim)
        self.decoder = Decoder(x_dim, h_dim, c_dim)

    def forward(self, x, c):
        mu_e, logvar_e = self.encoder(x, c)
        z = sample(mu_e, logvar_e)
        mu_d, logvar_d = self.decoder(z, c)
        return mu_e, logvar_e, mu_d, logvar_d

    def training_step(self, batch, batch_idx):
        x, c = batch
        output = self(x, c)
        loss = loss_function(x, *output)
        self.log("train_loss", loss)
        return loss

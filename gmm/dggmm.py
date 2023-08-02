import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from .vae import loss_function, sample


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, c_dim):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.c_dim = c_dim

        self.lstm = nn.LSTM(x_dim + c_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, x, c, hidden=None):
        n, t, _ = x.size()
        c = c.view(n, 1, -1).expand(-1, t, -1)
        x = torch.cat([x, c], 2)
        h, _ = self.lstm(x, hidden)
        h = F.relu(self.fc(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, c_dim):
        super().__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.c_dim = c_dim

        self.lstm = nn.LSTM(x_dim + x_dim + c_dim, h_dim, batch_first=True)
        self.fc = nn.Linear(h_dim, h_dim)
        self.mu = nn.Linear(h_dim, x_dim)
        self.logvar = nn.Linear(h_dim, x_dim)

    def forward(self, x_previous, z, c, hidden=None):
        n, t, _ = x_previous.size()
        c = c.view(n, 1, -1).expand(-1, t, -1)
        z = torch.cat([x_previous, z, c], 2)
        h, hidden = self.lstm(z, hidden)
        h = F.relu(self.fc(h))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar, hidden


class DGGMM(BaseModel):
    def __init__(self, x_dim, h_dim, c_dim, lr, step_size=None, gamma=None):
        super().__init__(lr, step_size, gamma)
        self.save_hyperparameters()

        self.encoder = Encoder(x_dim, h_dim, c_dim)
        self.decoder = Decoder(x_dim, h_dim, c_dim)

    def forward(self, x, c, hidden_e=None, hidden_d=None):
        mu_e, logvar_e = self.encoder(x, c, hidden_e)
        z = sample(mu_e, logvar_e)
        # TODO: We can probably compute `x_previous` once and reuse it
        x_previous = torch.zeros_like(x)
        x_previous[:, 1:] = x[:, :-1]
        mu_d, logvar_d, _ = self.decoder(x_previous, z, c, hidden_d)
        return mu_e, logvar_e, mu_d, logvar_d

    def training_step(self, batch, batch_idx):
        x, c = batch
        output = self(x, c)
        loss = loss_function(x, *output)
        self.log("train_loss", loss)
        return loss


def generate(decoder, c, size):
    n, t = size

    with torch.no_grad():
        c = torch.from_numpy(c).float().view(1, -1).expand(n, -1)
        y = torch.zeros((n, t + 1, decoder.x_dim), dtype=torch.float32)
        z = torch.randn(n, t, decoder.x_dim)

        hidden = None
        for i in range(t):
            x_previous = y[:, i : i + 1, :]
            z_t = z[:, i : i + 1, :]
            mu_d, logvar_d, hidden = decoder(x_previous, z_t, c, hidden=hidden)
            y_t = sample(mu_d, logvar_d)
            y[:, i + 1, :] = y_t[:, 0, :]

    return y[:, 1:, :]

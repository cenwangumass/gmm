import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv
from torch_scatter import scatter_sum

from .base import BaseModel


class Encoder(nn.Module):
    def __init__(self, h_dim, n_layers):
        super().__init__()
        self.conv = GatedGraphConv(h_dim, n_layers)

    def forward(self, data):
        x = self.conv(data.x, data.edge_index)
        x = scatter_sum(x, data.batch, dim=0)
        return x


class Decoder(nn.Module):
    def __init__(self, h_dim, y_dim):
        super().__init__()
        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, y_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y.view(-1)


class GMM(BaseModel):
    def __init__(self, h_dim, y_dim, n_layers, lr, step_size=None, gamma=None):
        super().__init__(lr, step_size, gamma)
        self.save_hyperparameters()

        self.encoder = Encoder(h_dim, n_layers)
        self.decoder = Decoder(h_dim, y_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = (y_pred - y).abs().mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = (y_pred - y).abs().mean()
        self.log("val_loss", loss)
        return loss

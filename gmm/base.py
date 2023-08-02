import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class BaseModel(pl.LightningModule):
    def __init__(self, lr, step_size=None, gamma=None):
        super().__init__()

        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        if self.step_size is None:
            return optimizer
        else:
            step_lr = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            return [optimizer], [step_lr]

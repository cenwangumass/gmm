import pytorch_lightning as pl
from gmm.gmm import GMM
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch_geometric.loader import DataLoader

from data import load_dataset


def main():
    train_dataset = load_dataset("data/train")
    val_dataset = load_dataset("data/val")
    train_dataloader = DataLoader(train_dataset, batch_size=500, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=2000, num_workers=1)

    model = GMM(
        h_dim=32,
        y_dim=1,
        n_layers=4,
        lr=0.002,
        step_size=500,
        gamma=0.99,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=1000,
        precision=16,
        check_val_every_n_epoch=100,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("model.ckpt")


if __name__ == "__main__":
    main()

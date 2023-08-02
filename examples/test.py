import torch
from gmm.gmm import GMM
from torch_geometric.loader import DataLoader

from data import load_dataset


def main():
    test_dataset = load_dataset("data/test")
    test_dataloader = DataLoader(test_dataset, batch_size=2000, num_workers=1)

    model = GMM.load_from_checkpoint("model.ckpt")

    x = next(iter(test_dataloader))
    with torch.no_grad():
        error = model.validation_step(x, None)
        print(error.item())


if __name__ == "__main__":
    main()

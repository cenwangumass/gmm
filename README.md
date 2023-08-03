# gmm

Install dependencies:

1. It is recommended to install the dependencies in a conda environment.
2. Install [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [`pytorch-lightning`](https://www.pytorchlightning.ai/index.html).

Run the example:

```bash
# Make gmm importable
export PYTHONPATH=$PWD

# Enter the `examples` directory
cd examples

# Run the training script
python train.py

# Test model performance
python test.py
```

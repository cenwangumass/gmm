import json
import pickle

import torch
from torch.utils.data import Dataset

from san import SAN, Node, convert_san_to_pytorch_geometric_graph


class GraphDataset(Dataset):
    def __init__(self, graphs, y, log=True):
        super().__init__()
        self.graphs = graphs
        self.y = torch.from_numpy(y)
        self.log = log

    def __getitem__(self, i):
        if self.log:
            return self.graphs[i], self.y[i].log()
        else:
            return self.graphs[i], self.y[i]

    def __len__(self):
        return len(self.graphs)


def load_dataset(filename):
    with open(f"{filename}.pkl", "rb") as f:
        data = pickle.load(f)

    converted = []
    graphs = json.loads(data["graphs"])
    for d in graphs:
        nodes = []
        for n in d["nodes"]:
            nodes.append(Node(n["distribution"]))

        san = SAN(nodes, d["edges"])
        g = convert_san_to_pytorch_geometric_graph(san)
        converted.append(g)

    y = data["output"]

    return GraphDataset(converted, y, log=True)

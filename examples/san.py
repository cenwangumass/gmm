import re

import networkx as nx
import numpy as np
import torch
import torch_geometric

_PATTERN = re.compile(r"(\w+)\((.*)\)")


class Node:
    def __init__(self, distribution):
        self.distribution = distribution

    def to_dict(self):
        return {"distribution": self.distribution}

    def __repr__(self) -> str:
        return f"Node(distribution={self.distribution})"


class SAN:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def to_dict(self):
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
        }


def convert_to_networkx_graph(san: SAN) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from([(i, n.to_dict()) for i, n in enumerate(san.nodes)])
    graph.add_edges_from(san.edges)
    return graph


def add_dummy_start_end(graph: nx.DiGraph) -> nx.DiGraph:
    n = graph.number_of_nodes()

    start_id = n
    end_id = n + 1

    graph.add_node(start_id)
    graph.add_node(end_id)

    for node in range(n):
        predecessors = list(graph.predecessors(node))
        if not predecessors:
            graph.add_edge(start_id, node)

        successors = list(graph.successors(node))
        if not successors:
            graph.add_edge(node, end_id)

    return graph


def get_distribution_parameters(text):
    result = _PATTERN.search(text)
    parameters = result.group(2).replace(" ", "").split(",")
    parameters = [float(p) for p in parameters]
    return parameters


def convert_to_pytorch_geometric_graph(graph: nx.DiGraph):
    n_params = len(get_distribution_parameters(graph.nodes[0]["distribution"]))

    # Check all nodes have the same number of distribution parameters
    for i in range(1, graph.number_of_nodes() - 2):
        if len(get_distribution_parameters(graph.nodes[i]["distribution"])) != n_params:
            raise ValueError("the number of distribution parameters must be the same")

    x_dim = 1 + n_params
    x = np.zeros([graph.number_of_nodes(), x_dim], dtype=np.float32)

    # Indicator of start node
    x[-2, 0] = 1

    for i in range(graph.number_of_nodes() - 2):
        node = graph.nodes[i]
        parameters = get_distribution_parameters(node["distribution"])
        x[i, 1:] = parameters

    # Clean up
    for node in graph:
        graph.nodes[node].pop("distribution", None)

    rv = torch_geometric.utils.from_networkx(graph)
    rv.x = torch.from_numpy(x)

    return rv


def convert_san_to_pytorch_geometric_graph(san: SAN):
    graph = convert_to_networkx_graph(san)
    graph = add_dummy_start_end(graph)
    return convert_to_pytorch_geometric_graph(graph)

import numpy as np
import networkx as nx

from examples.synthetic.utils import random_cpt, sample_from_cpt
from examples.utils import number_to_identifier


class Node:
    def __init__(self, id, nr_parents, seed=None):
        self.id = id
        self.rng = np.random.default_rng(seed)
        self.cpt = random_cpt(self.rng, nr_parents)

    def sample(self, parents_samples):
        return sample_from_cpt(self.rng, self.cpt, parents_samples)


class GenerateError(RuntimeError):
    pass


class BinaryVariablesModel:
    def __init__(self, seed=None):
        self.nodes = []
        self.graph: nx.DiGraph = None
        self.rng = np.random.default_rng(seed)

    def generate_new_dag(self, nodes, edges):
        # adjacency matrix
        adj = np.ones((nodes, nodes))
        adj = np.tril(adj, -1)
        edge_indices = np.nonzero(adj)
        nr_edges = len(edge_indices[0])
        if edges > nr_edges:
            GenerateError(
                f"Too many edges specified, can not be satisfied (want {edges} but maximum DAG has {nr_edges})"
            )
        nr_edges_to_rm = nr_edges - edges
        edges_to_rm = self.rng.choice(
            np.arange(nr_edges), nr_edges_to_rm, replace=False
        )
        adj[edge_indices[0][edges_to_rm], edge_indices[1][edges_to_rm]] = 0
        self.graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)

        # nodes and conditional probabilities
        self.nodes = []
        node_seeds = self.rng.integers(0, np.iinfo(np.int64).max, size=nodes)
        for node in range(nodes):
            nr_parents = int(np.sum(nx.to_numpy_array(self.graph), axis=1)[node])
            self.nodes.append(
                Node(
                    node,
                    nr_parents,
                    seed=node_seeds[node],
                )
            )

    def sample(self, number, intervention_var=-1, int_style="uniform"):
        # make a specified number of samples
        # apply an intervention on the variable with index intervention_var of type int_style
        # (if intervention_var is not a valid index, e.g. -1, no intervention is applied)
        values = np.zeros((number, len(self.nodes)), dtype=np.int8)
        adj = nx.to_numpy_array(self.graph)
        for node in self.nodes:
            parents = adj[node.id]
            if intervention_var == node.id:
                if type(int_style) == int:
                    node_samples = np.full(number, int_style)
                elif int_style == "uniform":
                    node_samples = self.rng.choice([0, 1], number)
                else:
                    raise GenerateError(
                        f"Intervetion style {int_style} is unknown. Sampling failed."
                    )
            elif sum(parents) == 0:
                node_samples = sample_from_cpt(node.rng, node.cpt, number)
            else:
                node_samples = sample_from_cpt(
                    node.rng, node.cpt, values[:, parents == 1]
                )
            values[:, node.id] = node_samples
        return values
    
    def relabel_nodes_to_ident(self):
        nx.relabel_nodes(self.graph, number_to_identifier, copy=False)


# bvm = BinaryVariablesModel(0)
# bvm.generate_new_dag(5, 5)
# print(np.average(bvm.sample(10000000), axis=0))
# print(np.average(bvm.sample(10000000, intervention_var=4), axis=0))

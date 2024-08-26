import numpy as np
import networkx as nx
import pickle

from examples.utils import number_to_identifier


class Node:
    def __init__(self, id, bias, noise_gen_params, seed=None):
        # id (int), bias to be added when sampling, noise generation function and parameters
        self.id = id  # int, also the column in a data array when sampling
        self.bias = bias
        self.rng = np.random.default_rng(seed)
        self.noise_gen_params = noise_gen_params

    def sample_noise(self, num_samples):
        return self.rng.normal(**self.noise_gen_params, size=num_samples)


class GenerateError(RuntimeError):
    pass


class ANM:
    def __init__(self, seed=None):
        self.nodes = []
        self.coeff_bounds = None
        self.neg_coeffs = None
        self.graph: nx.DiGraph = None
        self.rng = np.random.default_rng(seed)

    def _random_weights(self, size):
        # create weights within the coeff_bounds, possibly including negative values
        # by using a self.coeff_bounds[0] > 0, we can avoid small absolute values
        weights = self.rng.uniform(low=self.coeff_bounds[0], high=self.coeff_bounds[1], size=size)
        if self.neg_coeffs:
            factors = self.rng.choice([1, -1], size=size)
        weights *= factors
        return weights

    def get_node(self, id):
        # return node with the corresponding id
        for node in self.nodes:
            if node.id == id:
                return node

    def generate_new_dag(self, nodes, edges, coeff_bounds=(1, 10), neg_coeffs=True, bias_bounds=(-10, 10), noise_scale_bounds=(0.1, 1)):
        self.coeff_bounds = coeff_bounds
        self.neg_coeffs = neg_coeffs
        self.bias_bounds = bias_bounds
    
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

        # node bias (could also be the mean of the gaussian noise, but here coded as "bias")
        self.nodes = []
        biases = self.rng.uniform(low=bias_bounds[0], high=bias_bounds[1], size=nodes)
        node_seeds = self.rng.integers(0, np.iinfo(np.int64).max, size=nodes)
        #print(node_seeds)
        scales = list(self.rng.uniform(low=noise_scale_bounds[0], high=noise_scale_bounds[1], size=nodes))
        for node in range(nodes):
            self.nodes.append(Node(node, biases[node], {"loc": 0.0, "scale": scales[node]}, seed=node_seeds[node]))

    def sample(self, number, intervention_var=-1, int_style="uniform"):
        # make a specified number of samples
        values = np.zeros((number, len(self.nodes)))
        node_order = list(nx.topological_sort(self.graph))  # TODO not necessary?
        for node_id in node_order:
            if intervention_var == node_id:
                if type(int_style) == int:
                    int_values = np.full(number, int_style)
                elif int_style == "uniform":
                    int_values = self.rng.uniform(low=self.bias_bounds[0], high=self.bias_bounds[1], size=number)
                else:
                    raise GenerateError(
                        f"Intervetion style {int_style} is unknown. Sampling failed."
                    )
                values[:, node_id] = int_values
            else:
                in_edges = self.graph.in_edges(node_id)
                for edge in in_edges:
                    parent = edge[0]
                    assert edge[1] == node_id  # this should be unnecessary, I just want to make sure
                    # this column contains the values for the parent node (because nodes are iterated following the topology)
                    weight = nx.adjacency_matrix(self.graph)[parent, node_id]  # TODO better style?
                    values[:, node_id] += values[:, parent] * weight
                node = self.get_node(node_id)
                assert node.id == node_id  # this should be unnecessary, I just want to make sure
                values[:, node.id] += node.sample_noise(number) + node.bias
        return values
    
    def relabel_nodes_to_ident(self):
        nx.relabel_nodes(self.graph, number_to_identifier, copy=False)

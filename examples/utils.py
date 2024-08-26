import networkx as nx
import numpy as np
import torch
import random

# see https://pytorch.org/docs/stable/notes/randomness.html
# TODO despite all this, the model is still not truly deterministic

def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def deterministic_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def comp_avg_prob_diff(true_probs, spn_probs_0, spn_probs_1):
    # get into numpy
    # true_probs = true_probs.detach().cpu().numpy()
    # spn_probs = spn_probs.detach().cpu().numpy()
    spn_probs = spn_probs_1 / (spn_probs_0 + spn_probs_1)

    diffs = 0
    for i in range(true_probs.shape[0]):
        diffs += abs(true_probs[i] - spn_probs[i])

    diffs /= true_probs.shape[0]
    return diffs


def get_interventions_differences(einet, max_train_data, device):
    # assuming interventions on all variables are possible
    # also assuming that only the intervention variables is given as input, not the intervention value
    nr_variables = max_train_data.shape[0]

    # get observational most likely prediction of marginal distributions
    max_of_marginal = np.zeros(nr_variables)

    if False:  # marginalization with MPE, two problems: 1. it does NOT WORK with current implementatoin, 2. it takes quadratically more time than the simple MPE
        # no intervention
        intervention = torch.zeros((1, nr_variables)).to(device)
        # marginalize for each variable
        for i in range(nr_variables):
            # TODO HERE AND BELOW marginalization does not have any effect on MPE
            marg_list = list(range(nr_variables))
            marg_list.remove(i)
            einet.set_marginalization_idx(marg_list)
            # # we can use "zeros_like" and "ones_like" since all non-marginalized values (the single index i) has to be 0 and 1, the rest does not matter
            # x = torch.zeros(nr_variables)

            max_of_marginal[i] = einet.mpe(intervention)[0, i]
            einet.set_marginalization_idx(None)  # TODO unnecessary?

        # same for interventional
        max_of_marginal_i = np.zeros((nr_variables, nr_variables))
        # for any possible intervention
        for i in range(nr_variables):
            intervention = torch.zeros((1, nr_variables)).to(device)
            intervention[0, i] = 1
            # marginalize for each variable
            for j in range(nr_variables):
                marg_list = list(range(nr_variables))
                marg_list.remove(i)
                einet.set_marginalization_idx(marg_list)
                # # we can use "zeros_like" and "ones_like" since all non-marginalized values (the single index i) has to be 0 and 1, the rest does not matter
                # x = torch.zeros(nr_variables)

                max_of_marginal_i[i, j] = einet.mpe(intervention)[0, j]
                einet.set_marginalization_idx(None)  # TODO unnecessary?
    else:
        # no intervention
        intervention = torch.zeros((1, nr_variables)).to(device)
        max_of_marginal = einet.mpe(intervention)[0].cpu().numpy()

        # same for interventional (once per intervention)
        max_of_marginal_i = np.zeros((nr_variables, nr_variables))
        # for any possible intervention
        for i in range(nr_variables):
            intervention = torch.zeros((1, nr_variables)).to(device)
            intervention[0, i] = 1
            max_of_marginal_i[i] = einet.mpe(intervention)[0].cpu().numpy()

    # calculate difference and normalize
    edges = []
    for start_edge in range(max_of_marginal_i.shape[0]):
        for end_edge in range(max_of_marginal_i.shape[1]):
            # ignore edges to itself
            if start_edge == end_edge:
                pass
            diff = abs(max_of_marginal_i[start_edge, end_edge] - max_of_marginal[end_edge])
            # normalization
            diff /= max_train_data[end_edge]
            edges.append((start_edge, end_edge, diff))
    
    return edges


def greedy_dag(scores, gene_names):
    # we make a rather arbitrary cutoff choice: all changes in distributions smaller than 5% are not considered as possible edges (1. makes the graph more sparse, avoid very insignificant edges, 2. vastly improves runtime)
    
    # sort edges
    scores.sort(key=lambda x: x[2], reverse=True)

    # should output list of edges
    graph = nx.DiGraph()
    current_edge = scores.pop(0)
    while(current_edge[2] > 0.05):
        prev_graph = graph.copy()
        graph.add_edge(gene_names[current_edge[0]], gene_names[current_edge[1]])
        if not nx.is_directed_acyclic_graph(graph):
            graph = prev_graph
        current_edge = scores.pop(0)
        print(f"Current score: {current_edge[2]}", end="\r")
    return [e for e in graph.edges]


def number_to_identifier(num):
    identifier = ""
    while num >= 0:
        identifier = chr(ord('A') + num % 26) + identifier
        num //= 26
        num -= 1
        if num < 0:
            break
    return identifier   


def load_data_dict(file_name):
    data_np = np.load(file_name)
    nr_vars = data_np.shape[1]-1
    data_np_int = np.zeros((data_np.shape[0], nr_vars))
    for i in range(nr_vars):
        intervened_rows = data_np_int[np.where(data_np[:,-1]==i)]
        intervened_rows[:,i] = 1
        data_np_int[np.where(data_np[:,-1]==i)] = intervened_rows
    data_dict = {}
    for i in range(nr_vars):
        ident = number_to_identifier(i)
        data_dict[ident] = data_np[:, i:i+1]
        data_dict[f"Instr_{ident}"] = data_np_int[:, i:i+1] # for instrumental variable
    return data_dict

class EarlyStopper:
    def __init__(self, patience=1, nip=0.05):
        self.patience = patience
        # nip: necessary improvement percentage (see early stop)
        self.nip = nip
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            # the min loss is set to self.nip times this best loss lower such that early stopping also
            # occurs when improvements are very minor (e.g. useful when early stopping based on train data)
            self.min_validation_loss = validation_loss - abs(validation_loss * self.nip)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
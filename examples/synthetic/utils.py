import pickle

import numpy as np


def random_cpt(rng, nr_vars):
    # Return a random conditional probability table, i.e. one probability for each possible variables combination
    cpt = rng.random(2**nr_vars)
    return cpt


def sample_from_cpt(rng, cpt, var_settings):
    # Sample from a random conditional probability table
    # var_settings is an array of shape n, n_vars
    # n is the number of instances = samples to be generated
    # n_vars is the number of parents (same as sqrt(cpt size))
    # for nodes without parents, var_setting should be an integer
    if len(cpt) > 1:
        base_factor = 2 ** np.arange(var_settings.shape[1])
        cpt_indices = np.sum(var_settings * base_factor, axis=1)
    else:
        cpt_indices = np.zeros(var_settings, dtype=np.int8)
    probs = cpt[cpt_indices]
    sample_probs = rng.random(probs.shape[0])
    binary_samples = sample_probs < probs
    return binary_samples.astype(dtype=np.int8)


def save_model(file_name, model):
    with open(file_name, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file_name):
    with open(file_name, "rb") as handle:
        return pickle.load(handle)
    

def save_graph(file_name, model):
    with open(file_name, "w") as f:
        f.write("<NODES>\n")
        f.write("\n".join([str(node) for node in model.graph.nodes()]) + "\n\n")
        f.write("<EDGES>\n")
        for edge in model.graph.edges():
            f.write(str(edge[0]) + " -> " + str(edge[1]) + "\n")
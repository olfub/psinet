import argparse
from pathlib import Path

import numpy as np

from examples.synthetic.BinaryVariablesModel import BinaryVariablesModel
from examples.synthetic.ANM import ANM
from examples.synthetic.utils import save_model, save_graph


def main(args):
    if args.type in ["anm", "ANM"]:
        model = ANM(args.seed)
    else:
        model = BinaryVariablesModel(args.seed)
    model.generate_new_dag(args.nodes, args.edges)
    # last column contains intervention value (which variable to intervene on or -1 for no intervention)
    data = np.zeros((args.samples_per_int * (args.nodes + 1), args.nodes + 1))
    data[: args.samples_per_int, :-1] = model.sample(
        args.samples_per_int, intervention_var=-1
    )
    data[: args.samples_per_int, -1] = -1
    for i in range(args.nodes):
        data[
            args.samples_per_int * (i + 1) : args.samples_per_int * (i + 2), :-1
        ] = model.sample(args.samples_per_int, intervention_var=i)
        data[args.samples_per_int * (i + 1) : args.samples_per_int * (i + 2), -1] = i

    path = Path(f"{args.path}/{args.folder}/{args.type.upper()}_{args.nodes}_{args.edges}_{args.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)

    if args.type in ["bvm", "BVM"]:
        # calculate marginal probabilities based on samples
        probabilities = np.zeros((args.nodes + 1, args.nodes))
        probabilities[0] = np.average(model.sample(args.prob_samples), axis=0)
        for i in range(args.nodes):
            probabilities[i + 1] = np.average(model.sample(args.prob_samples, intervention_var=i), axis=0)
        np.save(str(path) + "_probs.npy", probabilities)

    save_model(str(path) + ".pkl", model)
    np.save(str(path) + ".npy", data)
    model.relabel_nodes_to_ident()
    save_graph(str(path) + ".cg", model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["BVM", "bvm", "ANM", "anm"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--nodes", default=5, type=int)
    parser.add_argument("--edges", default=5, type=int)
    parser.add_argument("--samples_per_int", default=10000, type=int)
    # samples used for calculating the marginal probabilities (not used for training, just one part of evaluation)
    parser.add_argument("--prob_samples", default=1000000, type=int)
    parser.add_argument("--path", default="datasets", type=str)
    parser.add_argument("--folder", default="temp", type=str)

    args = parser.parse_args()
    main(args)

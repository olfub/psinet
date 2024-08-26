import argparse

import numpy as np
import pandas as pd
import regex
from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from pgmpy.models import BayesianModel, LinearGaussianBayesianNetwork

from ciSPN.E1_helpers import get_experiment_name
from datasets.tabularDataset import TabularDataset

print("ok")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", choices=["ASIA", "CANCER", "EARTHQUAKE"], default="ASIA"
)
cli_args = parser.parse_args()

print("Arguments:", cli_args)

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = "BN"
conf.dataset = cli_args.dataset
conf.seed = 0


make_deterministic(conf.seed)

# setup experiments folder
# runtime_base_dir = environment["experiments"]["base"] / "E1" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E1" / "eval_logs"

experiment_name = get_experiment_name(
    conf.dataset, conf.model_name, conf.seed, None, None, None
)

# redirect logs
log_path = log_base_dir / (experiment_name + ".txt")
log_path.parent.mkdir(exist_ok=True, parents=True)
logger = PrintLogger(log_path)


print("Arguments:", cli_args)


# setup dataset
X_vars, Y_vars, interventionProvider = get_data_description(conf.dataset)
X_vars.pop()  # remove 'interventions' entry
dataset_paths_train = get_dataset_paths(conf.dataset, "train")
dataset_paths_test = get_dataset_paths(conf.dataset, "test")


def create_model(dataset_name, intervention):
    if dataset_name == "CHC":
        raise ValueError("Use BN_CHC_eval")
        # bn = BayesianModel() # isn't viable for continous variables
        bn = (
            LinearGaussianBayesianNetwork()
        )  # Error: "fit method has not been implemented for LinearGaussianBayesianNetwork." ...
        bn.add_edge("A", "F")
        bn.add_edge("A", "H")
        bn.add_edge("F", "H")
        bn.add_edge("H", "M")
        bn.add_edge("A", "D1")
        bn.add_edge("A", "D2")
        bn.add_edge("A", "D3")
        bn.add_edge("F", "D1")
        bn.add_edge("F", "D2")
        bn.add_edge("F", "D3")
        bn.add_edge("H", "D1")
        bn.add_edge("H", "D2")
        bn.add_edge("H", "D3")
        bn.add_edge("M", "D1")
        bn.add_edge("M", "D2")
        bn.add_edge("M", "D3")
    elif dataset_name == "ASIA":
        bn = BayesianModel()
        bn.add_edge("A", "T")
        bn.add_edge("T", "E")
        bn.add_edge("S", "L")
        bn.add_edge("L", "E")
        bn.add_edge("S", "B")
        bn.add_edge("E", "X")
        bn.add_edge("E", "D")
        bn.add_edge("B", "D")
    elif dataset_name == "CANCER":
        bn = BayesianModel()
        bn.add_edge("P", "C")
        bn.add_edge("S", "C")
        bn.add_edge("C", "X")
        bn.add_edge("C", "D")
    elif dataset_name == "EARTHQUAKE":
        bn = BayesianModel()
        bn.add_edge("B", "A")
        bn.add_edge("E", "A")
        bn.add_edge("A", "J")
        bn.add_edge("A", "M")
    else:
        raise ValueError("unknown dataset")

    if intervention is not None:
        bn.do([intervention], inplace=True)
    return bn


overall_samples = 0
correct_samples = 0

# learn a BN for every intervention
for dataset_path_train, dataset_path_test in zip(
    reversed(dataset_paths_train), reversed(dataset_paths_test)
):
    # extract intervention name from path
    intervention_name = regex.search(
        r"(?|do\((.*?)\)|(None))", dataset_path_train.name
    ).group(1)
    if intervention_name == "None":
        intervention_name = None

    # load data - we do not add intervention data, as it is the same within every dataset split anyways
    dataset_train = TabularDataset(
        [dataset_path_train], X_vars, Y_vars, None, store_as_torch_tensor=False
    )  # , part_transformer=interventionProvider)
    dataset_test = TabularDataset(
        [dataset_path_test], X_vars, Y_vars, None, store_as_torch_tensor=False
    )  # , part_transformer=interventionProvider)

    bn = create_model(conf.dataset, intervention_name)

    n_jobs = -1

    # put cond and class vars back together
    data = {
        **{n: dataset_train.X[:, i] for i, n in enumerate(X_vars)},
        **{n: dataset_train.Y[:, i] for i, n in enumerate(Y_vars)},
    }
    data = pd.DataFrame(data)
    bn.fit(data, n_jobs=n_jobs)

    # put condition vars
    test_data = {
        **{n: dataset_test.X[:, i] for i, n in enumerate(X_vars)},
    }
    test_data = pd.DataFrame(test_data)
    prediction = bn.predict(test_data, stochastic=False, n_jobs=n_jobs)
    prediction = np.vstack(
        [prediction.loc[:, var] for var in Y_vars]
    ).T  # read variables in correct order from frame (to_numpy does not guarantee correct ordering!)

    all = np.all(prediction == dataset_test.Y, axis=1)
    correct = np.sum(all)

    num_samples = len(dataset_test.X)
    print(f"Intervention: {intervention_name}")
    print(f"Correct {correct} out of {num_samples} ({correct/num_samples})")
    overall_samples += num_samples
    correct_samples += correct

accuracy = correct_samples / overall_samples
print(f"Total Accuracy: {accuracy}")

from ciSPN.evals.regressStats import RegressStats
from ciSPN.evals.visualize_problem import visualize_problem
from helpers.determinism import make_deterministic
from ciSPN.E1_helpers import get_experiment_name
from ciSPN.E2_helpers import create_dataloader
from helpers.configuration import Config
from ciSPN.datasets.particleCollisionDataset import ParticleCollisionDataset, attr_to_index
from datasets.galaxyCollisionDataset import GalaxyCollisionDataset

import torch
import argparse

from environment import environment, get_dataset_paths

from models.spn_create import load_spn

from libs.pawork.log_redirect import PrintLogger

from rtpt import RTPT

print_progress = True


parser = argparse.ArgumentParser()
# multiple seeds can be "-" separated, e.g. "0-1-2"
parser.add_argument("--seeds", default="0")
parser.add_argument("--model", choices=["ciSPN"], default='ciSPN')
parser.add_argument("--loss", choices=["NLLLoss"], default='NLLLoss')
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument("--loss2_factor", default="1.0")  # factor by which loss2 is added to the loss term
parser.add_argument("--dataset", choices=["PC", "GC"], default="PC")
parser.add_argument("--vis", action="store_true", default=False)  # whether to also generate a visualized example
parser.add_argument("--vis_args", default="")  # possible arguments for the visualization if "vis" is True
# neural network neurons and layers (TODO ideally should be read from a file and not need to be entered here)
# (however, for now like this (less coding work), just care to use the same arguments for train and test)
parser.add_argument("--nn_neurons", type=int, default=75)
parser.add_argument("--nn_layers", type=int, default=-1)  # -1 will be changed to a default depending on dataset
parser.add_argument("--log_name_add", default="")  # if not empty, give a unique name to the eval log
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = cli_args.model
conf.batch_size = 1000
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seeds = cli_args.seeds
conf.vis = cli_args.vis
conf.vis_args = cli_args.vis_args
conf.nn_neurons = cli_args.nn_neurons
conf.nn_layers = cli_args.nn_layers
conf.log_name_add = cli_args.log_name_add

conf.explicit_load_part = conf.model_name

all_seeds = [int(seed) for seed in conf.seeds.split("-")]
eval_strs = []
for seed in all_seeds:
    make_deterministic(seed)

    # setup experiments folder
    runtime_base_dir = environment["experiments"]["base"] / "E3" / "runtimes"
    log_base_dir = environment["experiments"]["base"] / "E3" / "eval_logs"
    output_base_dir = environment["experiments"]["base"] / "E3" / "outputs"

    experiment_name = get_experiment_name(conf.dataset, conf.model_name, True, seed, conf.loss_name,
                                          conf.loss2_name, conf.loss2_factor, E=3)
    load_dir = runtime_base_dir / experiment_name
    output_dir = output_base_dir / experiment_name

    # redirect logs
    add_to_name = f"_{conf.nn_neurons}_{conf.nn_layers}_{conf.log_name_add}_{seed}"
    log_path = log_base_dir / (experiment_name + add_to_name + ".txt")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = PrintLogger(log_path)

    print("Arguments:", cli_args)

    # setup dataset
    if conf.nn_layers == -1:
        num_layers = 1
    else:
        num_layers = conf.nn_layers
    if conf.dataset == "PC":
        data_base_dir = get_dataset_paths("PC", "test", get_base=True)
        dataset_paths = get_dataset_paths("PC", "test")
        dataset = ParticleCollisionDataset(data_base_dir, dataset_paths, seed)
    elif conf.dataset == "GC":
        data_base_dir = get_dataset_paths("GC", "test", get_base=True)
        dataset_paths = get_dataset_paths("GC", "test")
        dataset = GalaxyCollisionDataset(data_base_dir, dataset_paths, seed)
        if conf.nn_layers == -1:
            num_layers = 0
    else:
        raise RuntimeError(f"Unknown dataset ({conf.dataset}).")

    dataloader = create_dataloader(dataset, seed, num_workers=0, batch_size=conf.batch_size,
                                   multi_thread_data_loading=False, shuffle=True)

    num_condition_vars = dataset.num_observed_values
    num_vars = dataset.num_variables

    print(f"Loading {conf.explicit_load_part}")
    if conf.explicit_load_part == 'ciSPN':
        spn, _, _ = load_spn(num_condition_vars, load_dir=load_dir, num_layers=num_layers, num_neurons=conf.nn_neurons)
        eval_wrapper = spn
    else:
        raise ValueError(f"invalid load part {conf.explicit_load_part}")
    eval_wrapper.eval()

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # test performance on test set, unseen data
        stat = RegressStats(num_vars)

        placeholder_target_batch = None
        marginalized = None

        rtpt = RTPT(name_initials='FB', experiment_name='cfSPN Evaluation', max_iterations=len(dataloader))
        rtpt.start()

        def batch_processor(batch):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if type(batch[0]) in [tuple, list]:
                batch_input = (batch[0][0].to(device=device).float(), batch[0][1].to(device=device).float())
            else:
                batch_input = batch[0].to(device=device).float()
            batch_output = batch[1].to(device=device).float()
            return batch_input, batch_output

        for batch_num, batch in enumerate(dataloader):
            condition_batch, target_batch = batch_processor(batch)

            if placeholder_target_batch is None or placeholder_target_batch.shape != batch[1]:
                # zero out target vars to be sure
                placeholder_target_batch = torch.zeros_like(target_batch, dtype=torch.float, device=device)
                marginalized = torch.ones_like(placeholder_target_batch)

            reconstruction = eval_wrapper.predict(condition_batch, placeholder_target_batch, marginalized)

            if conf.vis and batch_num == 0:
                if conf.dataset == "PC":
                    parameters = [device, dataset.num_variables, placeholder_target_batch, marginalized]
                elif conf.dataset == "GC":
                    method = environment["datasets"]["GC_method"]
                    parameters = [device, dataset.num_variables, placeholder_target_batch, marginalized, method]
                visualize_problem(conf.dataset, eval_wrapper, output_dir, parameters, conf.vis_args)

            correct = stat.eval(target_batch, reconstruction)
            # use next line to get some kind of baseline performance (only if it makes sense for the dataset)
            # correct = stat.eval(target_batch, condition_batch[:, :num_vars])  # prediction = input

            if print_progress:
                print(f"Processed batch {batch_num+1} of {len(dataloader)}.", end="\r", flush=True)
            rtpt.step()

    # print stats to the console immediately, also ensure them being part of the logger for the respective seed only
    print(stat.get_eval_result_str())
    print("")
    print(stat.get_eval_result_str_per_feature())

    # also save the string to print everything again at the end, so that the results can be seen without scrolling
    # through the prints resulting from print_progress
    eval_strs.append(stat.get_eval_result_str())

    logger.close()

print("\nEvaluation for the different seeds:\n")
for eval_str in eval_strs:
    print(eval_str + "\n")

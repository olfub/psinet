import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from descriptions.description import get_data_description
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from models.nn_wrapper import NNWrapper
from models.spn_create import load_model_params, load_spn

from ciSPN.datasets.interventionHelpers import intervention_vars_dict
from ciSPN.E1_helpers import get_experiment_name
from ciSPN.models.model_creation import create_nn_model
from datasets.batchProvider import BatchProvider
from datasets.tabularDataset import TabularDataset

sns.set_theme()
np.set_printoptions(suppress=True)

print_progress = True


parser = argparse.ArgumentParser()
# multiple seeds can be "-" separated, e.g. "606-1011"
parser.add_argument("--seeds", default="606")  # 606, 1011, 3004, 5555, 12096
parser.add_argument("--model", choices=["mlp", "ciSPN"], default="mlp")
parser.add_argument(
    "--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default="MSELoss"
)
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument(
    "--loss2_factor", default="1.0"
)  # factor by which loss2 is added to the loss term
parser.add_argument(
    "--dataset",
    choices=[
        "CHC",
        "ASIA",
        "CANCER",
        "EARTHQUAKE",
        "WATERING",
        "TOY1",
        "TOY2",
        "TOY1I",
    ],
    default="CHC",
)  # CausalHealthClassification
parser.add_argument("--known-intervention", action="store_true", default=False)
parser.add_argument(
    "--samples", type=int, default=100
)  # number of samples for model pdf
parser.add_argument(
    "--skip_samples", type=int, default=0
)  # skip so many samples to get a different vis
parser.add_argument("--save", action="store_true", default=False)
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.known_intervention = cli_args.known_intervention
conf.model_name = cli_args.model
conf.batch_size = 1000
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seeds = cli_args.seeds

conf.explicit_load_part = conf.model_name

conf.samples = cli_args.samples
conf.skip_samples = cli_args.skip_samples
conf.save = cli_args.save

# setup experiments folder
runtime_base_dir = environment["experiments"]["base"] / "E1" / "runtimes"
log_base_dir = environment["experiments"]["base"] / "E1_2" / "eval_logs"
vis_base_dir = environment["experiments"]["base"] / "E1_2" / "visualizations"

# setup datasets
X_vars, Y_vars, providers = get_data_description(conf.dataset)

if len(Y_vars) > 8:
    raise ValueError(
        "The current implementation only supports up to 8 output variables (Y), the visualization parts "
        "of the code need to be changed to allow for more outputs."
    )

# this implementation assumes the intervention to be first in X_vars; this is not fundamentally necessary but requires
# changes in the code to account for that possibility
assert X_vars[0] == "intervention"
dataset_paths = get_dataset_paths(conf.dataset, "test")

all_seeds = [int(seed) for seed in conf.seeds.split("-")]
values_per_seed = {}

# TODO generalize for ground truth which are not binary variables
# dictionary with seed -> dictionary (intervention -> dictionary (variable -> dictionary (0 or 1 -> counts))
ground_truth = {}

experiment_folder = None
for seed in all_seeds:
    make_deterministic(seed)
    ground_truth[seed] = {}

    experiment_name = get_experiment_name(
        conf.dataset,
        conf.model_name,
        conf.known_intervention,
        seed,
        conf.loss_name,
        conf.loss2_name,
        conf.loss2_factor,
    )
    load_dir = runtime_base_dir / experiment_name

    # redirect logs
    log_path = log_base_dir / (experiment_name + ".txt")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = PrintLogger(log_path)

    # get path for visualizations in case results should be saved
    experiment_folder = (vis_base_dir / experiment_name).parent

    print("Arguments:", cli_args)

    # load dataset
    dataset = TabularDataset(
        dataset_paths,
        X_vars,
        Y_vars,
        conf.known_intervention,
        0,
        part_transformers=providers,
    )
    provider = BatchProvider(dataset, conf.batch_size, provide_incomplete_batch=True)
    data_x = dataset.X.cpu().numpy()
    data_y = dataset.Y.cpu().numpy()

    num_condition_vars = dataset.X.shape[1]
    num_target_vars = dataset.Y.shape[1]

    print(f"Loading {conf.explicit_load_part}")
    if conf.explicit_load_part == "ciSPN":
        spn, _, _ = load_spn(num_condition_vars, load_dir=load_dir)
        eval_wrapper = spn
    elif conf.explicit_load_part == "mlp":
        nn = create_nn_model(num_condition_vars, num_target_vars)
        load_model_params(nn, load_dir=load_dir)
        eval_wrapper = NNWrapper(nn)
    else:
        raise ValueError(f"invalid load part {conf.explicit_load_part}")
    eval_wrapper.eval()

    with torch.no_grad():

        def compute_pdf(dim, y_dims, batch_size, low, high, interv_vector, cond_vars):
            y_batch = np.zeros((batch_size, y_dims))
            y_batch[:, dim] = np.linspace(low, high, batch_size)
            marginalized = np.ones((batch_size, y_dims))
            marginalized[:, dim] = 0.0  # seemingly one sets the desired dim to 0
            # we assume that the intervention vector comes first
            x_batch = np.tile(
                np.concatenate((interv_vector, cond_vars)), (batch_size, 1)
            )
            x_batch = torch.Tensor(x_batch)
            y_batch = torch.Tensor(y_batch)
            marginalized = torch.Tensor(marginalized)
            if torch.cuda.is_available():
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
                marginalized = marginalized.cuda()

            out = eval_wrapper.forward(x_batch, y_batch, marginalized)
            values_x = (
                y_batch[:, dim].cpu().numpy()
            )  # range (all sampled points along axis)
            # given that out is log-likelihood, take exp() and these are the likelihoods for pdf
            vals_pdf = np.exp(out.cpu().numpy())
            return values_x, vals_pdf

        len_cond = len(X_vars) - 1
        only_interventional = (
            len_cond == 0
        )  # just the intervention vector, no other variables

        # we always assume that the intervention vector comes first
        if conf.known_intervention:
            # we know that for known_intervention, this intervention value is last
            # here, we only want the intervened variable, no matter the value (so that the visualization is over all
            # intervention values for this intervened variable)
            intervention_vector = np.unique(
                dataset.X[:, : -(len_cond + 1)].cpu().numpy(), axis=0
            )
        else:
            intervention_vector = np.unique(
                dataset.X[:, :-len_cond].cpu().numpy(), axis=0
            )
        zero_interv = np.zeros((intervention_vector.shape[1]))
        interventions = (
            [("None", zero_interv)] if zero_interv in intervention_vector else []
        )
        for row in intervention_vector[1:]:
            interventions.append(
                (intervention_vars_dict[conf.dataset][np.where(row == 1)[0][0]], row)
            )

        vals_per_intervention = {}
        for interv_desc, row in interventions:
            if interv_desc not in ground_truth[seed]:
                ground_truth[seed][interv_desc] = {}
            vals_per_var = {}
            if only_interventional:
                x_interv_only = data_x
            else:
                x_interv_only = data_x[:, : -(len(X_vars) - 1)]
            if conf.known_intervention:
                data_indices_for_intervention = np.where(
                    (x_interv_only[:, :-1] == row).all(axis=1)
                )[0]
            else:
                data_indices_for_intervention = np.where(
                    (x_interv_only == row).all(axis=1)
                )[0]
            for ind_var, var in enumerate(Y_vars):
                if var not in ground_truth[seed][interv_desc]:
                    ground_truth[seed][interv_desc][var] = {0: 0, 1: 0}
                x_values = np.zeros(conf.batch_size)
                pdf_values = np.zeros((conf.batch_size, 1))
                for sample in range(conf.samples):
                    sample += (
                        conf.skip_samples
                    )  # if you only print one sample but do not like the chosen one
                    if only_interventional:
                        conditionals = np.array([])
                    else:
                        if conf.known_intervention:
                            conditionals = data_x[
                                data_indices_for_intervention[sample], len_cond + 1 :
                            ]
                        else:
                            conditionals = data_x[
                                data_indices_for_intervention[sample], len_cond:
                            ]
                    gt_value = int(
                        data_y[data_indices_for_intervention[sample], ind_var]
                    )
                    ground_truth[seed][interv_desc][var][gt_value] += 1
                    interv_with_value = x_interv_only[
                        data_indices_for_intervention[sample]
                    ]
                    x_value, pdf_value = compute_pdf(
                        ind_var,
                        len(Y_vars),
                        conf.batch_size,
                        -0.5,
                        1.5,
                        interv_with_value,
                        conditionals,
                    )
                    x_values += x_value
                    pdf_values += pdf_value
                x_values /= conf.samples
                pdf_values /= conf.samples
                vals_per_var.update({var: (x_values, pdf_values)})
            vals_per_intervention.update({interv_desc: vals_per_var})
        values_per_seed[seed] = vals_per_intervention
    logger.close()  # TODO is it worth it to also log the code below?

if len(all_seeds) > 1:
    mean_vals_per_interv = {}
    std_per_interv = {}
    for interv_desc, row in interventions:
        for var in Y_vars:
            var_pdfs = []
            for seed in all_seeds:
                var_pdfs.append(values_per_seed[seed][interv_desc][var][1][:, 0])
            if interv_desc not in mean_vals_per_interv:
                mean_vals_per_interv[interv_desc] = [np.average(var_pdfs, axis=0)]
                std_per_interv[interv_desc] = [np.std(var_pdfs, axis=0)]
            else:
                mean_vals_per_interv[interv_desc].append(np.average(var_pdfs, axis=0))
                std_per_interv[interv_desc].append(np.std(var_pdfs, axis=0))

fig_len_x = 1
fig_len_y = len(Y_vars)
if len(Y_vars) > 6:
    fig_len_x += 1
    fig_len_y = int((fig_len_y + 1) / 2)  # +1 to make sure it is always rounded up

color_pairs = [
    ("#e480b5", "#db569d"),
    ("#40b3ef", "#13a0e9"),
    ("#ffbf2e", "#faad00"),
    ("#4fe547", "#26cb1d"),
    ("#ffff6b", "#ffff27"),
    ("#d54b37", "#892b1d"),
    ("#20e6d0", "#15beab"),
    ("#a575dc", "#8b4cd1"),
]
share_x = True
share_y = True

# all single seeds
colors = ["green", "blue", "orange", "cyan", "brown"]

# new variable names (only change the label here and not have to redo anything else) 
var_names = {
    "WATERING": {"M-cf": "M", "A-cf": "A", "B-cf": "B", "H-cf": "H"},
    "TOY1": {y_var: y_var[0] for y_var in Y_vars},
    "TOY2": {y_var: y_var[0] for y_var in Y_vars},
}

for interv_desc, row in interventions:
    fig, axs = plt.subplots(
        fig_len_x,
        fig_len_y,
        figsize=(fig_len_y * 3, 5 * fig_len_x),
        sharey=share_y,
        sharex=share_x,
        num=interv_desc,
    )

    for ind_var, var in enumerate(Y_vars):
        labels = np.array([0, 1])
        # add up counts for ground truth across all seeds
        first_seed = all_seeds[0]
        counts = np.array(
            [
                ground_truth[first_seed][interv_desc][var][0],
                ground_truth[first_seed][interv_desc][var][1],
            ]
        )
        for seed in all_seeds[1:]:
            counts += np.array(
                [
                    ground_truth[seed][interv_desc][var][0],
                    ground_truth[seed][interv_desc][var][1],
                ]
            )
        axs.flatten()[ind_var].bar(
            labels,
            counts / (len(all_seeds) * conf.samples),
            width=0.1,
            facecolor=color_pairs[ind_var][0],
            edgecolor=color_pairs[ind_var][1],
            label="Ground Truth",
        )
        # useful if different data is used between seeds; the following draws lines for the respective ground truths
        # for i, seed in enumerate(all_seeds):
        #     lw = 0.1  # line width
        #     axs.flatten()[ind_var].hlines(ground_truth[seed][interv_desc][var][0]/conf.samples, 0-lw, 0+lw, colors[i])
        #     axs.flatten()[ind_var].hlines(ground_truth[seed][interv_desc][var][1]/conf.samples, 1-lw, 1+lw, colors[i])

        # pdf
        if len(all_seeds) > 1:
            vals_pdf_mean = mean_vals_per_interv[interv_desc][ind_var]
            vals_pdf_std = std_per_interv[interv_desc][ind_var]
        vals_x = vals_per_intervention[interv_desc][var][0]

        model_name = "iSPN" if only_interventional else "cf-SPN"

        for col, seed in enumerate(all_seeds):
            if model_name == "cf-SPN":
                col += 1  # use blue instead of green for cf-SPN, (considering single seeds)
            seed_pdf = values_per_seed[seed][interv_desc][var][1][:, 0]
            axs.flatten()[ind_var].plot(
                vals_x,
                seed_pdf,
                label=f"{model_name} ({seed})"
                if len(all_seeds) > 1
                else f"{model_name}",
                color=colors[col],
                linestyle="solid",
                linewidth=1.5,
                zorder=10,
            )

        if len(all_seeds) > 1:
            axs.flatten()[ind_var].plot(
                vals_x,
                vals_pdf_mean,
                label=f"{model_name} (mean)",
                color="#CC4F1B",
                linestyle="solid",
                linewidth=1.5,
                zorder=10,
            )
            axs.flatten()[ind_var].fill_between(
                vals_x,
                vals_pdf_mean - vals_pdf_std,
                vals_pdf_mean + vals_pdf_std,
                alpha=0.5,
                edgecolor="#CC4F1B",
                facecolor="#FF9848",
                zorder=9,
            )

        axs.flatten()[ind_var].set_ylim(0, 1.3)
        axs.flatten()[ind_var].set_xticks([0, 1])
        axs.flatten()[ind_var].tick_params(axis="x", labelsize=20)
        axs.flatten()[ind_var].tick_params(axis="y", labelsize=16)
        if conf.dataset in var_names:
            axs.flatten()[ind_var].set_title(
                "{}".format(var_names[conf.dataset][var]), fontdict={"fontsize": 25}
            )
        else:
            axs.flatten()[ind_var].set_title(
                "{}".format(var), fontdict={"fontsize": 20}
            )
        axs.flatten()[ind_var].legend(prop={"size": 11.5})
    plt.tight_layout()
    if conf.save:
        experiment_folder.mkdir(exist_ok=True, parents=True)
        save_loc = (
            experiment_folder / f"{interv_desc}.pdf"
        )  # TODO more useful naming (especially if variable names
        # are changed
        plt.savefig(save_loc)
        print(f"Saved {interv_desc} @ {save_loc}")
    else:
        plt.show()
    plt.clf()
    plt.close()

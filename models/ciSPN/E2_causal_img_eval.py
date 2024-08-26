import argparse
import math

import torch
import torchvision
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from libs.pawork.log_redirect import PrintLogger
from models.spn_create import load_model_params, load_spn
from rtpt import RTPT

from ciSPN.E1_helpers import get_experiment_name
from ciSPN.E2_helpers import (
    create_cnn_for_spn,
    create_cnn_model,
    create_dataloader,
    img_batch_processor,
)
from ciSPN.evals.classifyStats import ClassifyStats
from ciSPN.models.nn_wrapper import NNWrapper
from datasets.hiddenObjectDataset import HiddenObjectDataset

print_progress = True


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)
parser.add_argument("--model", choices=["cnn", "ciCNNSPN"], default="cnn")
parser.add_argument(
    "--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default="MSELoss"
)
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument(
    "--loss2_factor", default="1.0"
)  # factor by which loss2 is added to the loss term
parser.add_argument("--dataset", choices=["hiddenObject"], default="hiddenObject")
parser.add_argument("--debug", default=None)  # disables dataloaders -> single thread
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = cli_args.model
conf.batch_size = 128
conf.num_workers = 8
conf.multi_thread_data_loading = (
    False if cli_args.debug == "true" else True
)  # otherwise we debug in multi-process setting ...
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed

conf.explicit_load_part = conf.model_name

if __name__ == "__main__":
    make_deterministic(conf.seed)

    # setup experiments folder
    runtime_base_dir = environment["experiments"]["base"] / "E2" / "runtimes"
    log_base_dir = environment["experiments"]["base"] / "E2" / "eval_logs"

    experiment_name = get_experiment_name(
        conf.dataset,
        conf.model_name,
        conf.seed,
        conf.loss_name,
        conf.loss2_name,
        conf.loss2_factor,
        E=2,
    )
    load_dir = runtime_base_dir / experiment_name

    # redirect logs
    log_path = log_base_dir / (experiment_name + ".txt")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = PrintLogger(log_path)

    print("Arguments:", cli_args)

    # setup dataset
    if cli_args.dataset == "hiddenObject":
        transforms = [torchvision.transforms.ToTensor()]
        image_transform = torchvision.transforms.Compose(transforms)

        dataset_split = "test"
        hidden_object_base_dir = get_dataset_paths(
            "hiddenObject", dataset_split, get_base=True
        )
        dataset = HiddenObjectDataset(
            hidden_object_base_dir, image_transform=image_transform, split=dataset_split
        )
    else:
        raise RuntimeError(f"Unknown dataset ({cli_args.dataset}).")

    dataloader = create_dataloader(
        dataset,
        conf.seed,
        num_workers=conf.num_workers,
        batch_size=conf.batch_size,
        multi_thread_data_loading=conf.multi_thread_data_loading,
        shuffle=False,
        drop_last=False,
    )

    num_condition_vars = dataset.num_observed_variables
    num_target_vars = dataset.num_hidden_variables

    print(f"Loading {conf.explicit_load_part}")
    if conf.explicit_load_part == "ciCNNSPN":
        spn, _, _ = load_spn(
            num_condition_vars, load_dir=load_dir, nn_provider=create_cnn_for_spn
        )
        eval_wrapper = spn
    elif conf.explicit_load_part == "cnn":
        nn = create_cnn_model(num_condition_vars, num_target_vars)
        load_model_params(nn, load_dir=load_dir)
        eval_wrapper = NNWrapper(nn)
    else:
        raise ValueError(f"invalid load part {conf.explicit_load_part}")
    eval_wrapper.eval()

    with torch.no_grad():
        # test performance on test set, unseen data
        stat = ClassifyStats()

        placeholder_target_batch = None
        marginalized = None

        num_batches = int(math.ceil(len(dataset) / conf.batch_size))
        rtpt = RTPT(
            name_initials="MW",
            experiment_name="CausalLossImg Eval",
            max_iterations=num_batches,
        )
        rtpt.start()
        for batch_num, batch in enumerate(dataloader):
            condition_batch, target_batch = img_batch_processor(batch)

            if (
                placeholder_target_batch is None
                or placeholder_target_batch.shape != batch["target"]
            ):
                # zero out target vars to be sure
                placeholder_target_batch = torch.zeros_like(
                    batch["target"], dtype=torch.float, device="cuda"
                )
                marginalized = torch.ones_like(placeholder_target_batch)

            reconstruction = eval_wrapper.predict(
                condition_batch, placeholder_target_batch, marginalized
            )

            correct = stat.eval(batch["target"], reconstruction, batch_num == 0)

            if print_progress:
                print(f"Processed batch {batch_num}.", end="\r")
            rtpt.step(f"batch {batch_num}/{num_batches}")

    print(stat.get_eval_result_str())
    logger.close()

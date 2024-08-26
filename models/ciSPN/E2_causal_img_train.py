import argparse
import pickle
import time

import torchvision
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from models.spn_create import save_model_params, save_spn
from trainers.losses import AdditiveLoss

from ciSPN.E1_helpers import create_loss, get_experiment_name
from ciSPN.E2_helpers import (
    create_cnn_for_spn,
    create_cnn_model,
    create_dataloader,
    get_E2_loss_path,
    img_batch_processor,
)
from ciSPN.libs.pawork.log_redirect import PrintLogger
from ciSPN.models.model_creation import create_spn_model
from ciSPN.models.nn_wrapper import NNWrapper
from ciSPN.trainers.dynamicTrainer import DynamicTrainer
from datasets.hiddenObjectDataset import HiddenObjectDataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)
parser.add_argument("--model", choices=["cnn", "ciCNNSPN"], default="cnn")
parser.add_argument(
    "--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default="MSELoss"
)
parser.add_argument("--lr", type=float, default=1e-5)  # default is 1e-3
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument(
    "--loss2_factor", default="1.0"
)  # factor by which loss2 is added to the loss term
parser.add_argument("--epochs", type=int, default=50)  # 40
parser.add_argument(
    "--loss_load_seed", type=int, default=None
)  # is set to seed if none
parser.add_argument("--dataset", choices=["hiddenObject"], default="hiddenObject")
parser.add_argument("--debug", default=None)  # disables dataloaders -> single thread
cli_args = parser.parse_args()

conf = Config()
conf.model_name = cli_args.model
conf.num_epochs = cli_args.epochs
conf.loss_load_seed = (
    cli_args.seed if cli_args.loss_load_seed is None else cli_args.loss_load_seed
)
conf.batch_size = 128
conf.num_workers = 8
conf.multi_thread_data_loading = (
    False if cli_args.debug == "true" else True
)  # otherwise we debug in multi-process setting ...
conf.lr = float(cli_args.lr)
conf.loss_name = cli_args.loss
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed


if __name__ == "__main__":
    make_deterministic(conf.seed, deterministic_cudnn=False)

    # setup experiments folder
    runtime_base_dir = environment["experiments"]["base"] / "E2" / "runtimes"
    log_base_dir = environment["experiments"]["base"] / "E2" / "logs"

    experiment_name = get_experiment_name(
        conf.dataset,
        conf.model_name,
        conf.seed,
        conf.loss_name,
        conf.loss2_name,
        conf.loss2_factor,
        E=2,
    )
    save_dir = runtime_base_dir / experiment_name
    save_dir.mkdir(exist_ok=True, parents=True)

    # redirect logs
    log_path = log_base_dir / (experiment_name + ".txt")
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logger = PrintLogger(log_path)

    print("Arguments:", cli_args)

    # setup dataset
    if cli_args.dataset == "hiddenObject":
        transforms = [torchvision.transforms.ToTensor()]
        image_transform = torchvision.transforms.Compose(transforms)

        dataset_split = "train"
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
    )

    num_condition_vars = dataset.num_observed_variables
    num_target_vars = dataset.num_hidden_variables

    # setup model
    if conf.model_name == "ciCNNSPN":
        rg, params, spn = create_spn_model(
            num_target_vars,
            num_condition_vars,
            conf.seed,
            nn_provider=create_cnn_for_spn,
            setup="hiddenObject",
        )
        model = spn
    elif conf.model_name == "cnn":
        nn = create_cnn_model(num_condition_vars, num_target_vars)
        model = NNWrapper(nn)

    model.print_structure_info()

    # setup loss
    loss, loss_ispn = create_loss(
        conf.loss_name,
        conf,
        nn_provider=create_cnn_for_spn,
        load_dir=runtime_base_dir
        / get_E2_loss_path("hiddenObject", conf.loss_load_seed),
    )

    if conf.loss2_name is not None:
        loss2, loss2_ispn = create_loss(
            conf.loss2_name,
            conf,
            nn_provider=create_cnn_for_spn,
            load_dir=runtime_base_dir
            / get_E2_loss_path("hiddenObject", conf.loss_load_seed),
        )
        final_loss = AdditiveLoss(loss, loss2, float(conf.loss2_factor))
    else:
        final_loss = loss

    trainer = DynamicTrainer(
        model,
        conf,
        final_loss,
        train_loss=False,
        pre_epoch_callback=None,
        optimizer="adam",
        lr=conf.lr,
    )

    t0 = time.time()
    loss_curve = trainer.run_training_dataloader(dataloader, img_batch_processor)
    training_time = time.time() - t0
    print(f"TIME {training_time:.2f}")

    # save results
    if conf.model_name == "ciCNNSPN":
        save_spn(save_dir, spn, params, rg, file_name="spn.model")
    elif conf.model_name == "cnn":
        save_model_params(save_dir, nn, file_name="nn.model")

    # save loss curve
    with open(save_dir / "loss.pkl", "wb") as f:
        pickle.dump(loss_curve, f)

    with open(save_dir / "runtime.txt", "wb") as f:
        pickle.dump(training_time, f)

    print(f'Final parameters saved to "{save_dir}"')
    logger.close()

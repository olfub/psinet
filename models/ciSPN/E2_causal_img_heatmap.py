import argparse
import math

import numpy as np
import torch
import torchvision
from environment import environment, get_dataset_paths
from helpers.configuration import Config
from helpers.determinism import make_deterministic
from matplotlib import cm
from models.spn_create import load_model_params, load_spn
from PIL import Image

from ciSPN.E1_helpers import get_experiment_name
from ciSPN.E2_helpers import create_cnn_for_spn, create_cnn_model, img_batch_processor
from ciSPN.gradcam.gradcam import GradCam
from ciSPN.models.nn_wrapper import NNWrapper
from datasets.hiddenObjectDataset import HiddenObjectDataset

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=606)
parser.add_argument("--model", choices=["cnn", "ciCNNSPN"], default="cnn")
parser.add_argument(
    "--loss", choices=["MSELoss", "NLLLoss", "causalLoss"], default="MSELoss"
)
parser.add_argument("--sample_id", type=int)
parser.add_argument("--loss2", choices=["causalLoss"], default=None)
parser.add_argument(
    "--loss2_factor", default="1.0"
)  # factor by which loss2 is added to the loss term
parser.add_argument("--dataset", choices=["hiddenObject"], default="hiddenObject")
cli_args = parser.parse_args()

conf = Config()
conf.dataset = cli_args.dataset
conf.model_name = cli_args.model
conf.loss_name = cli_args.loss
conf.sample_id = cli_args.sample_id
conf.loss2_name = cli_args.loss2
conf.loss2_factor = cli_args.loss2_factor
conf.dataset = cli_args.dataset
conf.seed = cli_args.seed

conf.explicit_load_part = conf.model_name

if __name__ == "__main__":
    make_deterministic(conf.seed, deterministic_cudnn=False)

    if conf.loss2_name is not None:
        raise RuntimeError("Not supported yet")  # see fixme for gradcam_path below

    # setup experiments folder
    runtime_base_dir = environment["experiments"]["base"] / "E2" / "runtimes"
    gradcam_dir = environment["experiments"]["base"] / "E2" / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    gradcam_path = (
        gradcam_dir / f"{conf.sample_id}_{conf.model_name}_{conf.loss_name}.png"
    )  # FIXME add loss2 and factor to path if needed

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

    batch = dataset[conf.sample_id]
    batch["image"] = torch.tensor(batch["image"])  # FIXME use transform
    batch["target"] = torch.tensor(batch["target"])
    condition_batch, target_batch = img_batch_processor(batch)
    condition_batch = condition_batch.unsqueeze(0)  # add batch dimension
    target_batch = target_batch.unsqueeze(0)  # add batch dimension

    grad_cam = GradCam(eval_wrapper)
    cam = grad_cam.generate_cam(condition_batch, target_batch)

    input_image = condition_batch[0, :3, :, :].cpu().numpy()
    input_image = np.moveaxis(input_image, 0, -1)

    cmap = cm.get_cmap("jet")
    color_cam = cam[:, :]
    color_cam = cmap(color_cam)
    color_cam = color_cam[:, :, :3]

    alpha = 0.5
    overlay = (alpha * input_image) + ((1 - alpha) * color_cam)

    print(f"saving to: {gradcam_path}")
    im = Image.fromarray((overlay * 255).astype(np.uint8))
    im.save(gradcam_path)

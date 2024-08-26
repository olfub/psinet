from models.ciSPN.models.spn_create import load_spn

from models.ciSPN.trainers.losses import CausalLoss, MSELoss, NLLLoss


def get_experiment_name(
    dataset,
    model_name,
    known_intervention,
    seed,
    loss_name=None,
    loss2_name=None,
    loss2_factor_str=None,
    E=1,
):
    intervention_info = "knownI" if known_intervention else "hiddenI"
    exp_name = f"E{E}_{dataset}_{model_name}_{intervention_info}"
    if loss_name is not None:
        exp_name += f"_{loss_name}"
    if loss2_name is not None:
        exp_name += f"_{loss2_name}_{loss2_factor_str}"

    exp_name = f"{exp_name}/{seed}"
    return exp_name


def get_loss_path(dataset_name, known_intervention, loss_load_seed):
    loss_folder = get_experiment_name(
        dataset_name, "ciSPN", known_intervention, loss_load_seed, "NLLLoss", None, None
    )
    return loss_folder


def create_loss(
    loss_name, conf, num_condition_vars=None, load_dir=None, nn_provider=None
):
    loss_ispn = None
    if loss_name == "NLLLoss":
        loss = NLLLoss()
    elif loss_name == "MSELoss":
        loss = MSELoss()
    elif loss_name == "causalLoss":
        loss_ispn, _, _ = load_spn(
            num_condition_vars, load_dir=load_dir, nn_provider=nn_provider
        )
        loss_ispn.eval()
        loss = CausalLoss(loss_ispn)
    else:
        raise ValueError(f"unknown loss name: {conf.loss_name}")

    return loss, loss_ispn

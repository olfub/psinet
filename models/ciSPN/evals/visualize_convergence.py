import glob
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns


# datasets = ["CausalHealthClassification", "ASIA", "CANCER", "EARTHQUAKE"]
# dataset_abrevs = ["CHC", "ASIA", "CANCER", "EARTHQUAKE"]
# loss = "MSELoss"
# eval_folder = "evals"
# model_name = "simpleMlp20"

datasets = ["hiddenObject"]
loss = "MSELoss"
runtimes_folders = [
    "runtimes_img_interv_pos/C_caCNNSPN_NLLLoss_hiddenObject_ep40",
    "runtimes_img_interv_pos/C_simpleCNNModel_causalLoss_hiddenObject_ep40",
    "runtimes_img_interv_pos/C_simpleCNNModel_MSELoss_hiddenObject_ep40",
]
captions = ["ciSPN", "NN with ciSPN", "NN with MSE"]
model_name = "simpleCNNModel"
epochs = 40


def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    ax.fill_between(x, mn, mx, alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    plt.errorbar(x, est, sd, linestyle="None")  # , marker='|')
    ax.margins(x=0)


for runtimes_folder, caption in zip(runtimes_folders, captions):
    fig, ax = plt.subplots()

    file_names = list(glob.glob(f"../../{runtimes_folder}/caCNNSPN_SEED*/loss.pkl"))
    file_names.sort()

    losses = []
    for file_name in file_names:
        with open(file_name, "rb") as f:
            losses.append(pickle.load(f))

    # losses = np.array(losses)

    losses = np.array(losses)

    mean = np.mean(losses, axis=1)
    std = np.std(losses, axis=1)

    # for loss in losses:
    #    plt.plot(loss)
    tsplot(ax, losses, color="#40b3ef")

    """
    tsplot(ax, models["MLP"]["accs"], color="#40b3ef")

    #ax.legend([f'MLP {dataset_abrev}'], loc="upper right")

    plt.rcParams["mathtext.fontset"] = "cm"

    plt.xticks(range(len(labels)), labels, rotation=60)
    plt.xlabel("$\\alpha$")
    plt.ylabel("Accuracy (%)")
    """

    plt.title(caption)
    plt.tight_layout()
    plt.show()

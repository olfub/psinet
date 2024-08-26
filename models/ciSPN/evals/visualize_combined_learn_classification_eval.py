import glob

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
dataset_abrevs = ["hiddenObject"]
loss = "MSELoss"
eval_folder = "evals_img_interv_pos"
model_name = "simpleCNNModel"


def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    plt.xlim(x[0] - 0.1, x[-1] + 0.1)
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)

    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    ax.fill_between(x, mn, mx, alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    plt.errorbar(
        x, est, sd, markersize=8, capsize=5
    )  # , linestyle='None') #, marker='|')
    ax.margins(x=0)


for dataset_abrev, dataset in zip(dataset_abrevs, datasets):
    # factors = [    "0.001", "0.005", "0.01", "0.02", "0.1", "1.0", "10", "100"]
    # labels = ["1e-3", "5e-3", "0.01", "0.02", "0.1", "1.0", "10", "100"]
    factors = ["0.001", "0.01", "0.1", "1.0", "10"]
    labels = ["1e-3", "0.01", "0.1", "1.0", "10"]

    # labelsize = 20
    matplotlib.rcParams.update({"font.size": 20})

    models = dict()

    print(f"Evaluating: {dataset}")

    model_accs_mean = []
    model_accs_std = []
    model_accs = []
    for factor in factors:
        file_names = list(
            glob.glob(
                f"../../{eval_folder}/combined/C1_{model_name}_{loss}_{dataset}_f{factor}_Seed*_eval.txt"
            )
        )
        file_names.sort()
        # print(file_names)

        accuracies = []
        for file_name in file_names:
            with open(file_name) as f:
                lines = f.readlines()
                accuracy = float(lines[-1].split(" ")[-1]) * 100
                accuracies.append(accuracy)

        mean = np.mean(accuracies)
        std = np.std(accuracies)
        # print(f"accs: {}")
        print(
            f"alpha> {factor} :",
            f"{[f'{a:.2f}' for a in accuracies]} (mean|std): {mean:.3f}, {std:.3f}",
        )
        # print(f"min: {np.min(accuracies)} | max: {np.max(accuracies)}")

        model_accs_mean.append(mean)
        model_accs_std.append(std)
        model_accs.append(accuracies)

    models["MLP"] = {
        "mean": model_accs_mean,
        "std": model_accs_std,
        "accs": np.array(model_accs).T,
    }

    argmax = np.argmax(models["MLP"]["mean"])
    val_max = models["MLP"]["mean"][argmax]
    print(f"=== MAX {dataset} === at alpha: {factors[argmax]} [acc: {val_max}]")

    # sns.lineplot(data=models["MLP"]["accs"], x="epoch", y="accuracy")
    # ax = sns.tsplot(data=models["MLP"]["accs"], ci="sd")

    # sns.lineplot(data=models["MLP"]["accs"], ci="sd, x="epoch", y="accuracy")

    # matplotlib.rc('xtick', labelsize=labelsize)
    # matplotlib.rc('ytick', labelsize=labelsize)

    fig, ax = plt.subplots()
    # def tsplot(ax, data, **kw):
    #    x = np.arange(data.shape[1])
    #    est = np.mean(data, axis=0)
    #    sd = np.std(data, axis=0)
    #    cis = (est - sd, est + sd)
    #    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    #    ax.plot(x,est,**kw)
    #    ax.margins(x=0)

    tsplot(ax, models["MLP"]["accs"], color="#40b3ef")

    # ax.legend([f'MLP {dataset_abrev}'], loc="upper right")

    plt.rcParams["mathtext.fontset"] = "cm"

    plt.xticks(range(len(labels)), labels, rotation=60)
    plt.xlabel("$\\alpha$")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()

    plt.show()

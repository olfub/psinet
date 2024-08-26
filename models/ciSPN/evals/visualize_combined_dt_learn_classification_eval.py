import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

datasets = ["ASIA", "CANCER", "EARTHQUAKE"]
dataset_abrevs = ["ASIA", "CANCER", "EARTHQUAKE"]
loss = "GICL"
eval_folder = "evals/dt"  # "evals_dt_combined_loss" # "evals"


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


for dataset, dataset_abrev in zip(datasets, dataset_abrevs):
    # factors = ["0.01", "0.02", "0.1", "0.2", "0.5", "0.8", "1.0", "10.0", "100.0"]
    # labels = ["0.01", "0.02", "0.1", "0.2", "0.5", "0.8", "1.0", "10.0", "100.0"]
    # factors = ["0.0001", "0.001", "0.005", "0.01", "0.02", "0.1", "0.2", "0.5", "0.8", "1.0", "10", "100"]
    # labels = ["1e-4", "1e-3", "0.005", "0.01", "0.02", "0.1", "0.2", "0.5", "0.8", "1.0", "10.0", "100.0"]
    factors = ["0.001", "0.01", "0.1", "1.0", "10"]
    labels = ["1e-3", "0.01", "0.1", "1.0", "10"]

    matplotlib.rcParams.update({"font.size": 20})

    models = dict()

    model_accs_mean = []
    model_accs_std = []
    model_accs = []
    for factor in factors:
        file_names = list(
            glob.glob(
                f"../../{eval_folder}/D1_{dataset}_{loss}_{factor}_Seed*_eval_log.txt"
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
        # print(f"accs: {accuracies}")
        # print(f"min: {np.min(accuracies)} | max: {np.max(accuracies)}")
        print(f"alpha> {factor} (mean|std):", f"{mean:.2f}, {std:.2f}")

        model_accs_mean.append(mean)
        model_accs_std.append(std)
        model_accs.append(accuracies)

    models["DT"] = {
        "mean": model_accs_mean,
        "std": model_accs_std,
        "accs": np.array(model_accs).T,
    }

    argmax = np.argmax(models["DT"]["mean"])
    val_max = models["DT"]["mean"][argmax]
    print(f"=== MAX {dataset} === at alpha: {factors[argmax]} [acc: {val_max}]")

    # sns.lineplot(data=models["DT"]["accs"], x="epoch", y="accuracy")
    # ax = sns.tsplot(data=models["DT"]["accs"], ci="sd")

    # sns.lineplot(data=models["DT"]["accs"], ci="sd, x="epoch", y="accuracy")

    fig, ax = plt.subplots()

    tsplot(ax, models["DT"]["accs"], color="#40b3ef")

    # ax.legend([f'DT {dataset_abrev}'], loc="upper right")

    plt.rcParams["mathtext.fontset"] = "cm"

    plt.xticks(range(len(factors)), labels, rotation=60)
    plt.xlabel("$\\alpha$")
    plt.ylabel("Accuracy (%)")
    plt.tight_layout()

    plt.show()

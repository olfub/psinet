import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = "CausalHealthClassification"
loss = "trainableCausalLoss"

eps = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
# eps = [0, 10, 20, 30, 40, 50, 60, 70]

models = dict()
for part in ["MLP", "iSPN"]:
    model_accs_mean = []
    model_accs_std = []
    model_accs = []
    for ep in eps:
        file_names = list(
            glob.glob(
                f"../../evals/performance/C1_simpleMlp20_{loss}_{dataset}_{ep}_Seed*_{part}_eval.txt"
            )
        )
        file_names.sort()
        print(file_names)

        accuracies = []
        for file_name in file_names:
            with open(file_name) as f:
                lines = f.readlines()
                accuracy = float(lines[-1].split(" ")[1]) * 100
                accuracies.append(accuracy)

        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(f"accs: {accuracies}")
        print(f"min: {np.min(accuracies)} | max: {np.max(accuracies)}")
        print("mean", f"{mean:.2f}")
        print("std", f"{std:.2f}")

        model_accs_mean.append(mean)
        model_accs_std.append(std)
        model_accs.append(accuracies)

    models[part] = {"mean": mean, "std": std, "accs": np.array(model_accs).T}


# sns.lineplot(data=models["MLP"]["accs"], x="epoch", y="accuracy")
# ax = sns.tsplot(data=models["MLP"]["accs"], ci="sd")

# sns.lineplot(data=models["MLP"]["accs"], ci="sd, x="epoch", y="accuracy")

fig, ax = plt.subplots()


def tsplot(ax, data, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


print(models["MLP"]["accs"])
tsplot(ax, models["MLP"]["accs"], color="#40b3ef")
tsplot(ax, models["iSPN"]["accs"], color="orange")

ax.legend(["MLP", "causalLoss"], loc="lower right")

plt.xlabel("Epoch")
plt.xticks(range(len(eps)), eps)
plt.xlabel("Start of loss training (Epoch)")
plt.ylabel("Accuracy (%)")
plt.show()

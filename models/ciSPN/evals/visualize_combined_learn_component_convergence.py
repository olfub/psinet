import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = "CausalHealthClassification"
dataset_abrv = "CHC"
loss = "MSELoss"

epochs = 80

# factors = ["0.0001", "0.001", "0.01", "0.1", "1.0", "10", "100"] # all factors
factors = ["0.1", "1.0", "10", "100"]  # higher factor
factors = ["0.0001", "0.001", "0.01"]  # lower factor
# factors = ["1.0"]

models = dict()

for factor in factors:
    file_names = list(
        glob.glob(
            f"../../runtimes_combined_loss/C_simpleMlp20_{loss}_l2_causalLoss_loss2factor_{factor}_uniform_interventions_{dataset_abrv}_ep80_100k/iSPN_SEED*/loss_components.pkl"
        )
    )
    file_names.sort()
    print(file_names)

    loss_curves_mse = []
    loss_curves_cl = []
    for file_name in file_names:
        with open(file_name, "rb") as f:
            loss_parts = np.array(pickle.load(f))
            loss_mse = loss_parts[:, 0][:: len(loss_parts[:, 0]) // epochs]
            loss_cl = loss_parts[:, 1][:: len(loss_parts[:, 0]) // epochs]
            loss_curves_mse.append(loss_mse)
            loss_curves_cl.append(loss_cl)

    # models[factor] = { "loss_curve": np.log(-np.array(loss_curves)) }
    models[factor] = {"mse": np.array(loss_curves_mse), "cl": np.array(loss_curves_cl)}


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


for factor in factors:
    tsplot(ax, models[factor]["mse"])
    tsplot(ax, models[factor]["cl"])


def flatten(t):
    return [item for sublist in t for item in sublist]


# ax.legend(flatten([("MSE", "causal loss") for factor in factors]), loc="upper left")
ax.legend(flatten([(f"{factor} MSE", f"{factor} causal loss") for factor in factors]))

plt.xlabel("Epoch #")
# plt.xticks(range(len(factors)), factors)
plt.ylabel("Loss")
plt.show()

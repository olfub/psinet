import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dataset = "CausalHealthClassification"
dataset_abrv = "CHC"
loss = "MSELoss"


factors = ["0.0001", "0.001", "0.005", "0.01", "0.02", "0.1", "1.0", "10", "100"]

models = dict()

for factor in factors:
    file_names = list(
        glob.glob(
            f"../../runtimes/C_simpleMlp20_{loss}_l2_causalLoss_loss2factor_{factor}_uniform_interventions_{dataset_abrv}_ep80_100k/iSPN_SEED*/loss.pkl"
        )
    )
    file_names.sort()
    print(file_names)

    loss_curves = []
    for file_name in file_names:
        with open(file_name, "rb") as f:
            loss_curve = pickle.load(f)
            loss_curves.append(loss_curve)

    # models[factor] = { "loss_curve": np.log(-np.array(loss_curves)) }
    models[factor] = {"loss_curve": np.array(loss_curves)}


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
    tsplot(ax, -models[factor]["loss_curve"])

ax.legend(factors, loc="upper right")

plt.xlabel("Epoch #")
plt.xticks(range(len(factors)), factors)
plt.ylabel("MSE + $\\alpha$ * CL $\it{(log scale)}$")
ax.set_yscale("log")
# plt.ylim(-50, 1.0)
plt.show()

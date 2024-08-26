import glob

import numpy as np

for dataset in ["ASIA", "CANCER", "EARTHQUAKE"]:
    for score in ["GiniIndex", "CausalLossScore"]:
        # print("CFG", dataset, score)
        file_names = glob.glob(
            f"../../evals/dt/D1_{dataset}_{score}_Seed*_eval_log.txt"
        )

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
        print(f"{dataset} {score} mean|std:", f"{mean:.2f}, {std:.2f}")


print("======= SCIKIT =======")

for dataset in ["ASIA", "CANCER", "EARTHQUAKE"]:
    for score in ["GiniIndex"]:
        print("CFG", dataset, score)
        file_names = glob.glob(f"../../evals/dt/D1_{dataset}_scikit_Seed*_eval_log.txt")

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
        print(f"{dataset} {score} mean|std:", f"{mean:.2f}, {std:.2f}")

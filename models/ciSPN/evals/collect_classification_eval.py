import glob

import numpy as np

eval_folder = "eval_logs"
models = [("ciSPN", "NLLLoss"), ("mlp", "MSELoss"), ("mlp", "causalLoss")]
datasets = ["CHC", "ASIA", "CANCER", "EARTHQUAKE"]

# eval_folder = "evals_img_interv_pos"
# models = [("caCNNSPN", "NLLLoss"), ("simpleCNNModel", "MSELoss"), ("simpleCNNModel", "causalLoss")]
# datasets = ["hiddenObject"]

seeds = [606, 1011, 3004, 5555, 12096]

for model, loss in models:
    for dataset in datasets:
        print("CFG", dataset, model, loss)
        # file_names = glob.glob(f"../../experiments/E1/{eval_folder}/E1_{dataset}_{model}_{loss}/*.txt")
        file_names = [
            f"../../experiments/E1/{eval_folder}/E1_{dataset}_{model}_{loss}/{seed}.txt"
            for seed in seeds
        ]

        accuracies = []
        for file_name in file_names:
            with open(file_name) as f:
                lines = f.readlines()
                accuracy = float(lines[-1].split(" ")[-1]) * 100
                accuracies.append(accuracy)

        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(
            f"mean: {mean:.3f}",
            f"std: {std:.3f}",
            f"[{', '.join(f'{a:.3f}' for a in accuracies)}]",
        )
    print("=======")

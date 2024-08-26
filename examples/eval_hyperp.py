# WIP file to systematically analyze results of hyperparameter searches
import os
import re
import pandas as pd

hyperp_folder="/workspaces/conditional-einsum/results/causal_bench_hyper3"

# courtesy of ChatGPT
def parse_namespace(namespace_str):
    # Define the pattern to match key-value pairs
    pattern = r"(\w+)\=([^\s,]+)"

    # Find all key-value pairs using regular expression
    matches = re.findall(pattern, namespace_str)

    # Create a dictionary from the matches
    namespace_dict = {key: value.strip("'") for key, value in matches}

    return namespace_dict

table = None
nr_rows = len(os.listdir(hyperp_folder))

for count, config in enumerate(os.listdir(hyperp_folder)):
    with open(hyperp_folder + "/" + config + "/einet_logs.txt") as f:
        lines = f.readlines()
    arguments = parse_namespace(lines[1])
    if len(lines) == 2:
        final_loss = 999999999
    else:
        final_loss = lines[-1].split("NLL: ")[-1][:-1]
    arguments["loss"] = final_loss
    if table is None:
        columns = list(arguments.keys())
        table = pd.DataFrame(columns=columns)
    table = table._append(arguments, ignore_index=True)

def avg_loss_by_properties(table, filter_dict):
    filtered_table = table.copy()
    for key, value in filter_dict.items():
        filtered_table = filtered_table[filtered_table[key] == value]
    # print(pd.to_numeric(filtered_table["loss"]).nsmallest(10))
    print("Mean:", pd.to_numeric(filtered_table["loss"]).mean())
    print("Median:", pd.to_numeric(filtered_table["loss"]).median())
    print("Max:", pd.to_numeric(filtered_table["loss"]).max())
    print("Min:", pd.to_numeric(filtered_table["loss"]).min())

print(pd.to_numeric(table["loss"]).nsmallest(10))
table = table.set_index("exp_id")
pass

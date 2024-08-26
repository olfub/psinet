from ciSPN.datasets.interventionHelpers import InterventionProvider

var_sets = {
    "CHC": {"X": ["A", "F", "H", "M", "intervention"], "Y": ["D1", "D2", "D3"]},
    "ASIA": {"X": ["A", "T", "B", "L", "E", "intervention"], "Y": ["S", "X", "D"]},
    "CANCER": {"X": ["S", "C", "intervention"], "Y": ["P", "X", "D"]},
    "EARTHQUAKE": {"X": ["B", "E", "A", "intervention"], "Y": ["J", "M"]},
    "WATERING": {
        "X": ["intervention", "M", "A", "B", "H"],
        "Y": ["M-cf", "A-cf", "B-cf", "H-cf"],
    },
    "TOY1": {
        "X": ["intervention", "C", "D", "E", "F", "G", "H"],
        "Y": ["C-cf", "D-cf", "E-cf", "F-cf", "G-cf", "H-cf"],
    },
    "TOY2": {
        "X": ["intervention", "C", "D", "E", "F", "G", "H"],
        "Y": ["C-cf", "D-cf", "E-cf", "F-cf", "G-cf", "H-cf"],
    },
    "TOY1I": {"X": ["intervention"], "Y": ["C", "D", "E", "F", "G", "H"]},
}


def get_dataset_abrev_old(dataset_name):
    if dataset_name == "CausalHealthClassification":
        dataset_name_abrev = "CHC"
    elif dataset_name in ["ASIA", "CANCER", "EARTHQUAKE"]:
        dataset_name_abrev = dataset_name
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return dataset_name_abrev


def get_data_description(dataset_abrv):
    interventionProvider = InterventionProvider(dataset_abrv)
    var_set = var_sets[dataset_abrv]
    X = var_set["X"]
    Y = var_set["Y"]

    providers = [interventionProvider]
    return X, Y, providers

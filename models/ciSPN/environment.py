from pathlib import Path

environment = {
    "runtime": {"initials": "FB"},
    "experiments": {
        "base": Path("experiments/"),
    },
    "datasets": {
        "base": Path("datasets/"),
        "CHC_train": {
            "base": Path("causalHealthClassification/"),
            "files": [
                "causalHealthClassification_do(A)=U(A)_N80000_train.pkl",
                "causalHealthClassification_do(F)=U(F)_N80000_train.pkl",
                "causalHealthClassification_do(H)=U(H)_N80000_train.pkl",
                "causalHealthClassification_do(M)=U(M)_N80000_train.pkl",
                "causalHealthClassification_None_N80000_train.pkl",
            ],
        },
        "CHC_test": {
            "base": Path("causalHealthClassification/"),
            "files": [
                "causalHealthClassification_do(A)=U(A)_N20000_test.pkl",
                "causalHealthClassification_do(F)=U(F)_N20000_test.pkl",
                "causalHealthClassification_do(H)=U(H)_N20000_test.pkl",
                "causalHealthClassification_do(M)=U(M)_N20000_test.pkl",
                "causalHealthClassification_None_N20000_test.pkl",
            ],
        },
        "ASIA_train": {
            "base": Path("ASIA/"),
            "files": [
                "ASIA_do(A)=UBin(A)_N80000_train.pkl",
                "ASIA_do(B)=UBin(B)_N80000_train.pkl",
                "ASIA_do(E)=UBin(E)_N80000_train.pkl",
                "ASIA_do(L)=UBin(L)_N80000_train.pkl",
                "ASIA_do(T)=UBin(T)_N80000_train.pkl",
                "ASIA_None_N80000_train.pkl",
            ],
        },
        "ASIA_test": {
            "base": Path("ASIA/"),
            "files": [
                "ASIA_do(A)=UBin(A)_N20000_test.pkl",
                "ASIA_do(B)=UBin(B)_N20000_test.pkl",
                "ASIA_do(E)=UBin(E)_N20000_test.pkl",
                "ASIA_do(L)=UBin(L)_N20000_test.pkl",
                "ASIA_do(T)=UBin(T)_N20000_test.pkl",
                "ASIA_None_N20000_test.pkl",
            ],
        },
        "CANCER_train": {
            "base": Path("CANCER/"),
            "files": [
                "CANCER_do(C)=UBin(C)_N80000_train.pkl",
                "CANCER_do(S)=UBin(S)_N80000_train.pkl",
                "CANCER_None_N80000_train.pkl",
            ],
        },
        "CANCER_test": {
            "base": Path("CANCER/"),
            "files": [
                "CANCER_do(C)=UBin(C)_N20000_test.pkl",
                "CANCER_do(S)=UBin(S)_N20000_test.pkl",
                "CANCER_None_N20000_test.pkl",
            ],
        },
        "EARTHQUAKE_train": {
            "base": Path("EARTHQUAKE/"),
            "files": [
                "EARTHQUAKE_do(A)=UBin(A)_N80000_train.pkl",
                "EARTHQUAKE_do(B)=UBin(B)_N80000_train.pkl",
                "EARTHQUAKE_do(E)=UBin(E)_N80000_train.pkl",
                "EARTHQUAKE_None_N80000_train.pkl",
            ],
        },
        "EARTHQUAKE_test": {
            "base": Path("EARTHQUAKE/"),
            "files": [
                "EARTHQUAKE_do(A)=UBin(A)_N20000_test.pkl",
                "EARTHQUAKE_do(B)=UBin(B)_N20000_test.pkl",
                "EARTHQUAKE_do(E)=UBin(E)_N20000_test.pkl",
                "EARTHQUAKE_None_N20000_test.pkl",
            ],
        },
        "hiddenObject_train": {"base": Path("hiddenObject_20000_4000/")},
        "hiddenObject_test": {"base": Path("hiddenObject_20000_4000/")},
        "WATERING_train": {
            "base": Path("WATERING/"),
            "files": [
                "WATERING_do(M-cf)=UBin(M-cf)_N80000_train.pkl",
                "WATERING_do(A-cf)=UBin(A-cf)_N80000_train.pkl",
                "WATERING_do(B-cf)=UBin(B-cf)_N80000_train.pkl",
                "WATERING_do(H-cf)=UBin(H-cf)_N80000_train.pkl",
                "WATERING_None_N80000_train.pkl",
            ],
        },
        "WATERING_test": {
            "base": Path("WATERING/"),
            "files": [
                "WATERING_do(M-cf)=UBin(M-cf)_N20000_test.pkl",
                "WATERING_do(A-cf)=UBin(A-cf)_N20000_test.pkl",
                "WATERING_do(B-cf)=UBin(B-cf)_N20000_test.pkl",
                "WATERING_do(H-cf)=UBin(H-cf)_N20000_test.pkl",
                "WATERING_None_N20000_test.pkl",
            ],
        },
        "TOY1_train": {
            "base": Path("TOY1/"),
            "files": [
                "TOY1_do(C-cf)=UBin(C-cf)_N80000_train.pkl",
                "TOY1_do(D-cf)=UBin(D-cf)_N80000_train.pkl",
                "TOY1_do(E-cf)=UBin(E-cf)_N80000_train.pkl",
                "TOY1_do(F-cf)=UBin(F-cf)_N80000_train.pkl",
                "TOY1_do(G-cf)=UBin(G-cf)_N80000_train.pkl",
                "TOY1_do(H-cf)=UBin(H-cf)_N80000_train.pkl",
                "TOY1_None_N80000_train.pkl",
            ],
        },
        "TOY1_test": {
            "base": Path("TOY1/"),
            "files": [
                "TOY1_do(C-cf)=UBin(C-cf)_N20000_test.pkl",
                "TOY1_do(D-cf)=UBin(D-cf)_N20000_test.pkl",
                "TOY1_do(E-cf)=UBin(E-cf)_N20000_test.pkl",
                "TOY1_do(F-cf)=UBin(F-cf)_N20000_test.pkl",
                "TOY1_do(G-cf)=UBin(G-cf)_N20000_test.pkl",
                "TOY1_do(H-cf)=UBin(H-cf)_N20000_test.pkl",
                "TOY1_None_N20000_test.pkl",
            ],
        },
        "TOY2_train": {
            "base": Path("TOY2/"),
            "files": [
                "TOY2_do(C-cf)=UBin(C-cf)_N80000_train.pkl",
                "TOY2_do(D-cf)=UBin(D-cf)_N80000_train.pkl",
                "TOY2_do(E-cf)=UBin(E-cf)_N80000_train.pkl",
                "TOY2_do(F-cf)=UBin(F-cf)_N80000_train.pkl",
                "TOY2_do(G-cf)=UBin(G-cf)_N80000_train.pkl",
                "TOY2_do(H-cf)=UBin(H-cf)_N80000_train.pkl",
                "TOY2_None_N80000_train.pkl",
            ],
        },
        "TOY2_test": {
            "base": Path("TOY2/"),
            "files": [
                "TOY2_do(C-cf)=UBin(C-cf)_N20000_test.pkl",
                "TOY2_do(D-cf)=UBin(D-cf)_N20000_test.pkl",
                "TOY2_do(E-cf)=UBin(E-cf)_N20000_test.pkl",
                "TOY2_do(F-cf)=UBin(F-cf)_N20000_test.pkl",
                "TOY2_do(G-cf)=UBin(G-cf)_N20000_test.pkl",
                "TOY2_do(H-cf)=UBin(H-cf)_N20000_test.pkl",
                "TOY2_None_N20000_test.pkl",
            ],
        },
        "TOY1I_train": {
            "base": Path("TOY1I/"),
            "files": [
                "TOY1I_do(C)=UBin(C)_N80000_train.pkl",
                "TOY1I_do(D)=UBin(D)_N80000_train.pkl",
                "TOY1I_do(E)=UBin(E)_N80000_train.pkl",
                "TOY1I_do(F)=UBin(F)_N80000_train.pkl",
                "TOY1I_do(G)=UBin(G)_N80000_train.pkl",
                "TOY1I_do(H)=UBin(H)_N80000_train.pkl",
                "TOY1I_None_N80000_train.pkl",
            ],
        },
        "TOY1I_test": {
            "base": Path("TOY1I/"),
            "files": [
                "TOY1I_do(C)=UBin(C)_N20000_test.pkl",
                "TOY1I_do(D)=UBin(D)_N20000_test.pkl",
                "TOY1I_do(E)=UBin(E)_N20000_test.pkl",
                "TOY1I_do(F)=UBin(F)_N20000_test.pkl",
                "TOY1I_do(G)=UBin(G)_N20000_test.pkl",
                "TOY1I_do(H)=UBin(H)_N20000_test.pkl",
                "TOY1I_None_N20000_test.pkl",
            ],
        },
        "PC_train": {"base": Path("PC/"), "files": ["PC_N192000_train.npy"]},
        "PC_test": {"base": Path("PC/"), "files": ["PC_N48000_test.npy"]},
        "PC_nr_particles": 3,
        "GC_train": {"base": Path("GC/"), "files": ["GC_N96000_train.npy"]},
        "GC_test": {"base": Path("GC/"), "files": ["GC_N24000_test.npy"]},
        "GC_method": "Euler_semi_implicit",
    },
}


def get_dataset_paths(name, mode, get_base=False):
    # mode = train or test
    dataset_cfg = environment["datasets"][f"{name}_{mode}"]
    base = environment["datasets"]["base"] / dataset_cfg["base"]
    if get_base:
        return base
    return [base / file for file in dataset_cfg["files"]]

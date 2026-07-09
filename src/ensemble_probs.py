import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse

from utils import (
    get_challenge_data_root,
    get_results_root,
    get_train_to_test_dataset_mapping,
)

def linear_weighted_ensemble(train_dataset_name, kmer_model_name, vj_model_name, dataset):
    challenge_root = get_challenge_data_root(dataset)
    results_root = get_results_root(dataset)

    orig_train_dataset_labels = (
        challenge_root / "train_datasets" / train_dataset_name / "metadata.csv"
    )
    kmer_train_dataset_preds = results_root / kmer_model_name / f"{train_dataset_name}_train_predictions.tsv"
    vj_train_dataset_preds = results_root / vj_model_name / f"{train_dataset_name}_train_predictions.tsv"
    val_indices = results_root / kmer_model_name / "split_indices" / f"{train_dataset_name}_val_indices.txt"

    kmer_train_dataset_preds = pd.read_csv(kmer_train_dataset_preds, sep="\t")
    vj_train_dataset_preds = pd.read_csv(vj_train_dataset_preds, sep="\t")
    with open(val_indices, "r") as f:
        val_indices = [line.strip() for line in f.readlines()]
    orig_train_dataset_labels = pd.read_csv(orig_train_dataset_labels)

    kmer_train_dataset_preds = kmer_train_dataset_preds.set_index("ID").loc[val_indices].reset_index()
    vj_train_dataset_preds = vj_train_dataset_preds.set_index("ID").loc[val_indices].reset_index()
    orig_train_dataset_labels = orig_train_dataset_labels.set_index("repertoire_id").loc[val_indices].reset_index()

    alphas = np.linspace(0, 1, 11)
    best_alpha = None
    best_auroc = 0

    for alpha in alphas:
        p_ensemble = alpha * kmer_train_dataset_preds["label_positive_probability"] + (1 - alpha) * vj_train_dataset_preds["label_positive_probability"]
        auroc = roc_auc_score(orig_train_dataset_labels["label_positive"], p_ensemble)
        if auroc > best_auroc:
            best_alpha = alpha
            best_auroc = auroc
    
    kmer_only_auroc = roc_auc_score(orig_train_dataset_labels["label_positive"], kmer_train_dataset_preds["label_positive_probability"])
    vj_only_auroc = roc_auc_score(orig_train_dataset_labels["label_positive"], vj_train_dataset_preds["label_positive_probability"])
    print(f"{train_dataset_name} - KMER ONLY AUROC: {kmer_only_auroc:.4f}")
    print(f"{train_dataset_name} - VJ ONLY AUROC: {vj_only_auroc:.4f}")
    print(f"{train_dataset_name} - Best Alpha: {best_alpha:.2f} - Best ENSEMBLE AUROC: {best_auroc:.4f}")
    print()

    # inference on test set
    kmer_test_dataset_preds = pd.read_csv(
        results_root / kmer_model_name / f"{train_dataset_name}_test_predictions.tsv", sep="\t"
    )
    vj_test_dataset_preds = pd.read_csv(
        results_root / vj_model_name / f"{train_dataset_name}_test_predictions.tsv", sep="\t"
    )
    p_ensemble = best_alpha * kmer_test_dataset_preds["label_positive_probability"] + (1 - best_alpha) * vj_test_dataset_preds["label_positive_probability"]

    test_dataset_preds = pd.DataFrame({
        "ID": kmer_test_dataset_preds["ID"],
        "dataset": kmer_test_dataset_preds["dataset"],
        "label_positive_probability": p_ensemble
    })
    output_dir = results_root / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset_preds.to_csv(output_dir / f"{train_dataset_name}_test_predictions.tsv", sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear alpha ensemble of k-mer and V/J models")
    parser.add_argument("--dataset", default="AIRR_ML_25_Phase2_data",
                        help="Dataset directory name or path.")
    parser.add_argument("--kmer_model_name", type=str, required=True, help="k-mer model name")
    parser.add_argument("--vj_model_name", type=str, required=True, help="VJ pairs model name")
    args = parser.parse_args()

    mapping = get_train_to_test_dataset_mapping(args.dataset)
    for train_dataset_name in mapping:
        linear_weighted_ensemble(
            train_dataset_name,
            args.kmer_model_name,
            args.vj_model_name,
            args.dataset,
        )

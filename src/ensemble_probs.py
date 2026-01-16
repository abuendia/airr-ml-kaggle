from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import argparse
from pathlib import Path

from utils import get_challenge_data_root, get_results_root


train_to_test_dataset_mapping = {
    "train_dataset_1": ["test_dataset_1"],
    "train_dataset_2": ["test_dataset_2"],
    "train_dataset_3": ["test_dataset_3"],
    "train_dataset_4": ["test_dataset_4"],
    "train_dataset_5": ["test_dataset_5"],
    "train_dataset_6": ["test_dataset_6"],
    "train_dataset_7": ["test_dataset_7_1", "test_dataset_7_2"],
    "train_dataset_8": ["test_dataset_8_1", "test_dataset_8_2", "test_dataset_8_3"],
}

def stacking_ensemble(train_dataset_name, kmer_model_name, vj_model_name):
    challenge_root = get_challenge_data_root()
    results_root = get_results_root()

    orig_train_dataset_labels = (
        challenge_root / "train_datasets" / "train_datasets" / train_dataset_name / "metadata.csv"
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

    X_meta = np.column_stack([kmer_train_dataset_preds["label_positive_probability"], vj_train_dataset_preds["label_positive_probability"]])
    y_meta = orig_train_dataset_labels["label_positive"]

    # 4-fold cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    fold_aurocs = []
    kmer_only_aurocs = []
    vj_only_aurocs = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_meta)):
        X_train_fold = X_meta[train_idx]
        X_val_fold = X_meta[val_idx]
        y_train_fold = y_meta[train_idx]
        y_val_fold = y_meta[val_idx]

        # Train meta-learner on training fold
        meta = LogisticRegression(penalty='l2', solver='liblinear', max_iter=10000)
        meta.fit(X_train_fold, y_train_fold)

        # Predict on validation fold
        y_pred_fold = meta.predict_proba(X_val_fold)[:, 1]

        # Compute AUROC for this fold
        fold_auroc = roc_auc_score(y_val_fold, y_pred_fold)
        fold_aurocs.append(fold_auroc)
        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

        kmer_only_auroc = roc_auc_score(y_val_fold, kmer_train_dataset_preds["label_positive_probability"][val_idx])
        vj_only_auroc = roc_auc_score(y_val_fold, vj_train_dataset_preds["label_positive_probability"][val_idx])
        kmer_only_aurocs.append(kmer_only_auroc)
        vj_only_aurocs.append(vj_only_auroc)

    # Overall AUROC across all folds
    print(f"{train_dataset_name} - KMER ONLY AUROC: {np.mean(kmer_only_aurocs):.4f} (+/- {np.std(kmer_only_aurocs):.4f})")
    print(f"{train_dataset_name} - VJ ONLY AUROC: {np.mean(vj_only_aurocs):.4f} (+/- {np.std(vj_only_aurocs):.4f})")
    print(f"{train_dataset_name} - ENSEMBLE AUROC: {np.mean(fold_aurocs):.4f} (+/- {np.std(fold_aurocs):.4f})")
    print()

    meta = LogisticRegression(penalty='l2', solver='liblinear', max_iter=10000)
    meta.fit(X_meta, y_meta)

    kmer_test_dataset_preds = results_root / kmer_model_name / f"{train_dataset_name}_test_predictions.tsv"
    vj_test_dataset_preds = results_root / vj_model_name / f"{train_dataset_name}_test_predictions.tsv"
    # run through meta model
    kmer_test_dataset_preds = pd.read_csv(kmer_test_dataset_preds, sep="\t")
    vj_test_dataset_preds = pd.read_csv(vj_test_dataset_preds, sep="\t")
    X_test_meta = np.column_stack([kmer_test_dataset_preds["label_positive_probability"], vj_test_dataset_preds["label_positive_probability"]])
    y_test_pred = meta.predict_proba(X_test_meta)[:, 1]

    test_dataset_preds = pd.DataFrame({
        "ID": kmer_test_dataset_preds["ID"],
        "dataset": train_dataset_name,
        "label_positive_probability": y_test_pred
    })
    output_dir = results_root / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset_preds.to_csv(output_dir / f"{train_dataset_name}_test_predictions.tsv", sep="\t", index=False)

def linear_weighted_ensemble(train_dataset_name, kmer_model_name, vj_model_name):
    challenge_root = get_challenge_data_root()
    results_root = get_results_root()

    orig_train_dataset_labels = (
        challenge_root / "train_datasets" / "train_datasets" / train_dataset_name / "metadata.csv"
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
        "dataset": train_dataset_name,
        "label_positive_probability": p_ensemble
    })
    output_dir = results_root / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    test_dataset_preds.to_csv(output_dir / f"{train_dataset_name}_test_predictions.tsv", sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stacking Ensemble of kmer and VJ models")
    parser.add_argument("--kmer_model_name", type=str, required=True, help="k-mer model name")
    parser.add_argument("--vj_model_name", type=str, required=True, help="VJ pairs model name")
    args = parser.parse_args()

    for train_dataset_name, test_dataset_names in train_to_test_dataset_mapping.items():
        linear_weighted_ensemble(train_dataset_name, args.kmer_model_name, args.vj_model_name)

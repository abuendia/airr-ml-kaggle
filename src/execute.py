import argparse
import os
from typing import List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from contextlib import redirect_stdout, redirect_stderr
import traceback
from pathlib import Path

from model import ImmuneStatePredictor
from utils import validate_dirs_and_files, save_tsv
from utils import concatenate_output_files
from utils import get_challenge_data_root, get_results_root, get_train_to_test_dataset_mapping
from write_final_output import write_submission


def _train_predictor(predictor: ImmuneStatePredictor, train_dir: str):
    """Trains the predictor on the training data."""
    print(f"Fitting model on examples in ` {train_dir} `...")
    predictor.fit(train_dir)


def _generate_predictions(predictor: ImmuneStatePredictor, test_dirs: List[str]) -> pd.DataFrame:
    """Generates predictions for all test directories and concatenates them."""
    all_preds = []
    for test_dir in test_dirs:
        print(f"Predicting on examples in ` {test_dir} `...")
        preds = predictor.predict_proba(test_dir)
        if preds is not None and not preds.empty:
            all_preds.append(preds)
        else:
            print(f"Warning: No predictions returned for {test_dir}")
    if all_preds:
        return pd.concat(all_preds, ignore_index=True)
    return pd.DataFrame()


def _save_predictions(predictions: pd.DataFrame, out_dir: str, train_dir: str, is_train_set=False) -> None:
    """Saves predictions to a TSV file."""
    if predictions.empty:
        raise ValueError("No predictions to save - predictions DataFrame is empty")

    if is_train_set:
        preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_train_predictions.tsv")
    else:
        preds_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_test_predictions.tsv")
    save_tsv(predictions, preds_path)
    print(f"Predictions written to `{preds_path}`.")


def _compute_ensemble_alpha(
    kmer_train_preds: pd.DataFrame,
    vj_train_preds: pd.DataFrame,
    val_ids,
    labels_path: str,
) -> float:
    """Compute best linear ensemble alpha on the held-out val set (inline, no disk reads)."""
    from sklearn.metrics import roc_auc_score
    labels_df = pd.read_csv(labels_path).set_index("repertoire_id")
    val_ids = list(val_ids)
    kmer_val = kmer_train_preds.set_index("ID").loc[val_ids, "label_positive_probability"].values
    vj_val = vj_train_preds.set_index("ID").loc[val_ids, "label_positive_probability"].values
    y_val = labels_df.loc[val_ids, "label_positive"].values
    if pd.Series(y_val).nunique() < 2:
        raise ValueError("The validation split contains only one class; AUROC cannot be computed.")
    best_alpha, best_auroc = 0.0, float("-inf")
    for alpha in np.linspace(0, 1, 11):
        auroc = roc_auc_score(y_val, alpha * kmer_val + (1 - alpha) * vj_val)
        if auroc > best_auroc:
            best_alpha, best_auroc = float(alpha), auroc
    print(f"Best ensemble alpha={best_alpha:.2f} (AUROC={best_auroc:.4f})")
    return best_alpha


def _save_ensemble_test_predictions(
    kmer_test_preds: pd.DataFrame,
    vj_test_preds: pd.DataFrame,
    alpha: float,
    ensemble_dir: str,
    dataset_name: str,
) -> None:
    """Write linear-weighted ensemble test predictions to the ensemble output directory."""
    keys = ["dataset", "ID"]
    combined = kmer_test_preds[keys + ["label_positive_probability"]].merge(
        vj_test_preds[keys + ["label_positive_probability"]],
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_kmer", "_vj"),
    )
    if len(combined) != len(kmer_test_preds) or len(combined) != len(vj_test_preds):
        raise ValueError("K-mer and VJ predictions do not contain identical (dataset, ID) pairs.")
    ensemble_preds = combined[keys].copy()
    ensemble_preds["label_positive_probability"] = (
        alpha * combined["label_positive_probability_kmer"]
        + (1 - alpha) * combined["label_positive_probability_vj"]
    )
    out_path = os.path.join(ensemble_dir, f"{dataset_name}_test_predictions.tsv")
    save_tsv(ensemble_preds, out_path)
    print(f"Ensemble predictions written to `{out_path}`.")


def _save_important_sequences(
    predictor: ImmuneStatePredictor,
    out_dir: str,
    train_dir: str,
    top_k: int = 50000,
) -> None:
    """Save the sequences ranked by the fitted k-mer model."""
    dataset_name = os.path.basename(train_dir)
    seqs = predictor.identify_associated_sequences(
        train_dir_path=train_dir,
        top_k=top_k,
    )
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences available to save")

    seqs_path = os.path.join(out_dir, f"{dataset_name}_important_sequences.tsv")
    save_tsv(seqs, seqs_path)
    print(f"Important sequences written to `{seqs_path}`.")


def _save_val_indices_and_performance(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves validation set patient IDs to a text file."""
    val_ids = predictor.model.val_ids_
    val_score = predictor.model.val_score_
    if val_ids is None or len(val_ids) == 0:
        print(f"Warning: No validation IDs available to save for {train_dir}")
        return

    dataset_id = os.path.basename(train_dir)
    split_indices_dir = os.path.join(out_dir, "split_indices")
    os.makedirs(split_indices_dir, exist_ok=True)
    
    val_indices_path = os.path.join(split_indices_dir, f"{dataset_id}_val_indices.txt")
    with open(val_indices_path, 'w') as f:
        for val_id in val_ids:
            f.write(f"{val_id}\n")
    print(f"Validation indices written to `{val_indices_path}`.")

    performance_path = os.path.join(split_indices_dir, f"{dataset_id}_performance.txt")
    with open(performance_path, 'w') as f:
        f.write(f"Validation score: {val_score}\n")
    print(f"Validation performance written to `{performance_path}`.")


def main(
        train_dir: str,
        test_dirs: List[str],
        out_dir: str,
        n_jobs: int,
        model_type: str,
        classifier_type: str,
        save_important_sequences: bool,
        dataset: str,
        vj_model_name: Optional[str] = None,
        vj_out_dir: Optional[str] = None,
        top_k: int = 50000,
    ) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)

    if model_type == 'both':
        dataset_name = os.path.basename(train_dir.rstrip(os.sep))

        # Train kmer model
        kmer_predictor = ImmuneStatePredictor(n_jobs=n_jobs, model_type='kmer', classifier_type=classifier_type)
        _train_predictor(kmer_predictor, train_dir)
        _save_val_indices_and_performance(kmer_predictor, out_dir, train_dir)
        kmer_train_preds = kmer_predictor.predict_proba(train_dir)
        _save_predictions(kmer_train_preds, out_dir, train_dir, is_train_set=True)
        kmer_test_preds = _generate_predictions(kmer_predictor, test_dirs)
        _save_predictions(kmer_test_preds, out_dir, train_dir, is_train_set=False)

        # Train VJ model
        if vj_out_dir is None:
            raise ValueError("vj_out_dir must be provided when model_type='both'")
        os.makedirs(vj_out_dir, exist_ok=True)
        vj_predictor = ImmuneStatePredictor(n_jobs=n_jobs, model_type='vj', classifier_type=classifier_type)
        _train_predictor(vj_predictor, train_dir)
        vj_train_preds = vj_predictor.predict_proba(train_dir)
        _save_predictions(vj_train_preds, vj_out_dir, train_dir, is_train_set=True)
        vj_test_preds = _generate_predictions(vj_predictor, test_dirs)
        _save_predictions(vj_test_preds, vj_out_dir, train_dir, is_train_set=False)

        # Compute ensemble alpha inline using the held-out val set
        labels_path = str(get_challenge_data_root(dataset) / "train_datasets" / dataset_name / "metadata.csv")
        alpha = _compute_ensemble_alpha(kmer_train_preds, vj_train_preds, kmer_predictor.model.val_ids_, labels_path)
        ensemble_dir = str(get_results_root(dataset) / "ensemble")
        _save_ensemble_test_predictions(kmer_test_preds, vj_test_preds, alpha, ensemble_dir, dataset_name)

        # Match origin/main: rank important sequences with the k-mer model only.
        if save_important_sequences:
            seqs = kmer_predictor.identify_associated_sequences(
                train_dir_path=train_dir,
                top_k=top_k,
            )
            if seqs is None or seqs.empty:
                raise ValueError("No important sequences available to save")
            seqs_path = os.path.join(out_dir, f"{dataset_name}_important_sequences.tsv")
            save_tsv(seqs, seqs_path)
            print(f"Important sequences written to `{seqs_path}`.")
        return

    # Single-model path (kmer or vj)
    predictor = ImmuneStatePredictor(
        n_jobs=n_jobs,
        model_type=model_type,
        classifier_type=classifier_type,
    )
    _train_predictor(predictor, train_dir)
    _save_val_indices_and_performance(predictor, out_dir, train_dir)
    train_predictions = predictor.predict_proba(train_dir)
    _save_predictions(train_predictions, out_dir, train_dir, is_train_set=True)
    if save_important_sequences:
        _save_important_sequences(
            predictor, out_dir, train_dir, top_k=top_k,
        )
    test_predictions = _generate_predictions(predictor, test_dirs)
    _save_predictions(test_predictions, out_dir, train_dir, is_train_set=False)


def _run_dataset_job(
        train_dir: str,
        test_dirs: List[str],
        out_dir: str,
        model_type: str,
        classifier_type: str,
        per_job_n_jobs: int,
        save_important_sequences: bool,
        dataset: str,
        vj_model_name: Optional[str] = None,
        vj_out_dir: Optional[str] = None,
        top_k: int = 50000,
    ) -> None:
    """Wrapper for running one train dataset end-to-end (train, predict, write outputs).

    Writes a dedicated log file per dataset job under: <out_dir>/logs/
    """
    dataset_id = os.path.basename(train_dir.rstrip(os.sep))
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset_id}.log")

    with open(log_path, "w") as log_fh, redirect_stdout(log_fh), redirect_stderr(log_fh):
        print(f"=== Dataset job start: {dataset_id} ===")
        print(f"train_dir: {train_dir}")
        print(f"test_dirs: {test_dirs}")
        print(f"model_type: {model_type}")
        print(f"classifier_type: {classifier_type}")
        print(f"per_job_n_jobs: {per_job_n_jobs}")
        print(f"pid: {os.getpid()}")
        print("=== Running ===")
        try:
            main(
                train_dir=train_dir,
                test_dirs=test_dirs,
                out_dir=out_dir,
                n_jobs=per_job_n_jobs,
                model_type=model_type,
                classifier_type=classifier_type,
                save_important_sequences=save_important_sequences,
                dataset=dataset,
                vj_model_name=vj_model_name,
                vj_out_dir=vj_out_dir,
                top_k=top_k,
            )
            print(f"=== Dataset job complete: {dataset_id} ===")
        except Exception:
            print(f"=== Dataset job FAILED: {dataset_id} ===")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
    parser.add_argument("--dataset", type=str, default="AIRR_ML_25_Phase1_data", help="Dataset name")
    parser.add_argument("--model_type", type=str, default='kmer', choices=['kmer', 'vj', 'both'],
                        help="Model type to use ('kmer', 'vj', or 'both'). "
                             "'both' trains kmer and vj together, computes ensemble alpha, "
                             "saves ensemble test predictions, and optionally saves important sequences.")
    parser.add_argument("--model_name", type=str, default='4mer-logreg', help="Model name to use.")
    parser.add_argument("--classifier_type", type=str, default='logistic', help="Classifier type to use.")
    parser.add_argument("--parallel_jobs", type=int, default=8, help="Number of datasets to run in parallel (default: 8).")
    parser.add_argument("--per_job_n_jobs", type=int, default=4, help="CPU cores to use within each dataset job.")
    parser.add_argument("--save_important_sequences", action='store_true', help="Whether to save important sequences.")
    parser.add_argument("--vj_model_name", type=str, default=None,
                        help="Output model name for VJ predictions when --model_type=both.")
    parser.add_argument("--dataset-ids", nargs="+", default=None,
                        help="Optional train dataset suffixes to run, e.g. --dataset-ids 1 7.")
    parser.add_argument("--top-k", type=int, default=50000,
                        help="Number of important sequences saved per train dataset.")
    args = parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type
    classifier_type = args.classifier_type
    parallel_jobs = int(args.parallel_jobs)
    per_job_n_jobs = int(args.per_job_n_jobs)
    save_important_sequences = args.save_important_sequences
    vj_model_name = args.vj_model_name
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    challenge_root = get_challenge_data_root(args.dataset)
    train_datasets_dir = challenge_root / "train_datasets"
    test_datasets_dir = challenge_root / "test_datasets"
    results_dir = get_results_root(args.dataset) / model_name

    os.makedirs(str(results_dir), exist_ok=True)

    vj_out_dir = None
    if model_type == 'both':
        if vj_model_name is None:
            parser.error("--vj_model_name is required when --model_type=both")
        vj_results_dir = get_results_root(args.dataset) / vj_model_name
        os.makedirs(str(vj_results_dir), exist_ok=True)
        vj_out_dir = str(vj_results_dir)

    mapping = get_train_to_test_dataset_mapping(args.dataset)
    if args.dataset_ids:
        requested = {str(dataset_id) for dataset_id in args.dataset_ids}
        mapping = {
            train_name: test_names
            for train_name, test_names in mapping.items()
            if train_name.removeprefix("train_dataset_") in requested
        }
        found = {name.removeprefix("train_dataset_") for name in mapping}
        missing = sorted(requested - found)
        if missing:
            parser.error(f"No mapped train/test directories for dataset IDs: {', '.join(missing)}")
    if not mapping:
        parser.error(f"No matching train/test dataset directories found under {challenge_root}")
    train_test_dataset_pairs = [
        (
            str(train_datasets_dir / train_name),
            [str(test_datasets_dir / test_name) for test_name in test_names],
        )
        for train_name, test_names in mapping.items()
    ]

    # Run each training dataset (and its mapped test dirs) in parallel.
    Parallel(n_jobs=parallel_jobs, backend="loky")(
        delayed(_run_dataset_job)(
            train_dir=train_dir,
            test_dirs=test_dirs,
            out_dir=str(results_dir),
            model_type=model_type,
            classifier_type=classifier_type,
            per_job_n_jobs=per_job_n_jobs,
            save_important_sequences=save_important_sequences,
            dataset=args.dataset,
            vj_model_name=vj_model_name,
            vj_out_dir=vj_out_dir,
            top_k=args.top_k,
        )
        for train_dir, test_dirs in train_test_dataset_pairs
    )
    if model_type == 'both':
        # Use the dataset-safe writer so repeated repertoire IDs are joined using
        # (dataset, ID) and Phase 2 rows follow the sample-submission order.
        write_submission(
            args.dataset,
            input_dir=get_results_root(args.dataset) / "ensemble",
            sequences_dir=results_dir,
            train_dataset_names=mapping.keys(),
            test_dataset_names=[
                test_name
                for test_names in mapping.values()
                for test_name in test_names
            ],
        )
    else:
        concatenate_output_files(str(results_dir))

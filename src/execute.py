import argparse
import os
from typing import List
import pandas as pd
from joblib import Parallel, delayed
from contextlib import redirect_stdout, redirect_stderr
import traceback
from pathlib import Path

from model import ImmuneStatePredictor
from utils import validate_dirs_and_files, save_tsv
from utils import get_dataset_pairs, concatenate_output_files
from utils import get_challenge_data_root, get_results_root


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


def _save_important_sequences(predictor: ImmuneStatePredictor, out_dir: str, train_dir: str) -> None:
    """Saves important sequences to a TSV file."""
    seqs = predictor.identify_associated_sequences(train_dir_path=train_dir)
    if seqs is None or seqs.empty:
        raise ValueError("No important sequences available to save")

    seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
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
        device: str, 
        model_type: str, 
        classifier_type: str,
        save_important_sequences: bool
    ) -> None:
    validate_dirs_and_files(train_dir, test_dirs, out_dir)
    predictor = ImmuneStatePredictor(
        n_jobs=n_jobs,
        device=device,
        model_type=model_type,
        classifier_type=classifier_type,
    )
    _train_predictor(predictor, train_dir)
    _save_val_indices_and_performance(predictor, out_dir, train_dir)
    train_predictions = _generate_predictions(predictor, [train_dir])
    _save_predictions(train_predictions, out_dir, train_dir, is_train_set=True)
    # if save_important_sequences:
    #     _save_important_sequences(predictor, out_dir, train_dir)
    # test_predictions = _generate_predictions(predictor, test_dirs)
    # _save_predictions(test_predictions, out_dir, train_dir, is_train_set=False)


def _run_dataset_job(
        train_dir: str, 
        test_dirs: List[str], 
        out_dir: str, 
        model_type: str, 
        classifier_type: str, 
        per_job_n_jobs: int, 
        save_important_sequences: bool, 
        device: str
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
        print(f"device: {device}")
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
                device=device,
            )
            print(f"=== Dataset job complete: {dataset_id} ===")
        except Exception:
            print(f"=== Dataset job FAILED: {dataset_id} ===")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Immune State Predictor CLI")
    parser.add_argument("--model_type", type=str, default='kmer', choices=['kmer', 'vj'],
                        help="Model type to use ('kmer' or 'vj').")
    parser.add_argument("--model_name", type=str, default='4mer-logreg', help="Model name to use.")
    parser.add_argument("--classifier_type", type=str, default='logistic', help="Classifier type to use.")
    parser.add_argument("--parallel_jobs", type=int, default=8, help="Number of datasets to run in parallel (default: 8).")
    parser.add_argument("--per_job_n_jobs", type=int, default=4, help="CPU cores to use within each dataset job.")
    parser.add_argument("--save_important_sequences", action='store_true', help="Whether to save important sequences.")
    parser.add_argument("--device", type=str, default='gpu', help="Device to use (e.g. 'cpu', 'gpu', 'cuda').")
    args = parser.parse_args()

    model_name = args.model_name
    model_type = args.model_type
    classifier_type = args.classifier_type
    parallel_jobs = int(args.parallel_jobs)
    per_job_n_jobs = int(args.per_job_n_jobs)
    save_important_sequences = args.save_important_sequences
    device = args.device

    challenge_root = get_challenge_data_root()
    train_datasets_dir = challenge_root / "train_datasets" / "train_datasets"
    test_datasets_dir = challenge_root / "test_datasets" / "test_datasets"
    results_dir = get_results_root() / model_name

    os.makedirs(str(results_dir), exist_ok=True)
    train_test_dataset_pairs = get_dataset_pairs(str(train_datasets_dir), str(test_datasets_dir))

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
            device=device,
        )
        for train_dir, test_dirs in train_test_dataset_pairs
    )
    concatenate_output_files(str(results_dir))

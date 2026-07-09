"""Build a final submission without conflating repertoire IDs across datasets."""

import argparse
from pathlib import Path

import pandas as pd

from utils import get_challenge_data_root, get_repo_root, get_results_root


OUTPUT_COLUMNS = [
    'ID', 'dataset', 'label_positive_probability',
    'junction_aa', 'v_call', 'j_call',
]


def _normalise_prediction_dataset(dataset: pd.Series) -> pd.Series:
    """Map legacy train-dataset labels in prediction files to test-dataset labels."""
    return dataset.astype(str).str.replace(
        r'^train_dataset_', 'test_dataset_', regex=True,
    )


def _load_test_predictions(
    predictions_dir: Path,
    train_dataset_names=None,
) -> pd.DataFrame:
    if train_dataset_names is None:
        files = sorted(predictions_dir.glob('train_dataset_*_test_predictions.tsv'))
    else:
        files = [
            predictions_dir / f'{name}_test_predictions.tsv'
            for name in sorted(train_dataset_names)
        ]
        missing_files = [path for path in files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f'Missing prediction files: {missing_files}')
    if not files:
        raise FileNotFoundError(f'No per-dataset test predictions found in {predictions_dir}')

    predictions = pd.concat((pd.read_csv(path, sep='\t') for path in files), ignore_index=True)
    required = {'ID', 'dataset', 'label_positive_probability'}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f'Prediction files are missing columns: {sorted(missing)}')

    predictions = predictions[['dataset', 'ID', 'label_positive_probability']].copy()
    predictions['dataset'] = _normalise_prediction_dataset(predictions['dataset'])
    if predictions.duplicated(['dataset', 'ID']).any():
        raise ValueError('Duplicate (dataset, ID) pairs found in prediction files.')
    return predictions


def _load_important_sequences(sequences_dir: Path, train_dataset_names=None) -> pd.DataFrame:
    if train_dataset_names is None:
        files = sorted(sequences_dir.glob('train_dataset_*_important_sequences.tsv'))
    else:
        files = [
            sequences_dir / f'{name}_important_sequences.tsv'
            for name in sorted(train_dataset_names)
        ]
        missing_files = [path for path in files if not path.is_file()]
        if missing_files:
            raise FileNotFoundError(f'Missing important-sequence files: {missing_files}')
    if not files:
        raise FileNotFoundError(f'No important-sequence files found in {sequences_dir}')
    return pd.concat((pd.read_csv(path, sep='\t') for path in files), ignore_index=True)


def _prediction_template(predictions: pd.DataFrame) -> pd.DataFrame:
    """Create a deterministic fallback when a competition template is unavailable."""
    template = predictions[['ID', 'dataset']].copy()
    template['label_positive_probability'] = 0.5
    template['junction_aa'] = -999.0
    template['v_call'] = -999.0
    template['j_call'] = -999.0
    return template[OUTPUT_COLUMNS]


def _resolve_sample_submission(dataset: str, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f'Sample submission does not exist: {path}')
        return path

    data_root = get_challenge_data_root(dataset)
    dataset_local = data_root / 'sample_submission.csv'
    if dataset_local.is_file():
        return dataset_local

    dataset_name = data_root.name.lower()
    phase = 'phase1' if 'phase1' in dataset_name else 'phase2' if 'phase2' in dataset_name else None
    if phase is not None:
        bundled_template = get_repo_root() / 'sample_csv' / f'sample_submission_{phase}.csv'
        if bundled_template.is_file():
            return bundled_template
    return None


def build_submission(
    sample_submission: pd.DataFrame,
    predictions: pd.DataFrame,
    important_sequences: pd.DataFrame,
) -> pd.DataFrame:
    """Attach each probability using the composite (dataset, ID) key."""
    test_mask = sample_submission['dataset'].astype(str).str.startswith('test_dataset')
    expected = sample_submission.loc[test_mask, ['dataset', 'ID']]
    matched = expected.merge(
        predictions,
        on=['dataset', 'ID'],
        how='left',
        validate='one_to_one',
    )
    if matched['label_positive_probability'].isna().any():
        missing = matched.loc[matched['label_positive_probability'].isna(), ['dataset', 'ID']]
        raise ValueError(f'Missing predictions for {len(missing)} test (dataset, ID) pairs.')
    extra = predictions.merge(expected, on=['dataset', 'ID'], how='left', indicator=True)
    if (extra['_merge'] == 'left_only').any():
        raise ValueError('Predictions contain unexpected test (dataset, ID) pairs.')

    test_rows = sample_submission.loc[test_mask].copy()
    test_rows['label_positive_probability'] = matched['label_positive_probability'].to_numpy()
    if important_sequences.empty:
        final = test_rows
    else:
        final = pd.concat([test_rows, important_sequences], ignore_index=True)
    return final.reindex(columns=OUTPUT_COLUMNS)


def write_submission(
    dataset: str,
    *,
    input_dir=None,
    sequences_dir=None,
    feat_importance_model='4mer-logreg',
    submission_name='submissions.csv',
    sample_submission_path=None,
    train_dataset_names=None,
    test_dataset_names=None,
) -> Path:
    """Write predictions in template order using the composite (dataset, ID) key."""
    results_root = get_results_root(dataset)
    predictions_dir = Path(input_dir) if input_dir else results_root / 'ensemble'
    sequences_dir = Path(sequences_dir) if sequences_dir else results_root / feat_importance_model

    predictions = _load_test_predictions(predictions_dir, train_dataset_names)
    important_sequences = _load_important_sequences(sequences_dir, train_dataset_names)
    template_path = _resolve_sample_submission(dataset, sample_submission_path)
    if template_path is None:
        sample_submission = _prediction_template(predictions)
        print(f'No sample submission found for {dataset}; using deterministic prediction order.')
    else:
        sample_submission = pd.read_csv(template_path)
        print(f'Using sample submission template: {template_path}')

    if test_dataset_names is not None:
        expected = set(test_dataset_names)
        sample_submission = sample_submission[
            sample_submission['dataset'].astype(str).isin(expected)
        ].copy()
        found = set(sample_submission['dataset'].astype(str).unique())
        missing = expected - found
        if missing:
            raise ValueError(f'Sample submission is missing test datasets: {sorted(missing)}')

    final = build_submission(sample_submission, predictions, important_sequences)
    output_path = predictions_dir / submission_name
    final.to_csv(output_path, index=False)
    print(f'Wrote {len(final)} rows to {output_path}')
    return output_path


def main(args: argparse.Namespace) -> None:
    write_submission(
        args.dataset,
        input_dir=args.input_dir,
        sequences_dir=args.sequences_dir,
        feat_importance_model=args.feat_importance_model,
        submission_name=args.submission_name,
        sample_submission_path=args.sample_submission,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write a dataset-safe final submission CSV.')
    parser.add_argument('--dataset', default='AIRR_ML_25_Phase2_data')
    parser.add_argument('--input-dir', default=None, help='Directory containing *_test_predictions.tsv files.')
    parser.add_argument('--sequences-dir', default=None, help='Directory containing important-sequence TSV files.')
    parser.add_argument('--feat-importance-model', default='4mer-logreg')
    parser.add_argument('--sample-submission', default=None, help='Optional explicit sample-submission CSV.')
    parser.add_argument('--submission-name', default='submissions.csv')
    main(parser.parse_args())

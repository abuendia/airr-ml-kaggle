"""Validate Phase 1 or Phase 2 submissions against the corresponding template."""

import argparse
from pathlib import Path

import pandas as pd
import pandas.api.types


COLUMN_NAMES = [
    'ID',
    'dataset',
    'label_positive_probability',
    'junction_aa',
    'v_call',
    'j_call',
]
SEQUENCE_COLUMNS = ['junction_aa', 'v_call', 'j_call']
STRING_DTYPES = {column: str for column in SEQUENCE_COLUMNS}
REPO_ROOT = Path(__file__).resolve().parent.parent


class ParticipantVisibleError(Exception):
    pass


def assert_submission_shape(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    if submission.shape != sample_submission.shape:
        raise ParticipantVisibleError(
            f'Submission shape is {submission.shape}, but expected '
            f'{sample_submission.shape} from the sample submission.'
        )


def assert_matching_dataset_counts(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    expected = sample_submission['dataset'].astype(str).value_counts().sort_index()
    actual = submission['dataset'].astype(str).value_counts().sort_index()
    if not actual.equals(expected):
        missing = expected.subtract(actual, fill_value=0)
        mismatched = missing[missing != 0].to_dict()
        raise ParticipantVisibleError(
            f'Submission dataset row counts do not match the template: {mismatched}'
        )


def assert_column_names(submission: pd.DataFrame) -> None:
    if submission.columns.tolist() != COLUMN_NAMES:
        raise ParticipantVisibleError(
            f'Submission columns must be exactly {COLUMN_NAMES}; '
            f'found {submission.columns.tolist()}.'
        )


def assert_column_types(submission: pd.DataFrame) -> None:
    dataset = submission['dataset'].astype(str)
    test_mask = dataset.str.startswith('test_dataset')
    train_mask = dataset.str.startswith('train_dataset')
    if not (test_mask | train_mask).all():
        raise ParticipantVisibleError('Submission contains unknown dataset names.')

    probabilities = submission.loc[test_mask, 'label_positive_probability']
    if probabilities.isnull().any():
        raise ParticipantVisibleError(
            'label_positive_probability contains missing values for test datasets.'
        )
    if not pandas.api.types.is_numeric_dtype(probabilities):
        raise ParticipantVisibleError(
            'label_positive_probability must be numeric for test datasets.'
        )
    if not probabilities.between(0, 1).all():
        raise ParticipantVisibleError(
            'label_positive_probability must be between 0 and 1 for test datasets.'
        )

    for column in SEQUENCE_COLUMNS:
        if submission[column].isnull().any():
            raise ParticipantVisibleError(f'{column} contains missing values.')
        if not submission.loc[train_mask, column].map(
            lambda value: isinstance(value, str) and value != '-999.0'
        ).all():
            raise ParticipantVisibleError(
                f"{column} must be a sequence annotation, not '-999.0', for train datasets."
            )
        if not submission.loc[test_mask, column].eq('-999.0').all():
            raise ParticipantVisibleError(
                f"{column} must be '-999.0' for test datasets."
            )


def assert_matching_test_pairs(
    sample_submission: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    """Require exact test `(dataset, ID)` pairs, including template row order."""
    expected = sample_submission.loc[
        sample_submission['dataset'].astype(str).str.startswith('test_dataset'),
        ['dataset', 'ID'],
    ].reset_index(drop=True)
    actual = submission.loc[
        submission['dataset'].astype(str).str.startswith('test_dataset'),
        ['dataset', 'ID'],
    ].reset_index(drop=True)

    if actual.duplicated().any():
        raise ParticipantVisibleError(
            'Submission contains duplicate test (dataset, ID) pairs.'
        )
    if not actual.equals(expected):
        raise ParticipantVisibleError(
            'Test (dataset, ID) pairs or their row order do not match the template.'
        )


def validate_submission(
    sample_submission: pd.DataFrame,
    submission: pd.DataFrame,
) -> None:
    """Validate a submission using all expectations derived from its template."""
    assert_column_names(submission)
    assert_submission_shape(submission, sample_submission)
    assert_matching_dataset_counts(submission, sample_submission)
    assert_column_types(submission)
    assert_matching_test_pairs(sample_submission, submission)
    print('Submission validation passed successfully.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Validate an AIRR-ML Phase 1 or Phase 2 submission.'
    )
    parser.add_argument(
        '--phase',
        choices=('1', '2'),
        default='2',
        help='Challenge phase used to choose relative default paths.',
    )
    parser.add_argument(
        '--sample-submission',
        type=Path,
        help='Template CSV; defaults to sample_csv/sample_submission_phase<phase>.csv.',
    )
    parser.add_argument(
        '--submission',
        type=Path,
        help='Submission CSV; defaults to AIRR_ML_25_Phase<phase>_data/results/ensemble/submissions.csv.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_path = args.sample_submission or (
        REPO_ROOT / 'sample_csv' / f'sample_submission_phase{args.phase}.csv'
    )
    submission_path = args.submission or (
        REPO_ROOT
        / f'AIRR_ML_25_Phase{args.phase}_data'
        / 'results'
        / 'ensemble'
        / 'submissions.csv'
    )
    if not sample_path.is_file():
        raise FileNotFoundError(f'Sample submission not found: {sample_path}')
    if not submission_path.is_file():
        raise FileNotFoundError(f'Submission not found: {submission_path}')

    sample_submission = pd.read_csv(sample_path, dtype=STRING_DTYPES)
    submission = pd.read_csv(submission_path, dtype=STRING_DTYPES)
    validate_submission(sample_submission, submission)


if __name__ == '__main__':
    main()

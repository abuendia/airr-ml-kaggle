import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import os
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from utils import (
    load_and_encode_kmers,
    load_and_encode_v_and_j_genes,
    load_full_dataset,
)


class KmerClassifier:
    """L1-regularized logistic regression for k-mer count data."""

    def __init__(self, c_values=None, cv_folds=5,
                 opt_metric='balanced_accuracy', random_state=123, n_jobs=1, ids=None):
        if c_values is None:
            c_values = np.logspace(-3, 1, num=9)
        self.c_values = c_values
        self.cv_folds = cv_folds
        self.opt_metric = opt_metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_C_ = None
        self.best_score_ = None
        self.cv_results_ = None
        self.model_ = None
        self.feature_names_ = None
        self.val_score_ = None
        self.ids_ = ids
        self.train_ids_ = None
        self.val_ids_ = None

    def _make_pipeline(self, C):
        """Create standardization + L1 logistic regression pipeline."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                penalty='l1', C=C, solver='liblinear',
                random_state=self.random_state, max_iter=10000
            ))
        ])

    def _get_scorer(self):
        """Get scoring function for optimization."""
        if self.opt_metric == 'balanced_accuracy':
            return 'balanced_accuracy'
        elif self.opt_metric == 'roc_auc':
            return 'roc_auc'
        else:
            raise ValueError(f"Unknown metric: {self.opt_metric}")

    def tune_and_fit(self, X, y, val_size=0.2):
        """Perform CV tuning on train split and fit, with optional validation split."""

        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if val_size > 0:
            X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
                X, y, self.ids_, test_size=val_size, random_state=self.random_state, stratify=y)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        self.train_ids_ = ids_train
        self.val_ids_ = ids_val
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                             random_state=self.random_state)
        scorer = self._get_scorer()

        results = []
        for C in self.c_values:
            pipeline = self._make_pipeline(C)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scorer,
                                     n_jobs=self.n_jobs)
            results.append({
                'C': C,
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores
            })

        self.cv_results_ = pd.DataFrame(results)
        best_idx = self.cv_results_['mean_score'].idxmax()
        self.best_C_ = self.cv_results_.loc[best_idx, 'C']
        self.best_score_ = self.cv_results_.loc[best_idx, 'mean_score']

        print(f"Best C: {self.best_C_} (CV {self.opt_metric}: {self.best_score_:.4f})")

        # Fit on training split with best hyperparameter
        self.model_ = self._make_pipeline(self.best_C_)
        self.model_.fit(X_train, y_train)

        if X_val is not None:
            if scorer == 'balanced_accuracy':
                self.val_score_ = balanced_accuracy_score(y_val, self.model_.predict(X_val))
            else:  # roc_auc
                self.val_score_ = roc_auc_score(y_val, self.model_.predict_proba(X_val)[:, 1])
            print(f"Validation {self.opt_metric}: {self.val_score_:.4f}")

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X):
        """Predict class labels."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model_.predict(X)

    def get_feature_importance(self):
        """
        Get feature importance from L1 coefficients.

        Returns:
            pd.DataFrame with columns ['feature', 'coefficient', 'abs_coefficient']
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")

        coef = self.model_.named_steps['classifier'].coef_[0]

        # For interpretability and consistency with sequence scoring, convert coefficients
        # back to the original (unstandardized) feature scale when a scaler is present.
        scaler = self.model_.named_steps.get('scaler')
        if scaler is not None and hasattr(scaler, 'scale_'):
            coef = coef / scaler.scale_

        if self.feature_names_ is not None:
            feature_names = self.feature_names_
        else:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        })

        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)

        return importance_df

    def score_all_sequences(
        self,
        sequences_df,
        sequence_col='junction_aa',
        *,
        include_gapped: bool = True,
        n_gaps: int = 1,
        gap_char: str = '_',
    ):
        """
        Score all sequences using model coefficients.

        Parameters:
            sequences_df: DataFrame with unique sequences
            sequence_col: Column name containing sequences

        Returns:
            DataFrame with added 'importance_score' column
        """
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        if self.feature_names_ is None or len(self.feature_names_) == 0:
            raise ValueError("feature_names_ is not set; fit the model with a DataFrame to capture feature names")

        # Weight map comes from get_feature_importance().
        importance_df = self.get_feature_importance()
        feature_to_weight = dict(zip(importance_df['feature'].astype(str), importance_df['coefficient'].astype(float)))

        k = len(self.feature_names_[0])
        use_gapped = bool(include_gapped) and isinstance(n_gaps, int) and (0 < n_gaps < k)

        scores = []
        total_seqs = len(sequences_df)
        for seq in tqdm(sequences_df[sequence_col], total=total_seqs, desc="Scoring sequences"):
            seq = str(seq)
            if len(seq) < k:
                scores.append(0.0)
                continue

            present = set()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                present.add(kmer)
                if use_gapped:
                    for positions in itertools.combinations(range(k), n_gaps):
                        chars = list(kmer)
                        for pos in positions:
                            chars[pos] = gap_char
                        present.add(''.join(chars))

            score = 0.0
            for feat in present:
                w = feature_to_weight.get(feat)
                if w is not None:
                    score += w
            scores.append(float(score))

        result_df = sequences_df.copy()
        result_df['importance_score'] = scores
        return result_df


class ImmuneStatePredictor:
    """
    A template for predicting immune states from TCR repertoire data.

    Participants should implement the logic for training, prediction, and
    sequence identification within this class.
    """

    def __init__(
            self,
            n_jobs: int = 1,
            model_type: str = 'kmer',
            classifier_type: str = 'logistic',
            device: str = 'cpu', **kwargs
        ):
        """
        Initializes the predictor.
        """
        self.train_ids_ = None
        total_cores = os.cpu_count()
        if n_jobs == -1:
            self.n_jobs = total_cores
        else:
            self.n_jobs = min(n_jobs, total_cores)
        
        self.model_type = model_type
        self.classifier_type = classifier_type
        self.device = device

        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: 'cuda' was requested but is not available. Falling back to 'cpu'.")
            self.device = 'cpu'
        else:
            self.device = device
        self.model = None
        self.important_sequences_ = None

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """
        if self.model_type == 'kmer':
            X_train_df, y_train_df = load_and_encode_kmers(train_dir_path) # Example of loading and encoding kmers
        elif self.model_type == 'vj':
            X_train_df, y_train_df = load_and_encode_v_and_j_genes(train_dir_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        X_train, y_train, train_ids = prepare_data(X_train_df, y_train_df,
                                                   id_col='ID', label_col='label_positive')

        # Select classifier based on type
        if self.classifier_type == 'logistic':
            self.model = KmerClassifier(
                c_values=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
                cv_folds=5,
                opt_metric='roc_auc',
                random_state=123,
                n_jobs=self.n_jobs,
                ids=train_ids,
            )
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")

        self.model.tune_and_fit(X_train, y_train)
        self.train_ids_ = train_ids
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predicts probabilities for examples in the provided path.

        Args:
            test_dir_path (str): Path to the directory with test TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        print(f"Making predictions for data in {test_dir_path}...")
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        if self.model_type == 'kmer':
            X_test_df, _ = load_and_encode_kmers(test_dir_path)
        elif self.model_type == 'vj':
            X_test_df, _ = load_and_encode_v_and_j_genes(test_dir_path)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        if self.model.feature_names_ is not None:
            X_test_df = X_test_df.reindex(columns=self.model.feature_names_, fill_value=0)

        repertoire_ids = X_test_df.index.tolist()
        probabilities = self.model.predict_proba(X_test_df)
        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        # to enable compatibility with the expected output format that includes junction_aa, v_call, j_call columns
        predictions_df['junction_aa'] = -999.0
        predictions_df['v_call'] = -999.0
        predictions_df['j_call'] = -999.0

        predictions_df = predictions_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        print(f"Prediction complete on {len(repertoire_ids)} examples in {test_dir_path}.")
        return predictions_df

    def identify_associated_sequences(self, train_dir_path: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identifies the top "k" important sequences (rows) from the training data that best explain the labels.

        Args:
            top_k (int): The number of top sequences to return (based on some scoring mechanism).
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call' columns.
        """
        dataset_name = os.path.basename(train_dir_path)
        full_df = load_full_dataset(train_dir_path)
        unique_seqs = full_df[['junction_aa', 'v_call', 'j_call']].dropna(subset=['junction_aa']).drop_duplicates()
        all_sequences_scored = self.model.score_all_sequences(unique_seqs, sequence_col='junction_aa')
        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        for col in ['junction_aa', 'v_call', 'j_call']:
            if col not in top_sequences_df.columns:
                top_sequences_df[col] = -999.0
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df['label_positive_probability'] = -999.0 # to enable compatibility with the expected output format
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']]

        return top_sequences_df


def prepare_data(X_df, labels_df, id_col='ID', label_col='label_positive'):
    """
    Merge feature matrix with labels, ensuring alignment.

    Parameters:
        X_df: DataFrame with samples as rows (index contains IDs)
        labels_df: DataFrame with ID column and label column
        id_col: Name of ID column in labels_df
        label_col: Name of label column in labels_df

    Returns:
        X: Feature matrix aligned with labels
        y: Binary labels
        common_ids: IDs that were kept
    """
    if id_col in labels_df.columns:
        labels_indexed = labels_df.set_index(id_col)[label_col]
    else:
        # Assume labels_df index is already the ID
        labels_indexed = labels_df[label_col]

    common_ids = X_df.index.intersection(labels_indexed.index)

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between feature matrix and labels")

    X = X_df.loc[common_ids]
    y = labels_indexed.loc[common_ids]

    print(f"Aligned {len(common_ids)} samples with labels")

    return X, y, common_ids

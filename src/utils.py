import os
import glob
from typing import Iterator, Optional, Tuple, Union, Iterable, List
from pathlib import Path
from functools import lru_cache
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict, Counter
import sys
import itertools

from sklearn.feature_extraction.text import TfidfVectorizer


@lru_cache(maxsize=1)
def get_repo_root() -> Path:
    """Return the repository root directory.

    Uses `requirements.txt` as the primary marker. Falls back to `<this_file>/..`.
    """
    start = Path(__file__).resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "requirements.txt").exists():
            return parent
    # Fallback: this file lives under <repo>/src/utils.py
    return start.parent.parent


@lru_cache(maxsize=1)
def get_challenge_data_root(dataset: str) -> Path:
    """Return the sibling `challenge_data/` directory (repo_root/challenge_data)."""
    return get_repo_root() / dataset


@lru_cache(maxsize=1)
def get_results_root(dataset: str) -> Path:
    """Return `<dataset>/results`, resolving relative datasets from the repo root."""
    return get_repo_root() / dataset / "results"


def get_train_to_test_dataset_mapping(dataset: str) -> dict:
    """
    Scans the train_datasets and test_datasets directories and matches
    train_dataset_x to any test_dataset_x or test_dataset_x_* entries,
    where x must be an exact match of the suffix after 'train_dataset_'.
    """
    challenge_root = get_challenge_data_root(dataset)
    train_datasets_dir = challenge_root / "train_datasets"
    test_datasets_dir = challenge_root / "test_datasets"
    if not train_datasets_dir.is_dir():
        raise FileNotFoundError(f"Train datasets directory does not exist: {train_datasets_dir}")
    if not test_datasets_dir.is_dir():
        raise FileNotFoundError(f"Test datasets directory does not exist: {test_datasets_dir}")

    test_groups: dict = defaultdict(list)
    for test_name in sorted(os.listdir(str(test_datasets_dir))):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(test_name)

    mapping = {}
    for train_name in sorted(os.listdir(str(train_datasets_dir))):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            matching = test_groups.get(train_id, [])
            if matching:
                mapping[train_name] = matching

    return mapping


def _count_contiguous_kmers_in_sequence(seq: str, k: int) -> Counter:
    """Count contiguous k-mers in a single amino-acid sequence."""
    counts = Counter()
    if not isinstance(seq, str):
        return counts
    if k <= 0 or len(seq) < k:
        return counts
    for i in range(len(seq) - k + 1):
        counts[seq[i:i + k]] += 1
    return counts


def _count_gapped_kmers_from_kmer(kmer: str, gap_char: str = '_', n_gaps: int = 1) -> Counter:
    """Generate gapped k-mers by replacing positions in a contiguous k-mer with a gap character.

    This follows the common "gapped k-mer" / wildcard-position definition: for a contiguous
    k-mer of length k, choose n_gaps positions and replace them with `gap_char`.
    The resulting feature string has the same length k.
    """
    counts = Counter()
    if not isinstance(kmer, str):
        return counts
    k = len(kmer)
    if k == 0:
        return counts
    if n_gaps <= 0:
        return counts
    if n_gaps >= k:
        return counts
    # Avoid accidentally colliding with standard amino acids.
    if len(gap_char) != 1:
        raise ValueError("gap_char must be a single character")

    for positions in itertools.combinations(range(k), n_gaps):
        chars = list(kmer)
        for pos in positions:
            chars[pos] = gap_char
        counts[''.join(chars)] += 1
    return counts


def _count_kmers_and_gapped_kmers_in_sequence(
    seq: str,
    k: int,
    *,
    include_gapped: bool = True,
    gap_char: str = '_',
    n_gaps: Union[int, Iterable[int]] = 1,
) -> Counter:
    """Count contiguous k-mers and (optionally) their gapped variants in a sequence."""
    counts = Counter()
    contiguous = _count_contiguous_kmers_in_sequence(seq, k)
    counts.update(contiguous)

    if not include_gapped:
        return counts

    if isinstance(n_gaps, int):
        gap_sizes = [n_gaps]
    else:
        gap_sizes = list(n_gaps)

    # De-duplicate while preserving a stable order.
    seen = set()
    gap_sizes = [g for g in gap_sizes if not (g in seen or seen.add(g))]

    for kmer, kmer_count in contiguous.items():
        for g in gap_sizes:
            # Skip invalid gap sizes for this k.
            if not isinstance(g, int):
                continue
            if g <= 0 or g >= k:
                continue
            gapped = _count_gapped_kmers_from_kmer(kmer, gap_char=gap_char, n_gaps=g)
            for gapped_kmer, gapped_count in gapped.items():
                counts[gapped_kmer] += gapped_count * kmer_count
    return counts


def _glob_tsv_files(data_dir: str) -> list:
    """Return a sorted list of all .tsv.gz and .tsv files in data_dir."""
    gz_files = glob.glob(os.path.join(data_dir, '*.tsv.gz'))
    tsv_files = glob.glob(os.path.join(data_dir, '*.tsv'))
    return sorted(set(gz_files) | set(tsv_files))


def _strip_tsv_extension(filename: str) -> str:
    """Strip .tsv.gz or .tsv extension from a basename."""
    if filename.endswith('.tsv.gz'):
        return filename[:-len('.tsv.gz')]
    if filename.endswith('.tsv'):
        return filename[:-len('.tsv')]
    return filename


def load_and_encode_kmers_tfidf(
    data_dir: str,
    ngram_range=(3, 6),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TF-IDF encoding of amino-acid k-mers (character n-grams).
    Each repertoire is treated as one document.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    docs = []
    metadata_records = []

    for item in tqdm(data_loader, total=len(os.listdir(data_dir)), desc="i/o for TF-IDF encoding"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = _strip_tsv_extension(os.path.basename(filename))
            label = None

        # concatenate all sequences into one "document"
        doc = " ".join(data_df['junction_aa'].dropna().astype(str).tolist())
        docs.append(doc)

        meta = {'ID': rep_id}
        if label is not None:
            meta['label_positive'] = label
        metadata_records.append(meta)

    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=ngram_range,
        lowercase=False,
        norm='l2',
        min_df=5,
        max_df=0.95,
    )

    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    features_df = pd.DataFrame(
        X.toarray(),
        index=[m['ID'] for m in metadata_records],
        columns=feature_names
    )

    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df


def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        for file_path in _glob_tsv_files(data_dir):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       a 'filename' column as an identifier.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A single, concatenated DataFrame containing all the data.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        total_files = len(_glob_tsv_files(data_dir))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = _strip_tsv_extension(os.path.basename(filename))
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df


def load_and_encode_kmers(
    data_dir: str,
    k: int = 4,
    include_gapped_kmers: bool = True,
    gapped_kmer_n_gaps: Union[int, Iterable[int]] = (1),
    gapped_kmer_gap_char: str = '_',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading and k-mer encoding of repertoire data.

    Args:
        data_dir: Path to data directory
        k: K-mer length
        include_gapped_kmers: Whether to add gapped k-mer (wildcard-position) features.
        gapped_kmer_n_gaps: Number(s) of wildcard positions per gapped k-mer (e.g. 1 or (1,2,3)).
        gapped_kmer_gap_char: Placeholder character to use for wildcard positions.

    Returns:
        Tuple of (encoded_features_df, metadata_df)
        metadata_df always contains 'ID', and 'label_positive' if available
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    repertoire_features = []
    metadata_records = []

    total_files = len(_glob_tsv_files(data_dir))

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = _strip_tsv_extension(os.path.basename(filename))
            label = None

        kmer_counts = Counter()
        for seq in data_df['junction_aa'].dropna():
            kmer_counts.update(
                _count_kmers_and_gapped_kmers_in_sequence(
                    str(seq),
                    k,
                    include_gapped=include_gapped_kmers,
                    gap_char=gapped_kmer_gap_char,
                    n_gaps=gapped_kmer_n_gaps,
                )
            )

        repertoire_features.append({
            'ID': rep_id,
            **kmer_counts,
        })

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, kmer_counts
    
    features_df = pd.DataFrame(repertoire_features).fillna(0).set_index('ID')
    features_df.fillna(0)
    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df


def load_and_encode_v_and_j_genes(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loading and k-mer encoding of repertoire data with v and j genes.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    vj_features = []
    metadata_records = []

    total_files = len(_glob_tsv_files(data_dir))

    for item in tqdm(data_loader, total=total_files, desc="Encoding v and j genes"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = _strip_tsv_extension(os.path.basename(filename))
            label = None

        v_gene_counts = build_v_gene_dict(data_df)
        j_gene_counts = build_j_gene_dict(data_df)

        vj_features.append({
            'ID': rep_id,
            **v_gene_counts,
            **j_gene_counts,
        })

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df, v_gene_counts, j_gene_counts

    features_df = pd.DataFrame(vj_features).fillna(0).set_index('ID')
    features_df.fillna(0)
    metadata_df = pd.DataFrame(metadata_records)

    return features_df, metadata_df
    

def build_v_gene_dict(data_df: pd.DataFrame):
    v_gene_counts = Counter()
    for v_gene in data_df['v_call'].dropna():
        v_gene_counts[v_gene] += 1
    return v_gene_counts


def build_j_gene_dict(data_df: pd.DataFrame):
    j_gene_counts = Counter()
    for j_gene in data_df['j_call'].dropna():
        j_gene_counts[j_gene] += 1
    return j_gene_counts


def save_tsv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep='\t', index=False)


def get_repertoire_ids(data_dir: str) -> list:
    """
    Retrieves repertoire IDs from the metadata file or filenames in the directory.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        list: A list of repertoire IDs.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        repertoire_ids = metadata_df['repertoire_id'].tolist()
    else:
        tsv_files = _glob_tsv_files(data_dir)
        repertoire_ids = [_strip_tsv_extension(os.path.basename(f)) for f in tsv_files]

    return repertoire_ids


def generate_random_top_sequences_df(n_seq: int = 50000) -> pd.DataFrame:
    """
    Generates a random DataFrame simulating top important sequences.

    Args:
        n_seq (int): Number of sequences to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns 'ID', 'dataset', 'junction_aa', 'v_call', 'j_call'.
    """
    seqs = set()
    while len(seqs) < n_seq:
        seq = ''.join(np.random.choice(list('ACDEFGHIKLMNPQRSTVWY'), size=15))
        seqs.add(seq)
    data = {
        'junction_aa': list(seqs),
        'v_call': ['TRBV20-1'] * n_seq,
        'j_call': ['TRBJ2-7'] * n_seq,
        'importance_score': np.random.rand(n_seq)
    }
    return pd.DataFrame(data)


def validate_dirs_and_files(train_dir: str, test_dirs: List[str], out_dir: str) -> None:
    assert os.path.isdir(train_dir), f"Train directory `{train_dir}` does not exist."
    train_tsv_gzs = glob.glob(os.path.join(train_dir, "*.tsv.gz"))
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs or train_tsv_gzs, f"No .tsv or .tsv.gz files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsv_gzs = glob.glob(os.path.join(test_dir, "*.tsv.gz"))
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs or test_tsv_gzs, f"No .tsv or .tsv.gz files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, f"test_write_permission.{os.getpid()}.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(
    out_dir: str,
    predictions_dir: Optional[str] = None,
    sequences_dir: Optional[str] = None,
) -> None:
    """
    Concatenates test predictions and important sequences TSV files into submissions.csv.

    Args:
        out_dir: Directory where submissions.csv is written.
        predictions_dir: Directory containing *_test_predictions.tsv files. Defaults to out_dir.
        sequences_dir: Directory containing *_important_sequences.tsv files. Defaults to out_dir.
    """
    if predictions_dir is None:
        predictions_dir = out_dir
    if sequences_dir is None:
        sequences_dir = out_dir

    predictions_pattern = os.path.join(predictions_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(sequences_dir, '*_important_sequences.tsv')

    predictions_files = sorted(glob.glob(predictions_pattern))
    sequences_files = sorted(glob.glob(sequences_pattern))

    df_list = []

    for pred_file in predictions_files:
        try:
            df = pd.read_csv(pred_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read predictions file '{pred_file}'. Error: {e}. Skipping.")
            continue

    for seq_file in sequences_files:
        try:
            df = pd.read_csv(seq_file, sep='\t')
            df_list.append(df)
        except Exception as e:
            print(f"Warning: Could not read sequences file '{seq_file}'. Error: {e}. Skipping.")
            continue

    output_columns = [
        'ID', 'dataset', 'label_positive_probability',
        'junction_aa', 'v_call', 'j_call',
    ]
    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(columns=output_columns)
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    concatenated_df = concatenated_df.reindex(columns=output_columns).fillna(-999.0)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")

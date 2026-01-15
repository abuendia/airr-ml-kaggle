import os
import glob
from typing import Iterator, Tuple, Union, Iterable, List
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

    Uses `pyproject.toml` as the primary marker. Falls back to `<this_file>/..`.
    """
    start = Path(__file__).resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: this file lives under <repo>/src/utils.py
    return start.parent.parent


@lru_cache(maxsize=1)
def get_challenge_data_root() -> Path:
    """Return the sibling `challenge_data/` directory (repo_root/../challenge_data)."""
    return get_repo_root().parent / "challenge_data"


@lru_cache(maxsize=1)
def get_results_root() -> Path:
    """Return the results root directory under this repo (repo_root/src/results)."""
    return get_repo_root() / "src" / "results"


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
            rep_id = os.path.basename(filename).replace(".tsv", "")
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
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
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
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
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

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc=f"Encoding {k}-mers"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
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
    Loading and encoding of repertoire data with V gene counts, J gene counts,
    and (V,J) gene-pair counts.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    data_loader = load_data_generator(data_dir=data_dir)

    vj_features = []
    metadata_records = []

    search_pattern = os.path.join(data_dir, '*.tsv')
    total_files = len(glob.glob(search_pattern))

    for item in tqdm(data_loader, total=total_files, desc="Encoding v and j genes"):
        if os.path.exists(metadata_path):
            rep_id, data_df, label = item
        else:
            filename, data_df = item
            rep_id = os.path.basename(filename).replace(".tsv", "")
            label = None

        v_counts = build_v_gene_dict(data_df)
        j_counts = build_j_gene_dict(data_df)
        vj_counts = build_vj_gene_pair_dict(data_df)

        # convert into percentages
        total_counts = sum(vj_counts.values())
        for key in vj_counts:
            vj_counts[key] = vj_counts[key] / total_counts if total_counts > 0 else 0.0

        total_v_counts = sum(v_counts.values())
        for key in v_counts:
            v_counts[key] = v_counts[key] / total_v_counts if total_v_counts > 0 else 0.0
        total_j_counts = sum(j_counts.values())
        for key in j_counts:
            j_counts[key] = j_counts[key] / total_j_counts if total_j_counts > 0 else 0.0

        # get rid of things that are less than 0.10 frequency
        for key in list(v_counts.keys()):
            if v_counts[key] < 0.01:
                del v_counts[key]
        for key in list(j_counts.keys()):
            if j_counts[key] < 0.01:
                del j_counts[key]
        for key in list(vj_counts.keys()):
            if vj_counts[key] < 0.01:
                del vj_counts[key]

        vj_features.append({
            'ID': rep_id,
            **v_counts,
            **j_counts,
        })

        metadata_record = {'ID': rep_id}
        if label is not None:
            metadata_record['label_positive'] = label
        metadata_records.append(metadata_record)

        del data_df

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


def build_vj_gene_pair_dict(data_df: pd.DataFrame, prefix: str = 'VJ__'):
    """Count (v_call, j_call) co-occurrences as additional features.

    Feature names are stored as strings with a prefix to avoid colliding with
    raw V and J gene feature names.
    """
    vj_counts = Counter()
    if 'v_call' not in data_df.columns or 'j_call' not in data_df.columns:
        return vj_counts

    for v_gene, j_gene in zip(data_df['v_call'], data_df['j_call']):
        if pd.isna(v_gene) or pd.isna(j_gene):
            continue
        vj_counts[f"{prefix}{v_gene}__{j_gene}"] += 1
    return vj_counts


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
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        repertoire_ids = [os.path.basename(f).replace('.tsv', '') for f in sorted(tsv_files)]

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
    train_tsvs = glob.glob(os.path.join(train_dir, "*.tsv"))
    assert train_tsvs, f"No .tsv files found in train directory `{train_dir}`."
    metadata_path = os.path.join(train_dir, "metadata.csv")
    assert os.path.isfile(metadata_path), f"`metadata.csv` not found in train directory `{train_dir}`."

    for test_dir in test_dirs:
        assert os.path.isdir(test_dir), f"Test directory `{test_dir}` does not exist."
        test_tsvs = glob.glob(os.path.join(test_dir, "*.tsv"))
        assert test_tsvs, f"No .tsv files found in test directory `{test_dir}`."

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, f"test_write_permission.{os.getpid()}.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"Failed to create or write to output directory `{out_dir}`: {e}")
        sys.exit(1)


def concatenate_output_files(out_dir: str) -> None:
    """
    Concatenates all test predictions and important sequences TSV files from the output directory.

    This function finds all files matching the patterns:
    - *_test_predictions.tsv
    - *_important_sequences.tsv

    and concatenates them to match the expected output format of submissions.csv.

    Args:
        out_dir (str): Path to the output directory containing the TSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame with predictions followed by important sequences.
                     Columns: ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
    """
    predictions_pattern = os.path.join(out_dir, '*_test_predictions.tsv')
    sequences_pattern = os.path.join(out_dir, '*_important_sequences.tsv')

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

    if not df_list:
        print("Warning: No output files were found to concatenate.")
        concatenated_df = pd.DataFrame(
            columns=['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call'])
    else:
        concatenated_df = pd.concat(df_list, ignore_index=True)
    submissions_file = os.path.join(out_dir, 'submissions.csv')
    concatenated_df.to_csv(submissions_file, index=False)
    print(f"Concatenated output written to `{submissions_file}`.")


def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs

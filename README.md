# AIRR-ML-25: Solution for Adaptive Immune Profiling Challenge

10th place solution for [AIRR-ML-25](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025)

## Dependencies

The sample submission CSVs, which define the required output rows and order, are stored with Git LFS. Install Git LFS before cloning or fetching the
repository. Create the conda env:

    conda create -n kaggle python==3.11
    conda activate kaggle
    conda install -c conda-forge git-lfs
    git lfs install

Fetch and materialize the LFS-managed files:

    git pull
    git lfs pull

To fetch only the sample CSVs:

    git lfs pull --include="sample_csv/*.csv"

Install the Python dependencies from the repository root:

    pip install -r requirements.txt

## Running the solution

The runner supports both dataset releases (Phases 1 and 2). Their repertoire
files use different formats: Phase 1 uses `.tsv`, while Phase 2 uses compressed
`.tsv.gz` files. Phase 1 contains 8 mapped training datasets and 11 test directories; Phase 2
contains 95 mapped training datasets and 95 test directories. Test directories
such as `test_dataset_7_1` and `test_dataset_7_2` are automatically associated
with `train_dataset_7`.

Pass the dataset directory as the first argument:

    bash src/run.sh AIRR_ML_25_Phase1_data
    bash src/run.sh AIRR_ML_25_Phase2_data

If no directory is supplied, the runner defaults to
`AIRR_ML_25_Phase2_data`. Results are kept separate under each dataset:

    AIRR_ML_25_Phase1_data/results/ensemble/submissions.csv
    AIRR_ML_25_Phase2_data/results/ensemble/submissions.csv

Final predictions are matched by the composite `(dataset, ID)` key and written
in the row order of `sample_submission_phase1.csv` or
`sample_submission_phase2.csv`, respectively.

For a targeted development run, dataset IDs and the number of reported
important sequences can be restricted:

    bash src/run.sh AIRR_ML_25_Phase1_data --dataset-ids 7 --top-k 25 --parallel_jobs 1 --per_job_n_jobs 1
    bash src/run.sh AIRR_ML_25_Phase2_data --dataset-ids 12 --top-k 25 --parallel_jobs 1 --per_job_n_jobs 1

This restriction reduces the number of dataset groups and output sequences, but
it still processes every repertoire in the selected group.

## Docker container

The project can also be built and run through Docker. Mount the desired dataset
at the corresponding path in `/app`:

    docker build -t airr-ml-kaggle .
    docker run -v $(pwd)/AIRR_ML_25_Phase2_data:/app/AIRR_ML_25_Phase2_data airr-ml-kaggle

To run Phase 1, mount its directory and override the default command:

    docker run \
        -v $(pwd)/AIRR_ML_25_Phase1_data:/app/AIRR_ML_25_Phase1_data \
        airr-ml-kaggle bash src/run.sh AIRR_ML_25_Phase1_data

## Modeling workflow

Code for the solution is in [src](./src). This approach includes the following modeling components:

1. Gapped k-mer logistic regression

    We train a logistic regression model on 4-mer frequencies derived from `junction_aa` sequences. These are 
    created by overlapping sliding windows over the sequence with stride 1. For generalizability, we add gapped k-mers,
    where we replace one token per k-mer with a wildcard "gap" token, increasing the feature set size.

2. Logistic regression on v and j gene counts

    We train a logistic regression model on counts for v-gene and j-gene identities. This is a simple mapping
    of each v- and j-gene identity to its frequency in the training set.

3. Linear alpha blend of the k-mer and V/J models

    For steps 1 and 2, we hold out a random 20% of patients as a validation set. We then use the predictions 
    for these patients to blend the probabilities from the models from steps 1 and 2. The blend is
    `alpha * p_kmer + (1 - alpha) * p_vj`; `alpha` is swept from 0 to 1 in increments of 0.1 and selected
    by validation AUROC. No stacked meta-model is used.

4. Top 50,000 sequences per dataset

    To predict the top 50,000 most influential sequences, each unique sequence is scored by summing the fitted
    k-mer coefficients for its distinct contiguous and one-gap 4-mers. Sequences are ranked by descending score.
    The V/J model and ensemble alpha are not used for sequence ranking.

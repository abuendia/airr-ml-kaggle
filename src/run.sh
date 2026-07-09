set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"

cd "$REPO_ROOT"

# Pass a dataset directory/name as the first argument. Remaining arguments are
# forwarded to execute.py.
DATASET="${1:-AIRR_ML_25_Phase2_data}"
if [[ $# -gt 0 ]]; then
    shift
fi

# Train k-mer + V/J models, select the linear blend alpha, rank important
# sequences with the k-mer model, and write the template-ordered submission.
"$PYTHON" src/execute.py \
    --dataset "$DATASET" \
    --model_type both \
    --model_name 4mer-logreg \
    --vj_model_name vj-logreg \
    --classifier_type logistic \
    --save_important_sequences \
    --parallel_jobs 8 \
    --per_job_n_jobs 4 \
    "$@"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Train k-mer logistic regression with gapping on junction_aa
python src/execute.py \
    --model_type kmer \
    --model_name 4mer-logreg \
    --classifier_type logistic \
    --save_important_sequences \
    --parallel_jobs 8 \
    --per_job_n_jobs 4 \

# Train logistic regression on vj counts
python src/execute.py \
    --model_type vj \
    --model_name vj-logreg \
    --classifier_type logistic \
    --parallel_jobs 8 \
    --per_job_n_jobs 4 \

# Use held-out validation set for meta-learner that combines models
python src/ensemble_probs.py \
    --kmer_model_name 4mer-logreg \
    --vj_model_name vj-logreg

# Write final submission file for kaggle
python src/write_final_output.py \
    --feat_importance_model 4mer-logreg \
    --submission_name 4mer-vj-logreg-ensemble

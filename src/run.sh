# Train k-mer logistic regression with gapping on junction_aa
CUDA_VISIBLE_DEVICES=0 python execute.py \
    --model_type kmer \
    --model_name 4mer-logreg \
    --classifier_type logistic \
    --parallel_jobs 8 \
    --per_job_n_jobs 4 \
    --device gpu

# Train logistic regression on vj counts
CUDA_VISIBLE_DEVICES=0 python execute.py \
    --model_type vj \
    --model_name vj-logreg \
    --classifier_type logistic \
    --parallel_jobs 8 \
    --per_job_n_jobs 4 \
    --device gpu

# Use held-out validation set to train meta-learner that combines models
CUDA_VISIBLE_DEVICES=0 python ensemble_probs.py \
    --kmer_model_name 4mer-logreg \
    --vj_model_name vj-logreg

# Write final submission file for kaggle
python write_final_output.py \
    --feat_importance_model 4mer-logreg \
    --submission_name 4mer-vj-logreg-ensemble

# AIRR-ML-25: Solution for Adaptive Immune Profiling Challenge

## Dependencies

The project's dependencies can be installed from the current directory by running `pip install -e`. 

The project can also be run through docker container as:

## Code

Code for the solution is in [src](./src). To run the end-to-end workflow, run the bash script as 
`bash src/run.sh`. This approach includes the following modeling components:

1. Gapped k-mer logistic regression

    We train a logistic regression model on 4-mer frequencies derived from `junction_aa` sequences. These are 
    created by overlapping sliding windows over the sequence with stride 1. For generalizability, we add gapped k-mers,
    where we replace one token per k-mer with a wildcard "gap" token, increasing the feature set size.

2. Logistic regression on v and j gene counts

    We train a logistic regression model on counts for v-gene and j-gene identities. This is a simple mapping
    of each v and j gene identity to its frequency in the training set.

3. Ensembled meta-learner for k-mer and v, j gene models

    For steps 1 and 2, we hold out a random 20% of patients as a validation set. We then use the predictions 
    for these patients to ensemble the probabilities from the models from 1 and 2. We train a meta-learner
    logistic regression using 4-fold cross validation on the validation set. The meta-learner is used to produce
    the final predictions.

4. Top 50,000 sequences per dataset

    To predict the top 50,000 most influential sequences, we simply take the coefficients from the model in step 1,
    and rank sequences by descending absolute value. We do not incorporate the v, j gene model.

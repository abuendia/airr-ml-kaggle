
import pandas as pd
import argparse


def main(args):
    ensemble_dir = args.input_dir
    train_datasets = ["train_dataset_1", "train_dataset_2", "train_dataset_3", "train_dataset_4", "train_dataset_5", "train_dataset_6", "train_dataset_7", "train_dataset_8"]

    preds = []
    for train_dataset in train_datasets:
        test_dataset_preds = pd.read_csv(f"{ensemble_dir}/{train_dataset}_test_predictions.tsv", sep="\t")
        preds.append(test_dataset_preds)
    combined_preds = pd.concat(preds)

    sample_submission = "/oak/stanford/groups/akundaje/abuen/kaggle/challenge_data/sample_submissions.csv"
    sample_submission_df = pd.read_csv(sample_submission)

    # label_positive_probability predictions
    num_preds = 0
    for index, row in sample_submission_df.iterrows():
        if "seq_top" in row["ID"]:
            continue
        sample_id = row["ID"]
        prob = combined_preds.loc[
            (combined_preds["ID"] == sample_id),
            "label_positive_probability"
        ].values[0]
        sample_submission_df.loc[index, "label_positive_probability"] = prob
        num_preds += 1

    # top 50,000 sequence predictions
    pred_prob_df = sample_submission_df.head(num_preds)
    feat_importance_dir = f"/oak/stanford/groups/akundaje/abuen/kaggle/airr-ml-kaggle/src/results/{args.feat_importance_model}"
    col = []
    for dataset in train_datasets:
        feat_imp_path = f"{feat_importance_dir}/{dataset}_important_sequences.tsv"
        feat_imp_df = pd.read_csv(feat_imp_path, sep="\t")
        col.append(feat_imp_df)
    all_important_seqs_df = pd.concat(col, axis=0)

    # concat
    final_df = pd.concat([pred_prob_df, all_important_seqs_df], axis=0)
    final_df.to_csv("/oak/stanford/groups/akundaje/abuen/kaggle/airr-ml-kaggle/src/results/ensemble/concat_with_importance.csv", sep=",", index=False)
    final_df.to_csv(f"{args.input_dir}/{args.submission_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write kaggle submission file")
    parser.add_argument("--input_dir", type=str, default='/oak/stanford/groups/akundaje/abuen/kaggle/airr-ml-kaggle/src/results/ensemble', help="Directory containing ensemble predictions")
    parser.add_argument("--feat_importance_model", type=str, default='4mer-logreg', help="Model name used for feature importance sequences")
    parser.add_argument("--submission_name", type=str, default='ensembled_preds', help="Name of the submission file")
    args = parser.parse_args()

    main(args)

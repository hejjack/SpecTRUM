import sys
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from pathlib import Path
import typer
from typing import List

app = typer.Typer(pretty_exceptions_enable=False)

def load_predictions(model_dir):
    """Loads the log file and prediction data."""
    model_name = Path(model_dir).parent.parent.name

    # Load predictions from df_best_predictions.jsonl
    pred_file = Path(model_dir) / 'df_best_predictions.jsonl'
    best_predictions = pd.read_json(pred_file, lines=True)
    return model_name, best_predictions

def validate_ground_truth(dfs):
    """Ensure that all models have the same ground truth (gt_smiles) values."""
    gt_smiles = dfs[0]['gt_smiles']
    for i, df in enumerate(dfs[1:], start=1):
        if not gt_smiles.equals(df['gt_smiles']):
            raise ValueError(f"Ground truth mismatch between models at index 0 and {i}.")
    print("Ground truth columns match across all models.")

def create_similarity_dataframe(dfs, models, column):
    """Creates a DataFrame of similarities for the given column."""
    similarity_df = pd.DataFrame()
    for model, df in zip(models, dfs):
        similarity_df[model] = df[column]
    return similarity_df

def perform_wilcoxon_test(similarity_df):
    """Performs the Wilcoxon signed-rank test on the similarity DataFrame."""
    cols = similarity_df.columns
    stat, p_value = wilcoxon(similarity_df[cols[0]], similarity_df[cols[1]])
    return stat, p_value

def perform_friedman_test(similarity_df):
    """Performs the Friedman test on the similarity DataFrame."""
    stat, p_value = friedmanchisquare(*[similarity_df[col] for col in similarity_df.columns])
    return stat, p_value

def perform_nemenyi_test(similarity_df):
    """Performs the Nemenyi post-hoc test."""
    posthoc = sp.posthoc_nemenyi_friedman(similarity_df)
    return posthoc

def significance_stars(p_value):
    """
    Given a p-value, return the significance level in stars:
    - *** for p < 0.001
    - ** for p < 0.01
    - * for p < 0.05
    - . for p < 0.1
    - no stars for p >= 0.1
    """
    if p_value < 0.001:
        return "*** highly significant"
    elif p_value < 0.01:
        return "** very significant"
    elif p_value < 0.05:
        return "* significant"
    elif p_value < 0.1:
        return ". close but not significant"
    else:
        return "not significant"

@app.command()
def main(additional_info: str = typer.Option(..., help="Additional information to be added to the output file name."),
         models_prediction_paths: List[str] = typer.Argument(..., help="Paths to the directories containing model predictions.")):
    models = []
    dfs = []
    fp_type = "morgan"
    fp_simil = "tanimoto"

    # Load predictions from all models
    for model_dir in models_prediction_paths:
        try:
            model_name, df = load_predictions(model_dir)
            models.append(model_name)
            dfs.append(df)
        except:
            print(f"Problem with loading {model_dir}")
            exit(1)

    # Validate that ground truth columns are the same across all models
    validate_ground_truth(dfs)

    # Create two DataFrames: one for 'prob_best_simil_{fp_type}_{fp_simil}', one for 'simil_best_simil_{fp_type}_{fp_simil}'
    probsort_df = create_similarity_dataframe(dfs, models, f'prob_best_simil_{fp_type}_{fp_simil}')
    similsort_df = create_similarity_dataframe(dfs, models, f'simil_best_simil_{fp_type}_{fp_simil}')

    if len(models) < 2:
        raise ValueError("At least two models are required for comparison.")
    elif len(models) == 2:
        test_name = "Wilcoxon signed-rank test"
        print("Only two models found. Performing Wilcoxon signed-rank test...")
        stat_prob, p_value_prob = perform_wilcoxon_test(probsort_df)
        stat_simil, p_value_simil = perform_wilcoxon_test(similsort_df)
    elif len(models) > 2:
        test_name = "Friedman test"
        print("More than two models found. Performing Friedman test...")
        stat_prob, p_value_prob = perform_friedman_test(probsort_df)
        stat_simil, p_value_simil = perform_friedman_test(similsort_df)

    print(f"{test_name} (probsort) statistic: {stat_prob}, p-value: {p_value_prob}")
    print(f"{test_name} (similsort) statistic: {stat_simil}, p-value: {p_value_simil}")

    # If significant difference, perform Nemenyi post-hoc test
    if len(models) > 2:
        if p_value_prob < 0.05:
            print("Performing Nemenyi post-hoc test for probsort similarities...")
            nemenyi_prob = perform_nemenyi_test(probsort_df)
            print(nemenyi_prob)
        else:
            print("No significant difference in probsort similarities.")

        if p_value_simil < 0.05:
            print("Performing Nemenyi post-hoc test for similsort similarities...")
            nemenyi_simil = perform_nemenyi_test(similsort_df)
            print(nemenyi_simil)
        else:
            print("No significant difference in similsort similarities.")

    # Save results to a file
    dataset_name = Path(models_prediction_paths[0]).parent.name
    outfile_name = f"model_comparison_on_{dataset_name}{'_' + additional_info if additional_info else ''}.txt"
    with open(Path(models_prediction_paths[0]).parent.parent.parent / outfile_name, 'w') as f:
        f.write("Statistical significance of model comparison\n\n")
        models_table = pd.DataFrame({"Model": similsort_df.columns,
                                     "probsort_mean_simil": probsort_df.mean().values,
                                     "similsort_mean_simil": similsort_df.mean().values})
        f.write(models_table.to_markdown() + "\n\n")
        f.write(f"{test_name} (probsort): statistic={stat_prob}, p-value={p_value_prob}   {significance_stars(p_value_prob)}\n")
        f.write(f"{test_name} (similsort): statistic={stat_simil}, p-value={p_value_simil}   {significance_stars(p_value_simil)}\n")

        if len(models) > 2:
            if p_value_prob < 0.05:
                f.write("\n\nSIGNIFICANT -> Nemenyi post-hoc test (probsort):\n")
                f.write(nemenyi_prob.to_markdown() + "\n")
            if p_value_simil < 0.05:
                f.write("\n\nSIGNIFICANT -> Nemenyi post-hoc test (similsort):\n")
                f.write(nemenyi_simil.to_markdown() + "\n")


if __name__ == "__main__":
    app()

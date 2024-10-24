import sys
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp
from pathlib import Path
import typer
from typing import List, Union, Dict
import yaml
import copy
import numpy as np
import ast

app = typer.Typer(pretty_exceptions_enable=False)


def discard_other_than_latest_eval(log_dict):
    """Discard all but the latest evaluation from the log file."""
    eval_keys = [key for key in log_dict.keys() if key.startswith("evaluation_")]
    eval_keys.sort(key=lambda x: int(x.split("_")[1]), reverse=True)
    log_dict["evaluation"] = log_dict[eval_keys[0]]

    for key in eval_keys[1:]:
        del log_dict[key]


def load_predictions(model_dir):
    """Loads log files and prediction data."""
    model_name = Path(model_dir).parent.parent.name

    # Load predictions from df_best_predictions.jsonl
    pred_file_path = Path(model_dir) / 'df_best_predictions.jsonl'
    log_file_path = Path(model_dir) / 'log_file.yaml'

    best_predictions = pd.read_json(pred_file_path, lines=True)
    with open(log_file_path, "r", encoding="utf-8") as f:
        log_dict = yaml.safe_load(f)
    discard_other_than_latest_eval(log_dict)

    return model_name, best_predictions, log_dict


def load_all_predictions(model_dirs):
    """Loads logfiles and prediction data of all given models."""
    models = []
    dfs = []
    log_dicts = []

    if not model_dirs:
        return models, dfs, log_dicts

    for model_dir in model_dirs:
        try:
            model_name, df, log_dict = load_predictions(model_dir)
            models.append(model_name)
            dfs.append(df)
            log_dicts.append(log_dict)
        except:
            raise ValueError(f"Problem with loading {model_dir}")

    return models, dfs, log_dicts


def validate_ground_truth(dfs):
    """Ensure that all models have the same ground truth (gt_smiles) values."""
    gt_smiles = dfs[0]['gt_smiles']
    for i, df in enumerate(dfs[1:], start=1):
        if not gt_smiles.equals(df['gt_smiles']):
            raise ValueError(f"Ground truth mismatch between models at index 0 and {i}.")
    print("Ground truth columns match across all models.")


def create_all_similarities_dataframe(dfs, models, column):
    """Creates a DataFrame of all predictions' similarities for the given column."""
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

def check_significance_of_inequality(similarity_df):
    """Check the significance of the inequality between models."""

    num_models = len(similarity_df.columns)
    if num_models < 2:
        raise ValueError("At least two models are required for comparison.")

    elif num_models == 2:
        test_name = "Wilcoxon signed-rank test"
        print("Only two models found. Performing Wilcoxon signed-rank test...")
        stat, p_value = perform_wilcoxon_test(similarity_df)

    elif num_models > 2:
        test_name = "Friedman test"
        print("More than two models found. Performing Friedman test...")
        stat, p_value = perform_friedman_test(similarity_df)

    # If significant difference, perform Nemenyi post-hoc test
    if num_models > 2:
        if p_value < 0.05:
            print("Performing Nemenyi post-hoc test...")
            nemenyi_prob = perform_nemenyi_test(similarity_df)
        else:
            nemenyi_prob = None
            print("No significant difference in probsort similarities.")

    return test_name, stat, p_value, nemenyi_prob


def compute_rate_of_wins(similarities1: pd.Series, similarities2: pd.Series):
    """Compute the rate of wins of the first list over the second list."""
    win_rate = (similarities1 > similarities2).sum() / len(similarities1)
    at_least_as_good_rate = (similarities1 >= similarities2).sum() / len(similarities1)

    return win_rate, at_least_as_good_rate


def compute_mean_difference(similarities1: pd.Series, similarities2: pd.Series):
    """Compute the mean difference between two lists of similarities."""
    return (similarities1 - similarities2).mean()


def compute_rate_of_wins_for_two_groups(similarity_df, models1, models2):
    """Compute the rate of wins of the first group over the second group."""
    wins_df = pd.DataFrame(index=models1, columns=models2)
    at_least_as_good_df = pd.DataFrame(index=models1, columns=models2)

    for model1 in models1:
        for model2 in models2:
            win_rate, at_least_as_good_rate = compute_rate_of_wins(similarity_df[model1], similarity_df[model2])
            wins_df.loc[model1, model2] = win_rate
            at_least_as_good_df.loc[model1, model2] = at_least_as_good_rate

    return wins_df, at_least_as_good_df


def compute_mean_differences_for_two_groups(similarity_df, models1, models2):
    """Compute the mean differences between the first group and the second group."""
    mean_diff_df = pd.DataFrame(index=models1, columns=models2)

    for model1 in models1:
        for model2 in models2:
            mean_diff = compute_mean_difference(similarity_df[model1], similarity_df[model2])
            mean_diff_df.loc[model1, model2] = mean_diff

    return mean_diff_df


def compute_db_search_performance(similarity_df, db_search_names):
    """Creates a table showing how close is the performance of spectral
    db searches (sss, hss) to the structural db searches (morgan_tanimoto).
    The table contains the rates of structural searches finding the
    closest structural candidate."""

    structural_searches = []
    spectral_searches = []
    for name in db_search_names:
        if "morgan_tanimoto" in name:
            structural_searches.append(name)
        else:
            spectral_searches.append(name)

    if not structural_searches or not spectral_searches:
        return None

    _, rate_of_closest_candidates = compute_rate_of_wins_for_two_groups(similarity_df, structural_searches, spectral_searches)

    return rate_of_closest_candidates


def get_info_from_logfile(log_dict: dict, value_path: str):
    """Extract a value from the log file based on the given path."""
    keys = value_path.split("/")
    value = copy.deepcopy(log_dict)
    for key in keys:
        value = value.get(key, False)
        if not value:
            raise ValueError(f"Value {key} at path {value_path} not found in the log file.")

    value = ast.literal_eval(str(value))
    assert isinstance(value, (str, int, float)), f"Value at path {value_path} is not a string, int or float."
    return value


def get_info_from_logfiles_for_all(log_dicts: Dict[str, Dict], value_paths: List[str]) -> pd.DataFrame:
    """Create a comparison table for all models with values from the log files.

    Parameters
    ----------
    log_dicts: dict
        Dictionary of log files. Keys are model names, values are log dictionaries
    value_paths: List[str]
        List of paths to the values in the log files to be extracted.
        The format is 'key1/key2/key3/...' for nested dictionaries.

    Return
    ------
    info_df: pd.DataFrame
        DataFrame with the extracted values (columns) for all models (rows).
    """
    value_names = [path.split("/")[-1] for path in value_paths]
    df = pd.DataFrame(index = list(log_dicts.keys()), columns = value_names)
    for model, log_dict in log_dicts.items():
        for path, name in zip(value_paths, value_names):
            try:
                df.loc[model, name] = get_info_from_logfile(log_dict, path)
            except ValueError as e:
                print(f"Error in model {model}: {e}")
                df.loc[model, name] = None
    return df


def compare_models(models_prediction_paths: List[str],
                   db_search_prediction_paths: Union[List[str], None],
                   fp_type: str = "morgan",
                   fp_simil: str = "tanimoto"):
    """
    Compare models based on their predictions and optionally database search predictions.

    Parameters:
    ----------
    models_prediction_paths: List[str]
        Paths to the directories containing model predictions.
    db_search_prediction_paths: Union[List[str], None]
        Paths to the directories containing database search predictions.
    fp_type: str
        Preferred fingerprint type used for similarity calculations.
    fp_simil: str
        Preferred similarity metric used for similarity calculations.

    Returns:
    -------
    output: dict
    Output dictionary with results and tables for the model comparison:

        output = {
            models: list_of_model_names,
            db_searches: list_of_db_search_names,
            mean_simils: df (sorted by similsort, both db_searches and models, bot probsort and similsort),
            significance_test: {
                test_name: str (Friedman/Wilcoxon),
                stat_probsort: float,
                p_value_probsort: float,
                stat_similsort: float,
                p_value_similsort: float,
                nemenyi_probsort: df or none,
                nemenyi_similsort: df or none
            },
            wins_over_db_search_probsort: df or none,
            wins_over_db_search_similsort: df or none,
            at_least_as_good_as_db_search_probsort: df or none,
            at_least_as_good_as_db_search_similsort: df or none,
            fpsd_score_probsort: df or none,
            fpsd_score_similsort: df or none,
            db_search_performance: df or none (what was the portion of predictions where db_search hit the MT closest molecule)
    """

    model_names, model_dfs, model_log_dicts = load_all_predictions(models_prediction_paths)
    db_search_names, db_search_dfs, db_search_log_dicts = load_all_predictions(db_search_prediction_paths)

    all_predictors, all_dfs = model_names + db_search_names, model_dfs + db_search_dfs
    all_log_dicts = {k: v for k, v in zip(all_predictors, model_log_dicts + db_search_log_dicts)}

    # Validate that ground truth columns are the same across all models
    validate_ground_truth(all_dfs)

    # Create two DataFrames: one for 'prob_best_simil_{fp_type}_{fp_simil}', one for 'simil_best_simil_{fp_type}_{fp_simil}'
    probsort_df = create_all_similarities_dataframe(all_dfs, all_predictors, f'prob_best_simil_{fp_type}_{fp_simil}')
    similsort_df = create_all_similarities_dataframe(all_dfs, all_predictors, f'simil_best_simil_{fp_type}_{fp_simil}')

    # do significance test for probsort and similsort
    test_name, stat_probsort, p_value_probsort, nemenyi_prob_probsort = check_significance_of_inequality(probsort_df)
    _, stat_similsort, p_value_similsort, nemenyi_prob_similsort = check_significance_of_inequality(similsort_df)

    mean_simils_df = pd.DataFrame({"Model": similsort_df.columns,
                                   "probsort_mean_simil": probsort_df.mean().values,
                                   "similsort_mean_simil": similsort_df.mean().values})
    exact_matches_df = get_info_from_logfiles_for_all(all_log_dicts,
                                                      ["evaluation/precise_preds_stats/rate_of_precise_preds_probsort",
                                                       "evaluation/precise_preds_stats/rate_of_precise_preds_similsort"] )

    if len(db_search_names) > 0:
        wins_over_db_search_probsort_df, at_least_as_good_probsort_df = compute_rate_of_wins_for_two_groups(probsort_df, model_names, db_search_names)
        wins_over_db_search_similsort_df, at_least_as_good_similsort_df = compute_rate_of_wins_for_two_groups(similsort_df, model_names, db_search_names)
        fpsd_score_probsort_df = compute_mean_differences_for_two_groups(probsort_df, model_names, db_search_names)
        fpsd_score_similsort_df = compute_mean_differences_for_two_groups(similsort_df, model_names, db_search_names)
        db_search_performance_df = compute_db_search_performance(similsort_df, db_search_names)
    else:
        wins_over_db_search_probsort_df, wins_over_db_search_similsort_df = None, None
        at_least_as_good_probsort_df, at_least_as_good_similsort_df = None, None
        fpsd_score_probsort_df, fpsd_score_similsort_df = None, None

    output = {
        "models": model_names,
        "db_searches": db_search_names,
        "mean_simils": mean_simils_df,
        "exact_matches": exact_matches_df,
        "significance_test": {
            "test_name": test_name,
            "stat_probsort": stat_probsort,
            "p_value_probsort": p_value_probsort,
            "stat_similsort": stat_similsort,
            "p_value_similsort": p_value_similsort,
            "nemenyi_probsort": nemenyi_prob_probsort,
            "nemenyi_similsort": nemenyi_prob_similsort
        },
        "wins_over_db_search_probsort": wins_over_db_search_probsort_df,
        "wins_over_db_search_similsort": wins_over_db_search_similsort_df,
        "at_least_as_good_as_db_search_probsort": at_least_as_good_probsort_df,
        "at_least_as_good_as_db_search_similsort": at_least_as_good_similsort_df,
        "fpsd_score_probsort": fpsd_score_probsort_df,
        "fpsd_score_similsort": fpsd_score_similsort_df,
        "db_search_performance": db_search_performance_df,
        "log_dicts": all_log_dicts
    }

    return output


@app.command()
def main(additional_info: str = typer.Option(..., help="Additional information to be added to the output file name."),
         models_prediction_paths: str = typer.Option(..., help="Paths to the directories containing model predictions separated by any whitespace."),
         db_search_prediction_paths: str = typer.Option(..., help="Paths to the directories containing database search predictions separated by any whitespace."),
         fp_type: str = typer.Option("morgan", help="Fingerprint type used for similarity calculations."),
         fp_simil: str = typer.Option("tanimoto", help="Similarity metric used for similarity calculations.")):

    models_paths_list = models_prediction_paths.split()
    db_search_paths_list = db_search_prediction_paths.split()

    comparison  = compare_models(models_paths_list, db_search_paths_list, fp_type, fp_simil)
    significance_test = comparison["significance_test"]
    # Save results to a file
    dataset_name = Path(models_paths_list[0]).parent.name
    outfile_name = f"model_comparison_on_{dataset_name}{'_' + additional_info if additional_info else ''}.txt"
    with open(Path(models_paths_list[0]).parent.parent.parent / outfile_name, 'w') as f:
        f.write("Statistical significance of model comparison\n\n")
        f.write(comparison["mean_simils"].to_markdown() + "\n\n")
        f.write(comparison["exact_matches"].to_markdown() + "\n\n")
        f.write(f"{significance_test['test_name']} (probsort): statistic={significance_test['stat_probsort']}, p-value={significance_test['p_value_probsort']}   {significance_stars(significance_test['p_value_probsort'])}\n")
        f.write(f"{significance_test['test_name']} (similsort): statistic={significance_test['stat_similsort']}, p-value={significance_test['p_value_probsort']}   {significance_stars(significance_test['p_value_similsort'])}\n")
        if len(comparison['models']) > 2:
            if significance_test['p_value_probsort'] < 0.05:
                f.write("\n\nSIGNIFICANT -> Nemenyi post-hoc test (probsort):\n")
                f.write(significance_test['nemenyi_probsort'].to_markdown() + "\n")
            if significance_test['p_value_similsort'] < 0.05:
                f.write("\n\nSIGNIFICANT -> Nemenyi post-hoc test (similsort):\n")
                f.write(significance_test['nemenyi_similsort'].to_markdown() + "\n")
        if comparison['db_searches'] is not None:
            f.write("\n\nRate of wins over database search predictions (probsort):\n")
            f.write(comparison['wins_over_db_search_probsort'].to_markdown() + "\n")
            f.write("\n\nRate of wins over database search predictions (similsort):\n")
            f.write(comparison['wins_over_db_search_similsort'].to_markdown() + "\n")
            f.write("\n\nRate of at least as good as database search predictions (probsort):\n")
            f.write(comparison['at_least_as_good_as_db_search_probsort'].to_markdown() + "\n")
            f.write("\n\nRate of at least as good as database search predictions (similsort):\n")
            f.write(comparison['at_least_as_good_as_db_search_similsort'].to_markdown() + "\n")
            f.write("\n\nMean difference in similarity to database search predictions (probsort):\n")
            f.write(comparison['fpsd_score_probsort'].to_markdown() + "\n")
            f.write("\n\nMean difference in similarity to database search predictions (similsort):\n")
            f.write(comparison['fpsd_score_similsort'].to_markdown() + "\n")
            if comparison['db_search_performance'] is not None:
                f.write("\n\nPerformance of database searches:\n")
                f.write(comparison['db_search_performance'].to_markdown() + "\n")


if __name__ == "__main__":
    app()

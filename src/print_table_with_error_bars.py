import argparse
import json
import os
from pathlib import Path

import numpy as np

from utils.constants import MODELS_MAPPER, OBJECT_CATEGORIES_NAMES, ATTRIBUTE_CATEGORIES_NAMES


def format_mean_std(value, std=None):
    if not std:
        return f"${(100 * value):.2f}$"
    return f"${(100 * value):.2f}_{{\\pm {(100 * std):.2f}}}$"


def bootstrap_accuracy(is_correct_list, n_bootstrap=1000):
    """Perform bootstrap resampling to estimate accuracy distribution."""
    n = len(is_correct_list)
    bootstrap_acc = []

    for _ in range(n_bootstrap):
        idxs = np.random.randint(0, n, size=n)  # Sample with replacement
        acc = np.mean(is_correct_list[idxs])
        bootstrap_acc.append(acc)

    return np.array(bootstrap_acc)


def _get_prediction_files(result_dir, ft_model_list=None):
    """Get all prediction files from the result directory."""
    prediction_files = []

    if ft_model_list is None:
        for filename in os.listdir(result_dir):
            if filename.endswith("_predictions.json"):
                model_name = filename.replace("_predictions.json", "")
                file_path = Path(result_dir) / filename
                prediction_files.append((file_path, model_name))
    else:
        for filename in os.listdir(result_dir):
            if (
                    filename.endswith("_predictions.json")
                    and any(ft_model in filename for ft_model in ft_model_list)
                    and "wo_contr" not in filename
            ):
                model_name = filename.replace("_predictions.json", "")
                file_path = Path(result_dir) / filename
                prediction_files.append((file_path, model_name))

    return prediction_files


def _generate_latex_row_modalities(model_name, data, print_modalities):
    """Generate a LaTeX table row for the given model and data."""
    latex_row = f"{MODELS_MAPPER.get(model_name, model_name)} "

    # Extract evaluation metrics
    metrics = _extract_evaluation_metrics(data, print_modalities)

    # Add contradiction metric
    latex_row += _format_metric_column(metrics["contradiction"])

    # Add modality-specific metrics if requested
    if print_modalities:
        for metric_name in ["image_only", "text_only", "irrelevant", "none"]:
            latex_row += _format_metric_column(metrics[metric_name])

    return latex_row + "\\\\"

def _generate_latex_row_finetune(model_name, data_contr, data_wo_contr):
    """Generate a LaTeX table row for the given model and data."""
    latex_row = f"{MODELS_MAPPER.get(model_name, model_name)} "

    # Extract evaluation metrics
    metrics_contr = _extract_evaluation_metrics(data_contr, print_modalities=True)
    metrics_wo_contr = _extract_evaluation_metrics(data_wo_contr, print_modalities=True)

    # Add FP metric
    latex_row += _format_metric_column(metrics_contr["contradiction"])
    latex_row += _format_metric_column(metrics_wo_contr["image_only"])
    latex_row += _format_metric_column(metrics_contr["contradiction"] + metrics_wo_contr["image_only"])
    latex_row += _format_metric_column(metrics_contr["none"])

    return latex_row + "\\\\"

def _generate_latex_row_categories(model_name, data, categories):
    """Generate a LaTeX table row for the given model and data."""
    latex_row = f"{MODELS_MAPPER.get(model_name, model_name)} "

    for category in categories:
        filtered_data = [sample for sample in data if sample['category'] == category]
        metrics = _extract_evaluation_metrics(filtered_data, None)
        latex_row += _format_metric_column(metrics["contradiction"])

    return latex_row + "\\\\"

def _extract_evaluation_metrics(data, print_modalities):
    """Extract evaluation metrics from the data entries."""
    metrics = {"contradiction": [], "image_only": [], "text_only": [], "irrelevant": [], "none": []}

    for entry in data:
        eval_data = entry.get("evaluation", {})

        contr = eval_data.get("matches_contradiction", False)
        image_only = eval_data.get("matches_image_only", False)
        text_only = eval_data.get("matches_text_only", False)
        irrelevant = eval_data.get("matches_irrelevant", False)
        matches_none = not (contr or image_only or text_only or irrelevant)
        if None in (contr, image_only, text_only, irrelevant):
            continue

        metrics["contradiction"].append(int(contr))

        if print_modalities:
            metrics["image_only"].append(int(image_only))
            metrics["text_only"].append(int(text_only))
            metrics["irrelevant"].append(int(irrelevant))
            metrics["none"].append(int(matches_none))

    return metrics


def _format_metric_column(metric_values):
    """Format a metric column for LaTeX output."""
    if not metric_values:
        return "& - "

    bootstrap_acc = bootstrap_accuracy(np.array(metric_values))
    mean_val = np.mean(bootstrap_acc)
    std_val = np.std(bootstrap_acc)

    return f"& {format_mean_std(mean_val, std_val)} "


def print_modality_preference_table(args, print_modalities=True):
    with open(args.test_data_path) as f:
        test_data = json.load(f)['samples']
    test_image_ids = [sample['image_id'] for sample in test_data]

    prediction_files = _get_prediction_files(args.result_dir)

    for file_path, model_name in prediction_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            filtered_data = [sample for sample in data if sample['image_id'] in test_image_ids]
            # print(f"Evaluating {len(filtered_data)} samples. ")

            latex_row = _generate_latex_row_modalities(model_name, filtered_data, print_modalities)
            print(latex_row)

def print_finetune_table(args):
    with open(args.test_data_path) as f:
        test_data = json.load(f)['samples']
    test_image_ids = [sample['image_id'] for sample in test_data]

    prediction_files = _get_prediction_files(args.result_dir, args.ft_model_list)

    for file_path, model_name in prediction_files:
        file_path_wo_contr = file_path.with_name(file_path.name.replace("_predictions.json", "_wo_contr_predictions.json"))
        with open(file_path, "r") as f:
            data_contr = json.load(f)
        with open(file_path_wo_contr, "r") as f:
            data_wo_contr = json.load(f)
        filtered_data = [sample for sample in data_contr if sample['image_id'] in test_image_ids]
        filtered_data_wo_contr = [sample for sample in data_wo_contr if sample['image_id'] in test_image_ids]

        latex_row = _generate_latex_row_finetune(model_name, filtered_data, filtered_data_wo_contr)
        print(latex_row)

def merge_predicitons_and_test_data(predictions, test_data):
    predictions_lookup = {sample['image_id']: sample for sample in predictions}

    merged_samples = []
    for sample in test_data:
        image_id = sample['image_id']
        pred_sample = predictions_lookup.get(image_id, {})
        merged_sample = {**sample, **pred_sample}  # merge dictionaries
        merged_samples.append(merged_sample)

    return merged_samples


def print_categories_table(args, category_type):
    with open(args.test_data_path) as f:
        test_data = json.load(f)['samples']
    test_image_ids = {sample['image_id'] for sample in test_data}  # use set for faster lookup

    if category_type == "object":
        category_names = OBJECT_CATEGORIES_NAMES
    elif category_type == "attribute":
        category_names = ATTRIBUTE_CATEGORIES_NAMES
    else:
        raise ValueError(f"Unknown category_type: {category_type}")

    prediction_files = _get_prediction_files(args.result_dir)
    for file_path, model_name in prediction_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            filtered_data = [sample for sample in data if sample['image_id'] in test_image_ids]

            merged_samples = merge_predicitons_and_test_data(filtered_data, test_data)

            relevant_samples = [sample for sample in merged_samples if sample['change_type'] == category_type]

            latex_row = _generate_latex_row_categories(model_name, relevant_samples, category_names)
            print(latex_row)


def print_llm_as_judge_table(args):
    prediction_files = _get_prediction_files(args.result_dir)

    for file_path, model_name in prediction_files:
        with open(file_path, "r") as f:
            data = json.load(f)

            latex_row = _generate_latex_row_llm_as_judge(model_name, data)
            print(latex_row)


def _generate_latex_row_llm_as_judge(model_name, data):
    categories = ["contradiction", "image_only", "text_only", "none"]
    judges = ["gemini-2.5-pro", "gpt-5"]

    latex_rows = ""
    metrics_by_category = {cat: {"Gemini": [], "GPT-5": [], "≥1": [], "Both": []} for cat in categories}

    for entry in data:
        evals = {}
        for judge in judges:
            evals[judge] = entry.get(f"evaluation_{judge}", {})

        for cat in categories:
            a_val = int(evals[judges[0]].get(f"matches_{cat}", False))
            b_val = int(evals[judges[1]].get(f"matches_{cat}", False))

            metrics_by_category[cat]["Gemini"].append(a_val)
            metrics_by_category[cat]["GPT-5"].append(b_val)
            metrics_by_category[cat]["≥1"].append(int(a_val or b_val))
            metrics_by_category[cat]["Both"].append(int(a_val and b_val))

    first_row = True
    for row_label in ["Gemini", "GPT-5", "≥1", "Both"]:
        row = ""
        if first_row:
            row += f"\\multirow{{4}}{{*}}{{{MODELS_MAPPER.get(model_name, model_name)}}} & "
            first_row = False
        else:
            row += "& "

        row += f"{row_label} "
        for cat in categories:
            row += _format_metric_column(metrics_by_category[cat][row_label])
        row += "\\\\\n"
        latex_rows += row

    return latex_rows


def main():
    parser = argparse.ArgumentParser(
        description="Get error bars and print table in latex format."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to test data.",
        default="../data/synthetic_captions_test_filtered.json",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        help="../results/multiple_choice or ../results/open_ended",
        default="../results/multiple_choice/",
    )
    parser.add_argument(
        "--ft_model_list",
        nargs="+",
        type=str,
        help="Filename prefixes of fine-tuned models to include in the finetune table.",
        default=[],
    )
    args = parser.parse_args()
    print("\n" + 40 * "-" + " MODALITY PREFERENCE TABLE " + 40 * "-" + "\n")
    print_modality_preference_table(args, print_modalities=True)
    print("\n" + 40 * "-" + " OBJECT CATEGORIES TABLE " + 40 * "-" + "\n")
    print_categories_table(args, "object")
    print("\n" + 40 * "-" + " ATTRIBUTE CATEGORIES TABLE " + 40 * "-" + "\n")
    print_categories_table(args, "attribute")
    if args.ft_model_list:
        print("\n" + 40 * "-" + " FINETUNE TABLE " + 40 * "-" + "\n")
        print_finetune_table(args)
    # print("\n" + 40 * "-" + " LLM AS JUDGE TABLE " + 40 * "-" + "\n")
    # print_llm_as_judge_table(args)


if __name__ == "__main__":
    main()

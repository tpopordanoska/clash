
import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from dataset.preprocessing import prepare_sample_for_multiple_choice
from models.base_model import model_factory

MAX_RETRIES = 3
RETRY_DELAY = 5

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../data/synthetic_captions_test_filtered.json",
        help="Path to the dataset JSON file",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=["gemini-2.5-pro"],
        help="Model name(s) to run. e.g. gemini-2.5-pro, gemini-2.5-flash-lite-preview-06-17, gpt-4.1-mini, gpt-5.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../results",
        help="Path to save the predictions json.",
    )
    parser.add_argument(
        "--question_format",
        type=str,
        default="open_ended",
        help="Question format to use: multiple_choice or open_ended",
    )
    parser.add_argument(
        "--wo_contr",
        action="store_true",
        help="Without contradictions: prompt with the original (non-conflicting) caption.",
    )
    parser.add_argument(
        "--num_icl_shots",
        type=int,
        default=0,
        help="Number of ICL shots (0 = no ICL). Each shot shows conflict + non-conflict example.",
    )
    return parser.parse_args()


def run_with_retries(model, sample, question_format, wo_contr=None, icl_samples=None, max_retries=MAX_RETRIES):
    for attempt in range(1, max_retries + 1):
        try:
            return run_model_on_sample(model, sample, question_format, wo_contr=wo_contr, icl_samples=icl_samples)
        except Exception as e:
            print(f"[Retry {attempt}/{max_retries}] Error on image_id {sample['image_id']}: {e}")
            if attempt < max_retries:
                time.sleep(RETRY_DELAY * attempt)
    print(f"Skipping sample {sample['image_id']} after {max_retries} failed attempts.")
    return None

def run_model_on_sample(model, sample, question_format, wo_contr=None, icl_samples=None):
    prepare_sample_for_multiple_choice(sample)
    model_input = model.prepare_model_input(
        sample, question_format=question_format, wo_contr=wo_contr, icl_samples=icl_samples or []
    )
    response = model.inference(model_input)
    print("------ ", response)
    prediction = {
        "image_id": sample["image_id"],
        "original_caption": sample["original_caption"],
        "conflicting_caption": sample.get("conflicting_caption"),
        "question": sample.get("question"),
        "prediction": response,
    }
    if question_format == "multiple_choice": 
        prediction["modality_answers"] = sample.get("modality_answers")
    else: 
        prediction["modality_answers"] = {
            "image_only": sample["modality_answers"]["image_only"]["answer"],
            "text_only": sample["modality_answers"]["text_only"]["answer"],
        }
    return prediction


def sample_icl_examples(data, current_sample, num_icl_shots):
    """Randomly sample ICL examples, excluding current sample."""
    if num_icl_shots <= 0:
        return []
    candidates = [s for s in data if s["image_id"] != current_sample["image_id"]]
    return random.sample(candidates, min(num_icl_shots, len(candidates)))


def main():
    args = parse_arguments()

    data = json.load(open(args.dataset_path))["samples"]

    for model_name in args.model_names:
        model = model_factory(model_name)(model_name=model_name)

        output_dir = Path(args.output_path) / args.question_format
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_wo_contr" if args.wo_contr else ""
        output_file = output_dir / f"{model_name}{suffix}_predictions.json"

        existing_preds = {}
        if output_file.exists():
            with open(output_file, "r") as f:
                predictions = json.load(f)
            existing_preds = {p["image_id"]: p for p in predictions}

        to_process = [
            sample for sample in data
            if sample["image_id"] not in existing_preds
            or not existing_preds[sample["image_id"]].get("prediction")  # empty check
        ]

        print(f"{model_name}: {len(to_process)} / {len(data)} samples need processing.")

        new_predictions = []
        try:
            if "gemini" in model_name.lower() or model_name.lower().startswith("gpt"):  # avoid minigpt4
                def process_sample(sample):
                    icl_samples = sample_icl_examples(data, sample, args.num_icl_shots)
                    return run_with_retries(model, sample, args.question_format, wo_contr=args.wo_contr, icl_samples=icl_samples)

                with ThreadPoolExecutor(max_workers=6) as executor:
                    futures = {
                        executor.submit(process_sample, sample): sample
                        for sample in to_process
                    }
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model_name}"):
                        result = future.result()
                        if result is not None:
                            new_predictions.append(result)
            else:
                for sample in tqdm(to_process, desc=f"{model_name}"):
                    try:
                        icl_samples = sample_icl_examples(data, sample, args.num_icl_shots)
                        new_predictions.append(
                            run_model_on_sample(model, sample, args.question_format, args.wo_contr, icl_samples=icl_samples)
                        )
                    except Exception as e:
                        print(f"Error processing sample {sample['image_id']}: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted by user, saving partial results...")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
        finally:
            for p in new_predictions:
                existing_preds[p["image_id"]] = p

            predictions = list(existing_preds.values())
            with open(output_file, "w") as f:
                json.dump(predictions, f, indent=4)
            print(f"Saved {len(predictions)} predictions for {model_name} to {output_file}")


if __name__ == "__main__":
    main()

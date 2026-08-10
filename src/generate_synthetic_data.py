import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from tqdm import tqdm

from dataset.schema import *
from models.prompts import (
    GENERATE_CONFLICTING_CAPTION_AND_ANSWER,
    GENERATE_MCQA_ANSWERS,
    ATTRIBUTE_CHECK,
    OBJECT_CHECK,
    ANSWERS_CHECK,
    QUESTION_CHECK,
)
from utils.constants import (
    BASE_DIR,
    GEMINI_API_KEY,
    OBJECT_CONDITIONS,
    ATTRIBUTE_CONDITIONS,
    ANSWERS_CONDITIONS,
    QUESTION_CONDITIONS,
)

def generate_conflicting_caption_and_answer_prompt(caption):
    return f"""{GENERATE_CONFLICTING_CAPTION_AND_ANSWER}
    The original image caption is: {caption}
    """


def generate_mcqa_answers_prompt(original_caption, conflicting_caption, question):
    return f"""{GENERATE_MCQA_ANSWERS} Input:
    - Original Caption: "{original_caption}"
    - Conflicting Caption: "{conflicting_caption}"
    - Question: "{question}"
"""


def attribute_check_prompt(changed_words):
    return f"""{ATTRIBUTE_CHECK} Input:
    - Original: {changed_words["original"]}
    - Conflicting: {changed_words["conflicting"]}
"""


def object_check_prompt(changed_words):
    return f"""{OBJECT_CHECK} Input:
        - Original: {changed_words["original"]}
        - Conflicting: {changed_words["conflicting"]}
"""


def answers_check_prompt(answers):
    return f"{ANSWERS_CHECK} Input: {answers['image_only']}, {answers['text_only']}, {answers['irrelevant_but_plausible']}"


def question_check_prompt(question, changed_words, answers):
    return f"""{QUESTION_CHECK} Input: 
        - Question: "{question}"
        - Changed words: {changed_words["original"]}, {changed_words["conflicting"]}
        - Answers: {answers["image_only"]}, {answers["text_only"]}, {answers["irrelevant_but_plausible"]}
"""


def with_retry(func, max_retries=3, backoff=2, jitter=0.5, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            wait = backoff * (2**attempt) + random.uniform(0, jitter)
            print(
                f"Error: {e} | Retrying in {wait:.2f}s (attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait)
    print(f"Failed after {max_retries} retries for {func.__name__}")
    return None


def query_llm(model_name, client, caption):
    prompt = generate_conflicting_caption_and_answer_prompt(caption)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        json_string = (
            response.text.strip().replace("```json\n", "").replace("\n```", "")
        )
        return json.loads(json_string)
    except json.JSONDecodeError as jde:
        print(f"JSON Decode Error for caption '{caption}': {jde}")
        print(f"Raw response:\n{response.text}")
        return {"conflicting_caption": None, "change_type": None, "question": None}
    except Exception as e:
        print(f"Error generating for caption '{caption}': {e}")
        return {"conflicting_caption": None, "change_type": None, "question": None}


def generate_answers(
    model_name, client, original_caption, conflicting_caption, question
):
    prompt = generate_mcqa_answers_prompt(
        original_caption, conflicting_caption, question
    )
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        json_string = (
            response.text.strip().replace("```json\n", "").replace("\n```", "")
        )
        return json.loads(json_string)
    except json.JSONDecodeError as jde:
        print(f"JSON Decode Error for caption '{original_caption}': {jde}")
        print(f"Raw response:\n{response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error for caption '{original_caption}': {e}")
        return None


def score_conditions(response_json, conditions):
    """
    Compare LLM response (dict or list of dicts) against expected conditions.
    Returns: (passed: bool, failed_conditions: list)
    """
    failed_conditions = []

    # Normalize to a list of dicts
    if isinstance(response_json, dict):
        response_items = [response_json]
    elif isinstance(response_json, list):
        response_items = response_json
    else:
        return False, ["invalid_response_format"]

    for condition, expected_value in conditions.items():
        matched = any(
            str(item.get(condition, "")).strip().lower()
            == expected_value.strip().lower()
            for item in response_items
            if isinstance(item, dict)
        )
        if not matched:
            failed_conditions.append(condition)

    passed = len(failed_conditions) == 0
    return passed, failed_conditions


def _process_sample(sample, client, model_name, check_config):
    failed_conditions = []

    # Step 1: Check changed words if applicable
    change_type = sample.get("change_type")
    if change_type in check_config:
        changed_words = sample.get("changed_words", [])
        prompt_fn, conditions = check_config[change_type]
        prompt = prompt_fn(changed_words)
        response_json = _call_llm_with_retry(client, model_name, prompt)
        if response_json is None:
            failed_conditions.append(f"{change_type}_parsing_error")
        else:
            _, fc = score_conditions(response_json, conditions)
            failed_conditions.extend(fc)

    # Step 2: Check answers if Step 1 passed
    if not failed_conditions:
        prompt = answers_check_prompt(sample.get("answers", []))
        response_json = _call_llm_with_retry(client, model_name, prompt)
        if response_json is None:
            failed_conditions.append("answers_parsing_error")
        else:
            _, fc = score_conditions(response_json, ANSWERS_CONDITIONS)
            failed_conditions.extend(fc)

    # Step 3: Check question if Steps 1 & 2 passed
    if not failed_conditions:
        prompt = question_check_prompt(
            sample.get("question", ""),
            sample.get("changed_words", []),
            sample.get("answers", []),
        )
        response_json = _call_llm_with_retry(client, model_name, prompt)
        if response_json is None:
            failed_conditions.append("question_parsing_error")
        else:
            _, fc = score_conditions(response_json, QUESTION_CONDITIONS)
            failed_conditions.extend(fc)

    # Return result
    if not failed_conditions:
        return "filtered", sample
    else:
        rejected_sample = sample.copy()
        rejected_sample["failed_conditions"] = failed_conditions
        return "rejected", rejected_sample


def run_filtering(client, data, model_name, max_workers=4):
    filtered = []
    rejected = []

    check_config = {
        "attribute": (attribute_check_prompt, ATTRIBUTE_CONDITIONS),
        "object": (object_check_prompt, OBJECT_CONDITIONS),
    }

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sample in data["samples"]:
            futures.append(
                executor.submit(
                    _process_sample, sample, client, model_name, check_config
                )
            )

        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Filtering samples"
        ):
            status, sample_res = f.result()
            if status == "filtered":
                filtered.append(sample_res)
            else:
                rejected.append(sample_res)

    return filtered, rejected


def _call_llm_with_retry(client, model_name, prompt, max_retries=5):
    """Call LLM with retry, return parsed JSON or None on failure."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            json_string = (
                response.text.strip().replace("```json\n", "").replace("\n```", "")
            )
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print("Retrying...")
            if attempt == max_retries - 1:
                print(f"Error occurred after {attempt} retries: ", e)
                return None
        except Exception as e:
            print("Retrying...")
            if attempt == max_retries - 1:
                print(f"Error occurred after {attempt} retries: ", e)
                return None


def run_checks(sample, checks_in_caption: list, checks_not_in_caption: list):
    def normalize(text: str) -> str:
        """Lowercase and remove punctuation."""
        if isinstance(text, list):
            text = " ".join(map(str, text))
        return re.sub(r"[^\w\s]", "", text.lower())

    def equivalent(phrase: str, target_phrase: str) -> bool:
        # special case: "one" vs "a"
        if phrase in {"one", "a"} and any(w in {"one", "a"} for w in target_phrase):
            return True

        if phrase in target_phrase:
            return True

        if target_phrase in phrase:
            return True

        phrase_words = phrase.split()
        target_words = target_phrase.split()

        for tw in target_words:
            for pw in phrase_words:
                if tw.startswith(pw[:3]) or pw.startswith(tw[:3]):
                    return True
        return False

    def resolve_keys(key):
        # Resolve nested keys like "changed_words.original"
        parts = key.split(".")
        value = sample
        for p in parts:
            value = value.get(p, "")
        value_norm = normalize(value)
        return value_norm

    failed_conditions = []
    if sample["original_caption"] == sample["conflicting_caption"]:
        failed_conditions.append(f"original_caption_same_as_conflicting_caption")
        return False, failed_conditions

    for key, caption_field in checks_in_caption:
        first_value_norm = resolve_keys(key)
        if "." in caption_field:
            second_value_norm = resolve_keys(caption_field)
            if not equivalent(first_value_norm, second_value_norm):
                failed_conditions.append(f"{key}_not_in_{caption_field}")
        else:
            caption = normalize(sample.get(caption_field, ""))

            if not equivalent(first_value_norm, caption):
                failed_conditions.append(f"{key}_not_in_{caption_field}")

    for key, caption_field in checks_not_in_caption:
        value_norm = resolve_keys(key)
        caption = normalize(sample.get(caption_field, ""))
        caption_words = caption.split()

        if value_norm in caption_words:
            failed_conditions.append(f"{key}_in_{caption_field}")

    passed = len(failed_conditions) == 0
    return passed, failed_conditions


def filter_samples_containment(data):
    filtered = []
    rejected = []

    checks_in = [
        ["answers.image_only", "changed_words.original"],
        ["answers.text_only", "changed_words.conflicting"],
        ["changed_words.original", "original_caption"],
        ["changed_words.conflicting", "conflicting_caption"],
        ["answers.image_only", "original_caption"],
        ["answers.text_only", "conflicting_caption"],
    ]
    checks_not_in = [
        ["answers.irrelevant_but_plausible", "conflicting_caption"],
        ["answers.irrelevant_but_plausible", "original_caption"],
    ]

    for sample in tqdm(data["samples"], desc="Filtering samples for containment"):
        passed, failed_conditions = run_checks(sample, checks_in, checks_not_in)
        if passed:
            filtered.append(sample)
        else:
            rejected_sample = sample.copy()
            rejected_sample["failed_conditions"] = failed_conditions
            rejected.append(rejected_sample)

    return filtered, rejected


def generate_data(args, client, max_workers=6, save_every=200):
    with open(args.source_annotations_path) as f:
        data = json.load(f)

    sampled_annotations = random.sample(data["annotations"], args.num_samples)
    synthetic_samples: List[Sample] = []
    processed_ids = set()

    # --- Resume from checkpoint if provided ---
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
            # Make sure we use the same model for generation if loading from checkpoint file
            assert (
                checkpoint_data["metadata"]["llm_model_generation"]
                == args.model_name_generation
            )
        if "samples" in checkpoint_data:
            for s in checkpoint_data["samples"]:
                synthetic_samples.append(
                    Sample(**s)
                )  # assumes Sample can be constructed from dict
                processed_ids.add(s["image_id"])
        print(
            f"Resumed from checkpoint {checkpoint_path}, "
            f"loaded {len(synthetic_samples)} samples."
        )

    # filter annotations to skip already processed ones
    remaining_annotations = [
        el for el in sampled_annotations if el["image_id"] not in processed_ids
    ]

    def process_element(element):
        result = with_retry(
            query_llm,
            model_name=args.model_name_generation,
            client=client,
            caption=element["caption"],
        )
        if result and result.get("conflicting_caption") is not None:
            answers = with_retry(
                generate_answers,
                model_name=args.model_name_generation,
                client=client,
                original_caption=element["caption"],
                conflicting_caption=result["conflicting_caption"],
                question=result["question"],
            )
            if answers is None:
                return None

            return Sample(
                image_id=element["image_id"],
                original_caption=element["caption"],
                conflicting_caption=result["conflicting_caption"],
                question=result["question"],
                change_type=result["change_type"],
                changed_words=result["changed_words"],
                answers=answers,
            )
        else:
            print(f" Failed to generate for: {element['caption']}")
            return None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(filename="synthetic_captions_checkpoint.json"):
        checkpoint_data = SyntheticDataset(
            metadata=DatasetMetadata.from_args(args), samples=synthetic_samples
        )
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(
                checkpoint_data.to_dict()
                if hasattr(checkpoint_data, "to_dict")
                else checkpoint_data,
                f,
                indent=4,
            )
        print(f"Checkpoint saved to: {output_path}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_element, el): el for el in remaining_annotations
        }

        for i, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Generating synthetic data",
            ),
            1,
        ):
            sample = future.result()
            if sample is not None:
                synthetic_samples.append(sample)

            if i % save_every == 0:
                save_checkpoint()

    print(f"\nSuccessfully generated {len(synthetic_samples)} examples.")

    final_data = SyntheticDataset(
        metadata=DatasetMetadata.from_args(args), samples=synthetic_samples
    )
    save_data(args, final_data)

    return final_data


def save_data(args, data, suffix=""):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = data.to_dict() if hasattr(data, "to_dict") else data

    data["metadata"]["num_samples"] = len(data["samples"])
    output_path = (
        output_dir / f"synthetic_captions_{args.model_name_generation}{suffix}.json"
    )
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to: {output_path}")


def append_and_save(args, new_data, suffix):
    path = (
        Path(args.checkpoint_path).parent
        / f"synthetic_captions_{args.model_name_generation}{suffix}.json"
    )

    # Load old data if it exists
    if path.exists():
        with open(path) as f:
            old_data = json.load(f)
        old_samples = old_data.get("samples", [])
    else:
        old_samples = []

    combined_samples = old_samples + new_data["samples"]

    save_data(
        args,
        {"metadata": new_data["metadata"], "samples": combined_samples},
        suffix=suffix,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and optionally filter synthetic captions with Gemini API."
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "filter", "generate+filter"],
        default="generate+filter",
    )
    parser.add_argument(
        "--model_name_generation",
        type=str,
        default="gemini-2.5-flash",
    )
    parser.add_argument(
        "--model_name_filtering",
        type=str,
        default="gemini-2.5-flash-lite-preview-06-17",
    )
    parser.add_argument("--output_dir", type=str, default="../data")
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # Generation
    parser.add_argument(
        "--source_annotations_path",
        type=str,
        default=Path(BASE_DIR) / "annotations" / "captions_train2017.json",
    )
    parser.add_argument("--num_samples", type=int, default=100)

    # Filtering
    parser.add_argument("--input_json", type=str, help="For filtering mode only")

    return parser.parse_args()


def main():
    args = parse_args()
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Export it before running this script (see README)."
        )
    client = genai.Client(api_key=GEMINI_API_KEY)

    data = None
    if args.mode in ["generate", "generate+filter"]:
        data = generate_data(args, client)

    if args.mode in ["filter", "generate+filter"]:
        if args.mode == "filter":
            if not args.input_json:
                raise ValueError("--input_json required in filter mode")
            with open(args.input_json) as f:
                data = json.load(f)
        else:
            data = data.to_dict()

        # 0. If starting from a checkpoint file, check if filtering has been done
        processed_ids = set()
        if args.checkpoint_path:
            filtered_path = (
                Path(args.checkpoint_path).parent
                / f"synthetic_captions_{args.model_name_generation}_filtered.json"
            )

            rejected_path = (
                Path(args.checkpoint_path).parent
                / f"synthetic_captions_{args.model_name_generation}_rejected.json"
            )

            if filtered_path.exists() and rejected_path.exists():
                with open(filtered_path) as f:
                    filtered_data = json.load(f)
                    for s in filtered_data.get("samples", []):
                        processed_ids.add(s["image_id"])

                with open(rejected_path) as f:
                    rejected_data = json.load(f)
                    for s in rejected_data.get("samples", []):
                        processed_ids.add(s["image_id"])

                print(
                    f"Found filtering results. "
                    f"Marked {len(processed_ids)} total samples as already processed."
                )

            # Filter out processed samples
            remaining_annotations = [
                el for el in data["samples"] if el["image_id"] not in processed_ids
            ]
            data["samples"] = remaining_annotations

        # 1. Fast filter (no LLM)
        filtered, rejected_containment = filter_samples_containment(data)

        # 2. Changed-words check & Answers check & Question check (LLM)
        filtered, rejected_llm = run_filtering(
            client, {"samples": filtered}, args.model_name_filtering
        )

        rejected = rejected_containment + rejected_llm
        append_and_save(
            args,
            {"metadata": data["metadata"], "samples": filtered},
            suffix="_filtered",
        )
        append_and_save(
            args,
            {"metadata": data["metadata"], "samples": rejected},
            suffix="_rejected",
        )


if __name__ == "__main__":
    main()

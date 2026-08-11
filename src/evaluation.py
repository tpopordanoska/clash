import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from nltk.stem import PorterStemmer
from tqdm import tqdm

from models.base_model import model_factory
from models.prompts import JUDGE_PROMPT

stemmer = PorterStemmer()
# Stopwords to remove
STOPWORDS = {"the", "a", "an"}
STRIP_WORDS = {"wooden": "wood", "golden": "gold", "sandy": "sand", "brightly": "bright"}

MAX_RETRIES = 3
N_WORKERS = 1

# mapping numbers ↔ words
NUM_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}

WORD_TO_NUM = {v: k for k, v in NUM_MAP.items()}

def generate_llm_as_judge_prompt(prediction, image_only, text_only):
    return f"""{JUDGE_PROMPT} Now classify: \n Image-only answer: {image_only} \n Text-only answer: {text_only} \n Prediction: {prediction}  
    """

def contains_word(ans, pred):
    if not ans:
        return False
    return re.search(rf"\b{re.escape(ans)}\b", pred, re.IGNORECASE) is not None

def add_open_ended_evaluation_llm(sample, model):
    if f'evaluation_{model.model_name}' in sample:
        # skip sample if already processed
        return sample
    pred = sample["prediction"]

    image_only = sample["modality_answers"]["image_only"]
    text_only = sample["modality_answers"]["text_only"]
    prompt = generate_llm_as_judge_prompt(pred, image_only, text_only)
    for attempt in range(MAX_RETRIES):
        try:
            response = model.inference_as_judge(prompt)
            label = response.strip().upper()

            matches_contradiction = (label == "CONFLICT")
            matches_image_only = (label == "IMAGE")
            matches_text_only = (label == "TEXT")
            matches_none = (label == "NONE")

            sample[f"evaluation_{model.model_name}_prediction"] = label
            sample[f"evaluation_{model.model_name}"] = {
                "matches_contradiction": matches_contradiction,
                "matches_image_only": matches_image_only,
                "matches_text_only": matches_text_only,
                "matches_none": matches_none,
            }

            return sample

        except Exception as e:
            wait_time = 2**attempt + random.random()
            print(f"[Retry {attempt + 1}/{MAX_RETRIES}] Error: {e}. Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)

    # If all retries failed, mark sample
    sample[f"evaluation_llm_{model.model_name}"] = {"error": "inference_failed"}
    return sample

def add_open_ended_evaluation_match(sample):
    def normalize(text):
        if not text:
            return ""
        tokens = re.findall(r"\w+", text.lower())
        norm_tokens = []
        for t in tokens:
            if t in STOPWORDS:
                continue

            # Map numbers -> words
            if t in NUM_MAP:
                t = NUM_MAP[t]
            t = STRIP_WORDS.get(t, t)
            # Stem
            t = stemmer.stem(t)
            norm_tokens.append(t)
        return " ".join(norm_tokens)

    def matches(ans, pred):
        if not ans:
            return False
        pred_norm = normalize(pred)
        ans_norm = normalize(ans)
        # print(f"pred norm: {pred_norm}, ans_norm: {ans_norm} ")
        return re.search(rf"\b{re.escape(ans_norm)}\b", pred_norm) is not None


    pred = sample["prediction"]
    ## key word: match or ==
    matches_contradiction_conflict = matches("conflict", pred)
    matches_contradiction_contradict = matches("contradict", pred)
    matches_contradiction = matches_contradiction_conflict or matches_contradiction_contradict
    matches_image_only = matches(sample["modality_answers"]["image_only"], pred)
    matches_text_only = matches(sample["modality_answers"]["text_only"], pred)

    matches_none = False
    if not matches_contradiction and not matches_text_only and not matches_image_only:
        matches_none = True
        # print(f"pred = {pred} ; image = {sample['modality_answers']['image_only']}, text = {sample['modality_answers']['text_only']}")

    sample["evaluation"] = {
        "matches_contradiction": matches_contradiction,
        "matches_image_only": matches_image_only,
        "matches_text_only": matches_text_only,
        "matches_none": matches_none,
    }
    return sample

def add_multiple_choice_evaluation(sample, mode="strict"):
    pred = sample["prediction"]
    if isinstance(pred, list):
        pred = pred[0] if pred else ""
    pred = str(pred)

    # --- Strict check: (A)/(B)/(C)/(D) ---
    matches = set(re.findall(r"\(([A-D])\)", pred))  # allow [B,B,B]
    if len(matches) == 1:
        final_choice = matches.pop()
        matches_contradiction = final_choice == sample["modality_answers"]["contradiction"]["choice"]
        matches_image_only = final_choice == sample["modality_answers"]["image_only"]["choice"]
        matches_text_only = final_choice == sample["modality_answers"]["text_only"]["choice"]
        matches_irrelevant = final_choice == sample["modality_answers"]["irrelevant_but_plausible"]["choice"]
    elif len(matches) > 1:
        # multiple bracketed answers → ambiguous
        matches_contradiction = matches_image_only = matches_text_only = matches_irrelevant = False
    else:
        # --- If mode is strict, stop here ---
        if mode == "strict":
            matches_contradiction = matches_image_only = matches_text_only = matches_irrelevant = False
        else:
            # --- Relaxed matching fallback ---
            if len(pred) == 1 and pred.isalpha():  # case: A/B/C/D
                final_choice = pred.upper()
                matches_contradiction = final_choice == sample["modality_answers"]["contradiction"]["choice"]
                matches_image_only = final_choice == sample["modality_answers"]["image_only"]["choice"]
                matches_text_only = final_choice == sample["modality_answers"]["text_only"]["choice"]
                matches_irrelevant = final_choice == sample["modality_answers"]["irrelevant_but_plausible"]["choice"]
            else:  # case: key word
                matches_contradiction = False
                matches_image_only = contains_word(sample["modality_answers"]["image_only"]["answer"], pred)
                matches_text_only = contains_word(sample["modality_answers"]["text_only"]["answer"], pred)
                matches_irrelevant = contains_word(sample["modality_answers"]["irrelevant_but_plausible"]["answer"],
                                                   pred)

    sample["evaluation"] = {
        "matches_contradiction": matches_contradiction,
        "matches_image_only": matches_image_only,
        "matches_text_only": matches_text_only,
        "matches_irrelevant": matches_irrelevant,
    }

    return sample

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM predictions.")
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="../results/multiple_choice/",
        help="Path to a predictions JSON file or a folder with multiple JSON files.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='gemini-2.5-pro',
        help="Judge model used for open-ended evaluation. Only used with --llm.",
    )
    parser.add_argument(
        "--match_mode",
        type=str,
        choices=["strict", "relaxed"],
        default="strict",
        help="Evaluation mode: 'strict' (bracket-letter-bracket only) or 'relaxed' (fallback logic).",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Score open-ended predictions with an LLM-as-judge (set via --model_name) "
             "instead of the default string matching. Requires the judge's API key.",
    )

    args = parser.parse_args()
    if os.path.isdir(args.predictions_path):
        json_files = [
            os.path.join(args.predictions_path, f)
            for f in os.listdir(args.predictions_path)
            if f.endswith(".json")
        ]
    elif args.predictions_path.endswith(".json"):
        json_files = [args.predictions_path]
    else:
        raise ValueError("predictions_path must be a JSON file or a directory containing JSON files.")

    for json_file in json_files:
        print(f"Processing {json_file} ... with {args.model_name}")

        if "open_ended" in json_file:
            if args.llm:
                model = model_factory(args.model_name)(model_name=args.model_name)
                eval_fcn = lambda sample: add_open_ended_evaluation_llm(sample, model)
            else:
                eval_fcn = lambda sample: add_open_ended_evaluation_match(sample)
        elif "multiple_choice" in json_file:
            eval_fcn = lambda sample: add_multiple_choice_evaluation(sample, args.match_mode)
        else:
            raise ValueError(f"Unknown question type in predictions path: {json_file}")

        predictions_dict = json.load(open(json_file))

        results = []
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            future_to_sample = {executor.submit(eval_fcn, s): s for s in predictions_dict}

            for future in tqdm(as_completed(future_to_sample), total=len(future_to_sample)):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    s = future_to_sample[future]
                    print("eval failed: ", e)
                    s["evaluation"] = {"error": str(e)}
                    results.append(s)

        with open(json_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Saved {len(results)} evaluations to {json_file}")


if __name__ == "__main__":
    main()
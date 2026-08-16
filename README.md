# CLASH: A Benchmark for Cross-Modal Contradiction Detection

[![Paper](https://img.shields.io/badge/Paper-b31b1b.svg)](https://openaccess.thecvf.com/content/CVPR2026F/papers/Popordanoska_CLASH_A_Benchmark_for_Cross-Modal_Contradiction_Detection_CVPRF_2026_paper.pdf)
[![Data](https://img.shields.io/badge/Data-4c1.svg)](#data)

Official code and data for **CLASH: A Benchmark for Cross-Modal Contradiction Detection**, published at **CVPR 2026 (Findings)**.


📄 **[Read the paper](https://openaccess.thecvf.com/content/CVPR2026F/papers/Popordanoska_CLASH_A_Benchmark_for_Cross-Modal_Contradiction_Detection_CVPRF_2026_paper.pdf)**

Vision-language models are usually evaluated on inputs where the image and its accompanying text agree. CLASH tests the opposite case: each sample pairs a COCO image with a caption that has been minimally edited to *contradict* it, together with a question whose answer depends on which modality you believe. A model that is genuinely grounded in both modalities should decline to answer and flag the conflict — instead of silently defaulting to one of them.

This makes CLASH a probe for **modality preference**: when the two sources disagree, does the model follow the image, follow the text, or notice the disagreement?

### The task at a glance

| | |
|---|---|
| 🖼️ **Image** (COCO `train2017`) | a bowl of **red** apples |
| 📝 **Caption shown to the model** | "A bowl full of **green** apples on top of a table" |
| ❓ **Question** | "What color are the apples in the bowl?" |
| 👁️ Image-only answer | Red |
| 📄 Text-only answer | Green |
| ✅ **Expected response** | *Conflicting information – cannot answer* |

A model answering "Red" reveals an image bias, "Green" a text bias, and only the last response demonstrates it actually cross-checked the two.

<!-- Optional teaser figure: drop the file at assets/teaser.png and uncomment.
<p align="center"><img src="assets/teaser.png" width="720" alt="CLASH overview"></p>
-->

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Gemini and OPENAI API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
export OPENAI_API_KEY="your-api-key-here"
```

Update `BASE_DIR` in [src/utils/constants.py](src/utils/constants.py) to point at your local COCO dataset directory (it should contain `annotations/captions_train2017.json`, from COCO's [2017 Train/Val annotations](https://cocodataset.org/#download)), or pass `--source_annotations_path` explicitly when running the script.

### Generating synthetic data

The imports in `src/generate_synthetic_data.py` are relative to the `src/` directory, so run it from there (or with `src` on your `PYTHONPATH`):

```bash
cd src
python generate_synthetic_data.py --num_samples 100 --output_dir ../data
```

Use `--mode generate`, `--mode filter`, or `--mode generate+filter` (default) to control whether the script generates conflicting captions, filters an existing file (`--input_json`), or does both.

### Running inference

Run one or more vision-language models over a CLASH file and save their predictions. Like the other scripts, run this from `src/`:

```bash
cd src
python inference.py \
  --dataset_path ../data/synthetic_captions_test_filtered.json \
  --model_names gemini-2.5-flash gpt-4.1-mini \
  --output_path ../results \
  --question_format open_ended
```

- `--model_names` currently supports Gemini (any name containing `gemini`) and OpenAI GPT (any name containing `gpt`) models, called through their respective APIs.
- `--question_format` is `open_ended` or `multiple_choice`.
- `--num_icl_shots` sets how many in-context (conflict / non-conflict) example pairs are shown per query (`0` disables ICL).
- `--wo_contr` runs the ablation where the model is shown the *original* (non-conflicting) caption instead, to test whether it's biased toward flagging conflicts regardless of input.

Predictions are written to `<output_path>/<question_format>/<model_name>_predictions.json` and resumed automatically if that file already exists.

### Evaluation

Score predictions produced by `inference.py`:

```bash
python evaluation.py --predictions_path ../results/open_ended/ --match_mode strict
```

- `--predictions_path` accepts either a single `*_predictions.json` file or a directory containing several.
- For `open_ended` predictions, add `--llm` to use an LLM-as-judge (set `--model_name` to the judge model, e.g. `gemini-2.5-pro`) instead of the default keyword-matching evaluator.
- For `multiple_choice` predictions, `--match_mode` controls whether only bracketed letter answers count (`strict`) or whether it falls back to keyword matching (`relaxed`).

### Printing result tables

Aggregate evaluated predictions into LaTeX tables with bootstrapped error bars:

```bash
python print_table_with_error_bars.py \
  --result_dir ../results/multiple_choice/ \
  --test_data_path ../data/synthetic_captions_test_filtered.json
```

By default this prints the modality-preference and category tables. Two cases switch it to a different table: `--ft_model_list` prints the fine-tuning table instead (see [Fine-tuned open-source models](#fine-tuned-open-source-models)), and pointing `--result_dir` at a directory named `llm_as_judge` prints only the judge-agreement table (see [LLM-as-judge agreement](#llm-as-judge-agreement)).

## Reproducing the main results

The `results/` directory contains the evaluated predictions for reproducing the main results in the paper, for the four API models we report (`gemini-2.5-flash-lite`, `gemini-2.5-pro`, `gpt-4.1-mini`, `gpt-5`) in both question formats:

```
results/
├── multiple_choice/<model_name>_predictions.json
└── open_ended/
    ├── <model_name>_predictions.json
    ├── finetune/<model_name>[_wo_contr]_predictions.json
    └── llm_as_judge/<model_name>_predictions.json
```

`open_ended/finetune/` holds the predictions for the fine-tuned open-source models (LLaVA-1.5-7b and mPLUG-Owl-1), used for the fine-tuning table — see [Fine-tuned open-source models](#fine-tuned-open-source-models). `open_ended/llm_as_judge/` holds open-source model predictions scored by several LLM judges, used for the judge-agreement table — see [LLM-as-judge agreement](#llm-as-judge-agreement).

Each entry keeps the original sample fields plus the model's `prediction` and an `evaluation` block with the `matches_*` flags that the tables aggregate.

### From the released predictions (no API keys needed)

All scripts resolve their default paths relative to `src/`, so run them from there:

```bash
cd src

# Multiple choice
python print_table_with_error_bars.py \
  --result_dir ../results/multiple_choice/ \
  --test_data_path ../data/synthetic_captions_test_filtered.json

# Open ended
python print_table_with_error_bars.py \
  --result_dir ../results/open_ended/ \
  --test_data_path ../data/synthetic_captions_test_filtered.json
```

Each run prints the modality-preference table, the object-category table and the attribute-category table as LaTeX rows, in that order. Columns are:

| Table | Columns |
|---|---|
| Modality preference | Contradiction, Image-only, Text-only, Irrelevant, None |
| Object / attribute categories | Contradiction rate per category, in the order of `OBJECT_CATEGORIES_NAMES` / `ATTRIBUTE_CATEGORIES_NAMES` in [src/utils/constants.py](src/utils/constants.py) |

Notes when comparing against the paper:

- Error bars come from 1000 bootstrap resamples with no fixed seed, so numbers move by ~0.1–0.2 points between runs. Set `np.random.seed(...)` if you need bit-identical output.
- The *Irrelevant* column is `0.00` for `open_ended`: open-ended questions have no distractor option, so no prediction can match it.

### From scratch (re-running the models)

Set `GEMINI_API_KEY` / `OPENAI_API_KEY` as described in [Setup](#setup), then, for each question format:

```bash
cd src

# 1. Inference — writes ../results/<question_format>/<model_name>_predictions.json
python inference.py \
  --dataset_path ../data/synthetic_captions_test_filtered.json \
  --model_names gemini-2.5-flash gemini-2.5-pro gpt-4.1-mini gpt-5 \
  --output_path ../results \
  --question_format multiple_choice

# 2. Evaluation — adds the "evaluation" block in place
python evaluation.py --predictions_path ../results/multiple_choice/ --match_mode strict

# 3. Tables
python print_table_with_error_bars.py \
  --result_dir ../results/multiple_choice/ \
  --test_data_path ../data/synthetic_captions_test_filtered.json
```

Inference resumes from an existing predictions file, so an interrupted run can simply be restarted with the same command. Because the API models are non-deterministic, freshly generated predictions will differ slightly from the released ones.

### Fine-tuned open-source models

The fine-tuning table reports LLaVA-1.5-7b and mPLUG-Owl-1 before and after LoRA fine-tuning on CLASH training data, in the open-ended format. Passing `--ft_model_list` switches `print_table_with_error_bars.py` to that table (the modality-preference and category tables are skipped):

```bash
cd src

python print_table_with_error_bars.py \
  --result_dir ../results/open_ended/finetune/ \
  --test_data_path ../data/synthetic_captions_test_filtered.json \
  --ft_model_list llava-v1.5-7b_predictions \
                  llava-v1.5-7b_lora_open_ended_single_predictions \
                  llava-v1.5-7b_lora_open_ended_single_30k_predictions \
                  mplug_owl1_predictions \
                  mplug_owl1_lora_open_ended_single_predictions \
                  mplug_owl1_lora_open_ended_30k_predictions
```

Each row combines two prediction files: `<model>_predictions.json` (the model sees the *conflicting* caption) and `<model>_wo_contr_predictions.json` (the same model on the *original* caption). Only the first is named on the command line — the second path is derived from it — but both must be present.

### LLM-as-judge agreement

The judge-agreement table measures how much the reported open-ended numbers depend on the choice of LLM judge, by scoring the same four open-source models (InternVL1.5, LLaVA-1.5-7b, mPLUG-Owl-1, mPLUG-Owl-2) with two independent judges. Pointing `--result_dir` at a directory named `llm_as_judge` prints only this table (the modality-preference and category tables are skipped):

```bash
cd src

python print_table_with_error_bars.py --result_dir ../results/open_ended/llm_as_judge/
```

Each model gets four rows — one per judge (`gemini-2.5-pro`, `gpt-5`), plus `≥1` (either judge flagged the category) and `Both` (both did).

## Data

The `data/` directory contains the CLASH samples as JSON files. Each file has the same top-level structure:

```json
{
  "metadata": { ... },
  "samples": [ ... ]
}
```

- `metadata`: information about how the file was generated, e.g. `created_on`, the LLM(s) used (`llm_model_generation`, `llm_model_filtering`, or `llm_model`), the source annotations file the captions were derived from (`source_annotations`, referring to COCO's `captions_train2017.json`), and `num_samples`.
- `samples`: a list of individual contradiction examples. Each sample has the form:

```json
{
  "image_id": 10478,
  "original_caption": "A bowl full of red apples on top of a table",
  "conflicting_caption": "A bowl full of green apples on top of a table",
  "question": "What color are the apples in the bowl?",
  "change_type": "attribute",
  "answers": {
    "image_only": "Red",
    "text_only": "Green",
    "irrelevant_but_plausible": "Yellow"
  },
  "changed_words": {
    "original": "red",
    "conflicting": "green"
  }
}
```

- `image_id`: the COCO `train2017` image ID the sample refers to. The images themselves are not included in this repository — download COCO's [2017 Train images](https://cocodataset.org/#download) and look up images by ID.
- `original_caption` / `conflicting_caption`: the caption matching the image and the caption that contradicts it.
- `question`: a question whose answer differs depending on whether it's answered from the image or from the (conflicting) text.
- `change_type`: the kind of edit that produced the conflicting caption (e.g. `object`, `attribute`).
- `answers`: candidate answers to `question` — the answer grounded in the image (`image_only`), the answer grounded in the conflicting text (`text_only`), and a plausible but incorrect distractor (`irrelevant_but_plausible`).
- `changed_words`: the specific span(s) edited between the original and conflicting captions.

### Files

| File | Samples | Description |
|---|---|---|
| `filtered_data_finetune.json` | 18,097 | Generated samples that passed the automated quality filters. |
| `merged_data_finetune.json` | 31,676 | Superset combining filtered and additional generated samples for fine-tuning. |
| `synthetic_captions_test_filtered.json` | 1,289 | Held-out filtered test split; samples include an extra `category` field. |

### Loading the data

```python
import json

with open("data/merged_data_finetune.json") as f:
    data = json.load(f)

metadata = data["metadata"]
samples = data["samples"]
print(f"Loaded {len(samples)} samples generated with {metadata['llm_model_generation']}")
```

## Citation

If you use CLASH in your research, please cite:

```bibtex
@inproceedings{popordanoska2026clash,
  title={CLASH: A benchmark for cross-modal contradiction detection},
  author={Popordanoska, Teodora and Li, Jiameng and Blaschko, Matthew B},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  pages={6051--6061},
  year={2026}
}
```

# CLASH: A Benchmark for Cross-Modal Contradiction Detection

This is the official code repository for CLASH: A Benchmark for Cross-Modal Contradiction Detection, published at CVPR 2026 (Findings). 

CLASH is a benchmark for evaluating the ability of vision-language models to detect contradictions between image and text modalities.


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

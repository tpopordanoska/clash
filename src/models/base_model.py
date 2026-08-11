from abc import ABC, abstractmethod

from models.prompts import (
    OPEN_ENDED_INSTRUCTION,
    MULTIPLE_CHOICE_INSTRUCTION,
    ICL_EXAMPLE_NON_CONFLICT,
    ICL_EXAMPLE_CONFLICT,
)
from utils.constants import BASE_DIR, IMAGE_SPLIT


class BaseModel(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def prepare_model_input(self, sample, **kwargs):
        pass

    @abstractmethod
    def inference(self, model_input):
        pass

    @staticmethod
    def image_path(image_id):
        """Resolve the COCO image file backing a sample."""
        return BASE_DIR / IMAGE_SPLIT / f"{image_id:012d}.jpg"

    def construct_prompt(self, question_format, sample, wo_contr=False):
        """
        Build the prompt for one sample.

        With wo_contr=True the model is shown the original (non-conflicting)
        caption instead, which measures how often it flags a conflict that
        isn't there.
        """
        caption = sample["original_caption" if wo_contr else "conflicting_caption"]

        if question_format == "multiple_choice":
            parts = [
                MULTIPLE_CHOICE_INSTRUCTION,
                caption,
                sample["question"],
                sample["formatted_choices_string"],
            ]
        elif question_format == "open_ended":
            parts = [OPEN_ENDED_INSTRUCTION, caption, sample["question"]]
        else:
            raise ValueError(f"Unsupported question format: {question_format}")

        return "\n".join(parts)

    def construct_icl_prompt(self, question_format, sample, icl_samples):
        """
        Build ICL prompt with paired examples (non-conflict + conflict per sample).

        Each ICL sample generates TWO examples:
        - Example Xa (non-conflict): original_caption -> image_only answer
        - Example Xb (conflict): conflicting_caption -> "Conflicting information" answer
        """
        # Select instruction based on format
        if question_format == "multiple_choice":
            instruction = MULTIPLE_CHOICE_INSTRUCTION
        else:
            instruction = OPEN_ENDED_INSTRUCTION

        instruction += "\nHere are examples showing how to identify conflicts:\n"

        # Build example texts (each ICL sample -> 2 examples: non-conflict + conflict)
        example_texts = []
        for idx, icl_sample in enumerate(icl_samples, start=1):
            # Example a: non-conflict (original caption -> image answer)
            non_conflict_text = ICL_EXAMPLE_NON_CONFLICT.format(
                idx=idx,
                original_caption=icl_sample["original_caption"],
                question=icl_sample["question"],
                image_only_answer=icl_sample["answers"]["image_only"],
            )
            # Example b: conflict (conflicting caption -> conflict answer)
            conflict_text = ICL_EXAMPLE_CONFLICT.format(
                idx=idx,
                conflicting_caption=icl_sample["conflicting_caption"],
                question=icl_sample["question"],
            )
            combined_example = non_conflict_text + "\n" + conflict_text
            example_texts.append((combined_example, icl_sample["image_id"]))

        target_text = [
            "\nNow answer the following:\n",
            sample["conflicting_caption"],
            sample["question"],
        ]

        if question_format == "multiple_choice":
            target_text.append(sample["formatted_choices_string"])

        target_text = "\n".join(target_text)

        return instruction, example_texts, target_text


def model_factory(model_name):
    if "gemini" in model_name:
        from models.gemini import GeminiModel

        return GeminiModel
    elif "gpt" in model_name and "minigpt" not in model_name:
        from models.openai import GPTModel

        return GPTModel

    else:
        raise ValueError(f"Unknown model name: {model_name}")

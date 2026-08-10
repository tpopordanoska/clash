import random

from utils.constants import CONTRADICTION_TEXT


def format_multiple_choice(answers_dict):
    all_answers = list(answers_dict.values()) + [CONTRADICTION_TEXT]
    random.shuffle(all_answers)

    choice_labels = ["A", "B", "C", "D"]
    formatted_choices = {
        label: answer for label, answer in zip(choice_labels, all_answers)
    }
    formatted_string = "\n".join(
        f"({label}) {text}" for label, text in formatted_choices.items()
    )
    return formatted_choices, formatted_string


def prepare_sample_for_multiple_choice(sample):
    formatted_choices, formatted_string = format_multiple_choice(sample["answers"])
    sample["formatted_choices"] = formatted_choices
    sample["formatted_choices_string"] = formatted_string

    modality_answers = {}
    for modality, ans_text in sample["answers"].items():
        choice_label = next(
            (label for label, text in formatted_choices.items() if text == ans_text),
            None,
        )
        modality_answers[modality] = {"choice": choice_label, "answer": ans_text}

    contradiction_label = next(
        (
            label
            for label, text in formatted_choices.items()
            if text == CONTRADICTION_TEXT
        ),
        None,
    )
    modality_answers["contradiction"] = {
        "choice": contradiction_label,
        "answer": CONTRADICTION_TEXT,
    }

    sample["modality_answers"] = modality_answers
    return sample



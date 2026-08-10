from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict


@dataclass
class Sample:
    image_id: int
    original_caption: str
    conflicting_caption: str
    question: str
    change_type: str  # e.g., object, attribute
    answers: Dict[str, str]
    changed_words: Dict[str, str]


@dataclass
class DatasetMetadata:
    created_on: str
    llm_model_generation: str
    llm_model_filtering: str
    source_annotations: str
    num_samples: int

    @classmethod
    def from_args(cls, args):
        return cls(
            created_on=datetime.now().isoformat(),
            llm_model_generation=args.model_name_generation,
            llm_model_filtering=args.model_name_filtering,
            source_annotations=args.source_annotations_path,
            num_samples=args.num_samples,
        )


@dataclass
class SyntheticDataset:
    metadata: DatasetMetadata
    samples: List[Sample]

    def to_dict(self):
        return asdict(self)

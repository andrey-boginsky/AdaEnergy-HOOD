
import json

from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional

import numpy as np
import torch

from torch.utils.data import Dataset

from transformers import (
    PreTrainedTokenizer,
)

# ============================================================
# CONSTANTS
# ============================================================

DATA_PATH = (
    Path(__file__).parent.parent
    / "data"
    / "data_full.json"
)

DEFAULT_MAX_LENGTH = 64

SPLITS = (
    "train",
    "val",
    "test",
)

OOD_LABEL_STR = "oos"

OOD_LABEL_ID = -1

# ============================================================
# LOAD CLINC150
# ============================================================

def load_clinc150(
    data_path: Path = DATA_PATH,
    include_ood: bool = True,
):

    """
    Load CLINC150 dataset.

    Returns:
        splits:
            {
                split_name: [
                    (
                        text,
                        label_name,
                        is_ood,
                    )
                ]
            }

        label2id:
            {
                intent_name: class_id
            }
    """

    with open(
        data_path,
        encoding="utf-8",
    ) as f:

        raw = json.load(f)

    splits = {}

    # --------------------------------------------------------
    # Build splits
    # --------------------------------------------------------

    for split in SPLITS:

        samples = [

            (
                text,
                label,
                False,
            )

            for text, label
            in raw[split]
        ]

        # ----------------------------------------------------
        # Add OOD samples
        # ----------------------------------------------------

        if include_ood:

            ood_key = f"oos_{split}"

            if ood_key in raw:

                samples.extend(

                    (
                        text,
                        OOD_LABEL_STR,
                        True,
                    )

                    for text, _
                    in raw[ood_key]
                )

        splits[split] = samples

    # --------------------------------------------------------
    # Build label mapping ONLY from ID train samples
    # --------------------------------------------------------

    train_labels = sorted({

        label

        for _, label, is_ood
        in splits["train"]

        if not is_ood
    })

    label2id = {

        label: idx

        for idx, label
        in enumerate(train_labels)
    }

    return (
        splits,
        label2id,
    )

# ============================================================
# DATASET
# ============================================================

class CLINC150Dataset(Dataset):

    """
    Dataset for Energy-Based Fine-Tuning.

    Design:
    --------

    ID samples:
        label = [0 ... num_classes-1]

    OOD samples:
        label = -1

    OOD samples are:
        - excluded from CE loss
        - used ONLY for energy regularization
    """

    def __init__(
        self,
        samples: List[
            Tuple[
                str,
                str,
                bool,
            ]
        ],

        label2id: Dict[
            str,
            int,
        ],

        tokenizer: PreTrainedTokenizer,

        max_length: int = DEFAULT_MAX_LENGTH,

        include_ood: bool = True,

        hard_ood_samples: Optional[
            List[
                Tuple[
                    str,
                    str,
                    bool,
                ]
            ]
        ] = None,
    ):

        self.label2id = label2id

        self.max_length = max_length

        self.ood_label = OOD_LABEL_ID

        if not include_ood:
            samples = [
                sample for sample in samples
                if not sample[2]  # is_ood флаг = False
            ]



        # ----------------------------------------------------
        # Merge hard OOD
        # ----------------------------------------------------

        all_samples = list(samples)

        if (
            hard_ood_samples
            and include_ood
        ):

            all_samples.extend(
                hard_ood_samples
            )

        # ----------------------------------------------------
        # Store raw fields
        # ----------------------------------------------------

        self.texts = [
            text
            for text, _, _
            in all_samples
        ]

        self.label_names = [
            label
            for _, label, _
            in all_samples
        ]

        self.is_ood = torch.tensor(

            [
                is_ood
                for _, _, is_ood
                in all_samples
            ],

            dtype=torch.bool,
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        self.labels = torch.tensor(

            [

                (
                    label2id[label]
                    if not is_ood
                    else self.ood_label
                )

                for _, label, is_ood
                in all_samples
            ],

            dtype=torch.long,
        )

        # ----------------------------------------------------
        # Batch tokenization
        # ----------------------------------------------------

        self.encodings = tokenizer(

            self.texts,

            max_length=max_length,

            padding="max_length",

            truncation=True,

            return_tensors="pt",
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.labels)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, idx):

        return {

            # ----------------------------------------------
            # Transformer inputs
            # ----------------------------------------------

            "input_ids":
                self.encodings[
                    "input_ids"
                ][idx],

            "attention_mask":
                self.encodings[
                    "attention_mask"
                ][idx],

            # ----------------------------------------------
            # Labels
            # ----------------------------------------------

            "label":
                self.labels[idx],

            "label_name":
                self.label_names[idx],

            # ----------------------------------------------
            # OOD info
            # ----------------------------------------------

            "is_ood":
                self.is_ood[idx],

            # ----------------------------------------------
            # Original text
            # ----------------------------------------------

            "text":
                self.texts[idx],

            # ----------------------------------------------
            # Index for debugging / analysis
            # ----------------------------------------------

            "index":
                idx,
        }

# ============================================================
# DATASET FACTORY
# ============================================================

def build_dataset(

    split_samples,

    label2id,

    tokenizer,

    max_length=DEFAULT_MAX_LENGTH,

    include_ood=True,

    hard_ood_samples=None,
):

    return CLINC150Dataset(

        samples=split_samples,

        label2id=label2id,

        tokenizer=tokenizer,

        max_length=max_length,

        include_ood=include_ood,

        hard_ood_samples=hard_ood_samples,
    )

# ============================================================
# DEBUG SUMMARY
# ============================================================

def dataset_summary(
    dataset,
    name = ''
):

    labels = dataset.labels.numpy()

    id_mask = labels != -1

    ood_mask = labels == -1

    print(name)

    print(
        f"  Total samples: {len(dataset)}"
    )

    print(
        f"  ID samples: {id_mask.sum()}"
    )

    print(
        f"  OOD samples: {ood_mask.sum()}"
    )

    print(
        f"  Unique ID labels: "
        f"{len(np.unique(labels[id_mask]))}"
    )


from typing import List
from typing import Tuple

from pathlib import Path


def load_hard_ood_from_txt(
    txt_path: Path,
) -> List[
    Tuple[
        str,
        str,
        bool,
    ]
]:

    """
    Load hard OOD samples.

    Format:
        one query per line

    Returns:
        [
            (
                text,
                "oos",
                True,
            )
        ]
    """

    if not txt_path.exists():

        raise FileNotFoundError(
            f"HARD OOD file not found:\n"
            f"{txt_path}"
        )

    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:

        texts = [

            line.strip()

            for line in f

            if line.strip()
        ]

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    texts = list(dict.fromkeys(texts))

    # --------------------------------------------------------
    # Convert to dataset format
    # --------------------------------------------------------

    samples = [

        (
            text,
            "oos",
            True,
        )

        for text in texts
    ]

    # --------------------------------------------------------
    # Stats
    # --------------------------------------------------------

    print("\n================================================")
    print("HARD OOD LOADED")
    print("================================================")

    print(
        f"File: {txt_path}"
    )

    print(
        f"Total HARD OOD samples: "
        f"{len(samples)}"
    )

    if len(samples) > 0:

        lengths = [
            len(x[0].split())
            for x in samples
        ]

        print(
            f"Avg words per sample: "
            f"{np.mean(lengths):.2f}"
        )

        print("\nExamples:")

        for i in range(
            min(5, len(samples))
        ):

            print(
                f"  [{i}] "
                f"{samples[i][0]}"
            )

    print("================================================")

    return samples
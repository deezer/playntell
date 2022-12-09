""" Utilities for processing captions """

import json
from typing import List, Tuple

from config import preprocessed_data_path


def _get_tokenizer() -> List[str]:
    try:
        import spacy

        spacy_en = spacy.load("en_core_web_sm")

        return lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
    except ImportError:
        print(
            "Please install SpaCy and the SpaCy English tokenizer. "
            "See the docs at https://spacy.io for more information."
        )
        raise
    except AttributeError:
        print(
            "Please install SpaCy and the SpaCy English tokenizer. "
            "See the docs at https://spacy.io for more information."
        )
        raise


tokenizer = _get_tokenizer()


def preprocess_caption(caption: str, return_tokens=True):
    """
    Transforms caption into lower case and drops special characters
    """
    punctuations = [
        "''",
        "'",
        "``",
        "`",
        "-LRB-",
        "-RRB-",
        "-LCB-",
        "-RCB-",
        ".",
        "?",
        "!",
        ",",
        ":",
        "-",
        "--",
        "...",
        ";",
        '"',
    ]
    caption = caption.lower().strip()
    caption_tokens = tokenizer(caption)

    # Remove punctuations
    caption_tokens = [token for token in caption_tokens if token not in punctuations]

    # Remove tokens with special characters
    caption_tokens = [
        token for token in caption_tokens if not ("\t" in token or "\r" in token)
    ]

    if return_tokens:
        return caption_tokens
    else:
        return " ".join(caption_tokens)


def max_caption_length(dataset_name: str) -> int:
    """
    Returns the maximum caption length in words in a given dataset name (curated-deezer or curated-spotify).
    """
    dataset = []
    for split in ["train", "val", "test"]:
        with open(f"{preprocessed_data_path}/{dataset_name}/splits/{split}.json") as f:
            dataset += json.load(f)

    return_value = 0

    for sample in dataset:
        caption_length = len(sample["caption"].split(" "))
        return_value = caption_length if caption_length > return_value else return_value

    return return_value


def extract_title_description_delimiter_formatting(caption: str):
    """
    Given a caption such as e.g. "bla <delimiter> bla bla bla",
    this method returns (title, description), in the e.g.: ("bla", "bla bla bla")
    """
    title_start_index = 0
    description_end_index = len(caption)
    try:
        title_end_index = caption.index("<delimiter>")
        description_start_index = title_end_index + 11
    except ValueError:
        title_end_index = len(caption)
        description_start_index = len(caption)

    title = caption[title_start_index:title_end_index]
    description = caption[description_start_index:description_end_index]

    title = title.strip()
    description = description.strip()

    return title, description


def extract_title_description(caption: str) -> Tuple[str, str]:
    """
    Given a caption such as e.g. "<title> bla <description> bla bla bla",
    this method returns (title, description), in the e.g.: ("bla", "bla bla bla")

    Test cases:
    - "<title> bla" -> ("bla", "")
    - "<description> this is the descr <title> this is the title") -> ("", "this is the descr <title> this is the title")
    - "<title> this is the title <description> this is the description") -> ("this is the title", "this is the description")
    - "<description> bla" -> ("", "bla")
    - "bla bla <title> bla" -> ("bla", "")
    - "bla bla <description> bla" -> ("", "bla")
    - "bla bla <title> ble <description> bla" -> ("ble", "bla")
    """
    try:
        start_idx_title = caption.index("<title>") + 7
    except ValueError:
        start_idx_title = len(caption)

    try:

        start_idx_description = caption.index("<description>") + 13
        end_idx_title = caption.index("<description>")

    except ValueError:
        start_idx_description = len(caption)
        end_idx_title = len(caption)

    end_idx_description = len(caption)

    title = caption[start_idx_title:end_idx_title]
    description = caption[start_idx_description:end_idx_description]

    title = title.strip()
    description = description.strip()

    return (title, description)

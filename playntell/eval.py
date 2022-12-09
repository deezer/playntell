import logging
import traceback
from time import sleep
from typing import Tuple

import pandas as pd
import torch
import transformers
from bert_score import score as bert_score

from audio_gpt import evaluation
from captions import extract_title_description

# silence Bert-Score logging messages
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


def accuracy(
    true: list,
    pred: list,
    what=["BLEU", "METEOR", "ROUGE", "CIDEr", "Bert-S"],
    caption="full",
) -> Tuple[dict, dict]:
    """
    `true` and `pred` are two lists of strings: predicted and true captions.
    This method compute several accuracy metrics to measure their similarity.
    `what` is the list of metrics to compute.
    `caption` is whether we compute metrics on the full captions or on title/description only.
    """
    one_values = {}
    each_values = {}

    # sanity check
    assert len(true) == len(pred)
    assert all(type(e) == str for e in true)
    assert all(type(e) == str for e in pred)
    assert all(
        metric in ["BLEU", "METEOR", "ROUGE", "CIDEr", "Bert-S"] for metric in what
    )
    assert caption in ["full", "title", "description"]

    # transform two lists in dictionaries, as required by audio_gpt's compute scores
    gts = {str(i): [true[i]] for i in range(len(true))}
    gen = {str(i): [pred[i]] for i in range(len(pred))}

    if caption == "full":
        pass
    elif caption == "title":
        gts = {k: [extract_title_description(v[0])[0]] for k, v in gts.items()}
        gen = {k: [extract_title_description(v[0])[0]] for k, v in gen.items()}
    elif caption == "description":
        gts = {k: [extract_title_description(v[0])[1]] for k, v in gts.items()}
        gen = {k: [extract_title_description(v[0])[1]] for k, v in gen.items()}

    # compute metrics
    for metric in what:
        if metric in ["BLEU", "METEOR", "ROUGE", "CIDEr"]:

            one_value, each_value = evaluation.compute_scores(gts, gen, metric)

            if metric == "BLEU":
                for i in range(4):
                    one_values[f"BLEU@{i+1}"] = one_value[i]
                    each_values[f"BLEU@{i+1}"] = each_value[i]
            else:
                one_values[metric] = one_value
                each_values[metric] = each_value

        if metric == "Bert-S":

            # compute Bert-score with recall and idf weighting:
            # best configuration in original paper for the image captioning task.
            while True:
                try:
                    _, recall, _ = bert_score(pred, true, idf=True, lang="en")
                    break
                except Exception:
                    print(traceback.format_exc())
                    print("Trying again in 5 seconds ...")
                    sleep(5)
            one_values["Bert-S"] = torch.mean(recall)
            each_values["Bert-S"] = recall

    for k, v in one_values.items():
        one_values[k] = float(v)
    for k, v in each_values.items():
        each_values[k] = [float(e) for e in v]

    return one_values, each_values


def diversity(training: list, pred: list, caption="full") -> dict:
    """
    This method compute several metrics to compute the diversity of predicted captions.
    `training` and `pred` are two lists of strings: training and true captions.
    `caption` is whether we compute metrics on the full captions or on title/description only.
    """
    return_value = {}

    if caption == "full":
        pass
    elif caption == "title":
        training = [extract_title_description(e)[0] for e in training]
        pred = [extract_title_description(e)[0] for e in pred]
    elif caption == "description":
        training = [extract_title_description(e)[1] for e in training]
        pred = [extract_title_description(e)[1] for e in pred]

    # Compute %Novel
    training_set = set(training)
    num_novel = 0

    for caption in pred:
        num_novel += 1 if caption not in training_set else 0

    return_value["perc_novel"] = (num_novel / len(pred)) * 100

    # Compute vocab
    vocabulary = set()

    for caption in pred:
        for word in caption.split(" "):
            vocabulary.add(word)

    return_value["vocab"] = len(vocabulary)

    return return_value


def stamp(one_values):
    """
    Utility to print the metrics.
    """
    d = {}

    for k, v in one_values.items():
        if type(v) == int:
            d[k] = str(v)
        elif type(v) == float:
            d[k] = "{:.3f}".format(v)
        else:
            raise ValueError(f"Unexpected type {type(v)} of metric {k}")

    d = {k: [v] for k, v in d.items()}

    return pd.DataFrame(d).to_string()

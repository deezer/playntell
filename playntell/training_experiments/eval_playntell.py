import argparse
import json
import os
import sys

# Append parent direcory to path to load modules correctly
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from captions import extract_title_description
from config import preprocessed_data_path
from eval import accuracy, diversity, stamp


def compute_metrics(path):
    """
    Benchmark some predictions according to accuracy and diversity metrics.
    `path` is the path to a json file which contains two lists: predictions and true captions.
    This method computes the metrics for full captions, and title/descriptio only.
    """
    # load data
    predictions, true_captions = json.load(open(path)).values()
    training = json.load(
        open(f"{preprocessed_data_path}/curated-deezer/splits/train.json")
    )
    training = [sample["caption"] for sample in training]

    # extract title, description and full caption
    t = [None, None, None]
    for i, caption in enumerate([predictions, true_captions, training]):

        l = [
            {
                **{
                    "full": s,
                },
                **dict(zip(("title", "description"), extract_title_description(s))),
            }
            for s in caption
        ]
        t[i] = l

    predictions, true_captions, training = tuple(t)

    # compute metrics for title, description and full caption

    t = [None, None, None]
    for i, k in enumerate(["title", "description", "full"]):
        pre = [d[k] for d in predictions]
        tru = [d[k] for d in true_captions]
        tra = [d[k] for d in training]

        acc, _ = accuracy(tru, pre)
        div = diversity(tra, pre)
        t[i] = {**acc, **div}

    return tuple(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    params = parser.parse_args()

    t = compute_metrics(params.path)
    for k, v in zip(["title", "description", "full"], t):
        print("\n")
        print(f"Metrics {k}:")
        print(stamp(v))

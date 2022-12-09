import argparse
import json

import numpy as np
from scipy.stats import ttest_rel
from tqdm import tqdm

from ..eval import accuracy


def significance_test(
    path_1: str,
    path_2: str,
    metrics=["BLEU", "METEOR", "ROUGE", "CIDEr", "Bert-S"],
) -> dict:
    """
    Statistically compare model outputs at `path_1` with outputs at `path_2` in terms of accuracy.

    Both `path_1` and `path_2` are paths to two json files, and each contains two lists: predictions and true captions.
    The output is a p-value for each accuracy metric.
    This method assumes that `path_1` is the most performing algorithm, and that `path_2` to be the least performing algorithm.
    Lower the p-value, higher the certainty that the `path_1` algorithm is more performing than the `path_2` algorithm.

    This methods computes the accuracy on the whole caption, i.e. title and description together
    """
    assert all(
        metric in ["BLEU", "METEOR", "ROUGE", "CIDEr", "Bert-S"] for metric in metrics
    )

    pred_path_1, true = json.load(open(path_1)).values()
    pred_path_2, true = json.load(open(path_2)).values()

    p_values = {}
    for metric in metrics:

        if metric in ["CIDEr", "Bert-S", "ROUGE"]:
            print(f"Computing significance for metric {metric}, simple t-test.")

            _, each_values_path_1 = accuracy(true, pred_path_1, [metric])
            _, each_values_path_2 = accuracy(true, pred_path_2, [metric])

            print(f"{metric} value algorithm 1: {np.mean(each_values_path_1[metric])}")
            print(f"{metric} value algorithm 2: {np.mean(each_values_path_2[metric])}")

            _, p = ttest_rel(each_values_path_1[metric], each_values_path_2[metric])
            print(f"{metric} p-value: {p}")

            p_values[metric] = p

        elif metric in ["BLEU", "METEOR"]:
            print(f"Computing significance for metric {metric}, bootstrap.")

            metric_values_path_1 = []
            metric_values_path_2 = []

            for _ in tqdm(range(1000)):
                indices = np.random.choice(len(true), len(true))

                bootstrap_replica_path_1 = [pred_path_1[i] for i in indices]
                bootstrap_replica_path_2 = [pred_path_2[i] for i in indices]
                bootstrap_replica_true = [true[i] for i in indices]

                value_path_1, _ = accuracy(
                    bootstrap_replica_true, bootstrap_replica_path_1, [metric]
                )
                value_path_2, _ = accuracy(
                    bootstrap_replica_true, bootstrap_replica_path_2, [metric]
                )

                metric_values_path_1.append(value_path_1)
                metric_values_path_2.append(value_path_2)

            for m in list(metric_values_path_1[0].keys()):
                values_path_1 = np.array([v[m] for v in metric_values_path_1])
                values_path_2 = np.array([v[m] for v in metric_values_path_2])

                print(
                    f"{m} value algorithm 1 across bootstrap replicas: {np.mean(values_path_1)}"
                )
                print(
                    f"{m} value algorithm 2 across bootstrap replicas: {np.mean(values_path_2)}"
                )

                p = 1 - (np.sum(values_path_1 > values_path_2) / 1000)
                p_values[m] = p

                print(f"{m} p-value: {p}")

    return p_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Input two paths to models output and run a statistical significance test for accuracty"
    )
    parser.add_argument("path_1", type=str)
    parser.add_argument("path_2", type=str)
    params = parser.parse_args()

    p_values = significance_test(params.path_1, params.path_2)
    print(p_values)

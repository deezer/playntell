import json
import os
import sys

# Append parent direcory to path to load modules correctly
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import pandas as pd
from config import preprocessed_data_path
from tqdm import tqdm

from eval import accuracy, diversity

"""
Replicates the tables that show on the paper.
"""


def diversity_table(d, caption="full"):
    dfs = []
    for ds in ["curated-deezer", "curated-spotify"]:
        training = json.load(
            open(f"{preprocessed_data_path}/curated-deezer/splits/train.json")
        )
        training = [sample["caption"] for sample in training]

        algorithms = list(d.keys())
        metrics = []

        for path in tqdm(d.values()):
            path = path.format(ds)
            predictions, _ = json.load(open(path)).values()
            div = diversity(training, predictions, caption)
            metrics.append(div)

        df = pd.DataFrame(metrics)
        mapper = {"perc_novel": "\\percnovel", "vocab": "\\vocabsize"}
        df = df.rename(mapper, axis=1)

        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    df = df.set_axis(algorithms)

    table = df.style.format("{:.1f}", subset=["\\percnovel"]).to_latex(
        hrules=True,
        column_format="c" * (len(df.columns) + 1),
        position_float="centering",
    )

    table = table.replace("\\toprule", "\\hline")
    table = table.replace("\\midrule", "\\hline")
    table = table.replace("\\bottomrule", "\\hline")

    table = (
        "\\addtolength{\\tabcolsep}{-2.5pt}"
        + table
        + "\\addtolength{\\tabcolsep}{+2.5pt}"
    )
    print(table)


def accuracy_table(d, caption="full"):
    dfs = []
    for ds in ["curated-deezer", "curated-spotify"]:

        algorithms = list(d.keys())
        metrics = []

        for path in tqdm(d.values()):
            path = path.format(ds)
            predictions, true_captions = json.load(open(path)).values()
            # acc, _ = accuracy(true_captions, predictions, caption=caption)
            acc, _ = accuracy(
                true_captions,
                predictions,
                caption=caption,
            )
            metrics.append(acc)

        columns = [
            "BLEU@1",
            "BLEU@2",
            "BLEU@3",
            "BLEU@4",
            "METEOR",
            "ROUGE",
            "CIDEr",
            "Bert-S",
        ]
        to_show = [
            f"\\bleuoneabbr{ds}",
            f"\\bleutwoabbr{ds}",
            f"\\bleuthreeabbr{ds}",
            f"\\bleufourabbr{ds}",
            f"\\meteorabbr{ds}",
            f"\\ciderabbr{ds}",
            f"\\rougeabbr{ds}",
            f"\\bertscoreabbr{ds}",
        ]
        mapper = {k: v for k, v in zip(columns, to_show)}

        df = pd.DataFrame(metrics) * 100
        df = df[columns]
        df = df.rename(mapper, axis=1)
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    df = df.set_axis(algorithms)
    table = (
        df.style.format("{:.1f}")
        .highlight_max(props="textbf:--rwrap;")
        .to_latex(
            hrules=True,
            column_format="c" * (len(columns) + 1),
            position_float="centering",
        )
    )

    table = table.replace("\\toprule", "\\hline")
    table = table.replace("\\midrule", "\\hline")
    table = table.replace("\\bottomrule", "\\hline")

    table = (
        "\\addtolength{\\tabcolsep}{-0.5pt}\n\\small\n"
        + table
        + "\\addtolength{\\tabcolsep}{+0.5pt}"
    )

    for ds in ["curated-deezer", "curated-spotify"]:
        table = table.replace(ds, "")
    print(table)


def qualitative():
    paths = {
        # "\\knn": "/data/playlist-captioning/p/curated-deezer/algorithm-data/knn/audio/predictions_curated-spotify_test.json",
        # "\\muscaps": "/data/playlist-captioning/p/curated-deezer/algorithm-data/manco/save/experiments/2022-02-21-08_45_19/predictions_beam_3_curated-spotify_test.json",
        # "\\dohr": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/rnn/white/s:True_epos:False/inference-curated-spotify-test.json",
        # "\\doht": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/transfomer/white/s:True_epos:False/inference-curated-spotify-test.json",
        # "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/curated-spotify-test.json",
    }
    d = {k: json.load(open(v))["predictions"] for k, v in paths.items()}
    d["Ground truth"] = json.load(open(list(paths.values())[-1]))["true_captions"]

    max_len = 110
    for i in range(len(list(d.values())[0])):

        skip = any(len(v[i]) > max_len for v in d.values())
        if skip:
            continue

        s = ""
        for k, v in d.items():

            value = "\\acaption{" + v[i] + "}"
            s += f"{k} & {value} \\\\ \n"

        s = s.replace("%", "\\%")
        s = s.replace("<eos>", "\\%")
        s += "\\hline\n"
        print(s)
        input()


if __name__ == "__main__":
    # print("Qualitative analysis")
    # qualitative()

    print("State of the art accuracy")
    accuracy_table(
        {
            # "\\knn": "/data/playlist-captioning/p/curated-deezer/algorithm-data/knn/audio/predictions_{}_test.json",
            # "\\muscaps": "/data/playlist-captioning/p/curated-deezer/algorithm-data/manco/save/experiments/2022-02-21-08_45_19/predictions_beam_3_{}_test.json",
            # "\\dohr": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/rnn/white/s:True_epos:False/inference-{}-test.json",
            # "\\doht": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/transfomer/white/s:True_epos:False/inference-{}-test.json",
            "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        }
    )

    print("State of the art accuracy, title only")
    accuracy_table(
        {
            # "\\knn": "/data/playlist-captioning/p/curated-deezer/algorithm-data/knn/audio/predictions_{}_test.json",
            # "\\muscaps": "/data/playlist-captioning/p/curated-deezer/algorithm-data/manco/save/experiments/2022-02-21-08_45_19/predictions_beam_3_{}_test.json",
            # "\\dohr": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/rnn/white/s:True_epos:False/inference-{}-test.json",
            # "\\doht": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/transfomer/white/s:True_epos:False/inference-{}-test.json",
            "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        },
        "title",
    )

    print("State of the art accuracy, description only")
    accuracy_table(
        {
            # "\\knn": "/data/playlist-captioning/p/curated-deezer/algorithm-data/knn/audio/predictions_{}_test.json",
            # "\\muscaps": "/data/playlist-captioning/p/curated-deezer/algorithm-data/manco/save/experiments/2022-02-21-08_45_19/predictions_beam_3_{}_test.json",
            # "\\dohr": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/rnn/white/s:True_epos:False/inference-{}-test.json",
            # "\\doht": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/transfomer/white/s:True_epos:False/inference-{}-test.json",
            "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        },
        "description",
    )

    print("State of the art diversity")
    diversity_table(
        {
            # "\\knn": "/data/playlist-captioning/p/curated-deezer/algorithm-data/knn/audio/predictions_{}_test.json",
            # "\\muscaps": "/data/playlist-captioning/p/curated-deezer/algorithm-data/manco/save/experiments/2022-02-21-08_45_19/predictions_beam_3_{}_test.json",
            # "\\dohr": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/rnn/white/s:True_epos:False/inference-{}-test.json",
            # "\\doht": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/transfomer/white/s:True_epos:False/inference-{}-test.json",
            "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        }
    )

    print("State of the art diversity, title only")
    diversity_table(
        {
            # "\\knn": "/data/playlist-captioning/p/curated-deezer/algorithm-data/knn/audio/predictions_{}_test.json",
            # "\\muscaps": "/data/playlist-captioning/p/curated-deezer/algorithm-data/manco/save/experiments/2022-02-21-08_45_19/predictions_beam_3_{}_test.json",
            # "\\dohr": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/rnn/white/s:True_epos:False/inference-{}-test.json",
            # "\\doht": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/transfomer/white/s:True_epos:False/inference-{}-test.json",
            "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        },
        "title",
    )

    print("State of the art diversity, description only")
    diversity_table(
        {
            # "\\knn": "/data/playlist-captioning/p/curated-deezer/algorithm-data/knn/audio/predictions_{}_test.json",
            # "\\muscaps": "/data/playlist-captioning/p/curated-deezer/algorithm-data/manco/save/experiments/2022-02-21-08_45_19/predictions_beam_3_{}_test.json",
            # "\\dohr": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/rnn/white/s:True_epos:False/inference-{}-test.json",
            # "\\doht": "/data/playlist-captioning/p/curated-deezer/algorithm-data/doh/exp/transfomer/white/s:True_epos:False/inference-{}-test.json",
            "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        },
        "description",
    )

    print("Ablation study.")
    accuracy_table(
        {
            # "\\onemode": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/audio/{}-test.json",
            # "\\twomodes": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/audio_artist/{}-test.json",
            # "\\threemodes": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        }
    )

    print("Sensitivity analysis")
    accuracy_table(
        {
            # "Random init": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/random/{}-test.json",
            # "\\ours": "/data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/{}-test.json",
        }
    )

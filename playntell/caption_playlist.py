import argparse
import json
import os
import shutil
from collections import Counter
from io import StringIO
from itertools import chain
from os.path import join
from pathlib import Path
import numpy as np

from audio_features.extractor import FeaturesExtractor, load_audio

discogs_tags_embedding_filename = (
    "/data/playlist-captioning/p/discogs_tags_embeddings.npy"
)


def process_tags(tags):

    tags_filtered = []
    for tag in tags:

        # tags pre-processing
        tag = tag.lower()
        tag = "rnb" if tag == "r&b" else tag

        tag = tag.replace("/", " ")
        tag = tag.replace("-", " ")
        tag = tag.replace(",", " ")
        tag = tag.replace(")", " ")
        tag = tag.replace("(", " ")
        tag = tag.replace("&", " ")
        tag = tag.replace("'", " ")
        tag = tag.replace(".", " ")

        l = tag.split()
        for e in l:
            e = e.strip()
            if len(e) > 1:
                tags_filtered.append(e)

    return list(set(tags_filtered))


# def store_tags_embeddings():
#     # prepare all discogs tags embeddings (once for all) and stor them in a file
#     # This is quite useful to avoid loading the very large word2vec model.

#     from gensim.models import KeyedVectors

#     df = pd.read_csv("../data/playlist-captioning/p/all_discogs_tags.csv")
#     discogs_tag = process_tags(df.discogs_tag)
#     w2v = KeyedVectors.load(
#         "/data/nfs/analysis/dafchar/MWE/static/gensim_model/model"
#     ).wv

#     embedding_dict = {}
#     for tag in discogs_tag:
#         try:
#             embedding_dict[tag] = w2v[tag]
#         except KeyError:
#             print(f"Tag '{tag}' has no w2v embedding, skipping ...")
#             continue
#     np.save(discogs_tags_embedding_filename, embedding_dict)


def compute_audio_embeddings(filename, model="MSD_vgg"):
    """
    Compute audio embedding from audio file
    """
    fe = FeaturesExtractor()

    try:
        audio_data = load_audio(filename, duration=30)

        features = fe.extract_features(audio_data, model=model)

        file_id = Path(filename).stem
        return features["pool5"]

    except Exception as err:
        raise err
        print("Audio embedding won't be computed because of the following error:")
        print(err)
        raise ValueError("No features computed")


def compute_playlist_embedding(playlist_description_json, data_path):
    """
    Compute playlist audio embeddings
    """

    with open(playlist_description_json, "r") as f:
        playlist_dict = json.load(f)

    for track in playlist_dict["tracks"]:
        tracks_embeddings = []

        try:
            track_file = join(data_path, track["id"])
            print(track_file)
            track_embeddings = compute_audio_embeddings(track_file)
            track_embeddings = np.reshape(track_embeddings, (10, 256))
            track_embeddings = np.mean(track_embeddings, axis=0)

        except ValueError as err:

            print(f"Issue with computation of embedding of track {track_file}")
            print(err)

            track_embeddings = np.zeros(256)

        tracks_embeddings.append(track_embeddings)

        playlist_embeddings = np.array(tracks_embeddings)
        output_path = join(data_path, "playlists_audio_embedding")
        os.makedirs(output_path, exist_ok=True)
        save_path = join(output_path, f"{playlist_dict['id']}.npy")
        np.save(
            save_path,
            playlist_embeddings,
        )


def prepare_artist_data(playlist_description_json, data_path):
    """
    Prepare artist distribution data
    """

    with open(playlist_description_json, "r") as f:
        playlist_desc = json.load(f)

    artist_counter = Counter([track["artist"] for track in playlist_desc["tracks"]])
    playlist_id = playlist_desc["id"]

    export_path = join(data_path, "playlists_artist_embedding")
    os.makedirs(export_path, exist_ok=True)

    # for playlist_id, artist_data in playlist_dict.items():
    count_list = [cnt for cnt in artist_counter.values()]
    d = sorted([el / sum(count_list) for el in count_list], reverse=True)

    artist_distribution = np.zeros((1, 10), dtype=np.float32)
    artist_distribution[0, : min(10, len(d))] = d[:10]
    export_filename = join(export_path, f"{playlist_id}.npy")
    np.save(export_filename, artist_distribution)

    playlist_dict = {
        playlist_id: [
            {"artist": art, "count": cnt} for art, cnt in artist_counter.most_common()
        ]
    }
    with open(join(data_path, "playlist2artists.json"), "w") as f:
        json.dump(playlist_dict, f, indent=4)


def prepare_tags_data(playlist_description_json, data_path):
    """
    Prepare playlist tags and embed them
    """

    with open(playlist_description_json, "r") as f:
        playlist_desc = json.load(f)

    playlist_id = playlist_desc["id"]
    embedding_dict = np.load(discogs_tags_embedding_filename, allow_pickle=True).item()

    export_path = join(data_path, "playlists_tags_embedding")
    os.makedirs(export_path, exist_ok=True)

    # for playlist_id, tags in playlist_tag_dict.items():

    tags = list(chain(*[track["tags"] for track in playlist_desc["tracks"]]))
    tags = process_tags(tags)

    embeddings = []
    for tag in tags:
        if tag in embedding_dict:
            embeddings.append(embedding_dict[tag])

    output_filename = join(export_path, f"{playlist_id}.npy")
    np.save(output_filename, np.stack(embeddings))

    samples = [{"id": playlist_id}]
    with open(join(data_path, "playlists.json"), "w") as f:
        json.dump(samples, f, indent=4)

    playlist_tag_dict = {playlist_id: tags}
    with open(join(data_path, "playlist2tags.json"), "w") as f:
        json.dump(playlist_tag_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folder_name",
        type=str,
        default="test_playlist",
        help="Folder where intermediary data and output will be stored (note that it will erase previous data)",
    )

    parser.add_argument(
        "playlist_description", type=str, help="json file describing the playlist"
    )

    args = parser.parse_args()

    playlist_description = args.playlist_description
    folder_name = args.folder_name

    data_path = f"/data/playlist-captioning/p/{folder_name}/"

    # prepare data
    prepare_artist_data(playlist_description, data_path)
    prepare_tags_data(playlist_description, data_path)
    compute_playlist_embedding(playlist_description, data_path)

    # release CUDA memory
    from numba import cuda

    device = cuda.get_current_device()
    device.reset()

    os.system(
        f"poetry run python3 playntell/infer.py --inference_dataset_name {folder_name}"
    )

    with open(
        f"/data/playlist-captioning/p/{folder_name}/inference/playntell/file.json",
        "r",
    ) as f:
        print(f.read())

import argparse
import json
import os
import pickle
import random

import numpy as np
import torch

from audio_gpt.data import (DataLoader, InferencePlaylists, PlaylistField,
                            RawField, TextField)
from audio_gpt.models.transformer import (DummyVisualEncoder,
                                          Transformer_audiogpt)
from captions import max_caption_length
from config import preprocessed_data_path
from training_experiments.train_playntell import get_captions

"""
Inference on new dataset of playlists `inference_dataset_name`.

It assumes to have the following files:
- `f"{preprocessed_data_path}/{inference_dataset_name}/samples.json"`: Ids of playlists to be inferred;
- `f"{preprocessed_data_path}/{inference_dataset_name}/playlist2artists.json"`: List of number of authored tracks by one artist in the playlist, and associated artist name, sorted by authored tracks.
- `f"{preprocessed_data_path}/{inference_dataset_name}/playlists_tags_embedding"`,
  `f"{preprocessed_data_path}/{inference_dataset_name}/playlists_audio_embedding"`,
  `f"{preprocessed_data_path}/{inference_dataset_name}/playlists_artist_embedding"`:
        Folders with tags, audio, artist embeddings associated to the playlist ids in the dataset.

It computes the captions with the model trained during `experiment_name`, 
and dumps the camputed captions in `f"{preprocessed_data_path}/{inference_dataset_name}/inference/{experiment_name}/"`

Note: I created a toy example, `inference_dataset_name`==dummy.
"""


def infer(experiment_name, inference_dataset_name):
    # Set-up folders
    model_folder = f"{preprocessed_data_path}/curated-deezer/algorithm-data/playntell/saved_models/{experiment_name}"
    inference_folder = f"{preprocessed_data_path}/{inference_dataset_name}/inference/{experiment_name}/"
    os.makedirs(inference_folder, exist_ok=True)

    # Load trained model parameters.
    params = json.load(open(f"{model_folder}/params.json"))

    # This part is very much copied from `src/playntell.py`.
    # TODO: code can be rewritten to avoid code duplication.

    #  Set-up variables.
    max_caption_len = max_caption_length("curated-deezer")
    playlist_features = params["playlist_feature"].split("_")
    artists_masking = any("artist" in feature for feature in playlist_features)
    srau = False if params["no_srau"] else True
    d_in = []
    for feature in playlist_features:
        if feature == "audio":
            d_in.append(256)
        elif feature == "tags":
            d_in.append(300)
        elif feature == "artist":
            d_in.append(10)

    # Set-up environmental variable.
    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    # Set-up cuda.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")

    # Pipeline for playlists.
    playlist_field = PlaylistField(playlist_features, inference_dataset_name)

    # Pipeline for text.
    text_field = TextField(
        init_token="<?",
        eos_token="<|endoftext|>",
    )

    # End of copied code.

    # Dataset.
    ## Some name contain "image", as this code is adapted from an image captioning code.
    dataset = InferencePlaylists(
        playlist_field,
        text_field,
        inference_dataset_name,
    )
    samples = dataset.get_samples

    # Load vocabulary.
    text_field.vocab = pickle.load(
        open(
            f"{preprocessed_data_path}/curated-deezer/algorithm-data/playntell//vocab_{experiment_name}.pkl",
            "rb",
        )
    )

    # Model.
    model = Transformer_audiogpt(
        text_field.vocab.stoi["<?"],
        DummyVisualEncoder(0, d_in),
        params["gpt_model_type"],
        params["decoder_layer"],
        tau=params["tau"],
        srau=srau,
        n_modes=len(d_in),
    ).to(device)

    # Load saved model.
    fname = f"{model_folder}/best.pth"
    data = torch.load(fname)
    torch.set_rng_state(data["torch_rng_state"])
    torch.cuda.set_rng_state(data["cuda_rng_state"])
    np.random.set_state(data["numpy_rng_state"])
    random.setstate(data["random_rng_state"])
    model.load_state_dict(data["state_dict"], strict=False)
    print(
        "Loading model saved at epoch %d, validation loss %f, and best cider %f"
        % (data["epoch"], data["val_loss"], data["best_cider"])
    )

    dict_dataset = samples.image_dictionary(
        {"image": playlist_field, "text": RawField()}
    )
    dataloader = DataLoader(
        dict_dataset,
        batch_size=1,
        num_workers=params["workers"],
    )

    # Compute predictions.
    _, gen = get_captions(
        model,
        dataloader,
        text_field,
        max_caption_len,
        artists_masking,
        params["formatting"],
        inference_dataset_name,
    )
    gen = [gen[k][0] for k in gen.keys()]

    # Dump predictions
    with open(f"{inference_folder}/file.json", "w") as outfile:
        output = {
            "predictions": gen,
        }
        json.dump(output, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="playntell",
        help="Experiment name that originated the saved model we want to resume.",
    )
    parser.add_argument(
        "--inference_dataset_name",
        type=str,
        default="dummy",
        help="The new dataset where we want to perform inference.",
    )
    args = parser.parse_args()

    infer(args.exp_name, args.inference_dataset_name)

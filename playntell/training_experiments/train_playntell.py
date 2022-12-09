import argparse
import json
import logging
import os
import pickle
import random
import sys

# Append parent direcory to path to load modules correctly
sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import torch
from torch.nn import NLLLoss
from tqdm import tqdm
from transformers import AdamW

from audio_gpt.data import (DataLoader, PlaylistField, Playlists, RawField,
                            TextField)
from audio_gpt.models.transformer import (DummyVisualEncoder,
                                          Transformer_audiogpt)
from captions import (extract_title_description_delimiter_formatting,
                      max_caption_length, preprocess_caption)
from config import preprocessed_data_path
from eval import accuracy, diversity, stamp

"""
Our proposed algorithm, PlayNTell.
The code is based on VisualGPT: https://github.com/Vision-CAIR/VisualGPT.
The predictions produced by PlayNTell, when ran with default parameters, are in: 
../data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/playntell/
"""

device = torch.device("cuda")
gpt2_model_path = "playntell/audio_gpt/data/encoder.json"


def evaluate_loss(model, dataloader, loss_fn, text_field):
    # Validation loss
    model.eval()
    running_loss = 0.0
    with tqdm(
        desc="Epoch %d - validation" % e, unit="it", total=len(dataloader)
    ) as pbar:
        with torch.no_grad():
            for it, (detections, captions) in enumerate(dataloader):

                detections, captions = detections.to(device), captions.to(device)
                out, past, _ = model(detections, captions)
                captions = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                loss = loss_fn(out.view(-1, len(text_field.vocab)), captions.view(-1))
                this_loss = loss.item()
                running_loss += this_loss
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()

    val_loss = running_loss / len(dataloader)
    return val_loss


def get_captions(
    model,
    dataloader,
    text_field,
    max_caption_len,
    artists_masking,
    formatting,
    dataset_name,
):
    import itertools

    if artists_masking:
        with open(
            f"{preprocessed_data_path}/{dataset_name}/playlist2artists.json"
        ) as f:
            playlist2artists = json.load(f)

    model.eval()

    gen = {}
    gts = {}
    with tqdm(desc="Evaluation", unit="it", total=len(dataloader)) as pbar:
        for it, (playlists_embeddings, caps_gt, playlists_ids) in enumerate(
            iter(dataloader)
        ):

            playlists_embeddings = playlists_embeddings.to(device)

            with torch.no_grad():
                out, _, _ = model.beam_search(
                    playlists_embeddings,
                    max_caption_len,
                    text_field.vocab.stoi["<|endoftext|>"],
                    3,
                    out_size=1,
                )

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = " ".join([k for k, g in itertools.groupby(gen_i)])
                gen["%d_%d" % (it, i)] = [
                    gen_i,
                ]
                gts["%d_%d" % (it, i)] = gts_i

            # replace artist tokens with artist names
            if artists_masking:
                for i in range(len(caps_gen)):

                    playlist_id = playlists_ids[i]
                    artists = playlist2artists[playlist_id]

                    for j, artist in enumerate(artists):
                        artist = preprocess_caption(
                            artist["artist"], return_tokens=False
                        )
                        token = f"artist{j}"

                        gen[f"{it}_{i}"] = [gen[f"{it}_{i}"][0].replace(token, artist)]

            for i in range(len(caps_gen)):

                if formatting == "standard":
                    pass
                elif formatting == "title_only":
                    gen[f"{it}_{i}"] = [f"<title> {gen[f'{it}_{i}'][0]}"]
                elif formatting == "description_only":
                    gen[f"{it}_{i}"] = [f"<description> {gen[f'{it}_{i}'][0]}"]
                elif formatting == "delimiter":
                    title, description = extract_title_description_delimiter_formatting(
                        gen[f"{it}_{i}"][0]
                    )
                    gen[f"{it}_{i}"] = [f"<title> {title} <description> {description}"]

            pbar.update()

    return gts, gen


def train_xe(model, dataloader, text_field, gpt_optimizer, dataloader_eval, args):
    # Training with cross-entropy
    model.train()
    running_loss = 0.0
    with tqdm(desc="Epoch %d - train" % e, unit="it", total=len(dataloader)) as pbar:
        for it, (detections, captions) in enumerate(dataloader):

            detections, captions = detections.to(device), captions.to(device)

            out, _, _ = model(detections, captions)

            captions_gt = captions[:, 1:].contiguous()
            out = out[:, :-1].contiguous()

            loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            gpt_optimizer.step()
            gpt_optimizer.zero_grad()

            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

    loss = running_loss / len(dataloader)
    return loss


if __name__ == "__main__":
    # Arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="playntell")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--head", type=int, default=12)
    parser.add_argument("--no_srau", action="store_true")
    parser.add_argument("--random_seed", type=int, default="42")
    parser.add_argument("--gpt_model_type", type=str, default="gpt")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--optimizer_type", type=str, default="adamw")
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--reinforcement_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_layer", type=int, default=12)
    parser.add_argument("--encoder_layer", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--playlist_feature", type=str, default="audio_tags_artist")
    parser.add_argument("--formatting", type=str, default="standard")
    args = parser.parse_args()

    # For the moment, the only training dataset is curated-deezer.
    args.__dict__["dataset_name"] = "curated-deezer"

    # Create folders where to dump logs, models, predictions and data in general.
    data_folder = (
        f"{preprocessed_data_path}/{args.dataset_name}/algorithm-data/playntell/"
    )
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(f"{data_folder}/logs", exist_ok=True)
    args.__dict__["log_file"] = f"{data_folder}/logs/{args.exp_name}"
    saved_models_folder = f"{data_folder}/saved_models/{args.exp_name}"
    os.makedirs(saved_models_folder, exist_ok=True)
    predictions_folder = f"{data_folder}/predictions/{args.exp_name}"
    os.makedirs(predictions_folder, exist_ok=True)

    # Dump params.
    json.dump(args.__dict__, open(f"{saved_models_folder}/params.json", "w"), indent=4)

    # Set-up variables.
    max_caption_len = max_caption_length(args.dataset_name)
    playlist_features = args.playlist_feature.split("_")
    artists_masking = any("artist" in feature for feature in playlist_features)
    srau = False if args.no_srau else True
    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )
    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    d_in = []
    for feature in playlist_features:
        if feature == "audio":
            d_in.append(256)
        elif feature == "tags":
            d_in.append(300)
        elif feature == "artist":
            d_in.append(10)

    # Set-up logging.
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info(args)

    # Set-up environmental variable.
    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    # Fix random seeds.
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set-up cuda.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Pipeline for playlists.
    playlist_field = PlaylistField(playlist_features, args.dataset_name)

    # Pipeline for text.
    text_field = TextField(
        init_token="<?",
        eos_token="<|endoftext|>",
    )

    # Dataset.
    ## Some name contain "image", as this code is adapted from an image captioning code.
    dataset = Playlists(
        playlist_field,
        text_field,
        artists_masking,
        args.formatting,
        args.dataset_name,
    )
    train_dataset, val_dataset, test_dataset = dataset.splits
    train_captions = [e.text for e in train_dataset.examples]
    dict_dataset_train = train_dataset.image_dictionary(
        {"image": playlist_field, "text": RawField()}
    )
    dict_dataset_val = val_dataset.image_dictionary(
        {"image": playlist_field, "text": RawField()}
    )

    # Vocabulary. Not dependent from dataset, but dependent on audio_gpt/data/encoder.json
    if not os.path.isfile(f"{data_folder}/vocab_{args.exp_name}.pkl"):
        print("Building vocabulary")
        text_field.build_GPT_vocab(gpt2_model_path)
        pickle.dump(
            text_field.vocab, open(f"{data_folder}/vocab_{args.exp_name}.pkl", "wb")
        )
    else:
        text_field.vocab = pickle.load(
            open(f"{data_folder}/vocab_{args.exp_name}.pkl", "rb")
        )

    # Model.
    model = Transformer_audiogpt(
        text_field.vocab.stoi["<?"],
        DummyVisualEncoder(0, d_in),
        args.gpt_model_type,
        args.decoder_layer,
        tau=args.tau,
        srau=srau,
        n_modes=len(d_in),
    ).to(device)

    # Optimisation.
    gpt_optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi["+="])

    best_cider = 0.0
    patience = 0
    for e in range(0, 1000000):  # Very large number, relying on early stopping.

        # Create dataloaders.
        dataloader_train = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
        )
        dataloader_val = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        dict_dataloader_val = DataLoader(
            dict_dataset_val,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )

        # Train for one epoch.
        train_loss = train_xe(
            model, dataloader_train, text_field, gpt_optimizer, dataloader_val, args
        )
        # Validation loss
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field)
        # Compute metrics
        gts, gen = get_captions(
            model,
            dict_dataloader_val,
            text_field,
            max_caption_len,
            artists_masking,
            args.formatting,
            args.dataset_name,
        )
        keys = gts.keys()
        gts = [gts[k][0] for k in keys]
        gen = [gen[k][0] for k in keys]
        scores, stds = accuracy(gts, gen)
        div = diversity(train_captions, gen)
        scores = {**scores, **div}
        val_cider = scores["CIDEr"]

        # Print metrics, ground-truth and generated captions
        logging.info("val metrics, current epoch")
        logging.info("\n" + stamp(scores))
        for t, p in zip(gts, gen):
            logging.info(f"\nTrue caption: {t}\nPredicted caption: {p}")

        # Update patience.
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        # If the last model is the best one, save it, and its predictions.
        if best:
            torch.save(
                {
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "random_rng_state": random.getstate(),
                    "epoch": e,
                    "val_loss": val_loss,
                    "val_cider": val_cider,
                    "state_dict": model.state_dict(),
                    "optimizer": gpt_optimizer.state_dict(),
                    "patience": patience,
                    "best_cider": best_cider,
                },
                f"{saved_models_folder}/best.pth",
            )
            with open(
                f"{predictions_folder}/{args.dataset_name}-val.json", "w"
            ) as outfile:
                output = {
                    "predictions": gen,
                    "true_captions": gts,
                }
                json.dump(output, outfile, indent=4)

        # Decide whether to exit train.
        exit_train = False
        if patience == 40:  # Patience is fixed to 40.
            print("patience reached.")
            exit_train = True

        # If exit, compute predictions on test sets.
        if exit_train:
            # Load back the best model
            fname = f"{saved_models_folder}/best.pth"
            data = torch.load(fname)
            model.load_state_dict(data["state_dict"], strict=False)
            print(
                "Resuming from epoch %d, validation loss %f, and best cider %f"
                % (data["epoch"], data["val_loss"], data["best_cider"])
            )

            # Compute predictions on test sets
            for dataset in ["curated-spotify", "curated-deezer"]:
                # Set up dataset.
                playlist_field = PlaylistField(playlist_features, dataset)
                dataset_spotify = Playlists(
                    playlist_field,
                    text_field,
                    artists_masking,
                    args.formatting,
                    dataset,
                )
                _, _, test_dataset = dataset_spotify.splits
                dict_dataset_test_spotify = test_dataset.image_dictionary(
                    {"image": playlist_field, "text": RawField()}
                )
                dict_dataloader_test_spotify = DataLoader(
                    dict_dataset_test_spotify,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                )

                # Compute predictions.
                gts, gen = get_captions(
                    model,
                    dict_dataloader_test_spotify,
                    text_field,
                    max_caption_len,
                    artists_masking,
                    args.formatting,
                    dataset,
                )
                keys = gts.keys()
                gts = [gts[k][0] for k in keys]
                gen = [gen[k][0] for k in keys]

                # Dump predictions
                with open(f"{predictions_folder}/{dataset}-test.json", "w") as outfile:
                    output = {
                        "predictions": gen,
                        "true_captions": gts,
                    }
                    json.dump(output, outfile, indent=4)

            # Actually exit train, interrupting the loop.
            break

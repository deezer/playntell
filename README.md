# PlayNTell

## Setup

Build the docker image and run it in a container while launching an interactive bash session:

```sh
$ make build
$ make run-bash
```

## Run algorithms

### PlayNTell: training & infererence:

To train the model on pre-processed deezer training data:
```sh
$ poetry run python3 playntell/training_experiments/train_playntell.py
```

Note: `playntell` accepts parameters. Two useful ones are:

- `--exp_name`: useful to distinguish the output of different runs. Default: "playntell";
- `--playlist_feature`: music modalities to be used. Default: "audio_tags_artist". E.g. if "audio_artist", only two modalities are used.

Note: `playntell` save its outputs in: ``../data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/``. In particular:

- model is stored as ``../data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/saved_models/{exp_name}/best.pth``;
- inference on test sets is stored in ``../data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/predictions/{exp_name}``;
- log file is store as ``../data/playlist-captioning/p/curated-deezer/algorithm-data/playntell/logs/{exp_name}``


Once the model trained, you can perform inference on preprocessed data (from deezer or spotify playlists) with:
```sh
$ poetry run python3 playntell/infer.py --exp_name playntell --inference_dataset_name curated-deezer
```


## Inference on new playlists:

You can use the playntell model to predict a caption for a playlist with:
```
$ poetry run python3 playntell/caption_playlist.py /data/playlist-captioning/p/test_playlist/playlist.json
```
the playlist.json files has two fields:
- "id" which is used of the playlist (could be any string)
- "tracks" is the list of tracks of the playlist. Each track must have the following field:
    - "id": the filename of the audio file.
    - "artist": the name of the main artist of the song in the audio file
    - "tags": a list of tags (in the discogs taxonomy) describing the track.

A dummy example of a playlist.json and audio files is provided for testing in `/data/playlist-captioning/p/test_playlist/` in the provided docker container




## Acknowledgements

This repo uses code from the following repo, with minor modifications:

* [muscaps](https://github.com/ilaria-manco/muscaps) (muscaps folder)
* [mood_flow_audio_features](https://github.deezerdev.com/rhennequin/mood_flow_audio_features) (audio_features folder)
* [ply_title_get](https://github.com/SeungHeonDoh/ply_title_gen) (ply_title_gen folder)
* [VisualGPT](https://github.com/Vision-CAIR/VisualGPT) (audio_gpt folder)
We took the code from VisualGPT, modified the model code in it and renamed it to AudioGPT

## Paper

This repo contains the code of the paper:
```
@InProceedings{Gabbolini2022,
  title={Bag of Tricks for Efficient Text Classification},
  author={Gabbolini, Giovanni and Hennequin, Romain and Epure, Elena},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month={December},
  year={2022}
}
```

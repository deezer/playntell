# coding: utf8
import os
import shutil
import warnings
from collections import Counter, OrderedDict
from itertools import chain

import h5py
import numpy as np
import six
import torch
from torch.utils.data.dataloader import default_collate
# from .tokenization_gpt2 import GPT2Tokenizer
from transformers import *

from config import preprocessed_data_path
from .dataset import Dataset
from .encoder import get_encoder
from .vocab import Vocab


class RawField(object):
    """Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """Preprocess an example if the `preprocessing` Pipeline is provided."""
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [
                batch,
            ]
        else:
            batch = list(zip(*batch))

        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out


class ImageDetectionsField(RawField):
    def __init__(
        self,
        preprocessing=None,
        postprocessing=None,
        detections_path=None,
        max_detections=100,
        sort_by_prob=False,
        load_in_tmp=True,
    ):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob

        tmp_detections_path = os.path.join("/tmp", os.path.basename(detections_path))

        if load_in_tmp:
            if not os.path.isfile(tmp_detections_path):
                if shutil.disk_usage("/tmp")[-1] < os.path.getsize(detections_path):
                    warnings.warn(
                        "Loading from %s, because /tmp has no enough space."
                        % detections_path
                    )
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
                    self.detections_path = tmp_detections_path
            else:
                self.detections_path = tmp_detections_path

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split("_")[-1].split(".")[0])

        try:
            f = h5py.File(self.detections_path, "r")
            precomp_data = f["%d_features" % image_id][()]
            if self.sort_by_prob:
                precomp_data = precomp_data[
                    np.argsort(np.max(f["%d_cls_prob" % image_id][()], -1))[::-1]
                ]
        except KeyError:
            warnings.warn("Could not find detections for %d" % image_id)
            precomp_data = np.random.rand(10, 2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate(
                [precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0
            )
        elif delta < 0:
            precomp_data = precomp_data[: self.max_detections]

        return precomp_data.astype(np.float32)


class PlaylistField(RawField):
    def __init__(self, embedding_types, dataset_name):
        def pad(data):
            # Pad all embeddings types of playlists in batch,
            # to maximum length of all embedding types

            max_embedding_rows = max(
                embedding.shape[0] for playlist in data for embedding in playlist
            )

            return_value = []
            for playlist_embeddings in data:
                padded_playlist_embeddings = [
                    np.pad(
                        embedding,
                        [(0, max_embedding_rows - embedding.shape[0]), (0, 0)],
                        mode="constant",
                    )
                    for embedding in playlist_embeddings
                ]
                padded_playlist_embeddings = np.concatenate(
                    padded_playlist_embeddings, axis=1
                )
                return_value.append(padded_playlist_embeddings)

            return_value = tuple(return_value)
            return return_value

        preprocessing = None
        postprocessing = pad
        self.embedding_types = embedding_types
        self.dataset_name = dataset_name
        assert all(
            e
            in [
                "audio",
                "tags",
                "artist",
            ]
            for e in embedding_types
        )
        super(PlaylistField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, playlist_id):
        embeddings = [
            np.load(
                f"{preprocessed_data_path}/{self.dataset_name}/playlists_{t}_embedding/{playlist_id}.npy",
                allow_pickle=True,
            ).astype(np.float32)
            for t in self.embedding_types
        ]

        return embeddings


class TextField(RawField):
    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,
        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    def __init__(
        self,
        use_vocab=True,
        init_token=None,
        eos_token=None,
        fix_length=None,
        dtype=torch.long,
        preprocessing=None,
        postprocessing=None,
        include_lengths=False,
        batch_first=True,
        pad_token="+=",
        unk_token="<|endoftext|>",
        pad_first=False,
        truncate_first=False,
        vectors=None,
    ):
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        self.vocab = None
        self.vectors = vectors

        # self.encoder = get_encoder()

        self.encoder = GPT2Tokenizer.from_pretrained(
            "gpt2", TOKENIZERS_PARALLELISM=False
        )

        super(TextField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x):
        x = x.split(" ")
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):

        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def build_vocab(self, *args, **kwargs):
        counter = Counter()
        sources = []
        for arg in args:
            if isinstance(arg, Dataset):
                sources += [
                    getattr(arg, name)
                    for name, field in arg.fields.items()
                    if field is self
                ]
            else:
                sources.append(arg)

        for data in sources:
            for x in data:
                x = self.preprocess(x)
                try:
                    counter.update(x)
                except TypeError:
                    counter.update(chain.from_iterable(x))

        specials = list(
            OrderedDict.fromkeys(
                [
                    tok
                    for tok in [
                        self.unk_token,
                        self.pad_token,
                        self.init_token,
                        self.eos_token,
                    ]
                    if tok is not None
                ]
            )
        )
        self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

    def build_GPT_vocab(self, data_file):
        import json

        counter = Counter()
        with open(data_file, "r") as f:
            data = json.load(f)

        for key in data.keys():
            counter.update([key])

        specials = list(
            OrderedDict.fromkeys(
                [
                    tok
                    for tok in [
                        self.unk_token,
                        self.pad_token,
                        self.init_token,
                        self.eos_token,
                    ]
                    if tok is not None
                ]
            )
        )
        self.vocab = self.vocab_cls(counter, specials=specials, json_file=data_file)

    def pad(self, minibatch):
        """Pad a batch of examples using this field.
        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just
        returns the padded list.
        """
        minibatch = list(minibatch)
        if self.fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = (
                self.fix_length + (self.init_token, self.eos_token).count(None) - 2
            )
        padded, lengths = [], []
        for x in minibatch:
            if self.pad_first:
                padded.append(
                    [self.pad_token] * max(0, max_len - len(x))
                    + ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                )
            else:
                padded.append(
                    ([] if self.init_token is None else [self.init_token])
                    + list(x[-max_len:] if self.truncate_first else x[:max_len])
                    + ([] if self.eos_token is None else [self.eos_token])
                    + [self.pad_token] * max(0, max_len - len(x))
                )
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self.include_lengths:
            return padded, lengths
        return padded

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a list of Variables.
        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.
        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError(
                "Field has include_lengths set to True, but "
                "input data is not a tuple of "
                "(data batch, batch lengths)."
            )
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

        if self.use_vocab:
            # for ex in arr:
            #     print(ex)

            arr = [
                self.encoder.encode(
                    " ".join(ex)
                    .replace(" <|endoftext|>", "<|endoftext|>")
                    .replace(" +=", "+=")
                )[: len(ex)]
                for ex in arr
            ]
            # for ex in arr:
            #     print(ex)

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)

            var = torch.tensor(arr, dtype=self.dtype, device=device)
        else:
            if self.vectors:
                arr = [[self.vectors[x] for x in ex] for ex in arr]
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype)
                )
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explictly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            arr = [
                numericalization_func(x) if isinstance(x, six.string_types) else x
                for x in arr
            ]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

            var = torch.cat(
                [torch.cat([a.unsqueeze(0) for a in ar]).unsqueeze(0) for ar in arr]
            )

        # var = torch.tensor(arr, dtype=self.dtype, device=device)
        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        if self.include_lengths:
            return var, lengths
        return var

    def decode(self, word_idxs, join_words=True):
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode(
                [
                    word_idxs,
                ],
                join_words,
            )[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)), join_words)[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0), join_words)[0]

        captions = []

        i = 0
        for wis in word_idxs:
            caption = []
            current_wis = self.encoder.decode(wis)

            for token in current_wis.split():
                if "<|endoftext|>" in token or "+=" in token:
                    token = token.split("<|endoftext|>")[0]
                    caption.append(token)
                    break
                else:
                    caption.append(token)

            if join_words:
                caption = " ".join(caption)

            captions.append(caption)

        return captions

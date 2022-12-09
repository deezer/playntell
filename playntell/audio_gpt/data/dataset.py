import collections
import itertools
import json
import os

from pycocotools.coco import COCO as pyCOCO
import numpy as np
import torch
from captions import extract_title_description, preprocess_caption
from config import preprocessed_data_path, raw_data_path
from .example import Example
from .utils import nostdout


class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [
                    batch,
                ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(
                    isinstance(t, torch.Tensor) for t in tensor
                ):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)

            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))

        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(
                value_batch_flattened
            )

            lengths = [
                0,
            ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) and any(
                isinstance(t, torch.Tensor) for t in value_tensors_flattened
            ):
                value_tensors = [
                    [vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]
                    for vt in value_tensors_flattened
                ]
            else:
                value_tensors = [
                    value_tensors_flattened[s:e]
                    for (s, e) in zip(lengths[:-1], lengths[1:])
                ]

            return value_tensors

        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert field in fields

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch, indices = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors, indices

        return collate

    def __getitem__(self, i):
        return (
            self.key_dataset[i],
            self.value_dataset[i],
            self.key_dataset.examples[i].image,
        )

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert "image" in fields
        assert "text" in fields
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields["image"]
        self.text_field = self.fields["text"]

    def image_set(self):
        img_list = [e.image for e in self.examples]
        image_set = unique(img_list)
        examples = [Example.fromdict({"image": i}) for i in image_set]
        dataset = Dataset(examples, {"image": self.image_field})
        return dataset

    def text_set(self):
        text_list = [e.text for e in self.examples]
        text_list = unique(text_list)
        examples = [Example.fromdict({"text": t}) for t in text_list]
        dataset = Dataset(examples, {"text": self.text_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields="image")
        return dataset

    def text_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields="text")
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


class COCO(PairedDataset):
    def __init__(
        self,
        image_field,
        text_field,
        img_root,
        ann_root,
        id_root=None,
        use_restval=True,
        cut_validation=False,
        train_percentage=1,
        split_train_data=False,
    ):
        roots = {}
        roots["train"] = {
            "img": os.path.join(img_root, "train2014"),
            "cap": os.path.join(ann_root, "captions_train2014.json"),
        }
        roots["val"] = {
            "img": os.path.join(img_root, "val2014"),
            "cap": os.path.join(ann_root, "captions_val2014.json"),
        }
        roots["test"] = {
            "img": os.path.join(img_root, "val2014"),
            "cap": os.path.join(ann_root, "captions_val2014.json"),
        }
        roots["trainrestval"] = {
            "img": (roots["train"]["img"], roots["val"]["img"]),
            "cap": (roots["train"]["cap"], roots["val"]["cap"]),
        }

        if id_root is not None:
            ids = {}
            ids["train"] = np.load(os.path.join(id_root, "coco_train_ids.npy"))

            ids["val"] = np.load(os.path.join(id_root, "coco_dev_ids.npy"))
            if cut_validation:
                ids["val"] = ids["val"][:5000]
            ids["test"] = np.load(os.path.join(id_root, "coco_test_ids.npy"))

            coco_restval_ids = np.load(os.path.join(id_root, "coco_restval_ids.npy"))
            if split_train_data:
                np.random.shuffle(ids["train"])
                np.random.shuffle(coco_restval_ids)
                ids["train"] = ids["train"][: int(len(ids["train"]) * train_percentage)]
                coco_restval_ids = coco_restval_ids[
                    : int(len(coco_restval_ids) * train_percentage)
                ]

            ids["trainrestval"] = (ids["train"], coco_restval_ids)

            if use_restval:
                roots["train"] = roots["trainrestval"]
                ids["train"] = ids["trainrestval"]

        else:
            ids = None

        with nostdout():
            (
                self.train_examples,
                self.val_examples,
                self.test_examples,
            ) = self.get_samples(roots, ids)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(COCO, self).__init__(examples, {"image": image_field, "text": text_field})

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, roots, ids_dataset=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ["train", "val", "test"]:
            if isinstance(roots[split]["cap"], tuple):
                coco_dataset = (
                    pyCOCO(roots[split]["cap"][0]),
                    pyCOCO(roots[split]["cap"][1]),
                )
                root = roots[split]["img"]
            else:
                coco_dataset = (pyCOCO(roots[split]["cap"]),)
                root = (roots[split]["img"],)

            if ids_dataset is None:
                ids = list(coco_dataset.anns.keys())
            else:
                ids = ids_dataset[split]

            if isinstance(ids, tuple):
                bp = len(ids[0])
                ids = list(ids[0]) + list(ids[1])
            else:
                bp = len(ids)

            for index in range(len(ids)):
                if index < bp:
                    coco = coco_dataset[0]
                    img_root = root[0]
                else:
                    coco = coco_dataset[1]
                    img_root = root[1]

                ann_id = ids[index]
                caption = coco.anns[ann_id]["caption"]
                img_id = coco.anns[ann_id]["image_id"]
                filename = coco.loadImgs(img_id)[0]["file_name"]

                example = Example.fromdict(
                    {"image": os.path.join(img_root, filename), "text": caption}
                )

                if split == "train":
                    train_samples.append(example)
                elif split == "val":
                    val_samples.append(example)
                elif split == "test":
                    test_samples.append(example)

        return train_samples, val_samples, test_samples


class InferencePlaylists(PairedDataset):

    """
    Ad-hoc dataset object for inference only, it doesn't require to know the captions, but just the playlist ids.
    This object relies on existing code, bridging the interfaces with a dummy caption: "to be inferred".
    """

    def __init__(self, playlist_field, text_field, dataset_name):
        with nostdout():
            self.samples = self.load(dataset_name)
        super(InferencePlaylists, self).__init__(
            self.sample, {"image": playlist_field, "text": text_field}
        )

    @property
    def get_samples(self):
        samples = PairedDataset(self.samples, self.fields)
        return samples

    @classmethod
    def load(cls, dataset_name):
        samples = []

        l = json.load(open(f"{preprocessed_data_path}/{dataset_name}/playlists.json"))

        for sample in l:
            playlist_id = sample["id"]
            caption = "to be inferred"

            example = Example.fromdict({"image": playlist_id, "text": caption})

            samples.append(example)

        return samples


class Playlists(PairedDataset):
    def __init__(
        self, playlist_field, text_field, artists_masking, formatting, dataset_name
    ):
        with nostdout():
            (
                self.train_examples,
                self.val_examples,
                self.test_examples,
            ) = self.get_samples(artists_masking, formatting, dataset_name)
        examples = self.train_examples + self.val_examples + self.test_examples
        super(Playlists, self).__init__(
            examples, {"image": playlist_field, "text": text_field}
        )

    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def get_samples(cls, artists_masking, formatting, dataset_name):
        train_samples = []
        val_samples = []
        test_samples = []

        for split in ["train", "val", "test"]:

            with open(
                f"{preprocessed_data_path}/{dataset_name}/splits/{split}.json"
            ) as f:
                l = json.load(f)

            if artists_masking:
                with open(
                    f"{preprocessed_data_path}/{dataset_name}/playlist2artists.json"
                ) as f:
                    playlist2artists = json.load(f)

            for sample in l:

                playlist_id = sample["playlist_id"]
                caption = sample["caption"]

                # alter caption based on different formatting strategies
                if split == "train":

                    if formatting == "standard":
                        pass
                    elif formatting == "title_only":
                        title, _ = extract_title_description(caption)
                        caption = title
                    elif formatting == "description_only":
                        _, description = extract_title_description(caption)
                        caption = description
                    elif formatting == "delimiter":
                        title, description = extract_title_description(caption)
                        caption = f"{title} <delimiter> {description}"
                    else:
                        raise ValueError(f"Bad formatting value: {formatting}")

                # Strategy for incorporating external info i.e. artist names
                if split == "train" and artists_masking:

                    artists = playlist2artists[playlist_id]

                    for i, artist in enumerate(artists):

                        artist = preprocess_caption(
                            artist["artist"], return_tokens=False
                        )
                        token = f"artist{i}"
                        caption = caption.replace(artist, token)

                example = Example.fromdict({"image": playlist_id, "text": caption})

                if split == "train":
                    train_samples.append(example)
                elif split == "val":
                    val_samples.append(example)
                elif split == "test":
                    test_samples.append(example)

        return train_samples, val_samples, test_samples

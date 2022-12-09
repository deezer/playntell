from torch.utils.data import DataLoader as TorchDataLoader

from .dataset import COCO, InferencePlaylists, Playlists
from .field import (ImageDetectionsField, Merge, PlaylistField, RawField,
                    TextField)


class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(
            dataset, *args, collate_fn=dataset.collate_fn(), **kwargs
        )

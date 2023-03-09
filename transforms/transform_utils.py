from typing import Mapping

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import generate_spatial_bounding_box
from monai.transforms.transform import MapTransform, Transform
from monai.utils import (
    convert_to_tensor,
)
from monai.utils.enums import TransformBackends


class LabelToBinaryLabel(Transform):
    """
    Convert labels to mask for other tasks.
    inp label [1, w, h, d],  channel have lbl. ex. [0, 1, 2]
    covert to binary label like [1, w, h, d], each channel only [0, 1] .
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor):
        img = convert_to_tensor(img, track_meta=get_track_meta())

        img[img>0] = 1
        
        return img.to(torch.uint8)


class LabelToBinaryLabeld(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.LabelToBinaryLabeld`.
    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
    """
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = LabelToBinaryLabel()

    def _convert(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            mask = self.converter(d[key])
            d[key] = mask
        return d

    def __call__(self, data):
        if isinstance(data, Mapping):
            return self._convert(data)
        else:
            return [self._convert(d) for d in data]

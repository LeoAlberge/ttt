from typing import Any, Optional, Tuple

import numpy as np
import torch
from autologging import logged

from src.python.core.volume import TTTVolume
from src.python.preprocessing.interpolation.utils import interpolate_to_fixed_size
from src.python.preprocessing.transform import TransformBlock


@logged
class PatchExtractor(TransformBlock):
    def __init__(self,
                 patch_size: tuple[int, int, int] = (96, 96, 96),
                 padding_value: float = 0
                 ):
        self._patch_size = patch_size
        self._padding_value = padding_value

    def _unfold_patches(self, data: torch.Tensor, origin_shape: torch.Size):
        return data.view(origin_shape)

    def _generate_patches(self, data: torch.Tensor):
        d, h, w = data.shape
        pd, ph, pw = self._patch_size

        num_patches_d = (d + pd - 1) // pd
        num_patches_h = (h + ph - 1) // ph
        num_patches_w = (w + pw - 1) // pw
        self.__log.info(
            f"input shape: {(d, h, w)}, nb patches: "
            f"{(num_patches_d, num_patches_h, num_patches_w)}")
        pad_h = ph - (h % ph)
        pad_w = pw - (w % pw)
        pad_d = pd - (d % pd)
        padding = (
            pad_w // 2,
            pad_w - pad_w // 2,
            pad_h // 2,
            pad_h - pad_h // 2,
            pad_d // 2,
            pad_d - pad_d // 2,
        )
        tensor = torch.nn.functional.pad(data, padding, mode='constant', value=self._padding_value)
        patches = tensor.unfold(0, pd, pd).unfold(1, ph, ph).unfold(2, pw, pw)
        patches = patches.contiguous().view(-1, pd, ph, pw)
        return patches

    def __call__(self, data: torch.Tensor, seg: torch.Tensor):
        return self._generate_patches(data), self._generate_patches(seg)


class SegmentationOneHotEncoding(TransformBlock):
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor):
        """

        Args:
            inputs:
            target: shape: D, H, W volume

        Returns: segmentation shape: D,H,W, C where C is number of classes

        """
        return (inputs,
                torch.nn.functional.one_hot(target.to(torch.int64),
                                            num_classes=self.num_classes)[0, :].permute(3, 0,
                                                                                        1, 2))


class CropOnSegmentation(TransformBlock):

    def __init__(self, margin: int = 5):
        self._margin = margin

    def __call__(self, inputs: np.ndarray, target: np.ndarray):
        positive_indices = target > 0
        indices = np.where(positive_indices)
        min_indices = np.min(indices, axis=1)
        max_indices = np.max(indices, axis=1)

        start_indices = np.maximum(np.zeros_like(min_indices), min_indices - self._margin)
        end_indices = np.minimum(target.shape, max_indices + self._margin + 1)

        cropped_segmentation = target[start_indices[0]:end_indices[0],
                               start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
        cropped_volume = inputs[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1],
                         start_indices[2]:end_indices[2]]
        return cropped_volume, cropped_segmentation


class VolumeReshaper:

    def __init__(self, target_size: Tuple[int, int, int]):
        self._target_size = target_size

    def __call__(self, inputs: np.ndarray, target: np.ndarray):
        inputs = interpolate_to_fixed_size(TTTVolume(inputs, np.zeros(3), np.ones(3), np.eye(3)),
                                           np.array(self._target_size)).data
        target = interpolate_to_fixed_size(TTTVolume(target, np.zeros(3), np.ones(3), np.eye(3)),
                                           np.array(self._target_size),
                                           method="nearest_neighbor").data
        return inputs, target


class VolumeNormalization(TransformBlock):

    def __init__(self,
                 clipping: Tuple[Optional[float], Optional[float]] = (-1000, 1000),
                 normalization: str = "unit"):
        self._clipping = clipping
        self._normalization = normalization

    def __call__(self, inputs: Any, target: Any):

        inputs = np.clip(inputs, self._clipping[0], self._clipping[1])
        if self._normalization == "unit":
            m, sd = inputs.mean(), inputs.std()
            sd = sd + 1e-5
            inputs = (inputs - m) / sd
        elif self._normalization == "none":
            pass
        else:
            raise NotImplementedError(f"{self._normalization} normalization is not implemented")
        return inputs, target


if __name__ == '__main__':
    # ds = TotalSegmentatorDataSet(
    #     r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201")
    x = torch.tensor(np.random.rand(152, 143, 541))
    x_p = PatchExtractor()._generate_patches(x)
    x_up = PatchExtractor()._unfold_patches(x_p, x.shape)
    np.testing.assert_allclose(x.numpy(), x_up.numpy())

    # print(x.shape, y.shape)
    #
    # y = SegmentationOneHotEncoding()(torch.tensor(np.zeros((152, 143, 541))))
    # print(y.shape)

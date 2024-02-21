from copy import deepcopy
from itertools import permutations
from typing import Any, Optional, Tuple

import numpy as np
import torch

from src.python.core.benchmarks import timeit
from src.python.core.volume import TTTVolume
from ttt_rs import trinilear_interpolation

from src.python.preprocessing.transform import TransformBlock


def interpolate_trilinearly(
        in_vol: TTTVolume,
        out_vol: TTTVolume,
        out_val: float = 0
):
    trinilear_interpolation(
        in_vol.data,
        tuple(in_vol.spacing.flatten()),
        tuple(in_vol.origin_lps.flatten()),
        tuple(in_vol.matrix_ijk_2_lps.flatten()),
        out_vol.data,
        tuple(out_vol.spacing.flatten()),
        tuple(out_vol.origin_lps.flatten()),
        tuple(out_vol.matrix_ijk_2_lps.flatten()),
        out_val

    )


@timeit("interpolate_to_target_spacing")
def interpolate_to_target_spacing(in_vol: TTTVolume, target_spacing: np.ndarray,
                                  out_val=0) -> TTTVolume:
    extent = in_vol.spacing * in_vol.data.shape[::-1]

    target_dimension = np.round(extent / target_spacing).astype(np.int32)
    new_spacing = extent / target_dimension
    out_vol = TTTVolume(
        data=np.zeros((target_dimension[::-1]), dtype=np.float32),
        spacing=new_spacing,
        origin_lps=deepcopy(in_vol.origin_lps),
        matrix_ijk_2_lps=deepcopy(in_vol.matrix_ijk_2_lps)
    )
    interpolate_trilinearly(in_vol, out_vol, out_val)
    return out_vol


def permute_to_identity_matrix(in_vol: TTTVolume) -> TTTVolume:
    permutation_list = list(permutations([0, 1, 2]))
    frobenius_distances = [np.linalg.norm(np.eye(3) - np.abs(in_vol.matrix_ijk_2_lps[:, p])) for p
                           in permutation_list]
    argmin_perm = int(np.argmin(frobenius_distances))
    perm = permutation_list[argmin_perm]
    axis_perm_mat = np.eye(3)[:, permutation_list[argmin_perm]]
    signed_perm_mat = np.diag(np.sign(np.diag(np.dot(in_vol.matrix_ijk_2_lps, axis_perm_mat))))
    flippers = np.array(np.where(np.diag(signed_perm_mat) < 0)).flatten().tolist()

    matrix_ijk_to_lps = np.dot(np.dot(in_vol.matrix_ijk_2_lps, axis_perm_mat), signed_perm_mat)
    return TTTVolume(
        data=np.flip(in_vol.data.transpose(perm), flippers).copy('C'),
        origin_lps=in_vol.origin_lps,
        spacing=in_vol.spacing[np.array(perm)],
        matrix_ijk_2_lps=matrix_ijk_to_lps

    )


class PatchExtractor(TransformBlock):
    def __init__(self, patch_size: tuple[int, int, int] = (96, 96, 96)):
        self._patch_size = patch_size

    def _generate_patches(self, data: torch.Tensor):
        d, h, w = data.shape
        pd, ph, pw = self._patch_size

        num_patches_d = (d + pd - 1) // pd
        num_patches_h = (h + ph - 1) // ph
        num_patches_w = (w + pw - 1) // pw

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
        tensor = torch.nn.functional.pad(data, padding, mode='constant', value=0)
        patches = tensor.unfold(0, pd, pd).unfold(1, ph, ph).unfold(2, pw, pw)
        patches = patches.contiguous().view(-1, pd, ph, pw)
        return patches

    def __call__(self, data: torch.Tensor, seg: torch.Tensor):
        return self._generate_patches(data), self._generate_patches(seg)


class SegmentationOneHotEncoding(TransformBlock):
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes

    def __call__(self, inputs: torch.Tensor, target: torch.Tensor):
        return inputs, torch.nn.functional.one_hot(target.to(torch.int64),
                                                   num_classes=self.num_classes)


class VolumeNormalization(TransformBlock):

    def __init__(self, clipping: Tuple[Optional[float], Optional[float]] = (-1000, 1000),
                 normalization: str = "unit"):
        self._clipping = clipping
        self._normalization = normalization

    def __call__(self, inputs: Any, target: Any):

        target = np.clip(target, self._clipping[0], self._clipping[1])
        if self._normalization == "unit":
            m, sd = inputs.mean(), inputs.std()
            sd = sd + 1e-5
            target = (target - m) / sd
        elif self._normalization == "none":
            pass
        else:
            raise NotImplementedError(f"{self._normalization} normalization is not implemented")
        return inputs, target


if __name__ == '__main__':
    # ds = TotalSegmentatorDataSet(
    #     r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201")

    # x, y = PatchExtractor()(torch.tensor(np.zeros((152, 143, 541))),
    #                         torch.tensor(np.zeros((152, 143, 541))))
    # print(x.shape, y.shape)
    #
    y = SegmentationOneHotEncoding()(torch.tensor(np.zeros((152, 143, 541))))
    print(y.shape)

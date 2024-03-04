from copy import deepcopy
from itertools import permutations

import numpy as np
from ttt_rs import trinilear_interpolation, neareast_neighbor_interpolation

from src.python.core.benchmarks import timeit
from src.python.core.volume import TTTVolume

def interpolate_trilinearly(
        in_vol: TTTVolume,
        out_vol: TTTVolume,
        out_val: float = 0
):
    trinilear_interpolation(
        in_vol.data.astype(np.float32, copy=False),
        tuple(in_vol.spacing.flatten()),
        tuple(in_vol.origin_lps.flatten()),
        tuple(in_vol.matrix_ijk_2_lps.flatten()),
        out_vol.data,
        tuple(out_vol.spacing.flatten()),
        tuple(out_vol.origin_lps.flatten()),
        tuple(out_vol.matrix_ijk_2_lps.flatten()),
        out_val

    )


def interpolate_nearest_neighbor(
        in_vol: TTTVolume,
        out_vol: TTTVolume,
        out_val: float = 0
):
    neareast_neighbor_interpolation(
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

@timeit("interpolate_to_target_spacing")
def interpolate_to_target_spacing(in_vol: TTTVolume, target_spacing: np.ndarray,
                                  out_val=0, method: str = "trilinear") -> TTTVolume:
    extent = in_vol.spacing * in_vol.data.shape[::-1]

    target_dimension = np.round(extent / target_spacing).astype(np.int32)
    new_spacing = extent / target_dimension

    if method == "trilinear":
        out_vol = TTTVolume(
            data=np.zeros((target_dimension[::-1]), dtype=np.float32),
            spacing=new_spacing,
            origin_lps=deepcopy(in_vol.origin_lps),
            matrix_ijk_2_lps=deepcopy(in_vol.matrix_ijk_2_lps)
        )
        interpolate_trilinearly(in_vol, out_vol, out_val)
    elif method == "nearest_neighbor":
        out_vol = TTTVolume(
            data=np.zeros((target_dimension[::-1]), dtype=np.uint8),
            spacing=new_spacing,
            origin_lps=deepcopy(in_vol.origin_lps),
            matrix_ijk_2_lps=deepcopy(in_vol.matrix_ijk_2_lps)
        )
        interpolate_nearest_neighbor(in_vol, out_vol, out_val)
    else:
        raise NotImplementedError(f"No {method} implemented")
    return out_vol


@timeit("interpolate_to_fixed_size")
def interpolate_to_fixed_size(in_vol: TTTVolume,
                              target_dimension: np.ndarray,
                              out_val=0, method: str = "trilinear") -> TTTVolume:
    extent = in_vol.spacing * in_vol.data.shape[::-1]
    new_spacing = extent / target_dimension

    if method == "trilinear":
        out_vol = TTTVolume(
            data=np.zeros((target_dimension[::-1]), dtype=np.float32),
            spacing=new_spacing,
            origin_lps=deepcopy(in_vol.origin_lps),
            matrix_ijk_2_lps=deepcopy(in_vol.matrix_ijk_2_lps)
        )
        interpolate_trilinearly(in_vol, out_vol, out_val)
    elif method == "nearest_neighbor":
        out_vol = TTTVolume(
            data=np.zeros((target_dimension[::-1]), dtype=np.uint8),
            spacing=new_spacing,
            origin_lps=deepcopy(in_vol.origin_lps),
            matrix_ijk_2_lps=deepcopy(in_vol.matrix_ijk_2_lps)
        )
        interpolate_nearest_neighbor(in_vol, out_vol, out_val)
    else:
        raise NotImplementedError(f"No {method} implemented")
    return out_vol



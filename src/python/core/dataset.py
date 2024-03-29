import os
from copy import deepcopy
from typing import Tuple, Optional, Callable, Dict, Any

import h5py
import numpy as np
from autologging import logged
from torch.utils.data import Dataset

from src.python.core.benchmarks import timeit
from src.python.core.volume import TTTVolume
from src.python.io.niifty_readers import read_nii
from src.python.preprocessing.interpolation.utils import interpolate_to_target_spacing, \
    permute_to_identity_matrix

TOTAL_SEG_CLASS_ID_TO_LABELS = {
    -1: "foreground",
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_right_4",
    93: "rib_right_3",
    94: "rib_left_1",
    95: "rib_left_2",
    96: "rib_left_3",
    97: "rib_left_4",
    98: "rib_left_5",
    99: "rib_left_6",
    100: "rib_left_7",
    101: "rib_left_8",
    102: "rib_left_9",
    103: "rib_left_10",
    104: "rib_left_11",
    105: "rib_left_12",
    106: "rib_right_1",
    107: "rib_right_2",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages"
}
TOTAL_SEG_LABELS_TO_CLASS_ID = {v: k for k, v in TOTAL_SEG_CLASS_ID_TO_LABELS.items()}


@logged
class TotalSegmentatorDataSet(Dataset):

    def __init__(self,
                 data_root_dir: str,
                 reshape_to_identity: bool = True,
                 target_spacing: Optional[Tuple[float, float, float]] = (1, 1, 1),
                 sub_classes: Dict[str, int] = None,
                 transform: Optional[Callable] = None,
                 size: Optional[int] = None):
        super().__init__()
        self._transform = transform
        self._data_root_dir = data_root_dir
        self._indexes = sorted([x for x in os.listdir(data_root_dir) if
                                os.path.isdir(os.path.join(self._data_root_dir, x))])

        self._reshape_to_identity = reshape_to_identity
        self._target_spacing = target_spacing

        self._sub_classes = sub_classes
        self._size = size

    def __len__(self):
        return len(self._indexes) if self._size is None else self._size

    def _get_indexed_item(self, index) -> Tuple[str, Any, Any]:
        if index == len(self):
            raise StopIteration
        index_name = self._indexes[index]
        try:
            dir = os.path.join(self._data_root_dir, index_name)
            ct = read_nii(os.path.join(dir, "ct.nii.gz"))
            seg = TTTVolume(np.zeros_like(ct.data,
                                          dtype=np.uint8),
                            spacing=deepcopy(ct.spacing),
                            origin_lps=deepcopy(ct.origin_lps),
                            matrix_ijk_2_lps=deepcopy(ct.matrix_ijk_2_lps))
            classes = self._sub_classes if self._sub_classes is not None else (
                TOTAL_SEG_LABELS_TO_CLASS_ID)
            for c, class_id in classes.items():
                seg_file = os.path.join(dir, "segmentations", f"{c}.nii.gz")
                if os.path.isfile(os.path.join(dir, "segmentations", f"{c}.nii.gz")):
                    tmp_seg = read_nii(seg_file)
                    seg.data[np.where(tmp_seg.data == 1)] = class_id

            if self._reshape_to_identity:
                ct = permute_to_identity_matrix(ct)
                seg = permute_to_identity_matrix(seg)
            if self._target_spacing is not None:
                ct = interpolate_to_target_spacing(ct, np.array(self._target_spacing))
                seg = interpolate_to_target_spacing(seg, np.array(self._target_spacing),
                                                    method="nearest_neighbor")

            if self._transform is not None:
                ct.data, seg.data = self._transform(ct.data, seg.data)
            return index_name, ct, seg
        except Exception as e:
            self.__log.error(f"error:{e} for index: {index_name}")
            return index_name, None, None

    def indexed_iter(self):
        for index in range(len(self)):
            yield self._get_indexed_item(index)

    @timeit("TotalSegmentatorDataSet.get_item")
    def __getitem__(self, index) -> Tuple[Any, Any]:
        index_name, ct, seg = self._get_indexed_item(index)
        if ct is not None:
            return ct.data, seg.data
        else:
            return None, None


class H5Dataset(Dataset):
    def __init__(self, h5_path: str, transform: Optional[Callable] = None, ):
        self._h5_file = h5py.File(h5_path)
        self._indexes = list(self._h5_file["inputs"])
        self._transform = transform

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        inputs, targets = self._h5_file["inputs"][self._indexes[index]][()], \
            self._h5_file["targets"][self._indexes[index]][()]
        if self._transform is not None:
            return self._transform(inputs, targets)
        return inputs, targets

    @property
    def indexes(self):
        return self._indexes


BTCV_CLASS_ID_TO_LABELS = {
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior vena cava",
    10: "portal vein and splenic vein",
    11: "pancreas",
    12: "right adrenal gland",
    13: "left adrenal gland",
    14: "duodenum",
}

BTCV_LABELS_TO_CLASS_ID = {v: k for k, v in BTCV_CLASS_ID_TO_LABELS.items()}


# TODO: Maybe re-factorisable with the total seg dataset
class BTCVDataset(Dataset):
    def __init__(self,
                 data_root_dir: str,
                 reshape_to_identity: bool = True,
                 target_spacing: Optional[Tuple[float, float, float]] = (1, 1, 1),
                 sub_classes: Dict[str, int] = None,
                 transform: Optional[Callable] = None,
                 size: Optional[int] = None):
        super().__init__()
        self._transform = transform
        self._data_root_dir = data_root_dir
        self._indexes = sorted([x for x in os.listdir(data_root_dir) if
                                os.path.isdir(os.path.join(self._data_root_dir, x))])

        self._reshape_to_identity = reshape_to_identity
        self._target_spacing = target_spacing

        self._sub_classes = sub_classes
        self._size = size

    def __len__(self):
        return len(self._indexes) if self._size is None else self._size

    @timeit("BTCVDataset.get_item")
    def __getitem__(self, index) -> Tuple[Any, Any]:
        if index == len(self):
            raise StopIteration
        index_name = self._indexes[index]
        try:
            dir = os.path.join(self._data_root_dir, index_name)
            ct = read_nii(os.path.join(dir, "ct.nii.gz"))
            seg = TTTVolume(np.zeros_like(ct.data,
                                          dtype=np.uint8),
                            spacing=deepcopy(ct.spacing),
                            origin_lps=deepcopy(ct.origin_lps),
                            matrix_ijk_2_lps=deepcopy(ct.matrix_ijk_2_lps))
            classes = self._sub_classes if self._sub_classes is not None else (
                TOTAL_SEG_LABELS_TO_CLASS_ID)
            for c, class_id in classes.items():
                seg_file = os.path.join(dir, "segmentations", f"{c}.nii.gz")
                if os.path.isfile(os.path.join(dir, "segmentations", f"{c}.nii.gz")):
                    tmp_seg = read_nii(seg_file)
                    seg.data[np.where(tmp_seg.data == 1)] = class_id

            if self._reshape_to_identity:
                ct = permute_to_identity_matrix(ct)
                seg = permute_to_identity_matrix(seg)
            if self._target_spacing is not None:
                ct = interpolate_to_target_spacing(ct, np.array(self._target_spacing))
                seg = interpolate_to_target_spacing(seg, np.array(self._target_spacing),
                                                    method="nearest_neighbor")

            if self._transform is not None:
                return self._transform(ct.data, seg.data)
            return ct.data, seg.data
        except Exception as e:
            self.__log.error(f"error:{e} for index: {index_name}")
            return None, None

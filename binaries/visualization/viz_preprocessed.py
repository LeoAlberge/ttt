import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.python.core.dataset import TotalSegmentatorDataSet, TOTAL_SEG_CLASS_ID_TO_LABELS, H5Dataset
from src.python.core.volume import TTTVolume
from src.python.preprocessing.io.niifty_readers import read_nii
from src.python.preprocessing.preprocessing import permute_to_identity_matrix, \
    interpolate_to_target_spacing, PatchExtractor, SegmentationOneHotEncoding
from src.python.preprocessing.transform import ComposeTransform, ToTensor


def main():
    sub_classes = {"liver": 1}
    x, y = next(iter(H5Dataset("res"))

    for i in range(x.shape[0]):
        outdir = f"out_p{i}"
        vol_data = x[i, :, :, :,].numpy()
        seg_data = y[i,  :, :, :, 1].numpy()
        os.makedirs(outdir, exist_ok=True)
        for k in range(vol_data.shape[0]):
            if seg_data[k, :, :].sum() > 0:
                plt.figure()
                plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
                plt.contour(seg_data[k, :, :],
                            levels=[0.5],
                            colors=["r"])
                plt.savefig(os.path.join(outdir, f"{k}.png"))


if __name__ == '__main__':
    main()

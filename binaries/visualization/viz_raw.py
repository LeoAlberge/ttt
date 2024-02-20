import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.python.core.dataset import TotalSegmentatorDataSet, TOTAL_SEG_CLASS_ID_TO_LABELS
from src.python.core.volume import TTTVolume
from src.python.preprocessing.io.niifty_readers import read_nii
from src.python.preprocessing.preprocessing import permute_to_identity_matrix, \
    interpolate_to_target_spacing, PatchExtractor, SegmentationOneHotEncoding
from src.python.preprocessing.transform import ComposeTransform, ToTensor


def main():
    sub_classes = {"liver": 1}

    # x, y = next(iter(TotalSegmentatorDataSet(
    #     r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201",
    #     sub_classes=sub_classes,
    #     transform=ComposeTransform([
    #         ToTensor(torch.float32, torch.int64),
    #         SegmentationOneHotEncoding(num_classes=2)
    #     ]))))
    #
    # for i in range(x.shape[0]):
    #     outdir = f"out"
    #     vol_data = x[:, :, :].numpy()
    #     seg_data = y[:, :, :, 1].numpy()
    #     os.makedirs(outdir, exist_ok=True)
    #     for k in range(vol_data.shape[0]):
    #         if seg_data[k, :, :].sum() > 0:
    #             plt.figure()
    #             plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
    #             plt.contour(seg_data[k, :, :],
    #                         levels=[0.5],
    #                         colors=["r"])
    #             plt.savefig(os.path.join(outdir, f"{k}.png"))

    x, y = next(iter(TotalSegmentatorDataSet(
        r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201",
        sub_classes=sub_classes,
        transform=ComposeTransform([
            ToTensor(torch.float32, torch.int64),
            PatchExtractor(),
            SegmentationOneHotEncoding(num_classes=2)
        ]))))

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

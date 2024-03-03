import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.python.core.dataset import TotalSegmentatorDataSet, TOTAL_SEG_CLASS_ID_TO_LABELS, \
    TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.preprocessing.preprocessing import PatchExtractor
from src.python.preprocessing.transform import ComposeTransform, ToTensor


def main():
    colors = {"liver": "red", "aorta": "blue", "kidney": "green"}

    sub_classes = {"liver": 1, "aorta":2,"kidney": 3}
    # x, y = next(iter(TotalSegmentatorDataSet(
    #     r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201",
    #     target_spacing=(2, 2, 2),
    #     sub_classes=sub_classes
    # )))
    #
    #
    #
    # outdir = f"out"
    # vol_data = x[:, :, :]
    # seg_data = y[:, :, :]
    # os.makedirs(outdir, exist_ok=True)
    # for k in range(vol_data.shape[0]):
    #     if seg_data[k, :, :].sum() > 0:
    #         plt.figure()
    #         plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
    #         plt.imshow(seg_data[k, :, :],  interpolation="bilinear", alpha=0.5)
    #         for l, id in sub_classes.items():
    #             if (seg_data[k, :, :] == id).sum() > 0:
    #                 plt.contour((seg_data[k, :, :] == id).astype(np.uint8), alpha=0.5, levels=[0.5], colors=[colors[l]])
    #
    #         plt.savefig(os.path.join(outdir, f"{k}.png"))
    #         plt.close()

    colors = {c: np.random.rand(3) for c in TOTAL_SEG_CLASS_ID_TO_LABELS.values()}
    sub_classes=TOTAL_SEG_LABELS_TO_CLASS_ID

    x, y = next(iter(TotalSegmentatorDataSet(
        r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201",
        target_spacing=(2,2,2),
        sub_classes=sub_classes,
        transform=ComposeTransform([
            ToTensor(torch.float32, torch.int64),
            PatchExtractor(),
            # SegmentationOneHotEncoding(num_classes=2)
        ]))))

    print(f"nb patches: {x.shape[0]}")
    for i in range(x.shape[0]):
        outdir = f"out/p{i}"
        vol_data = x[i, :, :, :,].numpy()
        seg_data = y[i, :, :, :].numpy()
        os.makedirs(outdir, exist_ok=True)
        for k in range(vol_data.shape[0]):
            if seg_data[k, :, :].sum() > 0:
                plt.figure()
                plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
                for l, id in sub_classes.items():
                    if (seg_data[k, :, :] == id).sum() > 0:
                        plt.contour((seg_data[k, :, :] == id).astype(np.uint8), alpha=0.5,
                                    levels=[0.5], colors=[colors[l]])

                plt.savefig(os.path.join(outdir, f"{k}.png"))
                plt.close()


if __name__ == '__main__':
    main()

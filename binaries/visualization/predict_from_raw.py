import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.python.core.dataset import H5Dataset, TOTAL_SEG_CLASS_ID_TO_LABELS, \
    TOTAL_SEG_LABELS_TO_CLASS_ID, TotalSegmentatorDataSet
from src.python.inference.sliding_window_inference import SlidingWindowInference
from src.python.io.niifty_readers import write_nii
from src.python.models.unetr import UnetR
from src.python.preprocessing.preprocessing import VolumeNormalization


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    with open("sb/subclasses.json") as f:
        subclasses = json.loads(f.read())
        num_classes = len(subclasses) + 1
    m = UnetR(nb_classes=num_classes, mlp_dim=1536, normalization="batch_norm")
    m.load_state_dict(torch.load('sb/16.pt', map_location=torch.device('cpu')))
    swi = SlidingWindowInference(m)

    ds = TotalSegmentatorDataSet(
        r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201",
        target_spacing=(1.5, 1.5, 1.5),
        sub_classes=subclasses,
        transform=VolumeNormalization())
    colors = ["red", "orange", "yellow", "blue", "purple","cyan", "olive"]
    for c, (case_name, vol, seg) in tqdm(enumerate(ds.indexed_iter())):
        vol_data = vol.data
        pred = swi.run(vol_data, step_size=(96,96,96), num_classes=num_classes)
        outdir = f"pred/{case_name}"
        os.makedirs(outdir, exist_ok=True)
        write_nii(seg, os.path.join(outdir, "gt.nii.gz"))
        seg.data = pred
        write_nii(seg, os.path.join(outdir, "pred.nii.gz"))
        write_nii(vol, os.path.join(outdir, "ct.nii.gz"))

        # for k in range(vol_data.shape[0]):
        #     plt.figure()
        #     plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
        #     # plt.imshow(pred[ k, :, :], cmap="hot", interpolation="bilinear", alpha=0.3,
        #     #            vmin=1, vmax=118)
        #     plt.colorbar()
        #
        #     plt.contour(
        #         (y[k, :, :] > 0).astype(np.uint8),
        #         levels=[0.5], colors=["green"])
        #     for color, c_id in zip(colors, subclasses.values()):
        #         plt.contour((pred[ k, :, :]==1).astype(np.uint8),
        #                     levels=[0.5], colors=[color])
        #     plt.savefig(os.path.join(outdir, f"{k}.png"))
        #     plt.close()


if __name__ == '__main__':
    main()

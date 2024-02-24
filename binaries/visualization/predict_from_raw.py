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
from src.python.models.unetr import UnetR
from src.python.preprocessing.preprocessing import VolumeNormalization


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    num_classes = 118
    m = UnetR(nb_classes=num_classes, mlp_dim=1536, normalization="batch_norm")
    m.load_state_dict(torch.load('4.pt', map_location=torch.device('cpu')))
    swi = SlidingWindowInference(m)

    ds = TotalSegmentatorDataSet(
        r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201",
        target_spacing=(1.5, 1.5, 1.5),
        sub_classes={"liver": 1},
        transform=VolumeNormalization())
    for c, (vol_data, y) in tqdm(enumerate(iter(ds))):
        if y.sum() > 0:
            pred = swi.run(vol_data, step_size=(96,96,96), num_classes=num_classes)
            outdir = f"pred/{c}"

            os.makedirs(outdir, exist_ok=True)
            for k in range(vol_data.shape[0]):
                plt.figure()
                plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
                plt.imshow(pred[ k, :, :], cmap="hot", interpolation="bilinear", alpha=0.3,
                           vmin=0, vmax=1)
                plt.colorbar()

                plt.contour(
                    (y[k, :, :] == 1).astype(np.uint8),
                    levels=[0.5], colors=["green"])
                plt.contour((pred[ k, :, :]==TOTAL_SEG_LABELS_TO_CLASS_ID["liver"]).astype(np.uint8),
                            levels=[0.5], colors=["red"])
                plt.savefig(os.path.join(outdir, f"{k}.png"))
                plt.close()


if __name__ == '__main__':
    main()

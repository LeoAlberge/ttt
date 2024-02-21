import os

import matplotlib.pyplot as plt
import numpy as np

from src.python.core.dataset import H5Dataset, TOTAL_SEG_CLASS_ID_TO_LABELS


def main():
    x, y = next(iter(H5Dataset("preprocessed.hdf5")))
    colors = {c: np.random.rand(3) for c in TOTAL_SEG_CLASS_ID_TO_LABELS.values()}
    outdir = "p2"
    vol_data = x[0, :, :, :, ]
    seg_data = y[0, :, :, :]
    os.makedirs(outdir, exist_ok=True)
    for k in range(vol_data.shape[0]):
        if seg_data[k, :, :].sum() > 0:
            plt.figure()
            plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
            for id, l in TOTAL_SEG_CLASS_ID_TO_LABELS.items():
                if (seg_data[k, :, :] == id).sum() > 0:
                    plt.contour((seg_data[k, :, :] == id).astype(np.uint8), alpha=0.5,
                                levels=[0.5], colors=[colors[l]])
            plt.savefig(os.path.join(outdir, f"{k}.png"))
            plt.close()


if __name__ == '__main__':
    main()

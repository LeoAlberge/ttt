import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.python.core.dataset import H5Dataset, TOTAL_SEG_CLASS_ID_TO_LABELS, \
    TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.models.unetr import UnetR


def main():
    colors = {c: np.random.rand(3) for c in TOTAL_SEG_CLASS_ID_TO_LABELS.values()}
    sub_classes = TOTAL_SEG_LABELS_TO_CLASS_ID
    m = UnetR(nb_classes=2, mlp_dim=1536, normalization="instance")
    m.load_state_dict(torch.load('0.pt', map_location=torch.device('cpu')))
    m.eval()
    ds =H5Dataset("preprocessed_100.hdf5")
    for c, (x, y) in enumerate(iter(ds)):
        outdir = f"pred/{c}"
        seg_data = torch.nn.functional.softmax(m.forward(torch.tensor(x[np.newaxis,:])), dim=1).detach().cpu().numpy()[0,1,:,:,:]
        vol_data = x[0, :, :, :, ]
        # seg_data = y[0, :, :, :]
        os.makedirs(outdir, exist_ok=True)
        for k in range(vol_data.shape[0]):
            if seg_data[k, :, :].sum() > 0:
                plt.figure()
                plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
                if seg_data[k, :, :].sum() > 0:
                    plt.contour((seg_data[k, :, :] == id).astype(np.uint8), alpha=0.5,
                                levels=[0.5], colors=["red"])
                plt.savefig(os.path.join(outdir, f"{k}.png"))
                plt.close()
        if c == 10:
            break

if __name__ == '__main__':
    main()

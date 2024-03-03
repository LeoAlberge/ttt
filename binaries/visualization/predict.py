import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.python.core.dataset import H5Dataset, TOTAL_SEG_CLASS_ID_TO_LABELS, \
    TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.models.unetr import UnetR


def main():

    with open("subclasses.json") as f:
        subclasses = json.loads(f.read())
        num_classes = len(subclasses) + 1
    colors = ["red"]
    subclasses = {"liver": 4}
    m = UnetR(nb_classes=num_classes, mlp_dim=1536, normalization="batch_norm")
    m.load_state_dict(torch.load('25.pt', map_location=torch.device('cpu')))
    m.eval()
    ds =H5Dataset("preprocessed_100.hdf5")
    for c, (x, y) in tqdm(enumerate(iter(ds))):
        if c <= 36:
            continue
        vol_data = x[0, :, :, :, ]
        if (y[0, :, :, :] == TOTAL_SEG_LABELS_TO_CLASS_ID["liver"]).sum() > 0:
            outdir = f"pred_batch/{c}"
            seg_data = torch.nn.functional.softmax(m.forward(torch.tensor(x[np.newaxis,:])), dim=1).detach().cpu().numpy()[0,:]
            os.makedirs(outdir, exist_ok=True)
            for k in range(vol_data.shape[0]):
                plt.figure()
                plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
                for color, c_id in zip(colors, subclasses.values()):
                    plt.imshow(seg_data[c_id, k, :, :], cmap="hot", interpolation="bilinear", vmin=0, vmax=1, alpha=0.3)
                    plt.colorbar()
                    plt.contour(seg_data[c_id, k, :, :] ,levels=[0.5], colors=[color])

                plt.savefig(os.path.join(outdir, f"{k}.png"))
                plt.close()


if __name__ == '__main__':
    main()

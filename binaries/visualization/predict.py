import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.python.core.dataset import H5Dataset, TOTAL_SEG_CLASS_ID_TO_LABELS, \
    TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.models.unetr import UnetR


def main():
    colors = {c: np.random.rand(3) for c in TOTAL_SEG_CLASS_ID_TO_LABELS.values()}
    sub_classes = TOTAL_SEG_LABELS_TO_CLASS_ID
    m = UnetR(nb_classes=2, mlp_dim=1536, normalization="batch_norm")
    m.load_state_dict(torch.load('3.pt', map_location=torch.device('cpu')))
    m.eval()
    ds =H5Dataset("preprocessed_100.hdf5")
    for c, (x, y) in tqdm(enumerate(iter(ds))):
        vol_data = x[0, :, :, :, ]
        if (y[0, :, :, :] == TOTAL_SEG_LABELS_TO_CLASS_ID["liver"]).sum() > 0:
            outdir = f"pred/{c}"
            seg_data = torch.nn.functional.softmax(m.forward(torch.tensor(x[np.newaxis,:])), dim=1).detach().cpu().numpy()[0,:]
            liver_proba = seg_data[1,:,:,:]
            seg_liver = np.argmax(seg_data, axis=0)


            os.makedirs(outdir, exist_ok=True)
            for k in range(vol_data.shape[0]):
                plt.figure()
                plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
                plt.imshow(liver_proba[k, :, :], cmap="hot", interpolation="bilinear", alpha=0.3, vmin=0, vmax=1)
                plt.colorbar()

                plt.contour((y[0, k, :, :] == TOTAL_SEG_LABELS_TO_CLASS_ID["liver"]).astype(np.uint8),
                            levels=[0.5], colors=["green"])
                plt.contour(seg_liver[k,:,:],
                            levels=[0.5], colors=["red"])
                plt.savefig(os.path.join(outdir, f"{k}.png"))
                plt.close()


if __name__ == '__main__':
    main()

import json
import os

import numpy as np
import torch
from torch import tensor
from torch.utils.data import Subset
from tqdm import tqdm

from src.python.core.dataset import H5Dataset
from src.python.core.volume import TTTVolume
from src.python.io.niifty_readers import write_nii
from src.python.models.unetr import UnetR
from src.python.training.metrics import SegmentationMultiDiceScores


def main():

    with open("subclasses2.json") as f:
        subclasses = json.loads(f.read())
        num_classes = len(subclasses) + 1

    m = UnetR(nb_classes=num_classes, mlp_dim=1536, normalization="batch_norm")
    m.load_state_dict(torch.load('liver/21.pt', map_location=torch.device('cpu')))
    m.eval()
    ds =H5Dataset(r"C:\Users\LeoAlberge\work\personnal\data\preprocessed_liver.hdf5")

    with open("liver/val_indexes.json", "r") as f:
        val_set = Subset(ds, indices=json.loads(f.read()))


    for c, (x, y) in tqdm(enumerate(iter(val_set))):
        vol_data = x[0, :, :, :, ]
        outdir = f"pred_batch/{c}"
        seg_data = np.argmax(m.forward(torch.tensor(x[np.newaxis,:])).detach().cpu(), axis=1)[0,...]
        print(seg_data.shape)
        os.makedirs(outdir, exist_ok=True)
        write_nii(TTTVolume(seg_data, np.zeros(3),  np.ones(3)*1.5, np.eye(3)), os.path.join(outdir, "pred.nii.gz"))
        write_nii(TTTVolume(vol_data, np.zeros(3),  np.ones(3)*1.5, np.eye(3)), os.path.join(outdir, "ct.nii.gz"))
        write_nii(TTTVolume(y[0,...], np.zeros(3),  np.ones(3)*1.5, np.eye(3)), os.path.join(outdir, "gt.nii.gz"))

            # for k in range(vol_data.shape[0]):
            #     plt.figure()
            #     plt.imshow(vol_data[k, :, :], cmap="gray", interpolation="bilinear")
            #     for color, c_id in zip(colors, subclasses.values()):
            #         plt.imshow(seg_data[c_id, k, :, :], cmap="hot", interpolation="bilinear", vmin=0, vmax=1, alpha=0.3)
            #         plt.colorbar()
            #         plt.contour(seg_data[c_id, k, :, :] ,levels=[0.5], colors=[color])
            #
            #     plt.savefig(os.path.join(outdir, f"{k}.png"))
            #     plt.close()


if __name__ == '__main__':
    main()

import argparse
import logging
import sys

import h5py
import numpy as np
from tqdm import tqdm

from src.python.core.dataset import TotalSegmentatorDataSet
from src.python.core.volume import TTTVolume
from src.python.io.niifty_readers import write_nii
from src.python.preprocessing.preprocessing import VolumeNormalization, \
    CropOnSegmentation, VolumeReshaper
from src.python.preprocessing.transform import ComposeTransform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root-dir",
                        default=r"C:\Users\LeoAlberge\work\personnal\data"
                                r"\Totalsegmentator_dataset_small_v201", required=False)
    parser.add_argument("--out-hdf5", default=r"res_full.hdf5", required=False)
    parser.add_argument("--size", default=None, required=False)
    parser.add_argument("--write-nii", default=False, required=False)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parser.parse_args()

    if args.size is None:
        size = None
    else:
        size = int(args.size)
    sub_classes = {"liver": 1}

    h5 = h5py.File(args.out_hdf5, "w")
    h5.create_group("inputs")
    h5.create_group("targets")
    dataset = TotalSegmentatorDataSet(
        args.data_root_dir,
        target_spacing=(1.5, 1.5, 1.5),
        sub_classes=sub_classes,
        transform=ComposeTransform([
            CropOnSegmentation(),
            VolumeReshaper((96, 96, 96)),
            VolumeNormalization(),
            lambda x, y: (x[None, :, :, :].astype(np.float32),
                          y[None, :, :, :].astype(np.uint8))
        ]),
        size=size)

    c = 0
    for _c, (inputs, targets) in tqdm(enumerate(iter(dataset)), total=len(dataset)):
        if inputs is None:
            continue
        if args.write_nii:
            write_nii(TTTVolume(inputs, np.zeros(3), np.ones(3), np.eye(3)), f"{c}_ct.nii.gz")
            write_nii(TTTVolume(targets, np.zeros(3), np.ones(3), np.eye(3)), f"{c}_gt.nii.gz")

        if targets.sum() > 0:
            h5["inputs"].create_dataset(f"{c}", data=inputs, compression="gzip")
            h5["targets"].create_dataset(f"{c}", data=targets, compression="gzip")
        c += 1
    h5.close()


if __name__ == '__main__':
    main()

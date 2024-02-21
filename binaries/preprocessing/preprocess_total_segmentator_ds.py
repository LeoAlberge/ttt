import argparse
import logging
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

from src.python.core.dataset import TotalSegmentatorDataSet
from src.python.preprocessing.preprocessing import PatchExtractor, VolumeNormalization
from src.python.preprocessing.transform import ComposeTransform, ToTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root-dir",
                        default=r"C:\Users\LeoAlberge\work\personnal\data"
                                r"\Totalsegmentator_dataset_small_v201", required=False)
    parser.add_argument("--out-hdf5", default=r"res_full.hdf5", required=False)
    parser.add_argument("--liver-only", default="false", required=False)
    parser.add_argument("--size", default=None, required=False)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parser.parse_args()

    if args.size is None:
        size = None
    else:
        size = int(args.size)
    if args.liver_only == "true":
        sub_classes = {"liver": 1}
    else:
        sub_classes = None

    h5 = h5py.File(args.out_hdf5, "w")
    h5.create_group("inputs")
    h5.create_group("targets")
    dataset = TotalSegmentatorDataSet(
        args.data_root_dir,
        target_spacing=(1.5, 1.5, 1.5),
        sub_classes=sub_classes,
        transform=ComposeTransform([
            VolumeNormalization(),
            ToTensor(torch.float32, torch.int64),
            PatchExtractor(),
            lambda x, y: (x[:, None, :, :, :].numpy().astype(np.float32),
                          y[:, None, :, :, :].numpy().astype(np.uint8))
        ]),
        size=size)

    c = 0
    for _c, (inputs, targets) in tqdm(enumerate(iter(dataset)), total=len(dataset)):
        if inputs is None:
            continue
        for i in range(inputs.shape[0]):
            h5["inputs"].create_dataset(f"{c}-{i}", data=inputs[i, :, :, :, :].astype(np.float32),
                                        compression="gzip")
            h5["targets"].create_dataset(f"{c}-{i}", data=targets[i, :, :, :, :],
                                         compression="gzip")
        c += 1
    h5.close()

if __name__ == '__main__':
    main()

import argparse
import json
import logging
import sys

import numpy as np
import torch
from torch.utils.data import random_split

from src.python.core.dataset import H5Dataset, TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.preprocessing.preprocessing import SegmentationOneHotEncoding
from src.python.preprocessing.transform import ComposeTransform, ToTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries"
                                r"\visualization\preprocessed_100.hdf5",
                        required=False)
    parser.add_argument("--logging",
                        default=r"INFO", required=False)
    parser.add_argument("--liver-only", default="false", required=False)

    args = parser.parse_args()

    if args.logging == "DEBUG":
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif args.logging == "INFO":
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    liver_only = args.liver_only.lower() == "true"
    num_classes = 2 if liver_only else 118
    if liver_only:
        transform_l = [
            lambda x, y: (x, (y == TOTAL_SEG_LABELS_TO_CLASS_ID["liver"]).astype(np.uint8)),
            ToTensor(torch.float32, torch.uint8),
            SegmentationOneHotEncoding(num_classes),
            ToTensor(torch.float32, torch.float32),
        ]
    else:
        transform_l = [
            ToTensor(torch.float32, torch.uint8),
            SegmentationOneHotEncoding(num_classes),
            ToTensor(torch.float32, torch.float32),
        ]

    ds = H5Dataset(args.dataset, transform=ComposeTransform(transform_l))

    train_set, val_set = random_split(ds, [0.8, 0.2])
    with open("train_indexes.json", "w") as f:
        f.write(json.dumps(train_set.indices))
    with open("val_indexes.json", "w") as f:
        f.write(json.dumps(val_set.indices))


if __name__ == '__main__':
    main()

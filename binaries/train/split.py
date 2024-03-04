import argparse
import json
import logging
import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.python.core.dataset import H5Dataset, TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.preprocessing.preprocessing import SegmentationOneHotEncoding
from src.python.preprocessing.transform import ComposeTransform, ToTensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries"
                                r"\visualization\preprocessed.hdf5",
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

    indexes_y = [x.split('-') for x in ds.indexes]
    patient_indexes = set([int(x[0]) for x in indexes_y])
    patient_to_patches = {}
    for c, ind in enumerate(indexes_y):
        patient_to_patches.setdefault(int(ind[0]), []).append(c)
    train_patient_set, val_patient_set = train_test_split(list(patient_indexes), test_size=0.2)

    with open("train_indexes.json", "w") as f:
        f.write(
            json.dumps([ind for p in train_patient_set for ind in patient_to_patches[p]]))
    with open("val_indexes.json", "w") as f:
        f.write(
            json.dumps([ind for p in val_patient_set for ind in patient_to_patches[p]]))


if __name__ == '__main__':
    main()

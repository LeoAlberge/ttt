import argparse
import json

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from tqdm import tqdm

from src.python.core.dataset import H5Dataset, TOTAL_SEG_CLASS_ID_TO_LABELS, \
    TOTAL_SEG_LABELS_TO_CLASS_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries"
                                r"\visualization\preprocessed.hdf5",
                        required=False)
    parser.add_argument("--subclasses",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries"
                                r"\visualization\sb\subclasses.json",
                        required=False)

    parser.add_argument("--train-indexes",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries"
                                r"\visualization\sb\train_indexes.json",
                        required=False)

    parser.add_argument("--val-indexes",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries"
                                r"\visualization\sb\val_indexes.json",
                        required=False)

    args = parser.parse_args()
    ds = H5Dataset(args.dataset)

    with open(args.train_indexes, "r") as f:
        train_set = Subset(ds, indices=json.loads(f.read()))
    with open(args.val_indexes, "r") as f:
        val_set = Subset(ds, indices=json.loads(f.read()))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, pin_memory=False, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, pin_memory=False,
                                               shuffle=True)

    print(len(val_loader), len(train_loader))
    with open(args.subclasses) as f:
        subclasses = json.loads(f.read())
        subclasses_ids = {TOTAL_SEG_LABELS_TO_CLASS_ID[s] for s in subclasses.keys()}
    n_patches = 1000

    for loader in [val_loader, train_loader]:
        label_distri_pixel = {}
        label_distri_patches = {}

        for counter, (_, y) in enumerate(tqdm(iter(loader))):
            uniques, counts = np.unique(y, return_counts=True)
            if counter == n_patches:
                break
            for un, c in zip(uniques, counts):
                if un == 0 or un in subclasses_ids:
                    label_distri_pixel.setdefault(un, 0)
                    label_distri_pixel[un] += c
                    label_distri_patches.setdefault(un, 0)
                    label_distri_patches[un] += 1
                    if un!=0:
                        label_distri_pixel.setdefault(-1, 0)
                        label_distri_patches.setdefault(-1, 0)
                        label_distri_patches[-1] += 1
                        label_distri_pixel[-1] += c

        def normalize_counts(counts_dict):
            total_count = sum(counts_dict.values())
            normalized_dict = {key: count / total_count for key, count in counts_dict.items()}
            return normalized_dict

        def normalize_count_by_len(counts_dict, total_count):
            normalized_dict = {key: count / total_count for key, count in counts_dict.items()}
            return normalized_dict

        patches_dist = normalize_counts(label_distri_pixel)
        pix_dist = normalize_count_by_len(label_distri_patches, n_patches)

        print(label_distri_pixel, normalize_counts(label_distri_pixel))
        print(label_distri_patches, normalize_count_by_len(label_distri_patches, n_patches))

        plt.figure(figsize=(8, 8))
        sns.barplot(x=list(patches_dist.values()),
                    y=[TOTAL_SEG_CLASS_ID_TO_LABELS.get(c, "background") for c in patches_dist],
                    hue=list(patches_dist.values()), palette="deep")

        plt.title('Patches Distribution')
        plt.show()

        plt.figure(figsize=(8, 8))
        sns.barplot(x=list(pix_dist.values()),
                    y=[TOTAL_SEG_CLASS_ID_TO_LABELS.get(c, "background") for c in pix_dist],
                    hue=list(pix_dist.values()), palette="deep")

        plt.title('Pixel Distribution')
        plt.show()

    # plt.figure(figsize=(8, 8))
    # plt.pie(list(pix_dist.values()), labels=[TOTAL_SEG_CLASS_ID_TO_LABELS.get(c, "background")
    # for c in pix_dist], autopct='%1.1f%%', startangle=140)
    # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.title('Pixelwise Distribution')
    # plt.show()


if __name__ == '__main__':
    main()

import argparse
import json

import numpy as np
import seaborn as sns
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
                                r"\visualization\subclasses.json",
                        required=False)

    parser.add_argument("--train-indexes",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries\train\train_indexes.json",
                        required=False)

    parser.add_argument("--val-indexes",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries\train\val_indexes.json",
                        required=False)



    args = parser.parse_args()
    ds = H5Dataset(args.dataset)

    with open(args.train_indexes, "r") as f:
        train_set = Subset(ds, indices=json.loads(f.read()))
    with open(args.val_indexes, "r") as f:
        val_set = Subset(ds, indices=json.loads(f.read()))




    label_distri_pixel= {}
    label_distri_patches= {}
    with open(args.subclasses) as f:
        subclasses = json.loads(f.read())
        subclasses_ids = {TOTAL_SEG_LABELS_TO_CLASS_ID[s] for s in subclasses.keys()}
    n_patches = 3000
    for counter, (_, y)  in enumerate(tqdm(iter(ds))):
        uniques, counts = np.unique(y, return_counts=True)
        if counter == n_patches:
            break
        for un, c in zip(uniques, counts):
            if un == 0 or un in subclasses_ids:
                label_distri_pixel.setdefault(un, 0)
                label_distri_pixel[un] += c
                label_distri_patches.setdefault(un, 0)
                label_distri_patches[un] +=1

    def normalize_counts(counts_dict):
        total_count = sum(counts_dict.values())
        normalized_dict = {key: count / total_count for key, count in counts_dict.items()}
        return normalized_dict

    def normalize_count_by_len(counts_dict, total_count):
        normalized_dict = {key: count / total_count for key, count in counts_dict.items()}
        return normalized_dict

    # print(label_distri_pixel, normalize_counts(label_distri_pixel))
    # print(label_distri_patches, normalize_count_by_len(label_distri_patches, n_patches))

    pix_dist = {0: 0.966324844376319, 5: 0.01932046961196357, 6: 0.00433202067725448,
                2: 0.001659412130971809, 52: 0.0029024259611096703, 1: 0.00300327192271244,
                3: 0.001664605216950865, 7: 0.0007929501027180955}
    patches_dist = {0: 1.0, 5: 0.2894, 6: 0.165, 2: 0.1144, 52: 0.1744, 1: 0.1284, 3: 0.1126,
                    7: 0.1032}

    # plt.figure(figsize=(8, 8))
    # plt.pie(list(pix_dist.values()), labels=[TOTAL_SEG_CLASS_ID_TO_LABELS.get(c, "background")
    # for c in pix_dist], autopct='%1.1f%%', startangle=140)
    # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.title('Pixelwise Distribution')
    # plt.show()

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



if __name__ == '__main__':
    main()

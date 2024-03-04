import argparse
import json
import logging
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import random_split, Subset

from src.python.core.dataset import H5Dataset, TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.models.unetr import UnetR, count_parameters
from src.python.preprocessing.preprocessing import SegmentationOneHotEncoding
from src.python.preprocessing.transform import ComposeTransform, ToTensor
from src.python.training.experiment_evaluator import ExperimentEvaluator, ExperimentEvaluationParams
from src.python.training.losses import CombinedSegmentationLoss
from src.python.training.metrics import SegmentationMultiDiceScores
from src.python.training.training_operator import TrainingOperatorParams, TrainingOperator, \
    ReloadWeightsConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries\visualization\preprocessed.hdf5",
                        required=False)
    parser.add_argument("--bs",
                        default=r"3", required=False)
    parser.add_argument("--logging",
                        default=r"INFO", required=False)
    parser.add_argument("--compiled", default="false", required=False)
    parser.add_argument("--subclasses", default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries\visualization\sb\subclasses.json", required=False)
    parser.add_argument("--num-workers", default="0", required=False)
    parser.add_argument("--val-indexes", default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries\visualization\sb\val_indexes.json", required=False)
    parser.add_argument("--exp-dir", default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries\visualization\sb", required=False)

    args = parser.parse_args()

    if args.logging == "DEBUG":
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif args.logging == "INFO":
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    num_workers = int(args.num_workers)
    bs = int(args.bs)
    if args.subclasses:
        with open(args.subclasses) as f:
            subclasses = json.loads(f.read())
            num_classes = len(subclasses) + 1
    else:
        subclasses = None
        num_classes = 118

    def map_new_classes(y, subclasses: Dict[str, int]) -> np.uint8:
        res = np.zeros_like(y, dtype=np.uint8)
        unique_classes = np.unique(y)
        for class_label, new_class_id in subclasses.items():
            if TOTAL_SEG_LABELS_TO_CLASS_ID[class_label] in unique_classes:
                res[np.where(y == TOTAL_SEG_LABELS_TO_CLASS_ID[class_label])] = new_class_id
        return res

    if subclasses:
        transform_l = [
            lambda x, y: (x, map_new_classes(y, subclasses)),
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

    with open(args.val_indexes, "r") as f:
        val_set = Subset(ds, indices=json.loads(f.read()))

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs,
                                             pin_memory=torch.cuda.is_available(),
                                             num_workers=num_workers)
    m = UnetR(nb_classes=num_classes, mlp_dim=1536, normalization="batch_norm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    if args.compiled.lower() == "true":
        m = torch.compile(m, mode="reduce-overhead")

    logging.info(f"Number of params {count_parameters(m)}")
    params = ExperimentEvaluationParams(
        model=m,
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"dices": SegmentationMultiDiceScores(device=device, apply_argmax=False, apply_softmax=True)},
        val_data_loader=val_loader,
        weights_dir=args.exp_dir,
        device=device,
    )
    t = ExperimentEvaluator(params)
    t.evaluate()


if __name__ == '__main__':
    main()

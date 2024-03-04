import argparse
import json
import logging
import sys
from typing import Dict

import numpy as np
import torch
from torch.utils.data import random_split, Subset

from src.python.core.dataset import H5Dataset, TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.models.unetr import UnetR, count_parameters
from src.python.preprocessing.preprocessing import SegmentationOneHotEncoding
from src.python.preprocessing.transform import ComposeTransform, ToTensor
from src.python.training.metrics import SegmentationMultiDiceScores
from src.python.training.training_operator import TrainingOperatorParams, TrainingOperator, \
    ReloadWeightsConfig, OptimConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=r"C:\Users\LeoAlberge\work\personnal\github\ttt\binaries"
                                r"\visualization\preprocessed_100.hdf5",
                        required=False)
    parser.add_argument("--epochs",
                        default=r"2", required=False)
    parser.add_argument("--bs",
                        default=r"6", required=False)
    parser.add_argument("--logging",
                        default=r"INFO", required=False)
    parser.add_argument("--compiled", default="false", required=False)
    parser.add_argument("--subclasses", default="subclasses.json", required=False)
    parser.add_argument("--num-workers", default="4", required=False)
    parser.add_argument("--reload-mode", default="pretrained", required=False)
    parser.add_argument("--evaluate", default="false", required=False)
    parser.add_argument("--steps-per-epoch", default=None, required=False)

    args = parser.parse_args()

    if args.logging == "DEBUG":
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif args.logging == "INFO":
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    evaluate = True if args.evaluate.lower() == "true" else False
    num_workers = int(args.num_workers)
    epoch = int(args.epochs)
    bs = int(args.bs)
    num_classes=15

    transform_l = [
        ToTensor(torch.float32, torch.uint8),
        SegmentationOneHotEncoding(num_classes),
        ToTensor(torch.float32, torch.float32),
    ]

    ds = H5Dataset(args.dataset, transform=ComposeTransform(transform_l))
    with open("train_indexes.json", "r") as f:
        train_set = Subset(ds, indices=json.loads(f.read()))
    with open("val_indexes.json", "r") as f:
        val_set = Subset(ds, indices=json.loads(f.read()))

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True,
                                              pin_memory=torch.cuda.is_available(),
                                              num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs,
                                             pin_memory=torch.cuda.is_available(),
                                             num_workers=num_workers)
    m = UnetR(nb_classes=num_classes, mlp_dim=1536, normalization="batch_norm")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = m.to(device)
    if args.compiled.lower() == "true":
        m = torch.compile(m, mode="reduce-overhead")

    logging.info(f"Number of params {count_parameters(m)}")
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)
    # metrics = {"mean_dice": MeanDiceScore(apply_argmax=True, device=device),
    #  "dices": SegmentationMultiDiceScores(apply_argmax=True, device=device)}
    params = TrainingOperatorParams(
        model=m,
        optim=OptimConfig(optimizer),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"dices": SegmentationMultiDiceScores(device=device)},
        train_data_loader=data_loader,
        val_data_loader=val_loader,
        nb_epochs=epoch,
        weights_dir=".",
        exp_dir=".",
        device=device,
        reload_weights=ReloadWeightsConfig(True, mode=args.reload_mode),
        evaluate=evaluate,
        nb_steps_per_epoch=args.steps_per_epoch
    )
    t = TrainingOperator(params)
    t.fit()


if __name__ == '__main__':
    main()

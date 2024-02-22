import argparse
import logging
import sys

import numpy as np
import torch
from torch.utils.data import random_split

from src.python.core.dataset import H5Dataset, TOTAL_SEG_LABELS_TO_CLASS_ID
from src.python.models.unetr import UnetR, count_parameters
from src.python.preprocessing.preprocessing import SegmentationOneHotEncoding
from src.python.preprocessing.transform import ComposeTransform, ToTensor
from src.python.training.losses import CombinedSegmentationLoss
from src.python.training.metrics import MeanDiceScore
from src.python.training.training_operator import TrainingOperatorParams, TrainingOperator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        default=r"preprocessed_100.hdf5", required=False)
    parser.add_argument("--epochs",
                        default=r"2", required=False)
    parser.add_argument("--bs",
                        default=r"6", required=False)
    parser.add_argument("--logging",
                        default=r"NONE", required=False)
    args = parser.parse_args()

    if args.logging == "DEBUG":
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    elif args.logging == "INFO":
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


    epoch = int(args.epochs)
    bs = int(args.bs)
    ds = H5Dataset(args.dataset, transform=ComposeTransform([
        # lambda x, y: (x, (y==TOTAL_SEG_LABELS_TO_CLASS_ID["liver"]).astype(np.uint8)),
        ToTensor(torch.float32, torch.uint8),
        SegmentationOneHotEncoding(118),
        ToTensor(torch.float32, torch.float32),
    ],
    ))
    train_set, val_set = random_split(ds, [0.8, 0.2])

    data_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs)


    m = UnetR(nb_classes=118).cuda()
    print(count_parameters(m))
    optimizer = torch.optim.Adam(m.parameters())
    params = TrainingOperatorParams(
        model=m,
        optimizer=optimizer,
        loss=CombinedSegmentationLoss(),
        metrics={"mean_dice": MeanDiceScore()},
        train_data_loader=data_loader,
        val_data_loader=val_loader,
        nb_epochs=epoch,
        weights_dir=".",
        cuda_enabled=True
    )
    t = TrainingOperator(params)
    t.fit()


if __name__ == '__main__':
    main()
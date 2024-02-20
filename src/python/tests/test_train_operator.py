import unittest

from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils import data

from models.unetr import UnetR
from training.training_operator import TrainingOperatorParams, TrainingOperator


class FakeData(data.Dataset):
    """A fake dataset that returns randomly generated images and returns them as PIL images

    Args:
        size (int, optional): Size of the dataset. Default: 1000 images
        image_size(tuple, optional): Size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): Number of classes in the dataset. Default: 10
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        random_offset (int): Offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(
            self,
            size: int = 1000,
            image_size: Tuple[int, int, int] = (1, 96, 96, 96),
            num_classes: int = 10,
    ) -> None:
        self.size = size
        self.num_classes = num_classes
        self.image_size = image_size

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # create random image that is consistent with the index id
        if index >= len(self):
            raise IndexError(f"{self.__class__.__name__} index out of range")
        rng_state = torch.get_rng_state()
        img = torch.randn(*self.image_size)
        target = torch.nn.functional.one_hot(
            torch.randint(0, self.num_classes, size=self.image_size, dtype=torch.long)[0],
            num_classes=self.num_classes).permute([3, 0, 1, 2])
        target = target.float()
        torch.set_rng_state(rng_state)
        return img, target

    def __len__(self) -> int:
        return self.size


class TrainOperatorTest(unittest.TestCase):

    def test_base_training(self):
        data_set = FakeData(1, num_classes=10)
        inputs, targets = next(iter(data_set))
        assert inputs.shape == torch.Size((1, 96, 96, 96))
        print(targets.shape)
        assert targets.shape == torch.Size((10, 96, 96, 96))

        data_loader = torch.utils.data.DataLoader(data_set)
        test_loader = torch.utils.data.DataLoader(data_set)

        m = UnetR(nb_classes=10)
        optimizer = torch.optim.Adam(m.parameters())
        params = TrainingOperatorParams(
            model=m,
            optimizer=optimizer,
            loss=torch.nn.MSELoss(),
            metrics={},
            train_data_loader=data_loader,
            val_data_loader=test_loader,
            nb_epochs=2

        )
        t = TrainingOperator(params)
        t.fit()
        assert t._logs[0]["metrics"]["val_loss"] is not None
        print(t._logs)
        assert False

import numpy as np
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class DiceSegmentationLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """

        Args:
            input: shape: B, C, D, H, W
            target:

        Returns:

        """
        num = torch.mul(torch.mul(input, target).sum(dim=[2, 3, 4]), 2)
        denum = torch.mul(input, input).sum(dim=[2, 3, 4]) + torch.mul(target, target).sum(
            dim=[2, 3, 4])
        dice = torch.mean(torch.divide(num, denum), dim=1)
        loss = torch.sub(1, dice)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "none":
            return loss
        raise NotImplementedError(f"{self.reduction} is not Defined")



class CombinedSegmentationLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self._ce = torch.nn.CrossEntropyLoss(size_average=size_average, reduction=reduction)
        self._dice_loss = DiceSegmentationLoss(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self._ce.forward(input, target)  + self._dice_loss(input, target)

if __name__ == '__main__':
    m1 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
    m2 = torch.tensor(np.ones((3, 5, 10, 10, 10)))

    print(DiceSegmentationLoss().forward(m1, m2).shape)

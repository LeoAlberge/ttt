import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

class DiceSegmentationLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        dice = torch.multiply(input, target)


class CombinedSegmentationLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self._ce = torch.nn.CrossEntropyLoss(size_average=size_average, reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self._ce.forward(input, target)
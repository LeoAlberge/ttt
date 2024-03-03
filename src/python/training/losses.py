import torch
import torchvision
from torch import Tensor
from torch.nn.modules.loss import _Loss


class DiceSegmentationLoss(_Loss):
    def __init__(self,
                 ignore_background: bool = False,
                 apply_softmax: bool = True,
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean',
                 eps: float = 1e-5) -> None:
        super().__init__(size_average, reduce, reduction)
        self._apply_softmax = apply_softmax
        self._ignore_background = ignore_background
        self._eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """

        Args:
            input: shape: B, C, D, H, W
            target:

        Returns:

        """
        if self._apply_softmax:
            input = torch.nn.functional.softmax(input, 1)

        if self._ignore_background:
            input = input[:, 1:, :, :, :]
            target = target[:, 1:, :, :, :]

        num = torch.mul(torch.mul(input, target).sum(dim=[2, 3, 4]), 2) + self._eps
        denum = torch.mul(input, input).sum(dim=[2, 3, 4]) + torch.mul(target, target).sum(
            dim=[2, 3, 4]) + self._eps
        dice = torch.mean(torch.divide(num, denum), dim=1)
        loss = torch.sub(1, dice)
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "none":
            return loss
        raise NotImplementedError(f"{self.reduction} is not Defined")


class CombinedSegmentationLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean',
                 ignore_dice_background: bool = True) -> None:
        super().__init__(size_average, reduce, reduction)
        self._ce = torch.nn.CrossEntropyLoss(size_average=size_average, reduction=reduction)
        self._dice_loss = DiceSegmentationLoss(reduction=reduction,
                                               ignore_background=ignore_dice_background)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce = self._ce.forward(input, target)
        dice = self._dice_loss(input, target)
        return torch.add(ce, dice)


class FocalLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torchvision.ops.sigmoid_focal_loss(input, target, reduction=self.reduction)

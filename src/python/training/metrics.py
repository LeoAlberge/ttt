from typing import Any, Iterable, Optional

import torch
from torcheval.metrics import Metric
from torcheval.metrics.metric import TSelf, TComputeReturn


class AbstractSegmentationMultiMetrics(Metric[torch.Tensor]):
    def __init__(self,
                 apply_softmax: bool = True,
                 apply_argmax: bool = False,
                 device: Optional[torch.device] = None):
        super().__init__(device=device)
        self._apply_softmax = apply_softmax
        self._apply_argmax = apply_argmax

    def _pre_process_inputs(self, inputs: torch.Tensor):
        """

        Args:
            inputs: shape: B,C, D, H,W

        """
        if self._apply_softmax:
            inputs = torch.nn.functional.softmax(inputs, 1)
        if self._apply_argmax:
            _, C, _, _, _ = inputs.shape
            inputs = torch.nn.functional.one_hot(torch.argmax(inputs, dim=1),
                                                 num_classes=C).permute(0, 4, 1, 2, 3)

        return inputs

    def compute(self: TSelf) -> TComputeReturn:
        return torch.mean(torch.cat(self.tmp, 0), dim=0)


class SegmentationMultiDiceScores(AbstractSegmentationMultiMetrics):
    def __init__(self,
                 apply_softmax: bool = False,
                 apply_argmax: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__(apply_softmax=apply_softmax,
                         apply_argmax=apply_argmax,
                         device=device)
        self._add_state("tmp", [])

    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        raise NotImplementedError()

    def _dices(self, input: torch.Tensor, target: torch.Tensor):
        num = torch.mul(torch.mul(input, target).sum(dim=[2, 3, 4]), 2)
        denum = torch.mul(input, input).sum(dim=[2, 3, 4]) + torch.mul(target, target).sum(
            dim=[2, 3, 4])
        return torch.divide(num, denum)

    def update(self: TSelf, input: torch.Tensor, target: torch.Tensor) -> TSelf:
        input = self._pre_process_inputs(input)
        self.tmp.append(self._dices(input, target))

    def compute(self: TSelf) -> TComputeReturn:
        return torch.nanmean(torch.cat(self.tmp, 0), dim=0)


class MeanDiceScore(SegmentationMultiDiceScores):
    def __init__(self,
                 apply_softmax: bool = False,
                 apply_argmax: bool = True,
                 device: Optional[torch.device] = None):
        super().__init__(apply_softmax=apply_softmax,
                         apply_argmax=apply_argmax,
                         device=device)
        self._add_state("tmp", [])

    def compute(self: TSelf) -> TComputeReturn:
        return torch.nanmean(torch.cat(self.tmp, -1))

    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        raise NotImplementedError()

    def update(self: TSelf, input: Any, target: Any) -> TSelf:
        input = self._pre_process_inputs(input)
        dice = torch.mean(self._dices(input, target), dim=1)
        self.tmp.append(dice)



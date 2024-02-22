from typing import Any, Iterable, Optional

import torch
from torcheval.metrics import Metric
from torcheval.metrics.metric import TSelf, TComputeReturn


class MeanDiceScore(Metric[torch.Tensor]):
    def __init__(self, device: Optional[torch.device] = None,):
        super().__init__(device=device)
        self._add_state("tmp", [])

    def compute(self: TSelf) -> TComputeReturn:
        return torch.mean(torch.cat(self.tmp, -1))

    def merge_state(self: TSelf, metrics: Iterable[TSelf]) -> TSelf:
        raise NotImplementedError()

    def update(self: TSelf, input: Any, target: Any) -> TSelf:
        num = torch.mul(torch.mul(input, target).sum(dim=[2, 3, 4]), 2)
        denum = torch.mul(input, input).sum(dim=[2, 3, 4]) + torch.mul(target, target).sum(
            dim=[2, 3, 4])
        dice = torch.mean(torch.divide(num, denum), dim=1)
        self.tmp.append(dice)


if __name__ == '__main__':
    import numpy as np

    m1 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
    m2 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
    mdc = MeanDiceScore()
    mdc.update(m1, m2)
    m = mdc.compute()
    assert m.numpy() == 1
    # mdc.reset()
    # m = mdc.compute()
    # print(m)
    m1 =np.ones((4, 3, 10, 10, 10))
    m1[:2,:,:,:,:] = 0
    m1 = torch.tensor(m1)
    m2 = torch.tensor(np.ones((4, 3, 10, 10, 10)))
    mdc = MeanDiceScore()
    mdc.update(m1, m2)
    m = mdc.compute()
    assert m.numpy() == 0.5
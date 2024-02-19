from dataclasses import dataclass
from typing import Any, Tuple
from typing import Dict

import torch.optim
from autologging import logged
from torch import nn
from torch.utils.data import DataLoader
from torcheval.metrics import Metric, Mean
from tqdm import tqdm


@dataclass
class TrainingOperatorParams:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    loss: Any
    metrics: Dict[str, Metric[torch.Tensor]]
    train_data_loader: DataLoader
    val_data_loader: DataLoader
    nb_epochs: int
    cuda_enabled: bool = False


@logged
class TrainingOperator:
    def __init__(self,
                 params: TrainingOperatorParams):
        self.inner = params
        self._logs = {}
        self._current_epoch = 0
        self._val_loss = Mean()
        self._train_loss = Mean()

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        d = self.inner.model.device
        inputs, outputs = batch
        self.__log.debug(f"inputs.shape: {inputs.shape}")
        self.__log.debug(f"outputs.shape: {outputs.shape}")
        if self.inner.cuda_enabled:
            inputs = inputs.cuda()
            outputs = outputs.cuda()
        with torch.set_grad_enabled(True):
            y = self.inner.model(inputs)
            l = self.inner.loss(y, outputs)
            self.__log.debug("loss", l)
            # clear gradients
            self.inner.optimizer.zero_grad()
            # backward
            l.backward()
            # update parameters
            self.inner.optimizer.step()
            self._train_loss.update(l)

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, outputs = batch
        if self.inner.cuda_enabled:
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        y = self.inner.model(inputs)
        l = self.inner.loss(y, outputs)
        self.__log.debug(f"val loss: {l}")
        self._val_loss.update(l)
        for m in self.inner.metrics.values():
            m.update(y, outputs)
        return l

    def on_epoch_end(self):
        self._logs[self._current_epoch]["metrics"][
            "train_loss"] = self._train_loss.compute().cpu().numpy().item()

    def on_val_end(self):
        self._logs[self._current_epoch]["metrics"][
            "val_loss"] = self._val_loss.compute().cpu().numpy().item()
        for name, m in self.inner.metrics.items():
            self._logs[self._current_epoch]["metrics"][
                name] = m.compute().detach().cpu().numpy().item()

    def fit(self):
        for i in range(self.inner.nb_epochs):
            self._logs.setdefault(self._current_epoch, {"metrics": {}})
            self._val_loss.reset()
            self._train_loss.reset()
            self.inner.model.train()
            for batch in tqdm(iter(self.inner.train_data_loader)):
                self.train_step(batch)
            self.inner.model.eval()
            for batch in iter(self.inner.val_data_loader):
                self.val_step(batch)
            self.on_val_end()
            self._current_epoch += 1


if __name__ == '__main__':
    import torch
    from torcheval.metrics import BinaryAUROC

    metric = BinaryAUROC()
    input = torch.tensor([0.1, 0.5, 0.7, 0.8])
    input2 = torch.tensor([0.9, 0.5, 0.7, 0.8])

    target = torch.tensor([1, 0, 1, 1])
    target2 = torch.tensor([1, 0, 1, 1])

    metric.update(input, target)
    print(metric.compute())
    metric = BinaryAUROC()
    metric.update(input2, target2)
    print(metric.compute())
    metric = BinaryAUROC()
    metric.update(input, target)
    metric.update(input2, target2)
    print(metric.compute())

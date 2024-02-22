import json
import os.path
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
    weights_dir: str
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

    def _preprocess(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, target = batch

        if self.inner.cuda_enabled:
            inputs = inputs.cuda()
            target = target.cuda()

        return inputs, target

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, outputs = self._preprocess(batch)
        self.__log.debug(f"inputs.shape: {inputs.shape}")
        self.__log.debug(f"outputs.shape: {outputs.shape}")

        with torch.set_grad_enabled(True):
            y = self.inner.model(inputs)
            self.__log.debug(f"y.shape: {y.shape}")
            l = self.inner.loss(y, outputs)
            self.__log.debug(f"loss: {l}")
            # clear gradients
            self.inner.optimizer.zero_grad()
            # backward
            l.backward()
            # update parameters
            self.inner.optimizer.step()
            self._train_loss.update(l.detach().cpu())

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, outputs = self._preprocess(batch)
        with torch.no_grad():
            y = self.inner.model(inputs)
            l = self.inner.loss(y, outputs)

        y = y.detach().cpu()
        outputs = outputs.detach().cpu()
        self.__log.debug(f"val loss: {l}")
        self._val_loss.update(l.detach().cpu())
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
            m.reset()
        torch.save(self.inner.model.state_dict(),
                   os.path.join(self.inner.weights_dir, f"{self._current_epoch}.pt"))
        with open("logs.json", "w") as f:
            f.write(json.dumps(self._logs))

    def fit(self):
        self._logs.setdefault(self._current_epoch, {"metrics": {}})
        self.inner.model.eval()
        for batch in iter(self.inner.val_data_loader):
            self.val_step(batch)
        self.on_val_end()
        self._current_epoch += 1
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

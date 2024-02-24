import json
import os.path
import re
from dataclasses import dataclass
from typing import Any, Tuple, Dict

import numpy as np
import torch.optim
from autologging import logged
from torch.utils.data import DataLoader
from torcheval.metrics import Metric, Mean
from tqdm import tqdm

from src.python.models.unetr import AbstractPretrainableModel


@dataclass
class ReloadWeightsConfig:
    enabled: bool = True
    mode: str = "reload_last"


@dataclass
class TrainingOperatorParams:
    model: AbstractPretrainableModel
    optimizer: torch.optim.Optimizer
    loss: Any
    metrics: Dict[str, Metric[torch.Tensor]]
    train_data_loader: DataLoader
    val_data_loader: DataLoader
    nb_epochs: int
    exp_dir: str
    weights_dir: str
    reload_weights: ReloadWeightsConfig
    device: torch.device
    evaluate: bool


@logged
class TrainingOperator:
    def __init__(self, params: TrainingOperatorParams):
        self.inner = params
        self._logs = {}
        self._current_epoch = 0
        self._val_loss = Mean(device=self.inner.device)
        self._train_loss = Mean(device=self.inner.device)
        self.reload_weights()

    def _log_path(self) -> str:
        return os.path.join(self.inner.exp_dir, "logs.json")

    def reload_weights(self):
        if self.inner.reload_weights.enabled:
            if os.path.isfile(self._log_path()):
                with open(self._log_path()) as f:
                    self._logs = json.load(f)
            weights_regex = re.compile("(\d+).pt")
            epoch_to_weights = {}
            for file in os.listdir(self.inner.weights_dir):
                if regex_res := weights_regex.search(file):
                    epoch = int(regex_res.group(1))
                    epoch_to_weights[epoch] = os.path.join(self.inner.weights_dir, file)
            if len(epoch_to_weights) > 0:
                last_epoch = np.max(list(epoch_to_weights.keys()))
                w_path = epoch_to_weights[last_epoch]
                if self.inner.reload_weights.mode == "pretrained":
                    self.inner.model.load_from_pretrained(w_path)
                elif self.inner.reload_weights.mode == "reload_last":
                    self.inner.model.load_state_dict(
                        torch.load(w_path, map_location=self.inner.device))
                else:
                    raise NotImplementedError(f"{self.inner.reload_weights.mode} does not exist")
                self._current_epoch = last_epoch + 1
                self.__log.info(f"Loaded weights from epoch {last_epoch}: {w_path}")  # type: ignore
                self.__log.info(f"Will start epoch: {self._current_epoch}")  # type: ignore

    def _preprocess(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, target = batch

        inputs = inputs.to(self.inner.device)
        target = target.to(self.inner.device)
        return inputs, target

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], logger: Any):
        inputs, outputs = self._preprocess(batch)
        self.__log.debug(f"inputs.shape: {inputs.shape}")  # type: ignore
        self.__log.debug(f"outputs.shape: {outputs.shape}")  # type: ignore

        with torch.set_grad_enabled(True):
            y = self.inner.model(inputs)
            self.__log.debug(f"y.shape: {y.shape}")  # type: ignore
            l = self.inner.loss(y, outputs)
            self.__log.debug(f"loss: {l}")  # type: ignore
            # clear gradients
            self.inner.optimizer.zero_grad()
            # backward
            l.backward()
            # update parameters
            self.inner.optimizer.step()
            l = l.detach().cpu()
            logger.set_postfix(loss=f"{l}")
            self._train_loss.update(l)

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, outputs = self._preprocess(batch)
        with torch.no_grad():
            y = self.inner.model(inputs)
            l = self.inner.loss(y, outputs)
            self.__log.debug(f"val loss: {l}")  # type: ignore
            self._val_loss.update(l)
            for m in self.inner.metrics.values():
                m.update(y, outputs)

    def on_epoch_start(self):
        self._logs.setdefault(self._current_epoch, {"metrics": {}})
        self._train_loss.reset()

    def on_eval_start(self):
        self.inner.model.eval()
        self._val_loss.reset()

    def on_epoch_end(self):
        self._logs[self._current_epoch]["metrics"][
            "train_loss"] = self._train_loss.compute().detach().cpu().numpy().item()
        torch.save(self.inner.model.state_dict(),
                   os.path.join(self.inner.weights_dir, f"{self._current_epoch}.pt"))

    def on_val_end(self):
        self._logs[self._current_epoch]["metrics"][
            "val_loss"] = self._val_loss.compute().detach().cpu().numpy().item()
        for name, m in self.inner.metrics.items():
            self._logs[self._current_epoch]["metrics"][
                name] = m.compute().detach().cpu().numpy().tolist()
            m.reset()
        self.__log.info(f"logs: {self._logs[self._current_epoch]}")  # type: ignore
        with open(self._log_path(), "w") as f:
            f.write(json.dumps(self._logs))

    def fit(self):
        for i in range(self.inner.nb_epochs):
            self.on_epoch_start()
            self.inner.model.train()
            with tqdm(iter(self.inner.train_data_loader),
                      desc=f"train epoch: {self._current_epoch}") as logger:
                for batch in logger:
                    self.train_step(batch, logger)
            self.on_epoch_end()
            if self.inner.evaluate:
                self.evaluate_epoch()
            self._current_epoch += 1

    def evaluate_epoch(self):
        self.on_eval_start()
        for batch in tqdm(iter(self.inner.val_data_loader),
                          desc=f"evaluating: {self._current_epoch}"):
            self.val_step(batch)
        self.on_val_end()

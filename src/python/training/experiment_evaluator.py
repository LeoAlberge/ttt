import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from autologging import logged
from torch.utils.data import DataLoader
from torcheval.metrics import Metric, Mean
from tqdm import tqdm

from src.python.models.unetr import AbstractPretrainableModel


@dataclass
class ExperimentEvaluationParams:
    model: AbstractPretrainableModel
    loss: Any
    metrics: Dict[str, Metric[torch.Tensor]]
    val_data_loader: DataLoader
    weights_dir: str
    device: torch.device


@logged
class ExperimentEvaluator:
    def __init__(self, params: ExperimentEvaluationParams):
        self.inner = params
        self._logs = {}
        self._val_loss = Mean(device=self.inner.device)
        self._epoch_to_weights = {}
        self._current_epoch= None
    def look_for_checkpoints(self):
        weights_regex = re.compile("(\d+).pt")
        for file in os.listdir(self.inner.weights_dir):
            if regex_res := weights_regex.search(file):
                epoch = int(regex_res.group(1))
                self._epoch_to_weights[epoch] = os.path.join(self.inner.weights_dir, file)

    def _preprocess(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, target = batch
        inputs = inputs.to(self.inner.device)
        target = target.to(self.inner.device)
        return inputs, target

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        inputs, outputs = self._preprocess(batch)
        with torch.no_grad():
            y = self.inner.model(inputs)
            l = self.inner.loss(y, outputs)
            self.__log.debug(f"val loss: {l}")  # type: ignore
            self._val_loss.update(l)
            for m in self.inner.metrics.values():
                m.update(y, outputs)

    def on_val_end(self):
        self._logs[self._current_epoch]["metrics"][
            "val_loss"] = self._val_loss.compute().detach().cpu().numpy().item()
        for name, m in self.inner.metrics.items():
            self._logs[self._current_epoch]["metrics"][
                name] = m.compute().detach().cpu().numpy().tolist()
            m.reset()
        self.__log.info(f"logs: {self._logs[self._current_epoch]}")  # type: ignore
        with open("logs.json", "w") as f:
            f.write(json.dumps(self._logs))

    def evaluate(self):
        self.look_for_checkpoints()
        for epoch, w_path in self._epoch_to_weights.items():
            self._current_epoch = epoch
            self._logs.setdefault(self._current_epoch, {"metrics": {}})

            self.inner.model.load_state_dict(
                torch.load(w_path, map_location=self.inner.device))
            self.inner.model.eval()
            self._val_loss.reset()
            for batch in tqdm(iter(self.inner.val_data_loader),
                              desc=f"evaluating: {epoch}"):
                self.val_step(batch)
            self.on_val_end()



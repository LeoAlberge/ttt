import abc
from typing import Any, List

import torch


class TransformBlock(abc.ABC):
    @abc.abstractmethod
    def __call__(self, inputs: Any, target: Any):
        pass


class ComposeTransform(TransformBlock):

    def __init__(self, transform_l: List[TransformBlock]):
        self._transform_l = transform_l

    def __call__(self, inputs: Any, target: Any):
        for t in self._transform_l:
            inputs, target = t(inputs, target)
        return inputs, target


class ToTensor(TransformBlock):
    def __init__(self, inputs_dtype: torch.dtype, output_dtype: torch.dtype):
        self._inputs_dtype = inputs_dtype
        self._output_dtype = output_dtype

    def __call__(self, inputs: Any, target: Any):
        return torch.tensor(inputs, dtype=self._inputs_dtype), torch.tensor(target,
                                                                            dtype=self._output_dtype)

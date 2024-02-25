from typing import Dict, List

import numpy as np
import torch
from autologging import logged
from torch import nn
from tqdm import tqdm

from src.python.core.benchmarks import timeit


def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


@logged
class SlidingWindowInference:
    def __init__(self, model: nn.Module):
        self._model = model
        self._model.eval()

    @timeit("predict patch")
    def _predict(self, patch):
        patch = torch.tensor(patch[np.newaxis, np.newaxis, :])
        return np.argmax(self._model.forward(patch).detach().cpu().numpy()[0, :], axis=0)
        # return torch.nn.functional.softmax(self._model.forward(patch),
        #                                    dim=1).detach().cpu().numpy()[0, :]


    @timeit("SlidingWindowInference.run")
    def run(self,
            input_volume: np.ndarray,
            patch_size=(96, 96, 96),
            step_size=(48, 48, 48),
            num_classes=2):
        input_shape = input_volume.shape

        # Pad the input volume
        padded_shape = tuple(
            (input_shape[i] + patch_size[i] - 1) // patch_size[i] * patch_size[i] for i in range(3))
        pad_width = [(0, padded_shape[i] - input_shape[i]) for i in range(3)]
        input_volume_padded = np.pad(input_volume, pad_width, constant_values=0)
        d, h, w = input_volume_padded.shape
        input_shape = input_volume_padded.shape

        output_volume = np.zeros((d, h, w), dtype=np.uint8)
        # count_volume = np.zeros(input_shape, dtype=np.int32)
        progress_bar = tqdm(desc="predicting over batches",
                            total=((input_shape[0] - patch_size[0]) // step_size[0] + 1) * (
                                    (input_shape[1] - patch_size[1]) // step_size[1] + 1) * (
                                          (input_shape[2] - patch_size[2]) // step_size[2] + 1))

        # Iterate over the input volume with sliding window
        for z in range(0, input_shape[0] - patch_size[0] + 1, step_size[0]):
            for y in range(0, input_shape[1] - patch_size[1] + 1, step_size[1]):
                for x in range(0, input_shape[2] - patch_size[2] + 1, step_size[2]):

                    # Extract patch
                    patch = input_volume_padded[
                            z:z + patch_size[0],
                            y:y + patch_size[1],
                            x:x + patch_size[2]]

                    # Perform inference on the patch (You should replace this with your DL model
                    # inference code)
                    # predicted_patch = your_dl_model_inference_function(patch)
                    # Here, for demonstration purposes, I'm just setting random values
                    predicted_patch = self._predict(patch)

                    # Aggregate predictions
                    output_volume[z:z + patch_size[0], y:y + patch_size[1],
                    x:x + patch_size[2]] += predicted_patch.astype(np.uint8)
                    # count_volume[:, z:z + patch_size[0], y:y + patch_size[1],
                    # x:x + patch_size[2]] += 1
                    progress_bar.update(1)

        # Calculate mean predictions
        # output_volume /= count_volume
        # class_mask = (count_volume > 0).astype(np.float32)
        # output_volume *= class_mask
        # output_volume = np.argmax(output_volume,0)
        res = unpad(output_volume, pad_width)
        return res

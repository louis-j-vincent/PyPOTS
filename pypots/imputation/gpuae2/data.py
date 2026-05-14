"""Dataset class for GP-UAE2 image sequences with explicit masks."""

from typing import Iterable, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def _to_float_tensor(array_like) -> torch.Tensor:
    if isinstance(array_like, torch.Tensor):
        return array_like.to(torch.float32)
    return torch.as_tensor(np.asarray(array_like), dtype=torch.float32)


class DatasetForGPVAE(Dataset):
    """Dataset for GP-UAE2.

    Expected format:
    - required keys: ``X``, ``mask``
    - optional validation keys: ``X_ori``, ``indicating_mask``
    - required shapes for ``X`` and ``mask``: ``(N, T, 28, 28)``
    """

    def __init__(
        self,
        data: Union[dict, str],
        return_X_ori: bool,
        return_y: bool,
        file_type: str = "hdf5",
    ):
        del return_y, file_type
        if isinstance(data, str):
            raise TypeError(
                "GP_UAE2 now accepts dict-style in-memory data only. "
                "Please pass a dict with keys 'X' and 'mask'."
            )
        if not isinstance(data, dict):
            raise TypeError(f"Expected `data` to be a dict, got {type(data)}.")

        self.return_X_ori = return_X_ori
        self.X = _to_float_tensor(data.get("X"))
        self.mask = _to_float_tensor(data.get("mask"))

        if self.X.ndim != 4 or self.mask.ndim != 4:
            raise ValueError(
                "Both `X` and `mask` must be 4D tensors with shape (N, T, 28, 28). "
                f"Got X{tuple(self.X.shape)} and mask{tuple(self.mask.shape)}."
            )
        if self.X.shape != self.mask.shape:
            raise ValueError(
                f"`X` and `mask` must have the same shape, got X{tuple(self.X.shape)} "
                f"and mask{tuple(self.mask.shape)}."
            )
        if self.X.shape[-2:] != (28, 28):
            raise ValueError(
                f"GP_UAE2 currently supports only 28x28 images. Got shape {tuple(self.X.shape)}."
            )

        if self.return_X_ori:
            if "X_ori" not in data:
                raise ValueError("Validation dataset must contain key `X_ori`.")
            if "indicating_mask" not in data:
                raise ValueError("Validation dataset must contain key `indicating_mask`.")
            self.X_ori = _to_float_tensor(data["X_ori"])
            self.indicating_mask = _to_float_tensor(data["indicating_mask"])
            if self.X_ori.shape != self.X.shape or self.indicating_mask.shape != self.X.shape:
                raise ValueError(
                    "`X_ori` and `indicating_mask` must match `X` shape. "
                    f"Got X_ori{tuple(self.X_ori.shape)} and indicating_mask{tuple(self.indicating_mask.shape)}."
                )

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Iterable:
        sample = [torch.tensor(idx), self.X[idx], self.mask[idx]]
        if self.return_X_ori:
            sample.extend([self.X_ori[idx], self.indicating_mask[idx]])
        return sample

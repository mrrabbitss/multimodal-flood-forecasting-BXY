from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .dataset import FloodSequenceDataset


def normalize_lead_times(values: Sequence[int]) -> tuple[int, ...]:
    lead_times = tuple(int(value) for value in values)
    if not lead_times or any(value < 1 for value in lead_times):
        raise ValueError("lead_times must contain positive integers")
    if len(set(lead_times)) != len(lead_times):
        raise ValueError("lead_times must not contain duplicates")
    return tuple(sorted(lead_times))


class MultiHorizonFloodDataset(FloodSequenceDataset):
    """Return one input window and aligned targets for several future leads."""

    def __init__(
        self,
        fused_dir: str | Path,
        event_indices: Sequence[int],
        input_len: int = 12,
        lead_times: Sequence[int] = (1, 3, 6, 12, 24),
        channel_names: str | Sequence[str] | None = None,
        target: str = "gt_depth",
    ) -> None:
        self.lead_times = normalize_lead_times(lead_times)
        super().__init__(
            fused_dir=fused_dir,
            event_indices=event_indices,
            input_len=input_len,
            lead_time=max(self.lead_times),
            channel_names=channel_names,
            target=target,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[index]
        artifact = self._load(item.file_path)
        all_channels = self._build_channels(artifact)
        end = item.start + self.input_len
        x = all_channels[item.start:end]
        targets = np.stack(
            [artifact[self.target][end - 1 + lead] for lead in self.lead_times],
            axis=0,
        ).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(targets)

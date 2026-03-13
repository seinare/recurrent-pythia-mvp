from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset


DEFAULT_SPLITS = {
    "train": (0, 90_000_000),
    "val": (90_000_000, 95_000_000),
    "test": (95_000_000, 100_000_000),
}


@dataclass(frozen=True)
class DatasetMetadata:
    data_path: Path
    split: str
    seq_len: int
    total_tokens: int
    num_sequences: int
    vocab_size: int


class CharDataset(Dataset[Dict[str, torch.Tensor]]):
    """Byte-level character dataset for enwik8/text8 style language modeling."""

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
        split: str = "train",
        stride: Optional[int] = None,
        split_offsets: Optional[dict[str, tuple[int, int]]] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.split = split
        self.stride = stride or seq_len
        self.split_offsets = split_offsets

        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.data_path}. "
                "Download enwik8/text8 first or pass the correct path."
            )

        raw_bytes = self.data_path.read_bytes()
        effective_offsets = self._resolve_split_offsets(raw_bytes)
        if split not in effective_offsets:
            raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(effective_offsets)}")
        self.split_offsets = self._resolve_split_offsets(raw_bytes)
        split_start, split_end = self.split_offsets[split]
        split_end = min(split_end, len(raw_bytes))
        if split_start >= len(raw_bytes):
            raise ValueError(
                f"Split '{split}' starts at {split_start}, but file only has {len(raw_bytes)} bytes."
            )

        self._buffer = torch.tensor(list(raw_bytes[split_start:split_end]), dtype=torch.long)
        if self._buffer.numel() <= self.seq_len:
            raise ValueError(
                f"Split '{split}' is too short for seq_len={seq_len}. "
                f"Available bytes: {self._buffer.numel()}."
            )

        # enwik8 byte modeling naturally uses 256 tokens; text8 will use the observed byte range.
        self._vocab = sorted(set(raw_bytes[: min(DEFAULT_SPLITS["train"][1], len(raw_bytes))]))
        self._token_to_id = {token: idx for idx, token in enumerate(self._vocab)}
        self._id_lut = self._build_lookup_table(self._token_to_id)
        self._encoded = self._id_lut[self._buffer]
        self._num_sequences = ((self._encoded.numel() - 1) - self.seq_len) // self.stride + 1

    def _resolve_split_offsets(self, raw_bytes: bytes) -> dict[str, tuple[int, int]]:
        if self.split_offsets is not None:
            return self.split_offsets

        if len(raw_bytes) >= DEFAULT_SPLITS["test"][1]:
            return DEFAULT_SPLITS

        total = len(raw_bytes)
        train_end = max(int(total * 0.9), 2)
        val_end = max(train_end + int(total * 0.05), train_end + 1)
        val_end = min(val_end, total - 1)
        return {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, total),
        }

    @staticmethod
    def _build_lookup_table(token_to_id: Dict[int, int]) -> torch.Tensor:
        lut = torch.full((256,), -1, dtype=torch.long)
        for token, idx in token_to_id.items():
            lut[token] = idx
        if (lut < 0).any():
            missing = int((lut < 0).sum().item())
            # For text8 and similar corpora, unseen bytes simply remain absent from the split vocabulary.
            # They will never appear in the encoded data if the vocabulary is built from the same corpus.
            if missing == 256:
                raise ValueError("Vocabulary lookup table is empty.")
        return lut

    def __len__(self) -> int:
        return self._num_sequences

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index < 0 or index >= self._num_sequences:
            raise IndexError(index)
        start = index * self.stride
        end = start + self.seq_len
        input_ids = self._encoded[start:end]
        labels = self._encoded[start + 1 : end + 1]
        return {
            "input_ids": input_ids.clone(),
            "labels": labels.clone(),
        }

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            data_path=self.data_path,
            split=self.split,
            seq_len=self.seq_len,
            total_tokens=int(self._encoded.numel()),
            num_sequences=self._num_sequences,
            vocab_size=self.vocab_size,
        )

    def decode(self, token_ids: torch.Tensor) -> bytes:
        ids = token_ids.detach().cpu().tolist()
        return bytes(self._vocab[idx] for idx in ids)


def build_dataset(
    data_path: str | Path,
    seq_len: int,
    split: str = "train",
    stride: Optional[int] = None,
) -> CharDataset:
    return CharDataset(data_path=data_path, seq_len=seq_len, split=split, stride=stride)

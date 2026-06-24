"""Pre-tokenized dataset: loads .pt shards with shard-level caching.

No index building, no full-shard loading — reads manifest, computes offsets on-the-fly.
"""

from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PreTokenizedDataset(Dataset):
    """Reads pre-tokenized shards. Caches a few shards for random access."""

    def __init__(self, tokenized_dir: str, split: str = "train",
                 skip_samples: int = 0, max_shard_cache: int = 4):
        self.split = split
        self.tokenized_dir = tokenized_dir
        self.max_shard_cache = max_shard_cache

        # Read manifest (tiny file, no shard loading)
        manifest_path = Path(tokenized_dir) / f"{split}_manifest.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        manifest = torch.load(manifest_path, map_location="cpu", weights_only=False)
        self.total_samples = manifest["total_samples"]
        shard_size = manifest["shard_size"]

        # Discover shard files
        shard_files = sorted(Path(tokenized_dir).glob(f"{split}.*.pt"))
        if not shard_files:
            raise FileNotFoundError(
                f"No pre-tokenized shards found in {tokenized_dir} for split '{split}'"
            )

        # Compute per-shard ranges from manifest (no file reads)
        self.shards = []  # [(path, start_sample, end_sample), ...]
        for sf in shard_files:
            start = len(self.shards) * shard_size
            end = min(start + shard_size, self.total_samples)
            if end > start:
                self.shards.append((sf, start, end))

        # Sequential access (no shuffle) for shard-cache efficiency.
        # Skip moves the starting point forward; remaining data is contiguous.
        self._offset = skip_samples
        self._length = self.total_samples - skip_samples

        # Shard cache
        self._cache = OrderedDict()

        print(f"  [{split}-pt] {self.total_samples} samples → {self._length} after skip="
              f"{skip_samples}  ({len(self.shards)} shards, sequential)")

    def _load_shard(self, path):
        """Load and cache a shard, evicting oldest if needed."""
        path = str(path)
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        if len(self._cache) >= self.max_shard_cache:
            self._cache.popitem(last=False)

        data = torch.load(path, map_location="cpu", weights_only=False)
        self._cache[path] = data
        return data

    def _global_to_local(self, global_idx):
        """Map a global sample index to (shard_path, local_offset)."""
        # Binary search (only 8 shards, instantaneous)
        lo, hi = 0, len(self.shards) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            sf, start, end = self.shards[mid]
            if global_idx < start:
                hi = mid - 1
            elif global_idx >= end:
                lo = mid + 1
            else:
                return sf, global_idx - start
        raise IndexError(f"global index {global_idx} out of range")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        global_idx = self._offset + idx  # sequential: offset + local index
        sf, local_idx = self._global_to_local(global_idx)
        data = self._load_shard(sf)
        return {
            "src_ids": data["src_ids"][local_idx].long(),
            "src_mask": data["src_mask"][local_idx],
            "tgt_ids": data["tgt_ids"][local_idx].long(),
            "decoder_input": data["decoder_input"][local_idx].long(),
            "tgt_pad_mask": data["tgt_pad_mask"][local_idx],
        }

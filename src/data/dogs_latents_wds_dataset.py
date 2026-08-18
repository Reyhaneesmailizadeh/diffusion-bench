"""Dogs latents WebDataset loader — reads pre-computed VAE/text/DINOv2 tensors."""
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import webdataset as wds


def _count_tar_samples(path: Path) -> int:
    """Count samples in a shard by counting '.latent.npy' members (one per sample).

    Reads only tar headers, not member contents, but scanning still costs ~1-3s per
    GB of shard on network storage, so callers should cache the result (see
    _shard_sample_counts below) rather than call this on every dataset load.
    """
    with tarfile.open(path) as tf:
        return sum(1 for name in tf.getnames() if name.endswith(".latent.npy"))


def _shard_sample_counts(data_dir: Path, tar_files: list) -> dict:
    """Get exact sample counts per shard, cached in a sidecar file keyed by (mtime, size).

    A full tar scan takes seconds to minutes depending on shard size, and every
    torchrun rank instantiates its own DogsLatentsWebDataset, so without caching this
    cost is paid repeatedly on every training launch. The cache is invalidated per-file
    if its mtime/size changes (e.g. shards regenerated).
    """
    cache_path = data_dir / ".shard_sample_counts.json"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    counts = {}
    dirty = False
    for f in tar_files:
        stat = f.stat()
        entry = cache.get(f.name)
        if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
            counts[f.name] = entry["count"]
            continue
        count = _count_tar_samples(f)
        counts[f.name] = count
        cache[f.name] = {"mtime": stat.st_mtime, "size": stat.st_size, "count": count}
        dirty = True

    if dirty:
        try:
            cache_path.write_text(json.dumps(cache))
        except OSError:
            pass  # e.g. read-only data dir; caching is an optimization, not required for correctness

    return counts


class DogsLatentsWebDataset:
    """
    Reads pre-computed latent shards produced by scripts/precompute_latents.py.

    Each tar sample contains:
        latent.npy      [C, H, W]              float16 → float32
        tokens.npy      [seq_len, dim]          float16 → float32
        attn_mask.npy   [seq_len]               bool
        dinov2.npy      [num_patches, dim]      float16 → float32 (present when RePA was used)
        attn_target.npy [dit_grid_size**2]      float16 → float32 (present when attn_align was used)

    Returns (latent, tokens, attn_mask, dinov2, attn_target) tuples where dinov2/attn_target
    are zero tensors (shape [1]) when not present, so WebLoader can collate homogeneously.
    """

    def __init__(self, data_dir: str, shuffle_buffer: int = 5000, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        tar_files = sorted(self.data_dir.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No tar shards found in {data_dir}. Run scripts/precompute_latents.py first.")
        self._shard_urls = [str(f) for f in tar_files]
        self._num_shards = len(self._shard_urls)
        self._num_samples = sum(_shard_sample_counts(self.data_dir, tar_files).values())

    @property
    def estimated_size(self) -> int:
        return self._num_samples

    @property
    def num_shards(self) -> int:
        return self._num_shards

    def _decode_sample(self, sample):
        try:
            latent = torch.from_numpy(np.load(io.BytesIO(sample["latent.npy"])).astype(np.float32))
            tokens = torch.from_numpy(np.load(io.BytesIO(sample["tokens.npy"])).astype(np.float32))
            attn_mask = torch.from_numpy(np.load(io.BytesIO(sample["attn_mask.npy"])))
            if "dinov2.npy" in sample:
                dinov2 = torch.from_numpy(np.load(io.BytesIO(sample["dinov2.npy"])).astype(np.float32))
            else:
                dinov2 = torch.zeros(1)
            if "attn_target.npy" in sample:
                attn_target = torch.from_numpy(np.load(io.BytesIO(sample["attn_target.npy"])).astype(np.float32))
            else:
                attn_target = torch.zeros(1)
            return latent, tokens, attn_mask, dinov2, attn_target
        except Exception:
            return None

    def create_pipeline(self, epoch: int = 0, shuffle: bool = True) -> wds.WebDataset:
        pipeline = wds.WebDataset(
            self._shard_urls,
            nodesplitter=wds.split_by_node if shuffle else None,
            shardshuffle=self._num_shards if shuffle else False,
            seed=self.seed + epoch,
        )
        if shuffle:
            pipeline = pipeline.shuffle(self.shuffle_buffer, initial=self.shuffle_buffer // 2)
        return (
            pipeline
            .map(self._decode_sample, handler=wds.ignore_and_continue)
            .select(lambda x: x is not None)
        )

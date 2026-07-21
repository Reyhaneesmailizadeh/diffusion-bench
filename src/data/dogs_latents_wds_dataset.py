"""Dogs latents WebDataset loader — reads pre-computed VAE/text/DINOv2 tensors."""
import io
from pathlib import Path

import numpy as np
import torch
import webdataset as wds

DOGS_LATENTS_NUM_SAMPLES = 26000


class DogsLatentsWebDataset:
    """
    Reads pre-computed latent shards produced by scripts/precompute_latents.py.

    Each tar sample contains:
        latent.npy    [C, H, W]              float16 → float32
        tokens.npy    [seq_len, dim]          float16 → float32
        attn_mask.npy [seq_len]               bool
        dinov2.npy    [num_patches, dim]      float16 → float32 (present when RePA was used)

    Returns (latent, tokens, attn_mask, dinov2) tuples where dinov2 is a zero tensor
    of shape [1] when not present (so WebLoader can collate homogeneously).
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

    @property
    def estimated_size(self) -> int:
        return self._num_shards * 500

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
            return latent, tokens, attn_mask, dinov2
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

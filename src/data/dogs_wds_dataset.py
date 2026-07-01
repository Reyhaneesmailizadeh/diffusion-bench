"""Dogs WebDataset loader for T2I training."""
from pathlib import Path
from typing import Optional

import webdataset as wds
from torchvision import transforms

DOGS_NUM_SAMPLES = 26000


def _filter_valid_samples(sample):
    return sample[0] is not None


class DogsWebDataset:
    """
    WebDataset wrapper for the dog breed dataset.

    Expects tar shards produced by scripts/pack_dogs_wds.py, where each
    sample contains a .jpg image and a .txt caption.
    Returns (image_tensor, caption_string) pairs.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 256,
        shuffle_buffer: int = 5000,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        tar_files = sorted(self.data_dir.glob("*.tar"))
        if not tar_files:
            raise ValueError(f"No tar shards found in {data_dir}. Run scripts/pack_dogs_wds.py first.")

        self._shard_urls = [str(f) for f in tar_files]
        self._num_shards = len(self._shard_urls)

        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    @property
    def estimated_size(self) -> int:
        return DOGS_NUM_SAMPLES

    @property
    def num_shards(self) -> int:
        return self._num_shards

    def _decode_sample(self, sample):
        image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")

        caption_raw = sample.get("txt", b"")
        caption = caption_raw.decode("utf-8").strip() if isinstance(caption_raw, bytes) else str(caption_raw).strip()

        if image is not None:
            image = self.transform(image)

        return image, caption

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
            .decode("pil", handler=wds.ignore_and_continue)
            .map(self._decode_sample, handler=wds.ignore_and_continue)
            .select(_filter_valid_samples)
        )

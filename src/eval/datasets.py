"""Evaluation dataset utilities for multi-dataset support."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset

from data.unified_dataloader import prepare_unified_dataloader


class ListDataset(Dataset):
    """Map-style dataset backed by a pre-loaded list of (image, condition) pairs."""
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

logger = logging.getLogger(__name__)


@dataclass
class EvalDatasetInfo:
    """Container for eval dataset info."""
    dataset: Dataset
    reference_npz: Optional[Union[str, List[str]]]  # Only required when 'fid' in metrics
    condition_type: str
    metrics: List[str] = field(default_factory=lambda: ['fid'])


def normalize_eval_datasets(datasets_cfg):
    """
    Normalize eval.datasets config to dict of {name: dataset_config}.

    Supported format:
        eval.datasets = {mscoco: {...}, mjhq: {...}}

    Returns dict of {name: dataset_config}. If no datasets are configured, returns an empty dict.
    """
    result = {}
    for name, cfg in datasets_cfg.items():
        if 'fid' in cfg['metrics'] and 'reference_npz' not in cfg:
            raise ValueError(f"eval.datasets.{name} missing 'reference_npz', required for FID eval")
        result[name] = cfg.copy()
        # set target to name if not explicitly provided (for simpleeval different versions)
        if 'target' not in result[name]:
            result[name]['target'] = name
    return result


def prepare_eval_datasets(
    eval_datasets_config: Dict[str, dict],
    image_size: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> Dict[str, EvalDatasetInfo]:
    """
    Prepare eval datasets from normalized config.

    Returns dict of {name: EvalDatasetInfo}.
    """
    eval_datasets = {}

    for ds_name, ds_cfg in eval_datasets_config.items():
        ds_cond_type = ds_cfg.get('condition_type', 'text')

        result = prepare_unified_dataloader(
            config=ds_cfg,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            condition_type=ds_cond_type,
            shuffle=False,
        )

        if result.is_iterable:
            logger.info(f"Materializing iterable eval dataset '{ds_name}' (conditions only)...")
            conditions = []
            for batch in result.loader:
                _, conds = batch  # discard images — only conditions needed for generation
                conditions.extend(conds if isinstance(conds, (list, tuple)) else conds.tolist())
            dataset = ListDataset(conditions)
        else:
            base = result.loader.dataset
            dataset = ListDataset([base[i][1] for i in range(len(base))])

        eval_datasets[ds_name] = EvalDatasetInfo(
            dataset=dataset,
            reference_npz=ds_cfg.get('reference_npz'),
            condition_type=ds_cond_type,
            metrics=ds_cfg.get('metrics', ['fid']),
        )
        logger.info(f"Eval dataset loaded: {ds_name}, {len(dataset)} samples")

    return eval_datasets

"""Distributed generation evaluation — FID, CLIPScore, VQAScore, GenEval, DPG-Bench."""

import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from tqdm import tqdm

from .clipscore import CLIPScoreEvaluator
from .distributed import create_eval_dataloader
from .dpgbench import DPGEvaluator
from .fid import _fid_from_moments, _inception_score
from .geneval import GenEvalEvaluator
from .vqascore import VQAScoreEvaluator


def _init_evaluators(metrics_to_compute: List[str], condition_type: str, device: torch.device):
    """Initialize metric evaluators and local score accumulators."""
    evaluators = {}
    local_scores = {}

    if 'clipscore' in metrics_to_compute and condition_type == 'text':
        evaluators['clipscore'] = CLIPScoreEvaluator(device=str(device))
        local_scores['clipscore'] = {'sum': 0.0, 'count': 0}

    if any(elem.startswith('vqascore') for elem in metrics_to_compute) and condition_type == 'text':
        vqascore_models = [elem for elem in metrics_to_compute if elem.startswith('vqascore')]
        vqascore_evaluators = {}
        for model_name in vqascore_models:
            model_name_ = model_name.split('_')[-1] if '_' in model_name else 'clip-flant5-xl'
            vqascore_evaluators[model_name] = VQAScoreEvaluator(model_name=model_name_, device=str(device))
            local_scores[model_name] = {'sum': 0.0, 'count': 0}
        evaluators['vqascore'] = vqascore_evaluators

    if 'geneval' in metrics_to_compute and condition_type == 'text':
        evaluators['geneval'] = GenEvalEvaluator(device=str(device))
        local_scores['geneval'] = {'sum': 0.0, 'count': 0}

    if 'dpgbench' in metrics_to_compute and condition_type == 'text':
        evaluators['dpgbench'] = DPGEvaluator(device=str(device))
        local_scores['dpgbench'] = {'sum': 0.0, 'count': 0}

    return evaluators, local_scores


def _aggregate_distributed_metrics(local_scores: dict, device: torch.device) -> Dict[str, float]:
    """All-reduce local score sums/counts across ranks and return averaged metrics."""
    metrics = {}
    for metric_name, scores in local_scores.items():
        sum_tensor = torch.tensor([scores['sum']], device=device, dtype=torch.float64)
        count_tensor = torch.tensor([scores['count']], device=device, dtype=torch.float64)
        dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        if count_tensor.item() > 0:
            metrics[metric_name] = sum_tensor.item() / count_tensor.item()
    return metrics


@torch.no_grad()
def evaluate_generation_distributed(
    model_fn,
    sample_fn,
    latent_size,
    additional_model_kwargs,
    use_guidance: bool,
    rae,
    val_dataset,
    num_samples: int,
    batch_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    experiment_dir: str,
    global_step: int,
    autocast_kwargs: dict,
    metric_batch_size: int = 128,
    reference_npz_path: Optional[str] = None,
    shared_tmpdir: Optional[str] = None,
    condition_type: str = "label",
    null_label: int = 1000,
    text_encoder=None,
    metrics_to_compute: Optional[List[str]] = None,
    cls_dim: Optional[int] = None,
) -> Optional[Dict[str, float]]:
    """
    Evaluate generation metrics using all GPUs in a distributed manner.

    FID is computed by extracting InceptionV3 features inline on each rank as images
    are generated (one batch at a time), then gathering the small feature vectors across
    ranks. No images are written to disk and no large arrays are loaded to GPU at once.

    Returns:
        Dictionary of metrics (only on rank 0, None on other ranks)
    """
    if metrics_to_compute is None:
        metrics_to_compute = ['fid']

    compute_fid = 'fid' in metrics_to_compute or 'inception_score' in metrics_to_compute

    # Sync all ranks before starting eval
    dist.barrier()

    loader = create_eval_dataloader(val_dataset, rank, world_size, num_samples, batch_size)
    evaluators, local_scores = _init_evaluators(metrics_to_compute, condition_type, device)

    # Each rank loads InceptionV3 for parallel inline feature extraction
    inception = None
    if compute_fid:
        inception = FeatureExtractorInceptionV3(
            name="inception-v3-compat", features_list=['2048', 'logits_unbiased']
        ).to(device).eval()

    local_feats: List[torch.Tensor] = []
    local_logits: List[torch.Tensor] = []

    iterator = tqdm(loader, desc=f"[Rank {rank}] Sampling", file=sys.stdout) if rank == 0 else loader

    with torch.inference_mode():
        for cond in iterator:
            if condition_type == "text":
                n = len(cond)
                z = torch.randn(n, *latent_size, device=device)
                enc_out = text_encoder(list(cond))
                context = enc_out["tokens"]
                context_attn_mask = enc_out["attention_mask"]
                if use_guidance:
                    z = torch.cat([z, z], dim=0)
                    enc_null = text_encoder([""] * n)
                    context = torch.cat([context, enc_null["tokens"]], dim=0)
                    context_attn_mask = torch.cat([context_attn_mask, enc_null["attention_mask"]], dim=0)
            else:
                n = cond.size(0)
                z = torch.randn(n, *latent_size, device=device)
                context = cond.to(device)
                context_attn_mask = None
                if use_guidance:
                    z = torch.cat([z, z], dim=0)
                    context = torch.cat([context, torch.full((n,), null_label, device=device)], dim=0)

            model_kwargs = dict(context=context, attn_mask=context_attn_mask, **additional_model_kwargs)
            if cls_dim is not None:
                cls_init = torch.randn(n, cls_dim, device=device)
                model_kwargs["cls_t"] = torch.cat([cls_init, cls_init], dim=0) if use_guidance else cls_init

            with autocast(**autocast_kwargs):
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                if use_guidance:
                    samples = samples.chunk(2, dim=0)[0]
                samples = rae.decode(samples).clamp(0, 1)

            # NCHW uint8 — shared format for InceptionV3 and other metrics
            gen_uint8 = samples.mul(255).clamp(0, 255).to(dtype=torch.uint8)

            # FID/IS: extract InceptionV3 features inline, discard the image
            if compute_fid:
                feats, logits = inception(gen_uint8)
                local_feats.append(feats.cpu())
                local_logits.append(logits.cpu())

            # Other metrics (clipscore etc.) need numpy NHWC
            if evaluators:
                gen_np = gen_uint8.permute(0, 2, 3, 1).cpu().numpy()
                if 'clipscore' in evaluators:
                    batch_scores = evaluators['clipscore'].compute_batch_scores(gen_np, list(cond))
                    local_scores['clipscore']['sum'] += batch_scores.sum().item()
                    local_scores['clipscore']['count'] += len(cond)
                if 'vqascore' in evaluators:
                    for model_name, evaluator in evaluators['vqascore'].items():
                        batch_scores = evaluator.compute_batch_scores(gen_np, list(cond))
                        local_scores[model_name]['sum'] += batch_scores.sum().item()
                        local_scores[model_name]['count'] += len(cond)
                if 'geneval' in evaluators:
                    batch_scores = evaluators['geneval'].compute_batch_scores(gen_np, list(cond))
                    local_scores['geneval']['sum'] += batch_scores.sum().item()
                    local_scores['geneval']['count'] += len(cond)
                if 'dpgbench' in evaluators:
                    batch_scores = evaluators['dpgbench'].compute_batch_scores(gen_np, list(cond))
                    local_scores['dpgbench']['sum'] += batch_scores.sum().item()
                    local_scores['dpgbench']['count'] += len(cond)

    # Free InceptionV3 VRAM before distributed ops
    del inception
    torch.cuda.empty_cache()

    # Aggregate non-FID metrics across ranks
    metrics = _aggregate_distributed_metrics(local_scores, device)

    # FID/IS: gather feature vectors from all ranks, compute on rank 0
    if compute_fid:
        local_feats_t = torch.cat(local_feats, dim=0)    # [local_n, 2048]
        local_logits_t = torch.cat(local_logits, dim=0)  # [local_n, 1008]

        # all_gather requires equal tensor sizes — pad the last rank if needed
        max_chunk = -(-num_samples // world_size)  # ceiling division
        local_n = local_feats_t.shape[0]
        if local_n < max_chunk:
            pad = max_chunk - local_n
            local_feats_t = torch.cat([local_feats_t, local_feats_t.new_zeros(pad, local_feats_t.shape[1])])
            local_logits_t = torch.cat([local_logits_t, local_logits_t.new_zeros(pad, local_logits_t.shape[1])])

        # Move to GPU for NCCL all_gather
        local_feats_t = local_feats_t.to(device)
        local_logits_t = local_logits_t.to(device)

        gathered_feats = [torch.zeros_like(local_feats_t) for _ in range(world_size)]
        gathered_logits = [torch.zeros_like(local_logits_t) for _ in range(world_size)]
        dist.all_gather(gathered_feats, local_feats_t)
        dist.all_gather(gathered_logits, local_logits_t)

        if rank == 0:
            all_feats = torch.cat(gathered_feats, dim=0)[:num_samples].cpu().double().numpy()
            all_logits = torch.cat(gathered_logits, dim=0)[:num_samples].cpu()

            if not reference_npz_path:
                raise ValueError("reference_npz_path is required for FID computation")
            ref = np.load(reference_npz_path)
            mu_ref = ref['mu'] if 'mu' in ref else ref['ref_mu']
            sigma_ref = ref['sigma'] if 'sigma' in ref else ref['ref_sigma']

            mu_gen = all_feats.mean(axis=0)
            sigma_gen = np.cov(all_feats, rowvar=False)

            metrics['fid'] = _fid_from_moments(mu_gen, sigma_gen, mu_ref, sigma_ref)
            metrics['inception_score'] = _inception_score(all_logits)

            print(f"[Eval] Step {global_step} Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")

    dist.barrier()
    return metrics if metrics else None

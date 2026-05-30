import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.model_forward import *

def build_epoch_config(epoch, epoch_stats, CONFIG):

    # Get loss weights from schedule
    shares = CONFIG['LOSS_SCHEDULE'].get(epoch, CONFIG['LOSS_SCHEDULE'][max(CONFIG['LOSS_SCHEDULE'].keys())])

    # First epoch: no stats, use defaults
    if epoch_stats is None:
        return {
            "ce_weight": shares.get("ce"),
            "energy_weight": shares.get("energy"),
        }

    # ========================================
    # Dynamic margins based on statistics
    # ========================================

    # Energy margins: push current percentiles further by N sigmas
    id_sigma = CONFIG['PARAMS'].get("energy_margin_id_sigma")
    ood_sigma = CONFIG['PARAMS'].get("energy_margin_ood_sigma")

    id_std = epoch_stats.get("energy_id_std")
    ood_std = epoch_stats.get("energy_ood_std")

    energy_margin_id_sigmastd = id_std * id_sigma
    energy_margin_ood_sigmastd = ood_std * ood_sigma

    energy_margin_id = epoch_stats.get("energy_id_p95") - energy_margin_id_sigmastd
    energy_margin_ood = epoch_stats.get("energy_ood_p05") + energy_margin_ood_sigmastd

    return {
        "ce_weight": shares.get("ce"),
        "energy_weight": shares.get("energy"),

        "energy_margin_id": energy_margin_id,
        "energy_margin_ood": energy_margin_ood,
        "energy_margin_id_sigmastd": energy_margin_id_sigmastd,
        "energy_margin_ood_sigmastd": energy_margin_ood_sigmastd,
    }

def train_epoch(
        model,
        loader,
        optimizer,
        scheduler,
        epoch,
        id2label,
        config,
        CONFIG,
        scaler=None,
        logger=False,
        debug=False,
        print_every=50,
):
    """
    Train for one epoch with detailed loss tracking.

    Returns:
        out_metrics: Dictionary with loss components, weights, and diagnostics
        epoch_records: List of per-sample records for this epoch
    """

    model.train()

    total_loss = 0.0
    total_n = 0

    # Accumulate ONLY raw loss components (for averaging)
    loss_acc = {}

    # Statistics for debugging
    stats = {
        "loss": [],
        "grad_norm": [],
        "logits_std": [],
        "ood_ratio": [],
    }

    # Initialize records for this epoch
    epoch_records = []

    for step, batch in enumerate(tqdm(loader, desc="TRAIN")):

        input_ids = batch["input_ids"].to(CONFIG['DEVICE'])
        attention_mask = batch["attention_mask"].to(CONFIG['DEVICE'])
        labels = batch["label"].to(CONFIG['DEVICE'])
        is_ood = batch["is_ood"].to(CONFIG['DEVICE'])

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=CONFIG['DEVICE'].type, enabled=CONFIG['USE_AMP']):

            out = model_forward(
                model,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": labels,
                    "is_ood": is_ood,
                },
                compute_uncertainty=True,
                output_hidden_states=True,
                output_attentions=False,
                use_pooled=False,
            )

            # создаем loss_batch на GPU
            loss_batch = {
                "label": labels,
                "is_ood": is_ood,
                "labels": labels,  # для совместимости
            }

            loss_fn = CONFIG['loss_fn']
            loss_dict = loss_fn(out, loss_batch, config)
            loss = loss_dict["loss"]

            # Accumulate EVERYTHING from loss_dict (both raw and weighted)
            for k, v in loss_dict.items():
                if k != "loss" and hasattr(v, 'item'):
                    loss_acc[k] = loss_acc.get(k, 0.0) + v.item()

            # Log per-sample details if requested
            if logger:
                epoch_records = train_logger(
                    records=epoch_records,
                    out=out,
                    batch=batch,
                    epoch=epoch,
                    batch_idx=step,
                    id2label=id2label,
                )

        # Skip if loss is not finite
        if not torch.isfinite(loss):
            continue

        # Backward pass with gradient clipping
        if CONFIG['USE_AMP']:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG['MAX_GRAD_NORM'],
            )

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG['MAX_GRAD_NORM'],
            )

            optimizer.step()

        # Update scheduler
        if scheduler:
            scheduler.step()

        # Track statistics
        bs = input_ids.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        # Update debug stats
        if debug:
            stats["loss"].append(loss.item())
            stats["grad_norm"].append(float(grad_norm))
            stats["logits_std"].append(out.logits.std(unbiased=False).item())
            stats["ood_ratio"].append(is_ood.float().mean().item())

            # Print debug information periodically
            if step % print_every == 0 and step > 0:
                print("\n" + "-" * 60)
                print(f"TRAIN STEP {step}")
                print("-" * 60)
                print(f"loss        : {np.mean(stats['loss'][-print_every:]):.6f}")
                print(f"grad_norm   : {np.mean(stats['grad_norm'][-print_every:]):.6f}")
                print(f"logits_std  : {np.mean(stats['logits_std'][-print_every:]):.6f}")
                print(f"ood_ratio   : {np.mean(stats['ood_ratio'][-print_every:]):.4f}")

    # ============================================================
    # Build UNIVERSAL output metrics
    # ============================================================

    # Start with primary loss
    out_metrics = {
        "train_loss": total_loss / total_n,
    }

    # Add ALL loss components (raw and weighted automatically detected)
    num_batches = len(loader)

    # Add all accumulated loss components (averaged)
    for k, v in loss_acc.items():
        out_metrics[k] = v / num_batches

    # Add config
    for k, v in config.items():
        out_metrics[k] = v

    # Add diagnostics if debug mode was enabled
    if debug and stats["grad_norm"]:
        out_metrics["avg_grad_norm"] = np.mean(stats["grad_norm"])
        out_metrics["avg_logits_std"] = np.mean(stats["logits_std"])
        out_metrics["avg_ood_ratio"] = np.mean(stats["ood_ratio"])

    return out_metrics, epoch_records

def evaluate(
    model,
    loader,
    device,
    metrics_fn,
):
    """
    Evaluate model on validation/test set.
    Returns classification and OOD metrics only.
    """
    model.eval()

    all_preds, all_labels, all_is_ood = [], [], []
    all_energy = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="EVAL"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("label")
            is_ood = batch.get("is_ood")

            if is_ood is None:
                raise ValueError("is_ood missing")

            labels = labels.to(device) if labels is not None else None
            is_ood = is_ood.to(device).bool()

            out = model_forward(
                model,
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label": labels,
                    "is_ood": is_ood,
                },
                compute_uncertainty=True,  # Need energy for OOD metrics
            )

            # Collect
            if labels is not None:
                all_preds.append(out.predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

            all_is_ood.append(is_ood.cpu().numpy())
            all_energy.append(out.energy.cpu().numpy())

    # Combine results
    preds = np.concatenate(all_preds) if all_preds else None
    labels = np.concatenate(all_labels) if all_labels else None
    is_ood = np.concatenate(all_is_ood)
    energy = np.concatenate(all_energy)

    results = {}

    # Classification metrics (ID only)
    if preds is not None:
        id_mask = ~is_ood
        results.update(
            metrics_fn.compute_classification_metrics(
                labels[id_mask],
                preds[id_mask],
            )
        )

    # OOD metrics
    results.update(
        metrics_fn.compute_ood_metrics(
            energy[~is_ood],
            energy[is_ood],
        )
    )

    return results


def compute_attention_entropy(attn):

    eps = 1e-12

    p = attn.clamp(min=eps)

    entropy = -(p * torch.log(p)).sum(dim=-1)

    return entropy.mean(dim=(0, 1, 2)).item()


def compute_attention_collapse(attn):

    max_attn = attn.max(dim=-1)[0]

    return max_attn.mean().item()


def compute_attention_disagreement(attn):

    # [B, H, T, T]

    head_mean = attn.mean(dim=1, keepdim=True)

    disagreement = (attn - head_mean).abs().mean()

    return disagreement.item()


def compute_epoch_stats(epoch_records):
    """
    Compute key statistics for ID and OOD samples from epoch records.

    Returns:
        Dictionary with stats for ID and OOD samples
    """
    if not epoch_records:
        return {}

    df = pd.DataFrame(epoch_records)

    # Split by is_ood
    id_df = df[~df['is_ood']]
    ood_df = df[df['is_ood']]

    stats = {}

    # Energy stats
    stats['energy_id_mean'] = id_df['energy'].mean()
    stats['energy_id_std'] = id_df['energy'].std()
    stats['energy_ood_mean'] = ood_df['energy'].mean()
    stats['energy_ood_std'] = ood_df['energy'].std()
    stats['energy_id_min'] = id_df['energy'].min()
    stats['energy_id_max'] = id_df['energy'].max()
    stats['energy_ood_min'] = ood_df['energy'].min()
    stats['energy_ood_max'] = ood_df['energy'].max()


    # Energy separation
    stats['energy_separation'] = stats['energy_ood_mean'] - stats['energy_id_mean']

    # New energy percentiles
    stats['energy_id_p99'] = id_df['energy'].quantile(0.99)
    stats['energy_id_p1'] = id_df['energy'].quantile(0.01)
    stats['energy_ood_p01'] = ood_df['energy'].quantile(0.01)
    stats['energy_ood_p99'] = ood_df['energy'].quantile(0.99)
    stats['energy_gap_1'] = stats['energy_ood_p01'] - stats['energy_id_p99']

    stats['energy_id_p95'] = id_df['energy'].quantile(0.95)
    stats['energy_ood_p05'] = ood_df['energy'].quantile(0.05)
    stats['energy_gap_5'] = stats['energy_ood_p05'] - stats['energy_id_p95']

    stats['energy_id_p90'] = id_df['energy'].quantile(0.90)
    stats['energy_ood_p10'] = ood_df['energy'].quantile(0.10)
    stats['energy_gap_10'] = stats['energy_ood_p10'] - stats['energy_id_p90']

    # Confidence stats
    stats['conf_id_mean'] = id_df['confidence'].mean()
    stats['conf_ood_mean'] = ood_df['confidence'].mean()
    stats['conf_separation'] = stats['conf_id_mean'] - stats['conf_ood_mean']

    # New confidence for correct vs wrong
    correct_df = id_df[id_df['correct'] == True]
    wrong_df = id_df[id_df['correct'] == False]
    stats['correct_conf_mean'] = correct_df['confidence'].mean() if len(correct_df) > 0 else 0
    stats['wrong_conf_mean'] = wrong_df['confidence'].mean() if len(wrong_df) > 0 else 0

    # Entropy stats
    stats['entropy_id_mean'] = id_df['entropy'].mean()
    stats['entropy_ood_mean'] = ood_df['entropy'].mean()
    stats['entropy_separation'] = stats['entropy_ood_mean'] - stats['entropy_id_mean']

    # Accuracy on ID samples
    stats['id_accuracy'] = id_df['correct'].mean()

    # New OOD detection performance using energy
    from sklearn.metrics import roc_auc_score
    all_energy = df['energy'].values
    all_labels = df['is_ood'].astype(int).values
    stats['energy_auroc'] = roc_auc_score(all_labels, all_energy)

    # Sample counts
    stats['num_id'] = len(id_df)
    stats['num_ood'] = len(ood_df)

    return stats


def train_logger(
    records, # список для текущей эпохи
    out,
    batch,
    epoch,
    batch_idx,
    id2label,
    ood_threshold=None,
):

    probs = out.probs.detach().cpu()
    logits = out.logits.detach().cpu()
    preds = out.predictions.detach().cpu()
    labels = out.labels.detach().cpu()
    is_ood = out.is_ood.detach().cpu().bool()
    energy = out.energy.detach().cpu()
    entropy = out.entropy.detach().cpu()
    msp = out.msp.detach().cpu()
    confidence = out.confidence.detach().cpu()


    topk_vals, _ = probs.topk(k=3, dim=-1)
    top1_prob = topk_vals[:, 0]
    top2_prob = topk_vals[:, 1]
    top3_prob = topk_vals[:, 2]
    probs_mean = probs.mean(dim=-1)
    probs_std = probs.std(dim=-1)
    probs_margin = top1_prob - top2_prob

    # ========================================================
    # HIDDEN
    # ========================================================

    hidden = out.hidden.detach().cpu()

    if hidden.dim() == 2:

        cls_vec = hidden

        cls_norm = cls_vec.norm(dim=-1)

        hidden_mean = hidden.mean(dim=1)

        hidden_std = hidden.std(dim=1)

        hidden_norm = hidden.norm(dim=-1)

        hidden_max = hidden.max(dim=1).values

        hidden_min = hidden.min(dim=1).values

    elif hidden.dim() == 3:

        cls_vec = hidden[:, 0]

        cls_norm = cls_vec.norm(dim=-1)

        hidden_mean = hidden.mean(dim=(1, 2))

        hidden_std = hidden.std(dim=(1, 2))

        hidden_norm = hidden.flatten(1).norm(dim=-1)

        hidden_max = hidden.amax(dim=(1, 2))

        hidden_min = hidden.amin(dim=(1, 2))

    else:

        raise ValueError(
            f"Unsupported hidden shape: {hidden.shape}"
        )

    # ========================================================
    # ATTENTION
    # ========================================================

    attention_entropy = torch.zeros(len(probs))

    attention_collapse = torch.zeros(len(probs))

    attention_disagreement = torch.zeros(len(probs))

    if out.attentions is not None:

        last_attn = out.attentions[-1].detach().cpu()

        for i in range(last_attn.size(0)):

            sample_attn = last_attn[i]

            attention_entropy[i] = (
                compute_attention_entropy(
                    sample_attn.unsqueeze(0)
                )
            )

            attention_collapse[i] = (
                compute_attention_collapse(
                    sample_attn.unsqueeze(0)
                )
            )

            attention_disagreement[i] = (
                compute_attention_disagreement(
                    sample_attn.unsqueeze(0)
                )
            )

    # ========================================================
    # PREDICTED OOD
    # ========================================================

    predicted_ood = torch.zeros_like(is_ood)

    if ood_threshold is not None:

        predicted_ood = (
            energy > ood_threshold
        )

    # ========================================================
    # SAVE
    # ========================================================

    texts = batch.get(
        "text",
        [""] * len(probs),
    )

    for i in range(len(probs)):

        label_id = int(labels[i])

        pred_id = int(preds[i])

        if label_id == -1:
            label_name = "OOD"
        else:
            label_name = id2label.get(
                label_id,
                str(label_id),
            )

        prediction_name = id2label.get(
            pred_id,
            str(pred_id),
        )

        records.append({

            # ================================================
            # INDEXING
            # ================================================

            "epoch":
                int(epoch),

            "batch_idx":
                int(batch_idx),

            "sample_idx":
                int(batch_idx * len(probs) + i),

            # ================================================
            # TEXT
            # ================================================

            "text":
                str(texts[i]),

            # ================================================
            # OOD
            # ================================================

            "is_ood":
                bool(is_ood[i]),

            "predicted_ood":
                bool(predicted_ood[i]),

            # ================================================
            # LABELS
            # ================================================

            "label":
                int(label_id),

            "label_name":
                label_name,

            "prediction":
                pred_id,

            "prediction_name":
                prediction_name,

            "correct":
                bool(pred_id == label_id),

            # ================================================
            # LOGITS
            # ================================================

            "max_logit":
                float(logits[i].max()),

            # ================================================
            # UNCERTAINTY
            # ================================================

            "energy":
                float(energy[i]),

            "entropy":
                float(entropy[i]),

            "msp":
                float(msp[i]),

            "confidence":
                float(confidence[i]),

            # ================================================
            # PROBS
            # ================================================

            "top1_prob":
                float(top1_prob[i]),

            "top2_prob":
                float(top2_prob[i]),

            "top3_prob":
                float(top3_prob[i]),

            "probs_mean":
                float(probs_mean[i]),

            "probs_std":
                float(probs_std[i]),

            "probs_margin":
                float(probs_margin[i]),

            # ================================================
            # CLS
            # ================================================

            "cls_norm":
                float(cls_norm[i]),

            # ================================================
            # HIDDEN
            # ================================================

            "hidden_mean":
                float(hidden_mean[i]),

            "hidden_std":
                float(hidden_std[i]),

            "hidden_norm":
                float(hidden_norm[i]),

            "hidden_max":
                float(hidden_max[i]),

            "hidden_min":
                float(hidden_min[i]),

            # ================================================
            # ATTENTION
            # ================================================

            "attention_entropy":
                float(attention_entropy[i]),

            "attention_collapse":
                float(attention_collapse[i]),

            "attention_disagreement":
                float(attention_disagreement[i]),
        })

    return records
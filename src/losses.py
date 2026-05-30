import torch
import torch.nn.functional as F

LOSS_REGISTRY = {}


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def ce_loss(out, batch, config):
    """
    Cross-entropy loss for ID classification.

    Returns:
        loss: weighted CE loss
        ce_loss: unweighted CE loss
        ce_loss_weighted: weighted CE loss
    """
    weight = config.get("ce_weight", 1.0)
    device = out.logits.device if out.logits is not None else torch.device("cpu")

    # Initialize result with default values
    result = {
        "loss": torch.tensor(0.0, device=device),
        "ce_loss": torch.tensor(0.0, device=device),
        "ce_loss_weighted": torch.tensor(0.0, device=device),
    }

    # Check if we can compute CE loss
    should_compute = True

    # Condition 1: labels must exist
    labels = batch.get("label")
    if should_compute and labels is None:
        should_compute = False

    if should_compute:
        id_mask = labels != -1
        if id_mask.sum() == 0:
            should_compute = False

    # Compute CE loss if conditions are met
    if should_compute:
        loss = F.cross_entropy(out.logits[id_mask], labels[id_mask])

        result = {
            "loss": weight * loss,
            "ce_loss": loss.detach(),
            "ce_loss_weighted": (weight * loss).detach(),
        }

    return result


LOSS_REGISTRY["ce"] = ce_loss


def energy_loss(out, batch, config):
    """
    Energy loss to push ID energies down (more negative) and OOD energies up (less negative).

    Returns:
        loss: weighted energy loss
        energy_loss: unweighted energy loss (id_loss + ood_loss)
        energy_loss_weighted: weighted energy loss
        energy_id_loss: MSE loss for ID energies
        energy_ood_loss: MSE loss for OOD energies
    """
    weight = config.get("energy_weight", 0.0)
    device = out.logits.device if out.logits is not None else torch.device("cpu")

    # Initialize result with default values
    result = {
        "loss": torch.tensor(0.0, device=device),
        "energy_loss": torch.tensor(0.0, device=device),
        "energy_loss_weighted": torch.tensor(0.0, device=device),
        "energy_id_loss": torch.tensor(0.0, device=device),
        "energy_ood_loss": torch.tensor(0.0, device=device),
    }

    # Check if we can compute energy loss
    should_compute = True

    # Condition 1: weight must be positive (or >0 for warmup)
    if weight <= 0:
        should_compute = False

    # Condition 2: energy and is_ood must exist
    if should_compute and (out.energy is None or out.is_ood is None):
        should_compute = False

    if should_compute:
        is_ood = out.is_ood.bool()
        id_energy = out.energy[~is_ood]
        ood_energy = out.energy[is_ood]

        # Condition 3: both groups must have samples
        if id_energy.numel() == 0 or ood_energy.numel() == 0:
            should_compute = False

    # Compute energy loss if conditions are met
    if should_compute:
        margin_id = config.get("energy_margin_id", -5.5)
        margin_ood = config.get("energy_margin_ood", -4.5)

        # ID: push down (more negative)
        id_loss = F.mse_loss(id_energy, torch.full_like(id_energy, margin_id))

        # OOD: push up (less negative)
        ood_loss = F.mse_loss(ood_energy, torch.full_like(ood_energy, margin_ood))

        loss = id_loss + ood_loss

        result = {
            "loss": weight * loss,
            "energy_loss": loss.detach(),
            "energy_loss_weighted": (weight * loss).detach(),
            "energy_id_loss": id_loss.detach(),
            "energy_ood_loss": ood_loss.detach(),
        }

    return result


LOSS_REGISTRY["energy"] = energy_loss



# ============================================================
# COMPOSITE LOSS
# ============================================================

def composite_loss(loss_names):
    """
    Simple composite loss - just sums weighted losses.
    No adaptive magic - keep it simple and stable.
    """

    def loss_fn(out, batch, config=None):
        config = config or {}

        total_loss = 0.0
        logs = {}

        for name in loss_names:
            fn = LOSS_REGISTRY[name]
            out_dict = fn(out, batch, config)

            total_loss = total_loss + out_dict["loss"]

            for k, v in out_dict.items():
                if k != "loss":
                    logs[k] = v

        logs["loss"] = total_loss

        return logs

    return loss_fn
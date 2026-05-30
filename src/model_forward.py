from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class IntentClassifier(nn.Module):
    """
    BERT-based intent classifier.

    Returns:
        - logits only (default)
        - or (logits, features) if return_features=True
        - or full dict if output_all=True
    """

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1, attn_implementation: str = "sdpa"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            attn_implementation=attn_implementation
        )
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        Args:
            output_hidden_states: возвращать last_hidden_state
            output_attentions: возвращать attention maps
            labels: для вычисления loss внутри модели

        Returns:
            Если output_all=False: logits или (logits, features)
            Если output_all=True: объект с полями logits, features, hidden, attentions
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # [CLS] token
        pooled = outputs.last_hidden_state[:, 0, :]
        features = self.dropout(pooled)
        logits = self.classifier(features)

        # Для совместимости с HuggingFace
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        else:
            loss = None

        # Базовый режим
        if not output_hidden_states and not output_attentions:
            if return_features:
                return logits, features
            return logits

        # Расширенный режим — возвращаем всё
        class Output:
            def __init__(self, logits, loss, features, hidden, attentions):
                self.logits = logits
                self.loss = loss
                self.features = features
                self.last_hidden_state = hidden
                self.attentions = attentions

        hidden = outputs.last_hidden_state if output_hidden_states else None
        attentions = outputs.attentions if output_attentions else None

        return Output(logits, loss, features, hidden, attentions)

@dataclass
class ForwardOutput:
    """
    Type-safe container for model forward pass results with uncertainty metrics.

    Attributes:
        logits: Prediction logits [batch_size, num_classes]
        outputs: Raw model output (for debugging)
        labels: Original labels (optional, for convenience)
        is_ood: Boolean mask for OOD samples (optional)

        # Uncertainty metrics (all follow: higher = more OOD-like)
        energy: Energy score [batch_size]
        entropy: Predictive entropy [batch_size]
        msp: Maximum Softmax Probability (inverted: -max_prob) [batch_size]
        probs: Full probability distribution [batch_size, num_classes]

        # Optional extras
        hidden: Hidden states
        attentions: Attention maps
        pooled: Pooled output
        loss: Computed loss
    """
    logits: torch.Tensor
    outputs: Any
    labels: Optional[torch.Tensor] = None
    is_ood: Optional[torch.Tensor] = None

    # Uncertainty metrics
    energy: Optional[torch.Tensor] = None
    entropy: Optional[torch.Tensor] = None
    msp: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None

    # Optional extras
    hidden: Optional[torch.Tensor] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    pooled: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    @property
    def predictions(self) -> Optional[torch.Tensor]:
        """Predicted class indices."""
        if self.logits is not None:
            return self.logits.argmax(dim=-1)
        return None

    @property
    def confidence(self) -> Optional[torch.Tensor]:
        """Raw confidence (max probability, higher = more confident)."""
        if self.probs is not None:
            return self.probs.max(dim=-1)[0]
        return None

    @property
    def has_uncertainty(self) -> bool:
        """Check if uncertainty metrics are computed."""
        return self.energy is not None

    @property
    def has_hidden(self) -> bool:
        return self.hidden is not None

    @property
    def has_attentions(self) -> bool:
        return self.attentions is not None

    @property
    def has_pooled(self) -> bool:
        return self.pooled is not None

    @property
    def has_loss(self) -> bool:
        return self.loss is not None

    def __repr__(self) -> str:
        return (
            f"ForwardOutput(\n"
            f"  logits_shape={tuple(self.logits.shape) if self.logits is not None else None},\n"
            f"  energy_shape={tuple(self.energy.shape) if self.energy is not None else None},\n"
            f"  has_uncertainty={self.has_uncertainty},\n"
            f"  has_hidden={self.has_hidden},\n"
            f"  has_attentions={self.has_attentions},\n"
            f"  has_pooled={self.has_pooled},\n"
            f"  has_loss={self.has_loss}\n"
            f")"
        )


# ============================================================
# UNCERTAINTY METRICS COMPUTATION
# ============================================================

def compute_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Energy score: higher = more OOD-like."""
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def compute_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Predictive entropy: higher = more OOD-like."""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def compute_msp(logits: torch.Tensor) -> torch.Tensor:
    """Maximum Softmax Probability (inverted): higher = more OOD-like."""
    probs = F.softmax(logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)
    return -max_probs


def compute_all_uncertainty_metrics(
        logits: torch.Tensor,
        temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute all uncertainty metrics at once.

    Returns:
        (energy, entropy, msp, probs)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Energy
    energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)

    # Entropy
    entropy = -(probs * log_probs).sum(dim=-1)

    # MSP (inverted)
    max_probs, _ = probs.max(dim=-1)
    msp = -max_probs

    return energy, entropy, msp, probs


# ============================================================
# ENHANCED MODEL FORWARD
# ============================================================

def model_forward(
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        use_pooled: bool = False,
        compute_uncertainty: bool = True,
        temperature: float = 1.0,
) -> ForwardOutput:
    """
    Safe universal forward pass with uncertainty metrics.

    Args:
        model: PyTorch model
        batch: Dict with 'input_ids', 'attention_mask', optionally 'label', 'is_ood'
        output_hidden_states: Extract hidden states
        output_attentions: Extract attention maps
        use_pooled: Extract pooled output
        compute_uncertainty: Compute energy, entropy, msp, probs
        temperature: Temperature for energy/entropy scaling

    Returns:
        ForwardOutput with all metrics
    """

    # Extract labels and OOD mask if present
    labels = batch.get("label")
    is_ood = batch.get("is_ood")

    # Build forward kwargs
    forward_kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }

    if output_hidden_states:
        forward_kwargs["output_hidden_states"] = True
    if output_attentions:
        forward_kwargs["output_attentions"] = True

    # Forward pass
    try:
        outputs = model(**forward_kwargs)
    except TypeError:
        # Model doesn't accept optional parameters
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            labels=labels,
        )

    # Extract logits
    logits = extract_logits(outputs)

    # Compute uncertainty metrics
    energy = None
    entropy = None
    msp = None
    probs = None

    if compute_uncertainty and logits is not None:
        energy, entropy, msp, probs = compute_all_uncertainty_metrics(logits, temperature)

    # Extract optional components
    hidden = None
    if output_hidden_states:
        try:
            hidden = extract_hidden(
                outputs,
                attention_mask=batch.get("attention_mask"),
                strategy="cls",
            )
        except (TypeError, AttributeError, IndexError):
            hidden = None

    attentions = None
    if output_attentions:
        attentions = extract_attentions(outputs)

    pooled = None
    if use_pooled:
        pooled = extract_pooled_output(outputs)

    loss = extract_loss(outputs)

    return ForwardOutput(
        logits=logits,
        outputs=outputs,
        labels=labels,
        is_ood=is_ood,
        energy=energy,
        entropy=entropy,
        msp=msp,
        probs=probs,
        hidden=hidden,
        attentions=attentions,
        pooled=pooled,
        loss=loss,
    )


# ============================================================
# HELPER FUNCTIONS (kept from original)
# ============================================================

def extract_logits(outputs: Any) -> torch.Tensor:
    """Extract logits from model output."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item) and item.dim() >= 2:
                return item
        for item in outputs:
            if torch.is_tensor(item):
                return item
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        for value in outputs.values():
            if torch.is_tensor(value) and value.dim() >= 2:
                return value
    raise TypeError(f"Cannot extract logits from {type(outputs)}")


def extract_hidden(outputs: Any, attention_mask: Optional[torch.Tensor] = None, strategy: str = "cls") -> torch.Tensor:
    """Extract hidden states."""
    hidden = None
    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
    elif hasattr(outputs, "hidden_states"):
        hidden = outputs.hidden_states[-1]
    elif isinstance(outputs, dict):
        if "last_hidden_state" in outputs:
            hidden = outputs["last_hidden_state"]
        elif "hidden_states" in outputs:
            hidden = outputs["hidden_states"][-1]
    elif isinstance(outputs, (tuple, list)):
        for item in outputs:
            if torch.is_tensor(item) and item.dim() == 3:
                hidden = item
                break
    if hidden is None:
        raise TypeError("Cannot extract hidden states")

    if strategy == "cls":
        return hidden[:, 0]
    elif strategy == "mean":
        if attention_mask is None:
            raise ValueError("attention_mask required for mean pooling")
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        return sum_embeddings / sum_mask.clamp(min=1e-9)
    elif strategy == "raw":
        return hidden
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def extract_attentions(outputs: Any) -> Optional[Tuple[torch.Tensor, ...]]:
    """Extract attention maps."""
    if hasattr(outputs, "attentions") and outputs.attentions is not None:
        return outputs.attentions
    if hasattr(outputs, "attn_weights") and outputs.attn_weights is not None:
        return outputs.attn_weights
    if isinstance(outputs, dict):
        for key in ["attentions", "attn_weights"]:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
    return None


def extract_pooled_output(outputs: Any) -> Optional[torch.Tensor]:
    """Extract pooled output."""
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "sentence_embedding") and outputs.sentence_embedding is not None:
        return outputs.sentence_embedding
    if isinstance(outputs, dict):
        for key in ["pooler_output", "sentence_embedding", "pooled_output"]:
            if key in outputs and outputs[key] is not None:
                return outputs[key]
    return None


def extract_loss(outputs: Any) -> Optional[torch.Tensor]:
    """Extract loss."""
    if hasattr(outputs, "loss"):
        loss = outputs.loss
        if loss is not None and torch.is_tensor(loss) and loss.dim() == 0:
            return loss
    if isinstance(outputs, dict):
        if "loss" in outputs:
            loss = outputs["loss"]
            if torch.is_tensor(loss) and loss.dim() == 0:
                return loss
    if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
        candidate = outputs[1]
        if torch.is_tensor(candidate) and candidate.dim() == 0:
            return candidate
        first = outputs[0]
        if torch.is_tensor(first) and first.dim() == 0:
            return first
    return None
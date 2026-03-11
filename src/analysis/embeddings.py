"""Extract hidden-layer embeddings from trained models.

Registers forward hooks to capture intermediate representations from any
model layer, then runs inference to collect embeddings for all trials.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    layer_name: Optional[str] = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Extract hidden-layer embeddings for all trials.

    If layer_name is None, uses the second-to-last module (before output projection).

    Args:
        model: Trained decoder model.
        dataloader: DataLoader yielding batches with 'features' and 'label_texts'.
        layer_name: Name of the module to hook (from model.named_modules()).
        device: Torch device.

    Returns:
        Dict with keys:
            'embeddings': np.ndarray [N, D] — mean-pooled hidden states.
            'labels': list[str] — ground truth labels per trial.
            'layer_name': str — name of hooked layer.
    """
    model.eval()
    model.to(device)

    # Find the target layer
    target_layer = _find_layer(model, layer_name)
    hooked_name = layer_name or _get_layer_name(model, target_layer)

    # Register hook
    captured = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        captured.append(output.detach().cpu())

    handle = target_layer.register_forward_hook(hook_fn)

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            label_texts = batch["label_texts"]
            captured.clear()

            _ = model(features)

            if captured:
                hidden = captured[0]  # [B, T, D] or [B, D]
                if hidden.ndim == 3:
                    # Mean-pool over time
                    hidden = hidden.mean(dim=1)  # [B, D]
                all_embeddings.append(hidden.numpy())
                all_labels.extend(label_texts)

    handle.remove()

    embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.empty((0, 0))

    logger.info(
        "Extracted %d embeddings (dim=%d) from layer '%s'",
        embeddings.shape[0], embeddings.shape[1] if embeddings.ndim == 2 else 0,
        hooked_name,
    )

    return {
        "embeddings": embeddings,
        "labels": all_labels,
        "layer_name": hooked_name,
    }


def save_embeddings(
    embeddings_dict: dict,
    output_dir: str | Path = "./outputs/embeddings",
    prefix: str = "model",
) -> Path:
    """Save extracted embeddings to disk.

    Args:
        embeddings_dict: Output from extract_embeddings().
        output_dir: Directory to save files.
        prefix: Filename prefix.

    Returns:
        Path to saved .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / f"{prefix}_embeddings.npz"
    np.savez(
        save_path,
        embeddings=embeddings_dict["embeddings"],
        labels=np.array(embeddings_dict["labels"], dtype=object),
        layer_name=embeddings_dict["layer_name"],
    )

    logger.info("Saved embeddings to %s", save_path)
    return save_path


def load_embeddings(path: str | Path) -> dict:
    """Load embeddings from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"].tolist(),
        "layer_name": str(data["layer_name"]),
    }


def _find_layer(model: nn.Module, layer_name: Optional[str]) -> nn.Module:
    """Find a specific layer by name, or auto-detect the pre-output layer."""
    if layer_name is not None:
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(
            f"Layer '{layer_name}' not found. Available: "
            f"{[n for n, _ in model.named_modules() if n]}"
        )

    # Auto-detect: find the last non-Linear layer (before output projection)
    modules = list(model.named_modules())
    # Reverse search for a recurrent/transformer/conv layer
    for name, module in reversed(modules):
        if isinstance(module, (nn.GRU, nn.LSTM, nn.TransformerEncoder)):
            return module
    # Fallback: second-to-last named child
    children = list(model.named_children())
    if len(children) >= 2:
        return children[-2][1]
    return children[-1][1] if children else model


def _get_layer_name(model: nn.Module, target: nn.Module) -> str:
    """Get the name of a module within a model."""
    for name, module in model.named_modules():
        if module is target:
            return name
    return "unknown"

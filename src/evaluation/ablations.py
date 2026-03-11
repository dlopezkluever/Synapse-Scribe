"""Ablation study runner and full evaluation suite.

Systematically evaluates model × decoding × augmentation combinations.
Generates per-character confusion matrices, per-subject metrics, and
statistical significance tests via paired bootstrap resampling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_cer, compute_wer, exact_match_accuracy
from src.decoding.greedy import greedy_decode_batch
from src.decoding.beam_search import beam_search_decode_batch
from src.decoding.lm_correction import load_lm_scorer, rescore_hypotheses
from src.data.dataset import CHAR_TO_IDX, VOCAB_SIZE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    model_name: str
    decoding_method: str  # "greedy", "beam", "beam+lm"
    augmentation: str     # "full", "none"
    feature_pathway: str  # "default", "temporal_conv", "linear_proj"
    cer: float
    wer: float
    exact_match: float
    n_samples: int
    inference_time_s: float
    predictions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("predictions")
        d.pop("references")
        return d


@dataclass
class BootstrapResult:
    """Result of paired bootstrap significance test."""

    model_a: str
    model_b: str
    metric: str
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_resamples: int
    significant: bool


# ---------------------------------------------------------------------------
# Per-character confusion matrix
# ---------------------------------------------------------------------------

def compute_confusion_matrix(
    predictions: list[str],
    references: list[str],
) -> np.ndarray:
    """Compute per-character confusion matrix.

    Returns:
        Matrix of shape [n_chars, n_chars] where rows = reference chars,
        columns = predicted chars. Characters are indexed a=0..z=25, space=26.
    """
    n_chars = 27  # a-z + space
    confusion = np.zeros((n_chars, n_chars), dtype=np.int64)

    char_to_idx = {chr(ord("a") + i): i for i in range(26)}
    char_to_idx[" "] = 26

    for pred, ref in zip(predictions, references):
        # Align characters using simple per-position comparison
        # For proper alignment, use edit distance alignment
        aligned_pred, aligned_ref = _align_strings(pred, ref)
        for p_ch, r_ch in zip(aligned_pred, aligned_ref):
            if r_ch in char_to_idx and p_ch in char_to_idx:
                confusion[char_to_idx[r_ch], char_to_idx[p_ch]] += 1

    return confusion


def _align_strings(pred: str, ref: str) -> tuple[str, str]:
    """Align two strings using Needleman-Wunsch for confusion analysis."""
    m, n = len(ref), len(pred)

    # DP table
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == pred[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Traceback
    aligned_ref = []
    aligned_pred = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (
            ref[i - 1] == pred[j - 1]
            or dp[i, j] == dp[i - 1, j - 1] + 1
        ):
            aligned_ref.append(ref[i - 1])
            aligned_pred.append(pred[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            aligned_ref.append(ref[i - 1])
            aligned_pred.append("-")
            i -= 1
        else:
            aligned_ref.append("-")
            aligned_pred.append(pred[j - 1] if j > 0 else "-")
            j -= 1

    return "".join(reversed(aligned_pred)), "".join(reversed(aligned_ref))


def get_char_labels() -> list[str]:
    """Return character labels for confusion matrix axes."""
    return [chr(ord("a") + i) for i in range(26)] + ["space"]


# ---------------------------------------------------------------------------
# Per-character error rates
# ---------------------------------------------------------------------------

def per_character_cer(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute CER broken down by character.

    Returns:
        Dict mapping character -> error rate for that character.
    """
    char_errors: dict[str, int] = {}
    char_counts: dict[str, int] = {}

    for pred, ref in zip(predictions, references):
        aligned_pred, aligned_ref = _align_strings(pred, ref)
        for p_ch, r_ch in zip(aligned_pred, aligned_ref):
            if r_ch == "-":
                continue
            char_counts[r_ch] = char_counts.get(r_ch, 0) + 1
            if p_ch != r_ch:
                char_errors[r_ch] = char_errors.get(r_ch, 0) + 1

    result = {}
    for ch in sorted(char_counts.keys()):
        count = char_counts[ch]
        errors = char_errors.get(ch, 0)
        result[ch] = errors / max(count, 1)

    return result


# ---------------------------------------------------------------------------
# Per-subject metrics
# ---------------------------------------------------------------------------

def per_subject_metrics(
    predictions: list[str],
    references: list[str],
    subjects: list[int],
) -> dict[int, dict[str, float]]:
    """Compute CER, WER, exact match per subject.

    Returns:
        Dict mapping subject_id -> {cer, wer, exact_match, n_samples}.
    """
    subject_preds: dict[int, list[str]] = {}
    subject_refs: dict[int, list[str]] = {}

    for pred, ref, subj in zip(predictions, references, subjects):
        subject_preds.setdefault(subj, []).append(pred)
        subject_refs.setdefault(subj, []).append(ref)

    results = {}
    for subj in sorted(subject_preds.keys()):
        p = subject_preds[subj]
        r = subject_refs[subj]
        results[subj] = {
            "cer": compute_cer(p, r),
            "wer": compute_wer(p, r),
            "exact_match": exact_match_accuracy(p, r),
            "n_samples": len(p),
        }

    return results


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    scores_a: list[float],
    scores_b: list[float],
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Paired bootstrap resampling test for model comparison.

    Tests whether model A significantly differs from model B on per-sample
    scores (e.g., per-trial CER).

    Args:
        scores_a: Per-sample scores for model A.
        scores_b: Per-sample scores for model B.
        n_resamples: Number of bootstrap resamples.
        seed: Random seed.
        alpha: Significance level.

    Returns:
        BootstrapResult with mean diff, CI, and p-value.
    """
    assert len(scores_a) == len(scores_b), "Score lists must have same length"

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    observed_diff = np.mean(scores_a - scores_b)

    rng = np.random.RandomState(seed)
    diffs = np.empty(n_resamples)

    n = len(scores_a)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        diffs[i] = np.mean(scores_a[indices] - scores_b[indices])

    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    # Two-sided p-value: fraction of resamples where diff crosses zero
    if observed_diff >= 0:
        p_value = float(np.mean(diffs <= 0)) * 2
    else:
        p_value = float(np.mean(diffs >= 0)) * 2
    p_value = min(p_value, 1.0)

    return BootstrapResult(
        model_a="",
        model_b="",
        metric="",
        mean_diff=float(observed_diff),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_resamples=n_resamples,
        significant=p_value < alpha,
    )


def per_sample_cer(predictions: list[str], references: list[str]) -> list[float]:
    """Compute per-sample CER for bootstrap testing."""
    from src.evaluation.metrics import _edit_distance
    results = []
    for p, r in zip(predictions, references):
        if not r:
            results.append(0.0 if not p else 1.0)
        else:
            results.append(_edit_distance(p, r) / len(r))
    return results


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_single_evaluation(
    model: nn.Module,
    dataloader,
    decoding_method: str = "greedy",
    beam_width: int = 100,
    lm_scorer=None,
    lm_alpha: float = 0.5,
    lm_beta: float = 0.0,
    device: str | torch.device = "cpu",
) -> tuple[list[str], list[str], float]:
    """Run inference on a dataloader and decode.

    Returns:
        (predictions, references, inference_time_seconds)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_refs = []
    t0 = time.time()

    for batch in dataloader:
        features = batch["features"].to(device)
        label_texts = batch["label_texts"]
        logits = model(features)

        if decoding_method == "greedy":
            decoded = greedy_decode_batch(logits)
        elif decoding_method in ("beam", "beam+lm"):
            batch_hyps = beam_search_decode_batch(
                logits, beam_width=beam_width, top_k=5,
            )
            if decoding_method == "beam+lm" and lm_scorer is not None:
                decoded = []
                for hyps in batch_hyps:
                    rescored = rescore_hypotheses(hyps, lm_scorer, lm_alpha, lm_beta)
                    decoded.append(rescored[0].text if rescored else "")
            else:
                decoded = [hyps[0].text if hyps else "" for hyps in batch_hyps]
        else:
            raise ValueError(f"Unknown decoding method: {decoding_method}")

        all_preds.extend(decoded)
        all_refs.extend(label_texts)

    elapsed = time.time() - t0
    return all_preds, all_refs, elapsed


def run_ablation_suite(
    models: dict[str, nn.Module],
    dataloader,
    decoding_methods: list[str] | None = None,
    lm_model_path: str | None = None,
    beam_width: int = 100,
    lm_alpha: float = 0.5,
    lm_beta: float = 0.0,
    device: str | torch.device = "cpu",
    output_dir: str | Path = "./outputs/results",
) -> list[AblationResult]:
    """Run full ablation suite over models × decoding methods.

    Args:
        models: Dict of model_name -> model.
        dataloader: Test DataLoader.
        decoding_methods: List of decoding methods to test.
        lm_model_path: Path to KenLM model (optional).
        beam_width: Beam width for beam search.
        lm_alpha: LM interpolation weight.
        lm_beta: Word insertion bonus.
        device: Torch device.
        output_dir: Directory to save results.

    Returns:
        List of AblationResult objects.
    """
    if decoding_methods is None:
        decoding_methods = ["greedy", "beam"]

    lm_scorer = load_lm_scorer(lm_model_path) if "beam+lm" in decoding_methods else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for model_name, model in models.items():
        for method in decoding_methods:
            logger.info("Evaluating %s with %s decoding...", model_name, method)

            preds, refs, elapsed = run_single_evaluation(
                model, dataloader,
                decoding_method=method,
                beam_width=beam_width,
                lm_scorer=lm_scorer,
                lm_alpha=lm_alpha,
                lm_beta=lm_beta,
                device=device,
            )

            cer = compute_cer(preds, refs)
            wer = compute_wer(preds, refs)
            em = exact_match_accuracy(preds, refs)

            result = AblationResult(
                model_name=model_name,
                decoding_method=method,
                augmentation="default",
                feature_pathway="default",
                cer=cer,
                wer=wer,
                exact_match=em,
                n_samples=len(preds),
                inference_time_s=elapsed,
                predictions=preds,
                references=refs,
            )
            results.append(result)

            logger.info(
                "  %s / %s: CER=%.4f, WER=%.4f, EM=%.4f, time=%.1fs",
                model_name, method, cer, wer, em, elapsed,
            )

    # Export results
    _export_results(results, output_dir)

    return results


def _export_results(results: list[AblationResult], output_dir: Path) -> None:
    """Export ablation results to JSON and CSV."""
    # JSON
    json_path = output_dir / "ablation_results.json"
    json_data = [r.to_dict() for r in results]
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    # CSV
    csv_path = output_dir / "ablation_results.csv"
    import csv
    if json_data:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=json_data[0].keys())
            writer.writeheader()
            writer.writerows(json_data)

    logger.info("Exported results to %s and %s", json_path, csv_path)


def run_significance_tests(
    results: list[AblationResult],
    n_resamples: int = 1000,
) -> list[BootstrapResult]:
    """Run pairwise bootstrap significance tests between all model pairs.

    Uses per-sample CER as the test metric.

    Returns:
        List of BootstrapResult for each pair.
    """
    sig_results = []

    # Group results by decoding method (compare models within same decoding)
    from collections import defaultdict
    by_method: dict[str, list[AblationResult]] = defaultdict(list)
    for r in results:
        by_method[r.decoding_method].append(r)

    for method, method_results in by_method.items():
        for i in range(len(method_results)):
            for j in range(i + 1, len(method_results)):
                ra = method_results[i]
                rb = method_results[j]

                scores_a = per_sample_cer(ra.predictions, ra.references)
                scores_b = per_sample_cer(rb.predictions, rb.references)

                if len(scores_a) != len(scores_b):
                    continue

                boot = paired_bootstrap_test(scores_a, scores_b, n_resamples)
                boot.model_a = ra.model_name
                boot.model_b = rb.model_name
                boot.metric = f"CER ({method})"

                sig_results.append(boot)

                logger.info(
                    "  %s vs %s (%s): diff=%.4f, p=%.4f, sig=%s",
                    boot.model_a, boot.model_b, method,
                    boot.mean_diff, boot.p_value, boot.significant,
                )

    return sig_results

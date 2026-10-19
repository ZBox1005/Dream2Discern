from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F


@contextmanager
def temporary_perturbation(auditor, perturbation):
    projections = auditor.projections()
    if any(object.__getattribute__(module, "_transient_v") is not None for module in projections.values()):
        raise RuntimeError("A temporary perturbation is already installed.")
    if perturbation is not None:
        if set(perturbation) != set(projections):
            raise ValueError("The perturbation must contain every selected projection exactly once.")
        for name, module in projections.items():
            if perturbation[name].shape != module.base.weight.shape:
                raise ValueError(f"Invalid perturbation shape: {name}")
    try:
        if perturbation is not None:
            for name, module in projections.items():
                object.__setattr__(module, "_transient_v", perturbation[name])
        yield
    finally:
        for module in projections.values():
            object.__setattr__(module, "_transient_v", None)


def _loss_chunks(auditor, microbatches):
    count = sum(batch["labels"].numel() for batch in microbatches)
    if count == 0:
        raise ValueError("A logical batch must contain labeled prefixes.")
    for batch in microbatches:
        logits = auditor(batch["input_ids"], batch["attention_mask"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float(), reduction="sum") / count
        if not torch.isfinite(loss):
            raise FloatingPointError("The batch prediction loss is nonfinite.")
        yield loss


def batch_loss(auditor, microbatches):
    return sum(_loss_chunks(auditor, microbatches))


def relative_norm(perturbation, scales):
    return torch.sqrt(sum((tensor.float().norm() / scales[name]) ** 2 for name, tensor in perturbation.items()))


def solve_dream(auditor, microbatches, gamma, rank=8):
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive.")
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer.")
    projections = auditor.projections()
    if not projections:
        raise ValueError("The auditor contains no perturbable projections.")
    scales = auditor.reference_scales()
    origin = {
        name: torch.zeros_like(module.base.weight, dtype=torch.float32, requires_grad=True)
        for name, module in projections.items()
    }
    names = list(origin)
    accumulated = [torch.zeros_like(origin[name]) for name in names]
    with temporary_perturbation(auditor, origin):
        for loss in _loss_chunks(auditor, microbatches):
            gradients = torch.autograd.grad(loss, tuple(origin.values()), allow_unused=True)
            for index, gradient in enumerate(gradients):
                if gradient is not None:
                    accumulated[index] = accumulated[index] + gradient.detach()
    if any(not torch.isfinite(gradient).all() for gradient in accumulated):
        raise FloatingPointError("A projection gradient is nonfinite.")
    denominator = torch.sqrt(sum((gradient.float() * scales[name]).pow(2).sum() for name, gradient in zip(names, accumulated))).clamp_min(1e-12)
    if not torch.isfinite(denominator):
        raise FloatingPointError("The scaled gradient norm is nonfinite.")
    candidate = {}
    for name, gradient in zip(names, accumulated):
        value = gamma * (scales[name] ** 2) * gradient.float() / denominator
        if min(value.shape) > rank:
            left, singular, right = torch.linalg.svd(value, full_matrices=False)
            value = (left[:, :rank] * singular[:rank]) @ right[:rank]
        candidate[name] = value.detach()
    radius = relative_norm(candidate, scales).clamp_min(1e-12)
    if not torch.isfinite(radius):
        raise FloatingPointError("The candidate radius is nonfinite.")
    if float(radius) > gamma:
        candidate = {name: value * (gamma / radius) for name, value in candidate.items()}
    zero = {name: torch.zeros_like(value) for name, value in candidate.items()}

    def evaluate(perturbation):
        with torch.no_grad(), temporary_perturbation(auditor, perturbation):
            return sum(float(loss.detach()) for loss in _loss_chunks(auditor, microbatches))

    clean_loss = evaluate(zero)
    accepted = zero
    accepted_loss = clean_loss
    fraction = 0.0
    attempts = 0
    for index in range(6):
        attempts += 1
        measured = evaluate(candidate)
        if measured > clean_loss:
            accepted = candidate
            accepted_loss = measured
            fraction = 2.0 ** (-index)
            break
        candidate = {name: value * 0.5 for name, value in candidate.items()}
    accepted = {name: value.detach() for name, value in accepted.items()}
    return accepted, {
        "clean_loss": clean_loss,
        "perturbed_loss": accepted_loss,
        "loss_increase": accepted_loss - clean_loss,
        "step_fraction": fraction,
        "relative_radius": float(relative_norm(accepted, scales)),
        "candidate_evaluations": attempts,
        "scaled_gradient_norm": float(denominator),
    }

#!/usr/bin/env python3
from __future__ import annotations
import numpy
import torch
from common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"

def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: dict[str, torch.FloatTensor],
    in_features: torch.FloatTensor,
) -> torch.FloatTensor:

    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    normed_features = in_features / rms
    output = normed_features * weights["weight"]
    return output

def test_rmsnorm():
    reference_weights = torch.load(FIXTURES_PATH / "rmsnorm_weights.pt")
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(FIXTURES_PATH / "rmsnorm_expected_output.pt")
    d_model = 64
    actual_output = run_rmsnorm(
        d_model=d_model, eps=1e-5, weights=reference_weights, in_features=in_features
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )
test_rmsnorm()
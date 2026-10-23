"""Small unit tests for the score-label helper (model downloads are not needed)."""

import torch

from processing.video_processor import get_fake_score


class FakeConfig:
    id2label = {0: "real", 1: "fake"}


class FakeModel:
    config = FakeConfig()


def test_fake_score_uses_fake_label():
    logits = torch.tensor([[0.0, 2.0]])
    score = get_fake_score(logits, FakeModel())
    assert score > 0.8


def test_fake_score_is_between_zero_and_one():
    logits = torch.tensor([[1.0, 1.0]])
    score = get_fake_score(logits, FakeModel())
    assert 0.0 <= score <= 1.0

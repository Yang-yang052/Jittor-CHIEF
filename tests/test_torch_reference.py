from __future__ import annotations

import numpy as np
import torch

from chief_jittor.metrics import classification_metrics
from chief_jittor.torch_reference import CHIEFTorch
from chief_jittor.weights import official_to_clean_names


def test_forward_shapes_and_attention_normalization():
    torch.manual_seed(7)
    model = CHIEFTorch(size_arg="small", dropout=False, n_classes=3, use_organ_context=False).eval()
    output = model(torch.randn(37, 768))
    assert output["bag_logits"].shape == (1, 3)
    assert output["attention"].shape == (1, 37)
    assert output["WSI_feature"].shape == (1, 768)
    np.testing.assert_allclose(output["attention"].sum().item(), 1.0, atol=1e-6)


def test_official_checkpoint_name_mapping():
    state = {
        "attention_net.0.weight": np.zeros((512, 768), dtype=np.float32),
        "attention_net.3.attention_c.bias": np.zeros((1,), dtype=np.float32),
        "classifiers.weight": np.zeros((2, 512), dtype=np.float32),
        "instance_classifiers.0.weight": np.zeros((2, 512), dtype=np.float32),
    }
    mapped = official_to_clean_names(state, dropout=True)
    assert set(mapped) == {
        "fc.weight", "gated_attention.attention_c.bias", "classifier.weight",
        "instance_classifiers.0.weight",
    }


def test_binary_auroc_uses_positive_class_probability():
    metrics = classification_metrics([0, 0, 1, 1], [
        np.array([0.9, 0.1]), np.array([0.8, 0.2]),
        np.array([0.2, 0.8]), np.array([0.1, 0.9]),
    ])
    assert metrics["macro_auroc"] == 1.0

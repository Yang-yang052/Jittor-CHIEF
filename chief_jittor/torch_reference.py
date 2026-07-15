from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import SIZE_DICT


class AttentionHead(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, attention_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(attention_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, dropout: bool):
        super().__init__()
        branch_a: list[nn.Module] = [nn.Linear(input_dim, attention_dim), nn.Tanh()]
        branch_b: list[nn.Module] = [nn.Linear(input_dim, attention_dim), nn.Sigmoid()]
        if dropout:
            branch_a.append(nn.Dropout(0.25))
            branch_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*branch_a)
        self.attention_b = nn.Sequential(*branch_b)
        self.attention_c = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_c(self.attention_a(x) * self.attention_b(x))
        return scores, x


class CHIEFTorch(nn.Module):
    """Clean PyTorch reference matching official models/CHIEF.py.

    The official source loads Text_emdding.pth inside __init__. This version
    accepts the embedding as an argument so it is testable and portable.
    """

    def __init__(
        self,
        size_arg: str = "small",
        dropout: bool = True,
        n_classes: int = 2,
        n_organs: int = 19,
        organ_dim: int = 768,
        use_organ_context: bool = True,
        organ_embedding: torch.Tensor | None = None,
    ):
        super().__init__()
        input_dim, hidden_dim, attention_dim = SIZE_DICT[size_arg]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.use_organ_context = use_organ_context
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25) if dropout else nn.Identity()
        self.gated_attention = GatedAttention(hidden_dim, attention_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)
        self.instance_classifiers = nn.ModuleList([nn.Linear(hidden_dim, 2) for _ in range(n_classes)])
        self.att_head = AttentionHead(hidden_dim, attention_dim)
        self.text_to_vision = nn.Sequential(
            nn.Linear(organ_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.25) if dropout else nn.Identity()
        )
        if organ_embedding is None:
            organ_embedding = torch.randn(n_organs, organ_dim)
        self.register_buffer("organ_embedding", organ_embedding.float().clone())

    def forward(self, features: torch.Tensor, organ_index: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if features.ndim == 3:
            if features.shape[0] != 1:
                raise ValueError("CHIEF uses variable-size bags and expects batch_size=1")
            features = features.squeeze(0)
        original = features
        hidden = self.dropout(self.relu(self.fc(features)))
        raw_attention, hidden = self.gated_attention(hidden)
        raw_attention = raw_attention.transpose(1, 0)
        attention = F.softmax(raw_attention, dim=1)
        pooled = torch.mm(attention, hidden)
        slide_embedding = torch.mm(attention, original)
        fused = pooled
        if self.use_organ_context:
            if organ_index is None:
                raise ValueError("organ_index is required when use_organ_context=True")
            organ = self.text_to_vision(self.organ_embedding[organ_index.reshape(-1)])
            fused = pooled + organ
        logits = self.classifier(fused)
        return {
            "bag_logits": logits,
            "bag_prob": F.softmax(logits, dim=1),
            "attention_raw": raw_attention,
            "attention": attention,
            "WSI_feature": slide_embedding,
            "WSI_feature_anatomical": fused,
        }

    def patch_probs(self, features: torch.Tensor, organ_index: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        result = self.forward(features, organ_index)
        hidden = self.dropout(self.relu(self.fc(features.squeeze(0) if features.ndim == 3 else features)))
        if self.use_organ_context:
            organ = self.text_to_vision(self.organ_embedding[organ_index.reshape(-1)])
            hidden = hidden + organ
        patch_logits = self.classifier(hidden)
        patch_prob = torch.sigmoid(result["attention_raw"].squeeze(0)) * F.softmax(patch_logits, dim=1)[:, 1]
        return {**result, "patch_prob": patch_prob}


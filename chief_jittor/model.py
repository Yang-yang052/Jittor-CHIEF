from __future__ import annotations

import jittor as jt
from jittor import nn

from .config import SIZE_DICT


class AttentionHead(nn.Module):
    def __init__(self, feature_dim: int, attention_dim: int):
        self.fc1 = nn.Linear(feature_dim, attention_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(attention_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int, dropout: bool):
        branch_a = [nn.Linear(input_dim, attention_dim), nn.Tanh()]
        branch_b = [nn.Linear(input_dim, attention_dim), nn.Sigmoid()]
        if dropout:
            branch_a.append(nn.Dropout(0.25))
            branch_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*branch_a)
        self.attention_b = nn.Sequential(*branch_b)
        self.attention_c = nn.Linear(attention_dim, 1)

    def execute(self, x):
        scores = self.attention_c(self.attention_a(x) * self.attention_b(x))
        return scores, x


class CHIEFJittor(nn.Module):
    """Jittor port of the official CHIEF WSI-level model."""

    def __init__(
        self,
        size_arg: str = "small",
        dropout: bool = True,
        n_classes: int = 2,
        n_organs: int = 19,
        organ_dim: int = 768,
        use_organ_context: bool = True,
        organ_embedding=None,
    ):
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
        text_layers = [nn.Linear(organ_dim, hidden_dim), nn.ReLU()]
        if dropout:
            text_layers.append(nn.Dropout(0.25))
        self.text_to_vision = nn.Sequential(*text_layers)
        if organ_embedding is None:
            organ_embedding = jt.randn((n_organs, organ_dim))
        elif isinstance(organ_embedding, jt.Var):
            organ_embedding = organ_embedding.float32()
        else:
            organ_embedding = jt.array(organ_embedding).float32()
        self.organ_embedding = organ_embedding.stop_grad()

    def execute(self, features, organ_index=None):
        if features.ndim == 3:
            if features.shape[0] != 1:
                raise ValueError("CHIEF uses variable-size bags and expects batch_size=1")
            features = features.squeeze(0)
        original = features
        hidden = self.dropout(self.relu(self.fc(features)))
        raw_attention, hidden = self.gated_attention(hidden)
        raw_attention = raw_attention.transpose(1, 0)
        attention = nn.softmax(raw_attention, dim=1)
        pooled = jt.matmul(attention, hidden)
        slide_embedding = jt.matmul(attention, original)
        fused = pooled
        if self.use_organ_context:
            if organ_index is None:
                raise ValueError("organ_index is required when use_organ_context=True")
            organ_index = organ_index.reshape((-1,)).int32()
            organ = self.text_to_vision(self.organ_embedding[organ_index])
            fused = pooled + organ
        logits = self.classifier(fused)
        return {
            "bag_logits": logits,
            "bag_prob": nn.softmax(logits, dim=1),
            "attention_raw": raw_attention,
            "attention": attention,
            "WSI_feature": slide_embedding,
            "WSI_feature_anatomical": fused,
        }

    def patch_probs(self, features, organ_index=None):
        result = self.execute(features, organ_index)
        flat = features.squeeze(0) if features.ndim == 3 else features
        hidden = self.dropout(self.relu(self.fc(flat)))
        if self.use_organ_context:
            organ = self.text_to_vision(self.organ_embedding[organ_index.reshape((-1,)).int32()])
            hidden = hidden + organ
        patch_logits = self.classifier(hidden)
        patch_prob = jt.sigmoid(result["attention_raw"].squeeze(0)) * nn.softmax(patch_logits, dim=1)[:, 1]
        return {**result, "patch_prob": patch_prob}


from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(labels: list[int], probabilities: list[np.ndarray]) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    y_pred = y_prob.argmax(axis=1)
    output = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if y_prob.shape[1] == 2:
            output["macro_auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            output["macro_auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro")
            )
    except ValueError:
        output["macro_auroc"] = float("nan")
    return output

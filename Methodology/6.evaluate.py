import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (multilabel_confusion_matrix, precision_recall_fscore_support,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import MultiLabelBinarizer

from config import CONCEPT_MAPS

def parse_and_map(value, col, is_true):
    if pd.isna(value) or str(value).strip().lower() in ("", "none", "nan"):
        return set()
    items = [i.strip() for i in str(value).split(",") if i.strip()]
    if is_true:
        m = CONCEPT_MAPS[col]
        return {m.get(i, i).lower() for i in items}
    return {i.lower() for i in items}


def metrics_bundle(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return {}
    exact = sum(t == p for t, p in zip(y_true, y_pred))
    partial = sum(len(t & p) > 0 for t, p in zip(y_true, y_pred))
    jac = [len(t & p) / len(t | p) if t | p else 0 for t, p in zip(y_true, y_pred)]

    mlb = MultiLabelBinarizer().fit(y_true + y_pred)
    yt, yp = mlb.transform(y_true), mlb.transform(y_pred)
    per_label_acc = (yt == yp).mean(axis=0)
    w = yt.sum(axis=0)

    return {
        "total_rows": n,
        "exact_match": round(exact / n, 3),
        "partial_match": round(partial / n, 3),
        "jaccard": round(float(np.mean(jac)), 3),
        "precision_micro": round(precision_score(yt, yp, average="micro", zero_division=0), 3),
        "recall_micro": round(recall_score(yt, yp, average="micro", zero_division=0), 3),
        "f1_micro": round(f1_score(yt, yp, average="micro", zero_division=0), 3),
        "acc_macro": round(float(per_label_acc.mean()), 3),
        "acc_weighted": round(float(np.average(per_label_acc, weights=w)), 3) if w.sum() else 0.0,
    }


def per_label_report(y_true, y_pred, col):
    mlb = MultiLabelBinarizer().fit(y_true + y_pred)
    yt, yp = mlb.transform(y_true), mlb.transform(y_pred)
    p, r, f, s = precision_recall_fscore_support(yt, yp, average=None, zero_division=0)
    df = pd.DataFrame({"Column": col, "Label": mlb.classes_, "Precision": p,
                       "Recall": r, "F1": f, "Support": s, "Predicted": yp.sum(axis=0)})
    return df.sort_values("F1", ascending=False).round(3)


def group_by_label_count(records):
    rows = []
    for k, g in records.groupby("n_true"):
        t, p = g["true"].tolist(), g["pred"].tolist()
        inter = [len(a & b) for a, b in zip(t, p)]
        tp = sum(inter)
        fp = sum(len(b) for b in p) - tp
        fn = sum(len(a) for a in t) - tp
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        rows.append({
            "true_labels": k, "samples": len(g),
            "exact_match": np.mean([a == b for a, b in zip(t, p)]),
            "partial_match": np.mean([i > 0 for i in inter]),
            "all_true_found": np.mean([a <= b for a, b in zip(t, p)]),
            "f1_micro": 2 * prec * rec / (prec + rec) if prec + rec else 0,
            "under_pred": (g["n_pred"] < g["n_true"]).mean(),
            "over_pred": (g["n_pred"] > g["n_true"]).mean(),
        })
    return pd.DataFrame(rows).round(3)


def plot_confusion(y_true, y_pred, cols=3):
    mlb = MultiLabelBinarizer().fit(y_true + y_pred)
    mcm = multilabel_confusion_matrix(mlb.transform(y_true), mlb.transform(y_pred))
    rows = (len(mlb.classes_) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axes = np.atleast_1d(axes).flatten()
    for ax, cm, label in zip(axes, mcm, mlb.classes_):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred No", "Pred Yes"], yticklabels=["True No", "True Yes"])
        ax.set_title(label, fontsize=8)
    for ax in axes[len(mlb.classes_):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

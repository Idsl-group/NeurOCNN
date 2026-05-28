import json
from pathlib import Path
import numpy as np
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
)
import matplotlib.pyplot as plt


def _jsonable(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (Path,)):
        return str(x)
    return x

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)

def append_csv_row(path: Path, row: dict, header_if_new=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    cols = list(row.keys())
    line = ",".join(str(row[c]) for c in cols) + "\n"
    if is_new and header_if_new:
        with open(path, "w") as f:
            f.write(",".join(cols) + "\n")
            f.write(line)
    else:
        with open(path, "a") as f:
            f.write(line)

def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def plot_learning_curves(epoch_csv: Path, out_png: Path):
    import csv
    epochs, tl, ta, vl, va = [], [], [], [], []
    with open(epoch_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            epochs.append(int(r["epoch"]))
            tl.append(float(r["train_loss"]))
            ta.append(float(r["train_acc"]))
            vl.append(float(r["val_loss"]))
            va.append(float(r["val_acc"]))

    plt.figure()
    plt.plot(epochs, tl, label="train_loss")
    plt.plot(epochs, vl, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.with_name(out_png.stem + "_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, ta, label="train_acc")
    plt.plot(epochs, va, label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(out_png.with_name(out_png.stem + "_acc.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm: np.ndarray, class_names, out_prefix: Path, normalize: bool = False):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_prefix.with_suffix(".npy"), cm)

    if normalize:
        cm_to_plot = cm.astype(np.float64)
        denom = cm_to_plot.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm_to_plot = cm_to_plot / denom
    else:
        cm_to_plot = cm.astype(np.int64)

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm_to_plot, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (norm)" if normalize else ""))
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = float(cm_to_plot.max()) / 2.0 if cm_to_plot.size > 0 else 0.0

    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            if normalize:
                txt = f"{float(cm_to_plot[i, j]):.2f}"
            else:
                txt = str(int(cm_to_plot[i, j]))
            plt.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if float(cm_to_plot[i, j]) > thresh else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close()



def compute_metrics_bundle(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, class_names=None):
    labels = list(range(n_classes))
    if class_names is None:
        class_names = [f"class_{i}" for i in labels]

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, labels=labels)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "acc": float(acc),
        "balanced_acc": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "cohen_kappa": float(kappa),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def module_parameter_table(model: nn.Module):
    rows = []
    for name, m in model.named_modules():
        ps = list(m.parameters(recurse=False))
        if len(ps) == 0:
            continue
        total = sum(p.numel() for p in ps)
        trainable = sum(p.numel() for p in ps if p.requires_grad)
        rows.append({
            "module": name if name != "" else "<root>",
            "type": m.__class__.__name__,
            "params_total": int(total),
            "params_trainable": int(trainable),
        })
    rows.sort(key=lambda r: r["params_total"], reverse=True)
    return rows

def save_model_summary(model: nn.Module, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        **count_parameters(model),
        "modules": module_parameter_table(model),
    }

    with open(out_dir / "model_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "model_arch.txt", "w") as f:
        f.write(str(model) + "\n")

    csv_path = out_dir / "model_modules.csv"
    with open(csv_path, "w") as f:
        f.write("module,type,params_total,params_trainable\n")
        for r in summary["modules"]:
            f.write(f'{r["module"]},{r["type"]},{r["params_total"]},{r["params_trainable"]}\n')

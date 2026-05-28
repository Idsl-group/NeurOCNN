from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from utils.logging import *
import torch
import torch.nn as nn
from pathlib import Path
from models.TemporalNeuralOperatorV1.TemporalNeuralOperator import *
from models.TemporalNeuralOperatorV3.TemporalNeuralOperatorV3 import *
from training.utils import train_one_epoch, evaluate, evaluate_with_outputs
from data_loaders.SleepEDF import create_nfold_dataloaders
from data_loaders.ECG import create_nfold_ecg_dataloaders
from data_loaders.helpers import load_all_epochs, compute_eeg_coords_1020, iter_subject_kfold_splits, split_train_val_groups, iter_kfold_splits, split_train_val
from models.modules.ConvBlocks import ConvBlocks
from models.NeuralCDE.NeuralCDE import NeuralCDE
from models.AttentionNeuralOperator.AttentionNeuralOperator import AttentionNeuralOperator
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np



def safe_multiclass_auc(y_true, logits, n_classes, average="macro"):
    if logits is None or len(y_true) == 0:
        return float("nan")
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs = probs / probs.sum(axis=1, keepdims=True)

    present = np.unique(y_true)
    if present.size < 2:
        return float("nan")

    try:
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average=average))
    except ValueError:
        return float("nan")



def run_kfold_cv(
    model_fn,
    npz_folder,
    dataset_name=None,
    n_classes=5,
    class_names=["W", "N1", "N2", "N3", "REM"],
    n_splits=5,
    val_size=0.1,
    seed=42,
    batch_size=64,
    num_workers=0,
    train_fs=100,
    train_len=30,
    test_fs_list=(80, 100, 128, 256, 512, 1024),
    test_len_list=(30,),
    keep_train_fs=False,
    num_epochs=40,
    lr=1e-3,
    save_dir=None,
    stratified=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_all, y_all, subj_all, ch_names, sfreq = load_all_epochs(npz_folder, dataset_name=dataset_name)
    base_fs = float(sfreq)
    print(f"Loaded all epochs at base_fs={base_fs} Hz")

    if ch_names is not None:
        coords = compute_eeg_coords_1020(ch_names)
        coords_t = torch.from_numpy(coords).float()
        n_channels = len(ch_names)
    else:
        coords_t = None
        n_channels = 1

    run_dir = None
    if save_dir is not None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        ds_tag = dataset_name if dataset_name is not None else "dataset"
        run_dir = Path(save_dir) / f"{ds_tag}_cv_{run_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)

        config = dict(
            dataset_name=dataset_name,
            npz_folder=str(npz_folder),
            n_splits=n_splits,
            val_size=val_size,
            seed=seed,
            batch_size=batch_size,
            num_workers=num_workers,
            train_fs=train_fs,
            train_len=train_len,
            test_fs_list=list(test_fs_list),
            test_len_list=list(test_len_list),
            keep_train_fs=keep_train_fs,
            num_epochs=num_epochs,
            lr=lr,
            base_fs=base_fs,
            n_channels=n_channels,
            n_classes=n_classes,
            ch_names=ch_names,
            device=str(device),
        )
        save_json(run_dir / "config.json", config)

    fold_summaries = []
    fs_acc_by_key = {}  # (fs, len) -> list of accs across folds


    for fold, trainval_idx, test_idx in iter_subject_kfold_splits(
        y_all, subj_all, n_splits=n_splits, seed=seed, stratified=stratified
    ) if subj_all is not None else iter_kfold_splits(y_all, n_splits=n_splits, seed=seed, stratified=stratified):
        print(f"\n========== Fold {fold}/{n_splits} ==========")

        if subj_all is not None:
            train_idx, val_idx = split_train_val_groups(
                trainval_idx, y_all, subj_all, val_size=val_size, seed=seed + 1000 + fold
            )
        else:
            train_idx, val_idx = split_train_val(trainval_idx, y_all, val_size=val_size, seed=seed + 1000 + fold)

        if dataset_name == "ECG":
            train_loader, val_loader, test_loaders, Fs_t = create_nfold_ecg_dataloaders(
                X_all, y_all,
                train_idx, val_idx, test_idx,
                base_fs=base_fs,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle_train=True,
                train_fs=train_fs,
                train_len=train_len,
                test_fs_list=list(test_fs_list),
                test_len_list=list(test_len_list),
                keep_train_fs=keep_train_fs,
            )
        else:
            train_loader, val_loader, test_loaders, Fs_t = create_nfold_dataloaders(
                X_all, y_all,
                train_idx, val_idx, test_idx,
                base_fs=base_fs,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle_train=True,
                train_fs=train_fs,
                train_len=train_len,
                test_fs_list=list(test_fs_list),
                test_len_list=list(test_len_list),
                keep_train_fs=keep_train_fs,
            )

        fold_dir = None
        if run_dir is not None:
            fold_dir = run_dir / f"fold_{fold:02d}"
            (fold_dir / "plots").mkdir(parents=True, exist_ok=True)
            (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (fold_dir / "predictions").mkdir(parents=True, exist_ok=True)
            (fold_dir / "metrics").mkdir(parents=True, exist_ok=True)

            if subj_all is not None:
                split_info = {
                    "fold": fold,
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                    "train_subjects": np.unique(subj_all[train_idx]).tolist(),
                    "val_subjects": np.unique(subj_all[val_idx]).tolist(),
                    "test_subjects": np.unique(subj_all[test_idx]).tolist(),
                }
            else:
                split_info = {
                    "fold": fold,
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                }
            save_json(fold_dir / "split.json", split_info)

        model = model_fn(n_channels, n_classes).to(device)
        save_model_summary(model, save_dir)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        if fold_dir is not None:
            save_json(fold_dir / "model_stats.json", count_parameters(model))

        best_val_acc = -1.0
        best_state = None
        fold_save_path = (fold_dir / "checkpoints" / f"best_model_fold{fold:02d}.pt") if fold_dir is not None else None
        last_save_path = (fold_dir / "checkpoints" / f"last_model_fold{fold:02d}.pt") if fold_dir is not None else None

        epoch_csv = (fold_dir / "epoch_log.csv") if fold_dir is not None else None

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, coords_t, Fs_t, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(
                model, val_loader, coords_t, Fs_t, criterion, device
            )

            dt = time.time() - t0
            print(
                f"Fold {fold:02d} | Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
                f"time={dt:.1f}s"
            )

            if epoch_csv is not None:
                append_csv_row(epoch_csv, {
                    "fold": fold,
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                    "sec": float(dt),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": float(val_acc),
                    "ch_names": ch_names,
                    "fold": fold,
                }
                if fold_save_path is not None:
                    torch.save(best_state, fold_save_path)

        if last_save_path is not None:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": num_epochs,
                "val_acc_last": float(val_acc),
                "ch_names": ch_names,
                "fold": fold,
            }, last_save_path)

        if best_state is not None:
            sd = best_state["model_state"]
            if isinstance(sd, dict) and "_metadata" in sd:
                sd = dict(sd)
                sd.pop("_metadata", None)
            model.load_state_dict(sd)

        if fold_dir is not None and epoch_csv is not None and epoch_csv.exists():
            plot_learning_curves(epoch_csv, fold_dir / "plots" / "learning_curves.png")

        test_loss, test_acc, fs_metrics = evaluate(
            model, test_loaders, coords_t, None, criterion, device
        )
        print(f"Fold {fold:02d} TEST: loss={test_loss:.4f}, acc={test_acc:.4f}")
        print(fs_metrics)

        if fold_dir is not None:
            save_json(fold_dir / "metrics" / "test_fs_metrics_raw.json", {
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "fs_metrics": fs_metrics,
                "best_epoch": int(best_state["epoch"]) if best_state is not None else None,
                "best_val_acc": float(best_val_acc),
            })

        items = []
        if isinstance(test_loaders, dict):
            items = list(test_loaders.items())
        elif isinstance(test_loaders, (list, tuple)):
            if len(test_loaders) > 0 and isinstance(test_loaders[0], (list, tuple)) and len(test_loaders[0]) == 2:
                items = list(test_loaders)
            else:
                items = [(f"test_{i}", ld) for i, ld in enumerate(test_loaders)]
        else:
            items = [("test", test_loaders)]

        all_y_true, all_y_pred, all_logits = [], [], []

        per_key_results = {}

        for key, ld in items:
            loss_k, acc_k, y_true_k, y_pred_k, logits_k = evaluate_with_outputs(
                model, ld, coords_t, None, criterion, device
            )

            all_y_true.append(y_true_k)
            all_y_pred.append(y_pred_k)
            all_logits.append(logits_k)

            key_name = key
            if isinstance(key, (list, tuple)) and len(key) == 2:
                key_name = f"fs{key[0]}_len{key[1]}"
            else:
                key_name = str(key)

            bundle = compute_metrics_bundle(y_true_k, y_pred_k, n_classes, class_names=class_names)
            bundle.update({"loss": float(loss_k), "acc_loader": float(acc_k), "key": key_name})

            per_key_results[key_name] = bundle

            if fold_dir is not None:
                cm = bundle.pop("confusion_matrix")
                save_json(fold_dir / "metrics" / f"{key_name}_metrics.json", bundle)

                save_confusion_matrix(cm, class_names, fold_dir / "metrics" / f"{key_name}_cm", normalize=False)
                save_confusion_matrix(cm, class_names, fold_dir / "metrics" / f"{key_name}_cm_norm", normalize=True)

                np.savez_compressed(
                    fold_dir / "predictions" / f"{key_name}_preds.npz",
                    y_true=y_true_k.astype(np.int64),
                    y_pred=y_pred_k.astype(np.int64),
                    logits=logits_k.astype(np.float32),
                )

        y_true_all = np.concatenate(all_y_true, axis=0) if len(all_y_true) else np.array([], dtype=np.int64)
        y_pred_all = np.concatenate(all_y_pred, axis=0) if len(all_y_pred) else np.array([], dtype=np.int64)
        logits_all_np = np.concatenate(all_logits, axis=0) if len(all_logits) else np.zeros((0, n_classes), dtype=np.float32)
        
        fold_f1_macro = float(f1_score(y_true_all, y_pred_all, average="macro")) if y_true_all.size else float("nan")
        fold_f1_weighted = float(f1_score(y_true_all, y_pred_all, average="weighted")) if y_true_all.size else float("nan")
        fold_auc_macro = safe_multiclass_auc(y_true_all, logits_all_np, n_classes, average="macro")
        fold_auc_weighted = safe_multiclass_auc(y_true_all, logits_all_np, n_classes, average="weighted")


        overall_bundle = compute_metrics_bundle(y_true_all, y_pred_all, n_classes, class_names=class_names)
        if fold_dir is not None:
            cm_all = overall_bundle.pop("confusion_matrix")
            save_json(fold_dir / "metrics" / "overall_test_metrics.json", {
                **overall_bundle,
                "test_loss_reported": float(test_loss),
                "test_acc_reported": float(test_acc),
                "f1_macro": fold_f1_macro,
                "f1_weighted": fold_f1_weighted,
                "auc_ovr_macro": fold_auc_macro,
                "auc_ovr_weighted": fold_auc_weighted,
                "best_epoch": int(best_state["epoch"]) if best_state is not None else None,
                "best_val_acc": float(best_val_acc),
            })
            save_confusion_matrix(cm_all, class_names, fold_dir / "metrics" / "overall_test_cm", normalize=False)
            save_confusion_matrix(cm_all, class_names, fold_dir / "metrics" / "overall_test_cm_norm", normalize=True)
            np.savez_compressed(
                fold_dir / "predictions" / "overall_test_preds.npz",
                y_true=y_true_all.astype(np.int64),
                y_pred=y_pred_all.astype(np.int64),
                logits=logits_all_np.astype(np.float32),
            )

        if isinstance(fs_metrics, (list, tuple)):
            for m in fs_metrics:
                key = m.get("fs", None)
                if key is None:
                    continue
                fs_acc_by_key.setdefault(tuple(key), []).append(float(m.get("acc", np.nan)))

        fold_summaries.append({
            "fold": fold,
            "best_val_acc": float(best_val_acc),
            "test_acc": float(test_acc),
            "f1_macro": fold_f1_macro,
            "f1_weighted": fold_f1_weighted,
            "auc_ovr_macro": fold_auc_macro,
            "auc_ovr_weighted": fold_auc_weighted,
        })

    test_accs = np.array([x["test_acc"] for x in fold_summaries], dtype=float)
    print("\n========== CV Summary ==========")
    print(f"Overall test acc: mean={np.nanmean(test_accs):.4f}, std={np.nanstd(test_accs):.4f}")

    if len(fs_acc_by_key) > 0:
        print("\nPer-(Fs,Len) acc across folds:")
        per_key_summary = {}
        for key in sorted(fs_acc_by_key.keys(), key=lambda x: (x[0], -1 if x[1] is None else x[1])):
            arr = np.array(fs_acc_by_key[key], dtype=float)
            mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
            print(f"  {key}: mean={mu:.4f}, std={sd:.4f}, n={len(arr)}")
            per_key_summary[str(key)] = {"mean": mu, "std": sd, "n": int(len(arr))}

        if run_dir is not None:
            save_json(run_dir / "cv_summary.json", {
                "fold_summaries": fold_summaries,
                "overall_test_acc_mean": float(np.nanmean(test_accs)),
                "overall_test_acc_std": float(np.nanstd(test_accs)),
                "per_fs_len_acc": per_key_summary,
            })
            summary_csv = run_dir / "cv_summary.csv"
            if not summary_csv.exists():
                with open(summary_csv, "w") as f:
                    f.write("fold,best_val_acc,test_acc\n")
            for row in fold_summaries:
                with open(summary_csv, "a") as f:
                    f.write(f"{row['fold']},{row['best_val_acc']},{row['test_acc']}\n")

    return fold_summaries, fs_acc_by_key

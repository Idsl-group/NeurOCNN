from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from scipy.signal import butter, sosfiltfilt


def get_subject_id_from_fname(fname: str, dataset_name=None) -> str:
    if dataset_name is not None and dataset_name.lower() == "isruc":
        base = Path(fname).stem
        sp = base.split("_")
        base = sp[0] + sp[1] + sp[2]
        return base
    elif dataset_name is not None and dataset_name.lower() == "hmc":
        base = Path(fname).stem
        sp = base.split("_")
        base = sp[0] + sp[1]
        return base
    else:
        base = Path(fname).stem
        base = base.split("_")[0]
        return base[:5]


def _lowpass_filter_epochs(X: np.ndarray, sfreq: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz is None:
        return X

    nyq = 0.5 * sfreq
    if not (0 < cutoff_hz < nyq):
        raise ValueError(f"cutoff_hz must be between 0 and Nyquist ({nyq:.3f} Hz). Got {cutoff_hz} Hz.")

    sos = butter(order, cutoff_hz, btype="low", fs=sfreq, output="sos")

    N = X.shape[-1]
    if N < 2:
        return X

    default_padlen = 3 * (2 * sos.shape[0] + 1)
    padlen = min(default_padlen, N - 1)

    X64 = X.astype(np.float64, copy=False)
    X_f = sosfiltfilt(sos, X64, axis=-1, padlen=padlen)
    return X_f.astype(np.float32, copy=False)


def load_all_epochs(npz_folder: str, lowpass_hz=35, filt_order=4, dataset_name="SleepEDF"):
    if dataset_name.lower() == "ecg":
        data = np.load("preprocessed_data/ECG/cinc2017_frames_8s_nonorm.npz")
        X_all = data["x"]
        y_all = data["y"]
        subj_all = None
        ch_names = None
        sfreq = data["fs"]
    else:
        folder = Path(npz_folder)
        npz_files = sorted(folder.glob("*.npz"))
        if not npz_files:
            raise RuntimeError(f"No .npz files found in {folder}")

        X_list, y_list, subj_list = [], [], []
        ch_names = None
        sfreq = None

        for f in npz_files:
            data = np.load(f, allow_pickle=True)
            X = data["X"]
            y = data["y"]
            ch = data["ch_names"] if "ch_names" in data else data["ch"]
            fs = float(data["sfreq"][0])

            if ch_names is None:
                ch_names = ch.tolist()
                sfreq = fs
            else:
                assert ch.tolist() == ch_names, f"Channel mismatch in {f}"
                assert np.isclose(fs, sfreq), f"Sampling rate mismatch in {f}"

            if lowpass_hz is not None:
                X = _lowpass_filter_epochs(X, sfreq=fs, cutoff_hz=lowpass_hz, order=filt_order)

            subj_id = get_subject_id_from_fname(f.name, dataset_name)
            n_ep = X.shape[0]

            X_list.append(X)
            y_list.append(y)
            subj_list.extend([subj_id] * n_ep)

        X_all = np.concatenate(X_list, axis=0).astype(np.float32)
        y_all = np.concatenate(y_list, axis=0).astype(np.int64)
        subj_all = np.array(subj_list)

    return X_all, y_all, subj_all, ch_names, sfreq


def make_subject_splits(y, subjects, train_size=0.7, val_size=0.15, random_state=42):
    n = len(y)
    dummy_X = np.zeros((n, 1))

    gss1 = GroupShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, temp_idx = next(gss1.split(dummy_X, y, groups=subjects))

    remaining_size = 1.0 - train_size
    val_rel = val_size / remaining_size
    gss2 = GroupShuffleSplit(
        n_splits=1, train_size=val_rel, random_state=random_state + 1
    )

    y_temp = y[temp_idx]
    subj_temp = subjects[temp_idx]
    val_idx_rel, test_idx_rel = next(
        gss2.split(np.zeros_like(y_temp), y_temp, groups=subj_temp)
    )

    val_idx = temp_idx[val_idx_rel]
    test_idx = temp_idx[test_idx_rel]

    return train_idx, val_idx, test_idx


def make_splits(y, train_size=0.7, val_size=0.15, random_state=42, stratified=True):
    y = np.asarray(y)
    n = len(y)
    idx_all = np.arange(n)

    if not (0 < train_size < 1) or not (0 < val_size < 1) or (train_size + val_size >= 1):
        raise ValueError("Require 0 < train_size < 1, 0 < val_size < 1, and train_size + val_size < 1.")

    dummy_X = np.zeros((n, 1))

    if stratified:
        splitter1 = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_rel, temp_rel = next(splitter1.split(dummy_X, y))
    else:
        splitter1 = ShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_rel, temp_rel = next(splitter1.split(dummy_X))

    train_idx = idx_all[train_rel]
    temp_idx  = idx_all[temp_rel]

    remaining_size = 1.0 - train_size
    val_rel_size = val_size / remaining_size

    y_temp = y[temp_idx]
    dummy_temp = np.zeros((len(temp_idx), 1))

    if stratified:
        splitter2 = StratifiedShuffleSplit(n_splits=1, train_size=val_rel_size, random_state=random_state + 1)
        val_rel, test_rel = next(splitter2.split(dummy_temp, y_temp))
    else:
        splitter2 = ShuffleSplit(n_splits=1, train_size=val_rel_size, random_state=random_state + 1)
        val_rel, test_rel = next(splitter2.split(dummy_temp))

    val_idx  = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    return train_idx, val_idx, test_idx


def iter_subject_kfold_splits(y, subjects, n_splits=5, seed=42, stratified=True):
    dummy_X = np.zeros((len(y), 1))

    if stratified:
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold, (trainval_idx, test_idx) in enumerate(splitter.split(dummy_X, y, groups=subjects), start=1):
                yield fold, trainval_idx, test_idx
            return
        except Exception as e:
            print(f"[WARN] StratifiedGroupKFold not available/failed ({e}). Falling back to GroupKFold.")

    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (trainval_idx, test_idx) in enumerate(splitter.split(dummy_X, y, groups=subjects), start=1):
        yield fold, trainval_idx, test_idx
        

def iter_kfold_splits(y, n_splits=5, seed=42, stratified=True, shuffle=True):
    y = np.asarray(y)

    if stratified:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)

    dummy_X = np.zeros((len(y), 1))

    for fold, (trainval_idx, test_idx) in enumerate(splitter.split(dummy_X, y), start=1):
        yield fold, trainval_idx, test_idx


def split_train_val_groups(trainval_idx, y, subjects, val_size=0.1, seed=123):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)

    y_tv = y[trainval_idx]
    subj_tv = subjects[trainval_idx]
    dummy_tv = np.zeros((len(trainval_idx), 1))

    train_rel, val_rel = next(gss.split(dummy_tv, y_tv, groups=subj_tv))
    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]
    return train_idx, val_idx


def split_train_val(trainval_idx, y, val_size=0.1, seed=123, stratified=True):
    trainval_idx = np.asarray(trainval_idx)
    y = np.asarray(y)

    y_tv = y[trainval_idx]
    dummy_tv = np.zeros((len(trainval_idx), 1))

    if stratified:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_rel, val_rel = next(splitter.split(dummy_tv, y_tv))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        train_rel, val_rel = next(splitter.split(dummy_tv))

    train_idx = trainval_idx[train_rel]
    val_idx   = trainval_idx[val_rel]
    return train_idx, val_idx

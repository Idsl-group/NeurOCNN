import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from data_loaders.helpers import (
    load_all_epochs,
    make_subject_splits,
)

class SleepEDFEEGDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = np.array(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        x = self.X[global_idx]
        y = self.y[global_idx]
        x = torch.from_numpy(x).float()
        y = int(y)
        return x, y


class SleepEDFEEGResampledTestDataset(Dataset):
    def __init__(self, X, y, indices, base_fs, train_fs, target_fs, target_len=None, keep_base_fs=False):
        self.X = X
        self.y = y
        self.indices = np.array(indices)
        self.base_fs = float(base_fs)
        self.train_fs = float(train_fs)
        self.target_fs = float(target_fs)
        self.target_len = target_len
        self.keep_base_fs = keep_base_fs

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def _resample_eeg(x_np, base_fs, train_fs, target_fs, keep_base_fs, target_len=None):
        C, N = x_np.shape
        duration = N / base_fs
        N_new = int(round(duration * target_fs))
        N_train = int(round(duration * train_fs))

        x = torch.from_numpy(x_np).float()
        x = x.unsqueeze(0)

        x_res = F.interpolate(
            x,
            size=N_new,
            mode="linear",
            align_corners=False,
        )
        if keep_base_fs:
            x_res = F.interpolate(
                x_res,
                size=N_train,
                mode="linear",
                align_corners=False,
            )
        if target_len is not None and keep_base_fs:
            target_length = int(target_len * train_fs)
            x_res = x_res[:, :, :target_length]
        elif target_len is not None:
            target_length = int(target_len * target_fs)
            x_res = x_res[:, :, :target_length]

        return x_res.squeeze(0)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        x_np = self.X[global_idx]
        y = int(self.y[global_idx])

        x = self._resample_eeg(x_np, self.base_fs, self.train_fs, self.target_fs, self.keep_base_fs, self.target_len)
        fs_t = torch.tensor(self.target_fs, dtype=torch.float32)

        return x, y, fs_t



def create_dataloaders(
    npz_folder,
    batch_size=64,
    train_size=0.8,
    val_size=0.1,
    num_workers=0,
    shuffle_train=True,
    train_fs=None,
    train_len=None,
    test_fs_list=None,
    test_len_list=None,
    keep_train_fs=False,
    seed=42
):
    g = torch.Generator()
    g.manual_seed(seed)

    X_all, y_all, subj_all, ch_names, sfreq = load_all_epochs(npz_folder)
    base_fs = float(sfreq)
    print(f"Loaded all epochs at base_fs={base_fs} Hz")

    train_idx, val_idx, test_idx = make_subject_splits(
        y_all, subj_all, train_size=train_size, val_size=val_size
    )

    if train_fs is None:
        train_ds = SleepEDFEEGDataset(X_all, y_all, train_idx)
        val_ds   = SleepEDFEEGDataset(X_all, y_all, val_idx)
        train_fs = base_fs
    else:
        train_ds = SleepEDFEEGResampledTestDataset(
                X_all, y_all, train_idx, base_fs=base_fs, train_fs=train_fs, target_fs=train_fs, target_len=train_len, keep_base_fs=False
            )
        val_ds   = SleepEDFEEGResampledTestDataset(
                X_all, y_all, val_idx, base_fs=base_fs, train_fs=train_fs, target_fs=train_fs, target_len=train_len, keep_base_fs=False
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, drop_last=False, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, generator=g
    )

    if test_fs_list is None:
        test_fs_list = [train_fs]

    if test_len_list is None:
        test_len_list = [None]

    test_loaders = {}
    for fs in test_fs_list:
        for length in test_len_list:
            test_ds = SleepEDFEEGResampledTestDataset(
                X_all, y_all, test_idx, base_fs=base_fs, train_fs=train_fs, target_fs=fs, target_len=length, keep_base_fs=keep_train_fs
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                generator=g
            )
            test_loaders[(fs, length)] = test_loader
            print(f"Created test_loader for Fs={fs} Hz, target_len={length} s, "
                  f"{len(test_ds)} samples, "
                  f"epoch length ~{len(test_ds[0][0][0])} time points")

    coords_t = None
    Fs_train = torch.tensor(train_fs, dtype=torch.float32)
    return train_loader, val_loader, test_loaders, coords_t, Fs_train, ch_names


def create_nfold_dataloaders(
    X_all, y_all,
    train_idx, val_idx, test_idx,
    base_fs,
    batch_size=64,
    num_workers=0,
    shuffle_train=True,
    train_fs=None,
    train_len=None,
    test_fs_list=None,
    test_len_list=None,
    keep_train_fs=False,
):
    if train_fs is None:
        train_ds = SleepEDFEEGDataset(X_all, y_all, train_idx)
        val_ds   = SleepEDFEEGDataset(X_all, y_all, val_idx)
        train_fs = float(base_fs)
    else:
        train_ds = SleepEDFEEGResampledTestDataset(
            X_all, y_all, train_idx,
            base_fs=float(base_fs),
            train_fs=float(train_fs),
            target_fs=float(train_fs),
            target_len=train_len,
            keep_base_fs=False,
        )
        val_ds = SleepEDFEEGResampledTestDataset(
            X_all, y_all, val_idx,
            base_fs=float(base_fs),
            train_fs=float(train_fs),
            target_fs=float(train_fs),
            target_len=train_len,
            keep_base_fs=False,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False
    )

    if test_fs_list is None:
        test_fs_list = [train_fs]
    if test_len_list is None:
        test_len_list = [None]

    test_loaders = {}
    for fs in test_fs_list:
        for length in test_len_list:
            test_ds = SleepEDFEEGResampledTestDataset(
                X_all, y_all, test_idx,
                base_fs=float(base_fs),
                train_fs=float(train_fs),
                target_fs=float(fs),
                target_len=length,
                keep_base_fs=keep_train_fs,
            )
            test_loaders[(float(fs), length)] = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, drop_last=False
            )

    Fs_train = torch.tensor(float(train_fs), dtype=torch.float32)
    return train_loader, val_loader, test_loaders, Fs_train

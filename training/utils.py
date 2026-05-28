import torch
from training.LossFunctions import accuracy_from_logits
import numpy as np

def train_one_epoch(model, loader, coords, Fs, optimizer, criterion, device, task="classification"):
    model.train()
    if coords is not None:
        coords = coords.to(device)
    Fs = Fs.to(device)
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for idx, (x, y, _) in enumerate(loader):
        print(f"Batch {idx+1}/{len(loader)}, GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", end="\r")
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x, Fs=Fs, coords=coords, T=None)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if task == "classification":
            acc = accuracy_from_logits(logits, y)
            running_acc += acc
        else:
            running_acc += 0.0
        running_loss += loss.item()
        n_batches += 1

    print()
    return running_loss / n_batches, running_acc / n_batches


@torch.no_grad()
def evaluate(model, test_loaders, coords, Fs, criterion, device, return_outputs=False, task="classification"):
    model.eval()
    if coords is not None:
        coords = coords.to(device)
    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0
    fs_metrics = []
    
    ys = []
    preds = []
    logits_all = []
    
    if Fs is None:
        y_true_list = []
        y_preds_list = []
        logits_list = []
        for fs, test_loader in test_loaders.items():
            n_fs_batches = 0
            running_fs_loss = 0.0
            running_fs_acc = 0.0
            for idx, (x, y, fs_t) in enumerate(test_loader):
                print(f"Batch {idx+1}/{len(test_loader)} {fs} Hz", end="\r")
                x = x.to(device)
                y = y.to(device)
                
                fs_t = fs_t[0].item()
                logits = model(x, Fs=fs_t, coords=coords, T=None)
                loss = criterion(logits, y)
                
                if task == "classification":
                    acc = accuracy_from_logits(logits, y)
                    running_acc += acc
                    running_fs_acc += acc
                else:
                    running_acc += 0.0
                    running_fs_acc += 0

                running_loss += loss.item()
                n_batches += 1
                n_fs_batches += 1

                running_fs_loss += loss.item()
                
                y_pred = torch.argmax(logits, dim=-1)
                ys.append(y.detach().cpu().numpy())
                preds.append(y_pred.detach().cpu().numpy())
                logits_all.append(logits.detach().cpu().numpy())
            
            fs_metrics.append({
                'fs': fs,
                'loss': running_fs_loss / n_fs_batches,
                'acc': running_fs_acc / n_fs_batches
            })
            
            y_true = np.concatenate(ys, axis=0)
            y_preds = np.concatenate(preds, axis=0)
            logits_np = np.concatenate(logits_all, axis=0)
            y_true_list.append(y_true)
            y_preds_list.append(y_preds)
            logits_list.append(logits_np)
        
        if return_outputs:
            return running_loss / n_batches, running_acc / n_batches, fs_metrics, y_true_list, y_preds_list, logits_list
        else:
            return running_loss / n_batches, running_acc / n_batches, fs_metrics
    
    else:
        for x, y, _ in test_loaders:
            x = x.to(device)
            y = y.to(device)

            logits = model(x, Fs=Fs, coords=coords, T=None)
            loss = criterion(logits, y)
            
            if task == "classification":
                acc = accuracy_from_logits(logits, y)
                running_acc += acc
            else:
                running_acc += 0.0

            running_loss += loss.item()
            n_batches += 1
        
        return running_loss / n_batches, running_acc / n_batches
    
    
    
@torch.no_grad()
def evaluate_with_outputs(model, loader, coords_t, Fs_t, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0

    ys = []
    preds = []
    logits_all = []
    
    coords_t = coords_t.to(device) if coords_t is not None else None

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                x, y = batch
                fs_batch = None
            elif len(batch) == 3:
                x, y, fs_batch = batch
            else:
                raise ValueError(f"Unexpected batch len={len(batch)}")
        elif isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
            fs_batch = batch.get("fs", None)
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")

        x = x.to(device)
        y = y.to(device)

        Fs_in = fs_batch.to(device) if fs_batch is not None else Fs_t

        Fs_in = Fs_in[0].item()
        logits = model(x, coords=coords_t, Fs=Fs_in)

        loss = criterion(logits, y)

        bs = y.numel()
        total_loss += loss.item() * bs
        total_n += bs

        y_pred = torch.argmax(logits, dim=-1)

        ys.append(y.detach().cpu().numpy())
        preds.append(y_pred.detach().cpu().numpy())
        logits_all.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    logits_np = np.concatenate(logits_all, axis=0)

    loss_avg = total_loss / max(total_n, 1)
    acc = (y_true == y_pred).mean() if y_true.size > 0 else float("nan")
    return float(loss_avg), float(acc), y_true, y_pred, logits_np

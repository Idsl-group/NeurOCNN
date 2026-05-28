import torch
import torch.nn as nn
from training.utils import train_one_epoch, evaluate
from data_loaders.SleepEDF import create_dataloaders
from utils.seed import seed_everything
from models.NeurOCNN.NeurOCNN import NeurOCNN


seed = 1000
seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading data...")
(train_loader,
    val_loader,
    test_loaders,
    coords_t,
    Fs_t,
    ch_names) = create_dataloaders(
    npz_folder="./preprocessed_data/SleepEDF",
    batch_size=64,
    train_size=0.8,
    val_size=0.1,
    num_workers=0,
    train_fs=100,
    train_len=30,
    test_fs_list=[80, 90, 100, 128, 256, 500, 512, 1024],
    test_len_list=[30],
    keep_train_fs=False,
)
print("Data loaded")

n_channels = len(ch_names)
n_classes = 5  # W, N1, N2, N3, REM

model = NeurOCNN(
    input_channels=n_channels,
    hidden_channels=128,
    output_channels=n_classes,
    kernel_duration=0.5,
    num_ctrl_points=15,
    convolution_type="spline",
    M=100,
    T_total=30,
    T_seg=5.0
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
best_state = None
num_epochs = 40

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, coords_t, Fs_t, optimizer, criterion, device
    )
    val_loss, val_acc = evaluate(
        model, val_loader, coords_t, Fs_t, criterion, device
    )

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f} | "
        f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "ch_names": ch_names,
        }
        print(f"  -> New best model!")

if best_state is not None:
    sd = best_state["model_state"]
    if isinstance(sd, dict) and "_metadata" in sd:
        sd = dict(sd)
        sd.pop("_metadata", None)
    model.load_state_dict(sd)

test_loss, test_acc, fs_metrics = evaluate(
    model, test_loaders, coords_t, None, criterion, device
)
print(f"\nTest: loss={test_loss:.4f}, acc={test_acc:.4f}")
print(fs_metrics)

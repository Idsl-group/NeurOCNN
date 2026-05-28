from models.NeurOCNN.NeurOCNN import NeurOCNN
from training.run_nfold_cv import run_kfold_cv
from utils.seed import seed_everything


model_name = "NeurOCNN"
model_number = 1

seed = 3
seed_everything(seed)

fold_summaries, fs_acc_by_key = run_kfold_cv(
    model_fn=lambda n_channels, n_classes: NeurOCNN(
        input_channels=n_channels,
        hidden_channels=128,
        output_channels=n_classes,
        kernel_duration=0.5,
        num_ctrl_points=15,
        M=50,
        T_total=30,
        T_seg=5,
    ),
    npz_folder="preprocessed_data/SleepEDF",
    dataset_name="SleepEDF",
    n_splits=5,
    val_size=0.1,
    seed=seed,
    batch_size=64,
    train_fs=100,
    train_len=30,
    test_fs_list=(80, 90, 100, 128, 200, 256, 300, 500, 512, 1000, 1024),
    test_len_list=(30,),
    keep_train_fs=False,
    num_epochs=20,
    stratified=True,
    save_dir=f"results/{model_name}_{model_number}",
)

print(fold_summaries)
print(fs_acc_by_key)

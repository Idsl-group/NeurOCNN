import torch
from torch import nn
import os
from typing import List
from pathlib import Path
from timm.models import create_model
from models.Labram import modeling_pretrain, modeling_finetune

# Download the checkpoints from https://github.com/935963004/LaBraM/tree/main/checkpoints
# and place them in the models/Labram/checkpoints folder

STANDARD_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def get_input_chans(ch_names):
    input_chans = [0] 
    for ch_name in ch_names:
        input_chans.append(STANDARD_1020.index(ch_name) + 1)
    return input_chans



class Labram(torch.nn.Module):
    def __init__(
            self,
            num_classes,
            device,
            epoch_length,
            sampling_rate,
            channel_names,
            model_name = "labram_base_patch200_1600_8k_vocab",
            input_size = 1600,
            layer_scale_init_value = 0.1, 
            drop_path = 0.1,
            codebook_size = 8192,
            codebook_dim = 32,
    ):
        super().__init__()
        self.device=device
        self.name=model_name
        self.input_size=input_size
        self.layer_scale_init_value=layer_scale_init_value
        self.drop_path=drop_path
        self.codebook_size=codebook_size
        self.codebook_dim=codebook_dim
        self.pretrained_weights_path = os.path.join(Path(__file__).parent.resolve(), "checkpoints/labram-base.pth")
        print(f"path: {self.pretrained_weights_path}")
        self.model = self.get_model()
        self.model.to(self.device)
        
        self.epoch_length = epoch_length
        self.sampling_rate = sampling_rate
        self.input_channels = get_input_chans(ch_names=channel_names)
        assert epoch_length % 15 == 0, "Epoch length should be a multiple of 15"
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=len(channel_names)*epoch_length*8192, out_features=num_classes)
        )
        

    def get_model(self):
        print("-"*60)
        print(f"Creating Model: {self.name}", end="\r")
        model = create_model(
            model_name = self.name,
            pretrained = True,
            drop_path_rate = self.drop_path,
            drop_block_rate = None,
            use_shared_rel_pos_bias = False,
            use_abs_pos_emb = True,
            init_values = self.layer_scale_init_value,
            vocab_size = self.codebook_size,
            init_ckpt = self.pretrained_weights_path
        )
        print(f"Creating Model: {self.name} | Done!")
        print("-"*60, end="\n\n")
        return model
    
    def load_pretrained_weights(self):
        init_ckpt = os.path.join(Path(__file__).parent.resolve(), self.pretrained_weights_path)
        checkpoint = torch.load(init_ckpt, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint["model"], strict=False)
        print("Loaded Pretrained LaBraM")
        return
    
    def unfreeze_tf_blocks(self, num_tf=4):
        for module_name, module in self.model.named_children():
            if module_name == "student":
                for submodule_name, submodule in module.named_children():
                    if submodule_name == "blocks":
                        for subsubmodule_name, subsubmodule in submodule.named_children():
                            if int(subsubmodule_name) > (11-num_tf):
                                subsubmodule.requires_grad_(True)
                            else:
                                pass
                    elif submodule_name == "norm" or submodule_name == "lm_head":
                        submodule.requires_grad_(True)
            elif module_name == "lm_head" or module_name == "projection_head":
                    module.requires_grad_(True)
    
    def forward(self, x:torch.Tensor, **kwargs):
        B, C, N = x.shape
        W, S = self.epoch_length, self.sampling_rate
        W_N = int(W/15)
        
        x = x.reshape(B, C, W, S) 
        x = x.reshape(B, C, W_N, 15, S) 
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B*W_N, C, 15, S)
        x = self.model(x, self.input_channels)
        x = x.reshape(B*W_N, C, 15, x.size(-1))
        x = x.reshape(B, W_N, C, 15, x.size(-1))
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, W, x.size(-1))
        return self.classifier(x)

#Implementation of PatchTST model from huggingface transformers library

import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForClassification

class PatchTST(nn.Module):
    def __init__(self, input_channels, num_classes, epoch_length, sampling_rate, 
                 patch_length=60, stride=30, d_model=128, n_heads=4, n_layers=3, 
                 dropout=0.0, head_dropout=0.0):
        super().__init__()
        
        self.context_length = int(epoch_length * sampling_rate)
        
        config = PatchTSTConfig(
            num_input_channels=input_channels,
            context_length=self.context_length,
            patch_length=patch_length,
            patch_stride=stride,
            d_model=d_model,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            attention_dropout=dropout,
            ff_dropout=dropout,
            positional_dropout=dropout, 
            path_dropout=dropout,
            head_dropout=head_dropout,
            ffn_dim=d_model * 4,
            num_targets=num_classes,  
            use_cls_token=False,
            pooling_type="mean"
        )
        
        self.model = PatchTSTForClassification(config)
        
    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 1)
        outputs = self.model(past_values=x)
        return outputs.prediction_logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation="elu", dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(1, kernel_size), 
            dilation=(1, dilation),
            padding="same",
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _get_activation(self, name):
        if name == "elu":
            return nn.ELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        else:
            return nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, complexity_factor, activation="elu"):
        super().__init__()
        scaled_out = int(out_channels * np.sqrt(complexity_factor))
        self.conv = ConvBlock(in_channels, scaled_out, kernel_size, activation=activation)
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

    def forward(self, x):
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (1, 0))
        
        conv_out = self.conv(x)
        if conv_out.shape[-1] % 2 != 0:
            conv_out = F.pad(conv_out, (1, 0))
            
        pooled = self.max_pool(conv_out)
        return pooled, conv_out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size, complexity_factor, activation="elu"):
        super().__init__()
        scaled_out = int(out_channels * np.sqrt(complexity_factor))
        
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2), mode="nearest"),
            ConvBlock(in_channels, scaled_out, kernel_size=2, activation=activation)
        )

        self.merge_conv = ConvBlock(
            scaled_out + skip_channels, 
            scaled_out, 
            kernel_size=kernel_size, 
            activation=activation
        )

    def crop_to_match(self, x, skip):
        diff = x.shape[-1] - skip.shape[-1]
        if diff > 0:
            start = diff // 2 + diff % 2
            x = x[..., start : start + skip.shape[-1]]
        elif diff < 0:
            pass
        return x

    def forward(self, x, skip):
        x = self.up_conv(x)
        x = self.crop_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        x = self.merge_conv(x)
        return x

class USleep(nn.Module):
    def __init__(
        self, 
        input_channels, 
        num_classes,
        depth=12, 
        init_filters=5, 
        complexity_factor=2, 
        kernel_size=9, 
        activation="elu",
        period_length=None
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.depth = depth
        self.period_length = period_length
        self.encoders = nn.ModuleList()
        current_filters = init_filters
        self.skip_channels_list = []
        last_out_ch = input_channels
        
        for i in range(depth):
            self.encoders.append(
                EncoderBlock(last_out_ch, current_filters, kernel_size, complexity_factor, activation)
            )
            
            this_out_ch = int(current_filters * np.sqrt(complexity_factor))
            self.skip_channels_list.append(this_out_ch)
            last_out_ch = this_out_ch
            current_filters = int(current_filters * np.sqrt(2))

        bottleneck_in_ch = last_out_ch
        bottleneck_out_ch = int(current_filters * np.sqrt(complexity_factor))
        self.bottleneck = ConvBlock(bottleneck_in_ch, bottleneck_out_ch, kernel_size, activation=activation)
        
        self.decoders = nn.ModuleList()
                
        self.decoders = nn.ModuleList()
        decoder_filters_val = current_filters
        
        last_out_ch = bottleneck_out_ch
        
        for i in range(depth):
            decoder_filters_val = int(np.ceil(decoder_filters_val / np.sqrt(2)))
            
            skip_ch = self.skip_channels_list[depth - 1 - i]
            
            self.decoders.append(
                DecoderBlock(
                    last_out_ch, 
                    skip_ch, 
                    decoder_filters_val, 
                    kernel_size, 
                    complexity_factor, 
                    activation
                )
            )
            
            last_out_ch = int(decoder_filters_val * np.sqrt(complexity_factor))
            
        self.dense_classifier = nn.Conv2d(last_out_ch, self.num_classes, kernel_size=1)
        
        transition_window = 1
        self.seq_conv1 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=(1, transition_window), padding="same")
        self.seq_conv2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=(1, transition_window), padding="same")
        
    def forward(self, x, **kwargs):
        x = x.unsqueeze(2)
        skips = []
        
        # Encoder
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for dec in self.decoders:
            skip = skips.pop()
            x = dec(x, skip)
        
        # Dense Classifier
        x = self.dense_classifier(x) 
        
        L = x.shape[-1] 
        
        if self.period_length is not None:
             pool_size = self.period_length
             if L % pool_size != 0:
                 pass
        else:
             pool_size = L
             
        x = F.avg_pool2d(x, kernel_size=(1, pool_size), stride=(1, pool_size)) 
        x = self.seq_conv1(x)
        x = F.elu(x) 
        x = self.seq_conv2(x)
        x = x.view(x.shape[0], self.num_classes)
        return x

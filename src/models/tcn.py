import torch
import torch.nn as nn
import torch.nn.utils as utils

class ChapedCausalConv1d(nn.Module):
    """
    1D Convolution explicitly padded to guarantee causal lookahead constraint.
    Output at index t only depends on inputs from index <= t.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(ChapedCausalConv1d, self).__init__()
        # Total padding needed to ensure output sequence length matches input
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=self.padding, dilation=dilation
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        res = self.conv(x)
        # Slice off the extra padding right-hand side elements to make it causal
        return res[:, :, :-self.padding] if self.padding > 0 else res


class TCNResidualBlock(nn.Module):
    """
    A single Residual block containing two layers of causal dilated convolutions,
    ReLU activations, dropout, and a residual skip connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, dropout=0.1):
        super(TCNResidualBlock, self).__init__()
        self.conv1 = ChapedCausalConv1d(in_channels, out_channels, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = ChapedCausalConv1d(out_channels, out_channels, kernel_size, stride, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Composite sequential path
        self.net = nn.Sequential(
            self.conv1, self.bn1, self.relu1, self.drop1,
            self.conv2, self.bn2, self.relu2, self.drop2
        )

        # Match dimensions for residual skip if channels change
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)


class CausalTCNLM(nn.Module):
    """
    Temporal Convolutional Network adapted for Industrial Sequence Modeling (Language Model framing).
    Accepts token IDs, processes through causal layers, and outputs step vocabulary logits.
    """
    def __init__(self, vocab_size=208, d_model=256, num_channels=[256, 256, 256, 256], kernel_size=3, dropout=0.15):
        super(CausalTCNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # Exponential dilation sequence: 1, 2, 4, 8...
            dilation_size = 2 ** i
            in_channels = d_model if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNResidualBlock(
                    in_channels, out_channels, kernel_size, 
                    stride=1, dilation=dilation_size, dropout=dropout
                )
            )
            
        self.tcn = nn.Sequential(*layers)
        # Map output features back to vocabulary logits
        self.lm_head = nn.Linear(num_channels[-1], vocab_size)
        
    def forward(self, input_ids):
        # Input shape: [Batch_Size, Seq_Len]
        x = self.embedding(input_ids) # -> [Batch_Size, Seq_Len, d_model]
        
        # Conv1D expects features as channel dimensions: [Batch_Size, Channels, Seq_Len]
        x = x.transpose(1, 2)
        
        features = self.tcn(x) # -> [Batch_Size, Last_Channel_Dim, Seq_Len]
        
        # Transpose back for classification head: [Batch_Size, Seq_Len, Last_Channel_Dim]
        features = features.transpose(1, 2)
        
        logits = self.lm_head(features) # -> [Batch_Size, Seq_Len, vocab_size]
        return logits
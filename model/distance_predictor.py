import torch
import torch.nn as nn

from model.resnet import ResidualNetwork
from model.transformer import LayerNorm
from model.sym_cnn import Sym_CNN2d_Network
"""
predict distance map from pair features
based on Hybrid CNN
"""

class DistanceNetwork(nn.Module):

    def __init__(self, n_feat, n_block=1, block_type='orig', p_drop=0.0):
        """
        Hybrid CNN for inter-residue relationship
        Input: 
            n_feat(int):feature dimension
            n_block(int):number of blocks 
            block_type(str):type of block
            p_drop(float):dropout ratio
        """
        super(DistanceNetwork, self).__init__()
        self.norm = LayerNorm(n_feat)
        self.proj = nn.Linear(n_feat, n_feat)
        self.drop = nn.Dropout(p_drop)
    
        self.resnet_theta = ResidualNetwork(n_block, n_feat, n_feat, 37, block_type=block_type, p_drop=p_drop)
        self.resnet_phi = ResidualNetwork(n_block, n_feat, n_feat, 19, block_type=block_type, p_drop=p_drop)

        # input_channel, output_channel, kernel_size, n_layers
        self.cnn_dist = Sym_CNN2d_Network(n_feat, n_feat, 37, 3, 3, p_drop=p_drop)
        self.cnn_omega = Sym_CNN2d_Network(n_feat, n_feat, 37, 3, 3, p_drop=p_drop)

    def forward(self, x):
        """
        Input: 
            x(float tensor):pair info (B, L, L, C)

        Output:
            logits_dist(float tensor):pair distance features 
            logits_omega(float tensor):pair omega angle features 
            logits_theta(float tensor):pair thate angle features
            logits_phi(float tensor):pair phi angle features
        """
        x = self.norm(x)
        x = self.drop(self.proj(x))
        x = x.permute(0,3,1,2).contiguous()
        # predict theta, phi (non-symmetric)
        logits_theta = self.resnet_theta(x)
        logits_phi = self.resnet_phi(x)

        # predict dist, omega
        x = 0.5 * (x + x.permute(0,1,3,2))
        logits_dist = self.cnn_dist(x)
        logits_omega = self.cnn_omega(x)

        return logits_dist, logits_omega, logits_theta, logits_phi

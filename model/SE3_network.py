import torch
import torch.nn as nn

from utils.equivariant_attention.fibers import Fiber
from utils.equivariant_attention.modules import GConvSE3
from utils.equivariant_attention.modules import GNormSE3
from utils.equivariant_attention.modules import GSE3Res
from utils.equivariant_attention.modules import GNormBias
from utils.equivariant_attention.modules import get_basis_and_r




class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers=1, num_channels=8, num_degrees=2, n_heads=1, div=2,
                 si_m='1x1', si_e='att',
                 l0_in_features=8, l0_out_features=8,
                 l1_in_features=3, l1_out_features=3,
                 num_edge_features=8, x_ij=None):

        '''
        Input:
            - num_layers(int): the number of layers
            - num_channels(int): the number of channels
            - num_degrees(int): the number of feature degrees
            - n_heads(int): the number of attention heads
            - div(int): channels divided number
            - si_m(str): self interaction type for middle layers
            - si_e(str): self interaction type for end layers
            - l0_in_features(int): the input dimension of residue node feature
            - l0_out_features(int): the output dimension of residue node feature
            - l1_in_features(int): input backbone's offset to CA atom
            - l1_out_features(int): output backbone's offset to CA atom
            - num_edge_features(int): the dimension of edge features
            - x_ij(str): the combination types of feature vectors (None, 'cat', 'add')
        '''
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = num_edge_features
        self.div = div
        self.n_heads = n_heads
        self.si_m, self.si_e = si_m, si_e
        self.x_ij = x_ij

        if l1_out_features > 0:
            fibers = {'in': Fiber(dictionary={0: l0_in_features, 1: l1_in_features}),
                           'mid': Fiber(self.num_degrees, self.num_channels),
                           'out': Fiber(dictionary={0: l0_out_features, 1: l1_out_features})}
        else:
            fibers = {'in': Fiber(dictionary={0: l0_in_features, 1: l1_in_features}),
                           'mid': Fiber(self.num_degrees, self.num_channels),
                           'out': Fiber(dictionary={0: l0_out_features})}

        blocks = self._build_gcn(fibers)
        self.Gblock = blocks

    def _build_gcn(self, fibers):
        # Equivariant layers
        '''
        Input:
            - fibers(dict): dimension for input, middle, output layers
        Output:
            - Gblock(list): graph convolution blocks
        '''
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads,
                                  learnable_skip=True, skip='cat',
                                  selfint=self.si_m, x_ij=self.x_ij))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(
            GSE3Res(fibers['mid'], fibers['out'], edge_dim=self.edge_dim,
                    div=1, n_heads=min(1, 2), learnable_skip=True,
                    skip='cat', selfint=self.si_e, x_ij=self.x_ij))
        return nn.ModuleList(Gblock)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, G, type_0_features, type_1_features):
        '''
        Input:
            - G(DGL graph data): residue graph
            - type_0_features(tensor): node feature in residue graph
            - type_1_features(tensor): backbone coordinate offset to CA atom
        Output:
            - h(dict): output residue feature and backbone coordinate offset to CA atom
        '''

        
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        h = {'0': type_0_features, '1': type_1_features}

        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        return h

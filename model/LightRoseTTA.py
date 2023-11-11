'''
Protein Structure Prediction Network
'''

import torch
import torch.nn as nn


from model.parse_msa_info import msa_parser
from model.refine_net import Refine_Network
from model.build_graph import Build_Graph_Network
class Predict_Network(nn.Module):
    def __init__(self, args):
        '''
        Input:
            - args(object):model argument
        '''
        super(Predict_Network, self).__init__()
        self._num_out_bins = 37
        self.parse_msa_model = msa_parser(args)
        self.build_graph_model = Build_Graph_Network(args)
        self.refine_model = Refine_Network(args)

    def forward(self, data):
        '''main framework

        Input:
            - data(object):input graph data
        Output:
            - xyz(tensor): atom 3d position
            - model_lddt(tensor):residue sequence LDDT
            - logits(tensor):residue pair distance and torsion angles
        '''
        # extract msa feature
        msa, prob_s, logits, seq1hot, idx = self.parse_msa_model(data.msa, data.xyz_t, data.t1d, data.t0d)

        # build protein graph and get backbone position
        bb_xyz, bb_state, node, edge = self.build_graph_model(data, msa, prob_s, seq1hot, idx)

        # refine and get all atom position
        xyz, model_lddt = self.refine_model(bb_xyz, bb_state, node, edge, seq1hot, idx, data.CA_atom_index)
        del msa
        del prob_s
        del seq1hot
        del idx

        del bb_xyz
        del bb_state
        del node
        del edge

        torch.cuda.empty_cache()
        return xyz, logits




        

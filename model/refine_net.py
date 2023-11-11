'''
refine backbone atom 3d position
'''

import torch
import torch.nn as nn

from model.transformer import LayerNorm
from model.SE3_network import SE3Transformer
from model.attention_module import make_graph as make_graph_topk




class Refine_Network(nn.Module):
    def __init__(self, args):
        super(Refine_Network, self).__init__()
        '''
        Refine the initial backbone atom coordinates

        Input:
            - args(argparse object): atom coordinates refinement parameters
        '''
        self.d_node = args.d_msa
        self.d_pair = args.d_hidden*2
        self.d_state = args.l0_out_feat
        self.norm_msa = LayerNorm(self.d_node)
        self.norm_pair = LayerNorm(self.d_pair)
        self.norm_state = LayerNorm(self.d_state)

        self.embed_x = nn.Linear(self.d_node+21+self.d_state, args.l0_in_feat) #ori_dim:args.l0_in_feat
        self.embed_e1 = nn.Linear(self.d_pair, args.edge_feat_dim)
        self.embed_e2 = nn.Linear(args.edge_feat_dim+36+1, args.edge_feat_dim)
        
        self.norm_node = LayerNorm(args.l0_in_feat)#ori_dim:args.l0_in_feat
        self.norm_edge1 = LayerNorm(args.edge_feat_dim)
        self.norm_edge2 = LayerNorm(args.edge_feat_dim)
        
        self.args = args
        self.se3 = SE3Transformer(args.num_layers,  
                            args.num_channels,
                            num_degrees=args.num_degrees,
                            n_heads=args.head,
                            div=args.div,
                            si_m = args.si_m,
                            si_e = args.si_e,
                            l0_in_features = args.l0_in_feat,
                            l0_out_features = args.l0_out_feat,
                            l1_in_features = args.l1_in_feat,
                            l1_out_features = args.l1_out_feat,
                            num_edge_features= args.edge_feat_dim)
        self.relu = nn.PReLU(num_parameters=1, init=0.25)
        self.pred_lddt = nn.Sequential(nn.Linear(args.l0_out_feat, 1), nn.Sigmoid())

    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, xyz, state, msa, pair, seq1hot, idx, CA_atom_index, top_k=64):
        '''
        Input:
            - xyz(tensor):backbone atom position
            - state(tensor):state information
            - msa(tensor):msa information
            - pair(tensor):pair information
            - seq1hot(tensor):sequence one hot vector
            - idx(tensor):residue index
            - CA_atom_index(tensor):CA atom index in sequence
            - top_k(tensor):top k for residue's nearest neighbor
        Output:
            - xyz_new(tensor):refined all atom position
            - lddt(tensor):protein LDDT
        '''
        # process node & pair features
        B, L = msa.shape[:2]

        node = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        state = self.norm_state(state)
        max_atom_num = 14
        node = torch.cat((node, seq1hot, state), dim=-1)
        node = self.norm_node(self.embed_x(node))
        pair = self.norm_edge1(self.embed_e1(pair))

        # define graph
        G = make_graph_topk(xyz, pair, idx, top_k=top_k)

        l0_feats = node
        node_coor = xyz

        l1_feats = node_coor - node_coor[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3) # (B*L, 14, 3)

        shift = self.se3(G, l0_feats.reshape(B*L, -1, 1), l1_feats)
        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        offset = shift['1'].reshape(B, L, -1, 3) # (B, L, 14, 3)

        CA_new = node_coor[:,:,1] + offset[:,:,1] # (B, L, 1, 3)
        CA_new_set = torch.stack([CA_new for i in range(3)], dim = 2)
        CA_new_set[:,0:1,:] = 0.0

        # update all atom position
        xyz_new = offset + CA_new_set
        xyz_new = xyz_new.reshape(B*L*3, 3)

        lddt = self.pred_lddt(self.norm_state(state)) 


        del node_coor
        del CA_new
        del CA_new_set

        torch.cuda.empty_cache()
        
        return xyz_new, lddt





    

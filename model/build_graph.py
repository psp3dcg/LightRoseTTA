'''
build residue graph and atom graph,
compute initial backbone atom postion
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

from model.transformer import LayerNorm
from model.init_str_generator import make_graph
from model.init_str_generator import get_seqsep 
from model.init_str_generator import UniMPBlock
from model.attention_module import get_bonded_neigh

    
class Res_Network(nn.Module):
    # residue(as node) graph network
    def __init__(self, 
                 r_node_dim_in=64, 
                 r_node_dim_hidden=64,
                 r_edge_dim_in=128, 
                 r_edge_dim_hidden=64, 
                 state_dim=8,
                 r_nheads=4, 
                 r_nblocks=3, 
                 r_dropout=0.5,
                 a_node_dim_in=43, 
                 a_node_dim_hidden=64, 
                 a_dropout=0.5):

        '''
        Input:
            - r_node_dim_in(int): the input dimension of residue graph node feature 
            - r_node_dim_hidden(int): the hidden dimension of residue graph node feature 
            - r_edge_dim_in(int): the input dimension of residue graph edge feature 
            - r_edge_dim_hidden(int): the hidden dimension of residue graph edge feature
            - state_dim(int): the dimenison of state feature 
            - r_nheads(int): the number of attention heads for residue graph network
            - r_nblocks(int): the number of blocks for residue graph network
            - r_dropout(float): dropout ratio for residue graph network
            - a_node_dim_in(int): the input dimension of atom graph node feature
            - a_node_dim_hidden(int): the hidden dimension of atom graph node feature
            - a_dropout(float): dropout ratio for atom graph network
        '''
        super(Res_Network, self).__init__()

        # embedding layers for node and edge features in residue graph
        self.norm_node = LayerNorm(r_node_dim_in)
        self.norm_edge = LayerNorm(r_edge_dim_in)

        self.embed_x = nn.Sequential(nn.Linear(r_node_dim_in+21, r_node_dim_hidden), LayerNorm(r_node_dim_hidden))
        self.embed_e = nn.Sequential(nn.Linear(r_edge_dim_in+2, r_edge_dim_hidden), LayerNorm(r_edge_dim_hidden))
        
        # residue graph network
        blocks = [UniMPBlock(r_node_dim_hidden,r_edge_dim_hidden,r_nheads,r_dropout) for _ in range(r_nblocks)]
        self.transformer = nn.Sequential(*blocks)

        self.final_res_gnn = UniMPBlock(r_node_dim_hidden,r_edge_dim_hidden,r_nheads,r_dropout)

        # atom graph network
        self.nhid = a_node_dim_hidden
        self.num_features = a_node_dim_in
        self.dropout_ratio = a_dropout
    
        self.conv1 = GraphConv(self.num_features, self.nhid)
        self.conv2 = GraphConv(self.nhid, self.nhid)
        self.conv3 = GraphConv(self.nhid, self.nhid)

        self.relu = nn.PReLU(num_parameters=1, init=0.25)

        
        # outputs
        self.get_xyz = nn.Linear(r_node_dim_hidden,9)
        self.norm_state = LayerNorm(r_node_dim_hidden)
        self.get_state = nn.Linear(r_node_dim_hidden, state_dim)

        self.softmax = torch.nn.Softmax(dim=0)
        self.proj_atom = nn.Linear(self.nhid, r_node_dim_hidden)
        self.res_atom_encoder = nn.Linear(r_node_dim_hidden*2, r_node_dim_hidden)
    


    def cos_similarity(self, x, y):
        '''
        Returns the cosine similarity batchwise

        Input:
            - x(tensor):feature x
            - y(tensor):feature y
        Output:
            - cos_sim(tensor):cosine similarity between each element of x and y
        '''


        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_sim = torch.matmul(x,torch.transpose(y,0,1))  # .transpose(1,2)

        return cos_sim

    def forward(self, seq1hot, idx, node, edge, data):
        '''
        Input:
            - seq1hot(tensor):residue sequence onehot vector
            - idx(tensor):residue index
            - node(tensor):residue graph node feature
            - edge(tensor):residue graph edge feature
            - data(torch geometric Data object):atom graph data
        Output:
            - xyz(tensor):backbone atom coordinates
            - state(tensor):sequence state information
        '''

        B, L = node.shape[:2]
        node = self.norm_node(node)
        edge = self.norm_edge(edge)
        
        node = torch.cat((node, seq1hot), dim=-1)
        node = self.embed_x(node)

        seqsep = get_seqsep(idx) 
        neighbor = get_bonded_neigh(idx)
        edge = torch.cat((edge, seqsep, neighbor), dim=-1)
        edge = self.embed_e(edge)

        # res graph
        G = make_graph(node, idx, edge)
        Gout = self.transformer(G)

        # atom graph
        x, edge_index = data.x, data.edge_index
        x1 = self.relu(self.conv1(x, edge_index))
        x2 = self.relu(self.conv2(x1, edge_index))
        x3 = self.relu(self.conv3(x2, edge_index))
        x3 = self.proj_atom((x1+x2+x3))

        # feature fusion
        cos_sim = self.cos_similarity(Gout.x, x3)
        new_cos_sim = self.softmax(cos_sim)
        filter_matrix = torch.bernoulli(new_cos_sim)
        new_cos_sim = filter_matrix * new_cos_sim
        new_x3 = torch.matmul(new_cos_sim, x3)
        Gout.x = self.res_atom_encoder(torch.cat([Gout.x, new_x3], dim=-1))

        # Output
        Gout = self.final_res_gnn(Gout)
        xyz = self.get_xyz(Gout.x)
        state = self.get_state(self.norm_state(Gout.x))

        torch.cuda.empty_cache()
        return xyz.reshape(B, L, 3, 3) , state.reshape(B, L, -1)


class Build_Graph_Network(nn.Module):
    def __init__(self, args):
        '''
        Input:
            - args(object):model argument
        '''
        super(Build_Graph_Network, self).__init__()
        self.residue_graph_network = Res_Network(r_node_dim_in=args.d_msa, r_node_dim_hidden=args.d_hidden,
                                                   r_edge_dim_in=args.d_hidden*2, r_edge_dim_hidden=args.d_hidden,
                                                   state_dim=args.l0_out_feat,
                                                   r_nheads=4, r_nblocks=3, r_dropout=args.p_drop)


        self.proj_edge = nn.Linear(args.edge_d_pair, args.d_hidden*2)
        

    def forward(self, data, node, edge, seq1hot, idx):
        '''
        Input:
            - data(torch geometric Data object):atom graph data
            - node(tensor):residue graph node feature
            - edge(tensor):residue graph node interaction
            - seq1hot(tensor):residue sequence onehot vec
            - idx(tensor):residue index
        Output:
            - bb_xyz(tensor):backbone atom coordinates
            - bb_state(tensor):updated residue state feature
            - node(tensor):residue graph node feature
            - edge(tensor):residue graph node interaction
        '''

        edge = self.proj_edge(edge)

        bb_xyz, bb_state = self.residue_graph_network(seq1hot, idx, node, edge, data)

 


        return bb_xyz, bb_state, node, edge
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import *
from model.resnet import ResidualNetwork
from model.SE3_network import SE3Transformer
from model.init_str_generator import InitStr_Network

# Attention module based on AlphaFold2's idea written by Minkyung Baek
#  - Iterative MSA feature extraction
#    - 1) MSA2Pair: extract pairwise feature from MSA --> added to previous residue-pair features
#                   architecture design inspired by CopulaNet paper
#    - 2) MSA2MSA:  process MSA features using Transformer (or Performer) encoder. (Attention over L first followed by attention over N)
#    - 3) Pair2MSA: Update MSA features using pair feature
#    - 4) Pair2Pair: process pair features using Transformer (or Performer) encoder.

def make_graph(xyz, pair, idx, top_k=64, kmin=9):
    '''
    Make residue graph
    Input:
        - xyz(tensor): current backbone cooordinates (B, L, 3, 3)
        - pair(tensor): pair features from Trunk (B, L, L, E)
        - idx(tensor): residue index from ground truth pdb
        - top_k(int): the threshold of the number of neighbor residues
        - kmin(int): the threshold of distance between sequentially connected adjacent residues
    Output:
        - G: defined graph
    '''

    B, L = xyz.shape[:2]
    device = xyz.device
    
    # distance map from current CA coordinates
    D = torch.cdist(xyz[:,:,1,:], xyz[:,:,1,:]) + torch.eye(L, device=device).unsqueeze(0)*999.9  # (B, L, L)
    # seq sep
    sep = idx[:,None,:] - idx[:,:,None]
    sep = sep.abs() + torch.eye(L, device=device).unsqueeze(0)*999.9
    
    # get top_k neighbors
    D_neigh, E_idx = torch.topk(D, min(top_k, L), largest=False) # shape of E_idx: (B, L, top_k)
    topk_matrix = torch.zeros((B, L, L), device=device)
    topk_matrix.scatter_(2, E_idx, 1.0)

    # put an edge if any of the 3 conditions are met:
    #   1) |i-j| <= kmin (connect sequentially adjacent residues)
    #   2) top_k neighbors
    cond = torch.logical_or(topk_matrix > 0.0, sep < kmin)
    b,i,j = torch.where(cond)
   
    src = b*L+i
    tgt = b*L+j
    G = dgl.graph((src, tgt), num_nodes=B*L).to(device)
    G.edata['d'] = (xyz[b,j,1,:] - xyz[b,i,1,:]).detach() # no gradient through basis function
    G.edata['w'] = pair[b,i,j]

    

    return G 

def get_bonded_neigh(idx):
    '''
    Get bonded neighbor residues

    Input:
        - idx(tensor): residue indices of given sequence (B,L)
    Output:
        - neighbor(tensor): bonded neighbor information with sign (B, L, L, 1)
    '''
    neighbor = idx[:,None,:] - idx[:,:,None]
    neighbor = neighbor.float()
    sign = torch.sign(neighbor) # (B, L, L)
    neighbor = torch.abs(neighbor)
    neighbor[neighbor > 1] = 0.0
    neighbor = sign * neighbor 
    return neighbor.unsqueeze(-1)

def rbf(D):
    '''
    Distance radial basis function

    Input:
        - D(tensor): distance matrix
    Output:
        - RBF(tensor): radial basis
    '''

    D_min, D_max, D_count = 0., 20., 36
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu[None,:]
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

class CoevolExtractor(nn.Module):
    def __init__(self, n_feat_proj, n_feat_out, p_drop=0.1):
        '''
        Coevolution information extractor

        Input:
            - n_feat_proj(int): the dimension of project feature 
            - n_feat_out(int): the dimension of output feature 
            - p_drop(float): dropout ratio
        '''
        super(CoevolExtractor, self).__init__()
        self.norm_2d = LayerNorm(n_feat_proj*n_feat_proj)
        # project down to output dimension (pair feature dimension)
        self.proj_2 = nn.Linear(n_feat_proj**2, n_feat_out)
    def forward(self, x_down, x_down_w):
        '''
        Input:
            - x_down(tensor): the projected msa feature
            - x_down_w(tensor): the projected msa feature with attention weight
        Output:
            - pair(tensor): pair embeddings 
        '''
        B, N, L = x_down.shape[:3]
        
        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down_w) # outer-product & average pool
        pair = pair.reshape(B, L, L, -1)
        pair = self.norm_2d(pair)
        pair = self.proj_2(pair) # (B, L, L, n_feat_out) # project down to pair dimension
        return pair

class MSA2Pair(nn.Module):
    def __init__(self, n_feat=64, n_feat_out=128, n_feat_proj=32,
                 n_resblock=1, p_drop=0.1, n_att_head=8):
        '''
        Update pair feature through msa

        Input:
            - n_feat(int): the dimension of input feature
            - n_feat_out(int): the dimension of output feature
            - n_feat_proj(int): the dimension of feature
            - n_resblock(int): the number of resnet blocks
            - p_drop(float): dropout ratio 
            - n_att_head(int): the number of attention heads
        '''
        super(MSA2Pair, self).__init__()
        # project down embedding dimension (n_feat --> n_feat_proj)
        self.norm_1 = LayerNorm(n_feat)
        self.proj_1 = nn.Linear(n_feat, n_feat_proj)
        
        self.encoder = SequenceWeight(n_feat_proj, 1, dropout=p_drop)
        self.coevol = CoevolExtractor(n_feat_proj, n_feat_out)

        # ResNet to update pair features 
        self.norm_down = LayerNorm(n_feat_proj)
        self.norm_orig = LayerNorm(n_feat_out)
        self.norm_new  = LayerNorm(n_feat_out)
        self.update = ResidualNetwork(n_resblock, n_feat_out*2+n_feat_proj*4+n_att_head, n_feat_out, n_feat_out, p_drop=p_drop)

    def forward(self, msa, pair_orig, att):
        '''
        Input:
            - msa(tensor): MSA embeddings
            - pair_orig(tensor): original pair embeddings 
            - att(tensor): attention weight
        Output:
            - pair(tensor): updated pair info 
        '''
        B, N, L, _ = msa.shape
        # project down to reduce memory
        msa = self.norm_1(msa)
        x_down = self.proj_1(msa) # (B, N, L, n_feat_proj)
        
        # get sequence weight
        x_down = self.norm_down(x_down)
        w_seq = self.encoder(x_down).reshape(B, L, 1, N).permute(0,3,1,2)
        feat_1d = w_seq*x_down
        pair = self.coevol(x_down, feat_1d)

        # average pooling over N of given MSA info
        feat_1d = feat_1d.sum(1)
        
        # query sequence info
        query = x_down[:,0] # (B,L,K)
        feat_1d = torch.cat((feat_1d, query), dim=-1) # additional 1D features
        # tile 1D features
        left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)
        right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)
        # update original pair features through convolutions after concat
        pair_orig = self.norm_orig(pair_orig)
        pair = self.norm_new(pair)
        pair = torch.cat((pair_orig, pair, left, right, att), -1)
        pair = pair.permute(0,3,1,2).contiguous() # prep for convolution layer
        pair = self.update(pair)
        pair = pair.permute(0,2,3,1).contiguous() # (B, L, L, C)

        return pair

class MSA2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8, n_feat=256, r_ff=4, p_drop=0.1,
                 performer_N_opts=None, performer_L_opts=None):
        '''
        MSA self-update process
        Input:
            - n_layer(int): the number of layers
            - n_att_head(int): the number of attention heads
            - n_feat(int): the dimension of msa feature
            - r_ff(int): the times of dimensional expansion for feedforward network
            - p_drop(float): dropout ratio
            - performer_N_opts(dict): the options of performer for attention along N
            - performer_L_opts(dict): the options of performer for attention along L
        Output:
            - pair(tensor): updated pair info 
        '''
        super(MSA2MSA, self).__init__()
        # attention along L
        enc_layer_1 = EncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   use_tied=True)
                                   #performer_opts=performer_L_opts)
        self.encoder_1 = Encoder(enc_layer_1, n_layer)
        
        # attention along N
        enc_layer_2 = EncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                   heads=n_att_head, p_drop=p_drop,
                                   performer_opts=performer_N_opts)
        self.encoder_2 = Encoder(enc_layer_2, n_layer)

    def forward(self, x):
        '''
        Input:
            - x(tensor): MSA embeddings
        Output:
            - x(tensor): updated MSA embeddings
            - att(tensor): attention weights
        '''
        B, N, L, _ = x.shape
        # attention along L
        x, att = self.encoder_1(x, return_att=True)
        # attention along N
        x = x.permute(0,2,1,3).contiguous()
        x = self.encoder_2(x)
        x = x.permute(0,2,1,3).contiguous()
        return x, att

class Pair2MSA(nn.Module):
    def __init__(self, n_layer=1, n_att_head=4, n_feat_in=128, n_feat_out=256, r_ff=4, p_drop=0.1):
        '''
        Update MSA through pair feature

        Input:
            - n_layer(int): the number of layers
            - n_att_head(int): the number of attention heads
            - n_feat_in(int): the dimension of input feature
            - n_feat_out(int): the dimension of output feature
            - r_ff(int): the times of dimensional expansion for feedforward network
            - p_drop(float): dropout ratio
        '''
        super(Pair2MSA, self).__init__()
        enc_layer = DirectEncoderLayer(heads=n_att_head, \
                                       d_in=n_feat_in, d_out=n_feat_out,\
                                       d_ff=n_feat_out*r_ff,\
                                       p_drop=p_drop)
        self.encoder = CrossEncoder(enc_layer, n_layer)

    def forward(self, pair, msa):
        '''
        Input:
            - pair(tensor): pair embeddings
            - msa(tensor): MSA embeddings
        Output:
            - out(tensor): updated MSA embeddings    
        '''
        out = self.encoder(pair, msa) # (B, N, L, K)
        return out

class Pair2Pair(nn.Module):
    def __init__(self, n_layer=1, n_att_head=8, n_feat=128, r_ff=4, p_drop=0.1,
                 performer_L_opts=None):
        '''
        Pair self update

        Input:
            - n_layer(int): the number of layers
            - n_att_head(int): the number of attention heads
            - n_feat(int): the dimension of features
            - r_ff(int): the times of dimensional expansion for feedforward network
            - p_drop(float): dropout ratio
            - performer_L_opts(dict): the options of performer for attention along L
        '''
        super(Pair2Pair, self).__init__()
        enc_layer = AxialEncoderLayer(d_model=n_feat, d_ff=n_feat*r_ff,
                                      heads=n_att_head, p_drop=p_drop,
                                      performer_opts=performer_L_opts)
        self.encoder = Encoder(enc_layer, n_layer)
    
    def forward(self, x):
        '''
        Input:
            - x(tensor): pair embeddings
        Output:
            - x(tensor): updated pair embeddings
        '''
        return self.encoder(x)

class Str2Str(nn.Module):
    def __init__(self, d_msa=64, d_pair=128, 
            SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}, p_drop=0.1):
        '''
        Structure information self update

        Input:
            - d_msa(int): dimension of MSA embedding
            - d_pair(int): dimension of pair embedding
            - SE3_param(dict): parameters for SE(3)-transformer
            - p_drop(float): dropout ratio
        '''
        super(Str2Str, self).__init__()

        # initial node & pair feature process
        self.norm_msa = LayerNorm(d_msa)
        self.norm_pair = LayerNorm(d_pair)
        self.encoder_seq = SequenceWeight(d_msa, 1, dropout=p_drop)
    
        self.embed_x = nn.Linear(d_msa+21, SE3_param['l0_in_features'])
        self.embed_e = nn.Linear(d_pair, SE3_param['num_edge_features'])
        
        self.norm_node = LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge = LayerNorm(SE3_param['num_edge_features'])
        
        self.se3 = SE3Transformer(**SE3_param)
    
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, msa, pair, xyz, seq1hot, idx, top_k=64):
        '''
        Input:
            - msa(tensor): MSA embedding
            - pair(tensor): pair embedding
            - xyz(tensor): initial backbone atom position 
            - seq1hot(tensor): sequence one-hot vector
            - idx(tensor): residue index
            - top_k(int): the threshold of the number of neighbor residues
        Output:
            - xyz_new(tensor): the updated backbone atom position
            - state(tensor): the updated residue embedding
        '''

        # process msa & pair features
        B, N, L = msa.shape[:3]
        msa = self.norm_msa(msa)
        pair = self.norm_pair(pair)
        
        w_seq = self.encoder_seq(msa).reshape(B, L, 1, N).permute(0,3,1,2)
        msa = w_seq*msa
        msa = msa.sum(dim=1)
        msa = torch.cat((msa, seq1hot), dim=-1)
        msa = self.norm_node(self.embed_x(msa))
        pair = self.norm_edge(self.embed_e(pair))
        
        # define graph
        G = make_graph(xyz, pair, idx, top_k=top_k)
        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2) # l1 features = displacement vector to CA
        l1_feats = l1_feats.reshape(B*L, -1, 3) # 
        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, msa.reshape(B*L, -1, 1), l1_feats)
        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        offset = shift['1'].reshape(B, L, -1, 3) # (B, L, 3, 3)
        CA_new = xyz[:,:,1] + offset[:,:,1]
        N_new = CA_new + offset[:,:,0]
        C_new = CA_new + offset[:,:,2]
        xyz_new = torch.stack([N_new, CA_new, C_new], dim=2)

        return xyz_new, state

class Str2MSA(nn.Module):
    def __init__(self, d_msa=64, d_state=32, inner_dim=32, r_ff=4,
                 distbin=[8.0, 12.0, 16.0, 20.0], p_drop=0.1):
        '''
        Update MSA through structure information

        Input:
            - d_msa(int): the dimension of MSA embedding
            - d_state(int): the dimension of residue embedding  
            - inner_dim(int): the dimension of inner embedding  
            - r_ff(int): the times of dimensional expansion for feedforward network
            - distbin(list): distance bins
            - p_drop(float): dropout ratio 
        '''
        super(Str2MSA, self).__init__()
        self.distbin = distbin
        n_att_head = len(distbin)

        self.norm_state = LayerNorm(d_state)
        self.norm1 = LayerNorm(d_msa)
        self.attn = MaskedDirectMultiheadAttention(d_state, d_msa, n_att_head, d_k=inner_dim, dropout=p_drop) 
        self.dropout1 = nn.Dropout(p_drop,)

        self.norm2 = LayerNorm(d_msa)
        self.ff = FeedForwardLayer(d_msa, d_msa*r_ff, p_drop=p_drop)
        self.dropout2 = nn.Dropout(p_drop,)
        
    def forward(self, msa, xyz, state):
        '''
        Input:
            - msa(tensor): MSA embedding
            - xyz(tensor): initial backbone atom position 
            - state(tensor): residue embedding
        Output:
            - msa(tensor): the updated MSA embedding
        '''
        dist = torch.cdist(xyz[:,:,1], xyz[:,:,1]) # (B, L, L)

        mask_s = list()
        for distbin in self.distbin:
            mask_s.append(1.0 - torch.sigmoid(dist-distbin))
        mask_s = torch.stack(mask_s, dim=1) # (B, h, L, L)
        
        state = self.norm_state(state)
        msa2 = self.norm1(msa)
        msa2 = self.attn(state, state, msa2, mask_s)
        msa = msa + self.dropout1(msa2)

        msa2 = self.norm2(msa)
        msa2 = self.ff(msa2)
        msa = msa + self.dropout2(msa2)

        return msa

class IterBlock(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_pair=128, n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None):
        '''
        The block of coevolution module

        Input:
            - n_layer(int): the number of layers
            - d_msa(int): the dimension of MSA embeddings
            - d_pair(int): the dimension of pair embeddings
            - n_head_msa(int): the number of MSA attention heads
            - n_head_pair(int): the number of pair attention heads
            - r_ff(int): the times of dimensional expansion for feedforward network
            - n_resblock(int): the number of resnet block  
            - p_drop(float): dropout ratio 
            - performer_L_opts(dict): the options of performer for attention along L
            - performer_N_opts(dict): the options of performer for attention along N
        '''
        super(IterBlock, self).__init__()
        
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=32,
                                 n_resblock=n_resblock, p_drop=p_drop, n_att_head=n_head_msa)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=4, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)

    def forward(self, msa, pair):
        '''
        Input:
            - msa(tensor): initial MSA embedding
            - pair(tensor): initial residue pair embeddings  
        Output:
            - msa(tensor): the updated MSA embedding
            - pair(tensor): the updated residue pair embedding
        '''
            
        # 1. process MSA features
        msa, att = self.msa2msa(msa) 
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair, att) 

        # 3. process pair features
        pair = self.pair2pair(pair) 
        
        # 4. update MSA features using updated pair features
        msa = self.pair2msa(pair, msa)
        

        return msa, pair

class IterBlock_w_Str(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_pair=128, n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        '''
        The block of coevolution module with structure information

        Input:
            - n_layer(int): the number of layers
            - d_msa(int): the dimension of MSA embedding
            - d_pair(int): the dimension of pair embedding
            - n_head_msa(int): the number of MSA attention heads
            - n_head_pair(int): the number of pair attention heads 
            - r_ff(int): the times of dimensional expansion for feedforward network 
            - n_resblock(int): the number of resnet block  
            - p_drop(float): dropout ratio
            - performer_L_opts(dict): the options of performer for attention along L
            - performer_N_opts(dict): the options of performer for attention along N
            - SE3_param(dict): SE(3)-tranformer parameters

        '''
        super(IterBlock_w_Str, self).__init__()
        
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=32,
                                 n_resblock=n_resblock, p_drop=p_drop, n_att_head=n_head_msa)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=4, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair, SE3_param=SE3_param, p_drop=p_drop)
        self.str2msa = Str2MSA(d_msa=d_msa, d_state=SE3_param['l0_out_features'],
                               r_ff=r_ff, p_drop=p_drop)

    def forward(self, msa, pair, xyz, seq1hot, idx, top_k=64):
        '''
        Input:
            - msa(tensor): MSA embedding
            - pair(tensor): pair embedding
            - xyz(tensor): initial atom position
            - seq1hot(tensor): sequence one-hot vector
            - idx(tensor): residue index
            - top_k(int): the threshold of the number of neighbor residues
        Output:
            - msa(tensor): the updated MSA embedding
            - pair(tensor): the updated pair embedding
            - xyz(tensor): the updated 
        '''  
            
        # 1. process MSA features
        msa, att = self.msa2msa(msa)
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair, att)

        # 3. process pair features
        pair = self.pair2pair(pair)
        
        # 4. update MSA features using updated pair features
        msa = self.pair2msa(pair, msa)
        

        xyz, state = self.str2str(msa.float(), pair.float(), xyz.float(), seq1hot, idx, top_k=top_k)
        msa = self.str2msa(msa, xyz, state)
            
        return msa, pair, xyz

class FinalBlock(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_pair=128, n_head_msa=4, n_head_pair=8, r_ff=4,
                 n_resblock=1, p_drop=0.1, performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        '''
        The final block of coevolution module

        Input:
            - n_layer(int): the number of layers
            - d_msa(int): the dimension of MSA embedding
            - d_pair(int): the dimension of pair embedding
            - n_head_msa(int): the number of MSA attention heads
            - n_head_pair(int): the number of pair attention heads 
            - r_ff(int): the times of dimensional expansion for feedforward network 
            - n_resblock(int): the number of resnet block  
            - p_drop(float): dropout ratio
            - performer_L_opts(dict): the options of performer for attention along L
            - performer_N_opts(dict): the options of performer for attention along N
            - SE3_param(dict): SE(3)-tranformer parameters
        ''' 
        super(FinalBlock, self).__init__()
        
        self.msa2msa = MSA2MSA(n_layer=n_layer, n_att_head=n_head_msa, n_feat=d_msa,
                               r_ff=r_ff, p_drop=p_drop,
                               performer_N_opts=performer_N_opts,
                               performer_L_opts=performer_L_opts)
        self.msa2pair = MSA2Pair(n_feat=d_msa, n_feat_out=d_pair, n_feat_proj=32,
                                 n_resblock=n_resblock, p_drop=p_drop, n_att_head=n_head_msa)
        self.pair2pair = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                   n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                   performer_L_opts=performer_L_opts)
        self.pair2msa = Pair2MSA(n_layer=n_layer, n_att_head=4, 
                                 n_feat_in=d_pair, n_feat_out=d_msa, r_ff=r_ff, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair, SE3_param=SE3_param, p_drop=p_drop)
        self.norm_state = LayerNorm(SE3_param['l0_out_features'])
        self.pred_lddt = nn.Linear(SE3_param['l0_out_features'], 1)

    def forward(self, msa, pair, xyz, seq1hot, idx):
        '''
        Input:
            - msa(tensor): MSA embedding
            - pair(tensor): pair embedding
            - xyz(tensor): initial atom position
            - seq1hot(tensor): sequence one-hot vector
            - idx(tensor): residue index
        Output:
            - msa(tensor): the updated MSA embedding
            - pair(tensor): the updated pair embedding
            - xyz(tensor): the updated 
            - lddt(tensor): the lddt value of build-in lddt predictor 
        '''   
            
        # 1. process MSA features
        msa, att = self.msa2msa(msa)
        
        # 2. update pair features using given MSA
        pair = self.msa2pair(msa, pair, att)

        # 3. process pair features
        pair = self.pair2pair(pair)
       
        msa = self.pair2msa(pair, msa)

        xyz, state = self.str2str(msa.float(), pair.float(), xyz.float(), seq1hot, idx, top_k=32)
        
        lddt = self.pred_lddt(self.norm_state(state))
        return msa, pair, xyz, lddt.squeeze(-1)

class IterativeFeatureExtractor(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_layer=4, d_msa=256, d_pair=128, d_hidden=64,
                 n_head_msa=8, n_head_pair=8, r_ff=4, 
                 n_resblock=1, p_drop=0.1,
                 performer_L_opts=None, performer_N_opts=None,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        '''
        The coevolution module

        Input:
            - n_module=4, 
            - n_module_str=4, 
            - n_layer(int): the number of layers
            - d_msa(int): the dimension of MSA embedding
            - d_pair(int): the dimension of pair embedding
            - d_hidden(int): the dimension o hidden embedding
            - n_head_msa(int): the number of MSA attention heads
            - n_head_pair(int): the number of pair attention heads 
            - r_ff(int): the times of dimensional expansion for feedforward network 
            - n_resblock(int): the number of resnet block  
            - p_drop(float): dropout ratio
            - performer_L_opts(dict): the options of performer for attention along L
            - performer_N_opts(dict): the options of performer for attention along N
            - SE3_param(dict): SE(3)-tranformer parameters
        ''' 
        super(IterativeFeatureExtractor, self).__init__()
        self.n_module = n_module
        self.n_module_str = n_module_str
        #
        self.initial = Pair2Pair(n_layer=n_layer, n_att_head=n_head_pair,
                                 n_feat=d_pair, r_ff=r_ff, p_drop=p_drop,
                                 performer_L_opts=performer_L_opts)

        if self.n_module > 0:
            self.iter_block_1 = IterBlock(n_layer=n_layer, 
                                                      d_msa=d_msa, d_pair=d_pair,
                                                      n_head_msa=n_head_msa,
                                                      n_head_pair=n_head_pair,
                                                      r_ff=r_ff,
                                                      n_resblock=n_resblock,
                                                      p_drop=p_drop,
                                                      performer_N_opts=performer_N_opts,
                                                      performer_L_opts=performer_L_opts
                                                      )
        
        self.init_str = InitStr_Network(node_dim_in=d_msa, node_dim_hidden=d_hidden,
                                        edge_dim_in=d_pair, edge_dim_hidden=d_hidden,
                                        nheads=4, nblocks=3, dropout=p_drop)

        if self.n_module_str > 0:
            self.iter_block_2 = IterBlock_w_Str(n_layer=n_layer, 
                                                      d_msa=d_msa, d_pair=d_pair,
                                                      n_head_msa=n_head_msa,
                                                      n_head_pair=n_head_pair,
                                                      r_ff=r_ff,
                                                      n_resblock=n_resblock,
                                                      p_drop=p_drop,
                                                      performer_N_opts=performer_N_opts,
                                                      performer_L_opts=performer_L_opts,
                                                      SE3_param=SE3_param
                                                      )
        
        self.final = FinalBlock(n_layer=n_layer, d_msa=d_msa, d_pair=d_pair,
                               n_head_msa=n_head_msa, n_head_pair=n_head_pair, r_ff=r_ff,
                               n_resblock=n_resblock, p_drop=p_drop,
                               performer_L_opts=performer_L_opts, performer_N_opts=performer_N_opts,
                               SE3_param=SE3_param)
    
    def forward(self, msa, pair, seq1hot, idx):
        '''
        Input:
            - msa(tensor): MSA embedding
            - pair(tensor): pair embedding
            - seq1hot(tensor): sequence one-hot vector
            - idx(tensor): residue index
        Output:
            - msa(tensor): the updated MSA embedding
            - pair(tensor): the updated pair embedding
            - xyz(tensor): the updated atom coordinates
            - lddt(tensor): the lddt value of build-in lddt predictor 
        ''' 

        pair = self.initial(pair)
        if self.n_module > 0:
            for i_m in range(self.n_module):
                # extract features from MSA & update original pair features
                msa, pair = self.iter_block_1(msa, pair)
        xyz = self.init_str(seq1hot, idx, msa, pair)
        top_ks = [128, 128, 64, 64]
        if self.n_module_str > 0:
            for i_m in range(self.n_module_str):
                msa, pair, xyz = self.iter_block_2(msa, pair, xyz, seq1hot, idx, top_k=top_ks[i_m])
        msa, pair, _, _ = self.final(msa, pair, xyz, seq1hot, idx)

        return msa[:,0], pair

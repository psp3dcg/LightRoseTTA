'''
process with protein msa and template information
'''
import torch
import torch.nn as nn

from model.embeddings import MSA_emb
from model.embeddings import Templ_emb
from model.embeddings import Pair_emb_wo_templ
from model.embeddings import Pair_emb_w_templ
from utils.kinematics import xyz_to_t2d
from model.distance_predictor import DistanceNetwork
from model.attention_module import IterativeFeatureExtractor









class msa_info_parser(nn.Module):
    def __init__(self, n_module=4, n_module_str=4, n_layer=4,\
                 d_msa=64, d_pair=128, d_templ=64,\
                 n_head_msa=4, n_head_pair=8, n_head_templ=4,
                 d_hidden=64, r_ff=4, n_resblock=1, p_drop=0.0, 
                 performer_L_opts=None, performer_N_opts=None, 
                 use_templ=False, device_type='cuda', 
                 SE3_param= {'l0_in_features':16, 
                            'l0_out_features':16, 
                            'num_edge_features':16}):

        '''
        Process protein msa information

        Input:
            - n_module(int):number of modules
            - n_module_str(int):number of str modules
            - n_layer(int):number of layers in module
            - d_msa(int):msa dimension
            - d_pair(int):msa dimension
            - d_templ(int):msa dimension
            - n_head_msa(int):number of msa attention head
            - n_head_pair(int):number of pair attention head
            - n_head_templ(int):number of template attention head
            - d_hidden(int):hidden feature dimension
            - r_ff():number of ffindex
            - n_resblock(int):number of resnet block
            - p_drop(float):dropout ratio
            - performer_L_opts(int):
            - performer_N_opts(int):
            - use_templ(bool):flag of using template or not
            - device_type(str):cpu or cuda
            - SE3_param(dict):SE(3) network parameter
        '''
        super(msa_info_parser, self).__init__()
        self.use_templ = use_templ
        self.device = device_type
        #
        self.msa_emb = MSA_emb(d_model=d_msa, p_drop=p_drop, max_len=5000)
        if use_templ:
            self.templ_emb = Templ_emb(d_templ=d_templ, n_att_head=n_head_templ, r_ff=r_ff, 
                                       performer_opts=performer_L_opts, p_drop=0.0)
            self.pair_emb = Pair_emb_w_templ(d_model=d_pair, d_templ=d_templ, p_drop=p_drop)
        else:
            self.pair_emb = Pair_emb_wo_templ(d_model=d_pair, p_drop=p_drop)
        #
        
        self.feat_extractor = IterativeFeatureExtractor(n_module=n_module,\
                                                        n_module_str=n_module_str,\
                                                        n_layer=n_layer,\
                                                        d_msa=d_msa, d_pair=d_pair, d_hidden=d_hidden,\
                                                        n_head_msa=n_head_msa, \
                                                        n_head_pair=n_head_pair,\
                                                        r_ff=r_ff, \
                                                        n_resblock=n_resblock,
                                                        p_drop=p_drop,
                                                        performer_N_opts=performer_N_opts,
                                                        performer_L_opts=performer_L_opts,
                                                        SE3_param=SE3_param)
        self.c6d_predictor = DistanceNetwork(d_pair, p_drop=p_drop)
        #

    def forward(self, msa, seq, idx, t1d=None, t2d=None, prob_s=None):
        '''
        Input:
            - msa(tensor):msa information
            - seq(tensor):sequence information
            - idx(tensor):residue index
            - t1d(tensor):template 1d information
            - t2d(tensor):template 2d informaiton
            - prob_s(tensor):residue pair distance and angle classes
        Output:
            - msa(tensor):processed msa information
            - prob_s(tensor):residue pair distance and angle classes
            - logits(tensor):residue pair distance and angle features
            - seq1hot(tensor):residue sequence one hot vector
            - idx(tensor):residue index
        '''
        seq1hot = torch.nn.functional.one_hot(seq, num_classes=21).float()
        idx = idx.to(self.device)
        
        B, N, L = msa.shape
        # Get embeddings
        msa = self.msa_emb(msa, idx)

        if self.use_templ:
            tmpl = self.templ_emb(t1d, t2d, idx)
            pair = self.pair_emb(seq, idx, tmpl)
        else:
            pair = self.pair_emb(seq, idx)
        # Extract features
    
        msa, pair = self.feat_extractor(msa, pair, seq1hot, idx)


        # Predict 6D coords
        logits = self.c6d_predictor(pair)
        prob_s = list()
        for l in logits:
            prob_s.append(nn.Softmax(dim=1)(l)) # (B, C, L, L)
        prob_s = torch.cat(prob_s, dim=1).permute(0,2,3,1)

        torch.cuda.empty_cache()

        return msa, prob_s, logits, seq1hot, idx

    


class msa_parser(nn.Module):
    def __init__(self, args):
        '''
        Feature preprocess before parsing 

        Input:
            - args(argparse object): msa parsing parameters
        '''
        super(msa_parser, self).__init__()
        self.SE3_param = {'l0_in_features':4, 
                          'l0_out_features':4, 
                          'num_edge_features':4}
        self.model = msa_info_parser(n_module=args.n_module, n_module_str=args.n_module_str, n_layer=args.n_layer,\
                                            d_msa=args.d_msa, d_pair=args.d_pair, d_templ=args.d_templ,\
                                            n_head_msa=args.n_head_msa, n_head_pair=args.n_head_pair, n_head_templ=args.n_head_templ,
                                            d_hidden=args.d_hidden, r_ff=args.r_ff, n_resblock=args.n_resblock, p_drop=args.p_drop, 
                                            performer_L_opts=args.performer_L_opts, performer_N_opts=args.performer_N_opts, 
                                            use_templ=args.use_templ, device_type=args.device, SE3_param=self.SE3_param)

    def forward(self, msa, xyz_t, t1d, t0d):
        '''
        Input:
            - msa(tensor):msa information
            - xyz_t(tensor):template backbone atom 3d position
            - t1d(tensor):template 1d information
            - t0d(tensor):template 0d information
        Output:
            - msa(tensor):processed msa information
            - prob_s(tensor):residue pair distance and angle classes
            - logits(tensor):residue pair distance and angle features
            - seq1hot(tensor):residue sequence one hot vector
            - idx(tensor):residue index
        '''
        
        if len(list(msa.size())) < 3:
            msa = msa.unsqueeze(0)
        if len(list(xyz_t.size())) < 5:
            xyz_t = xyz_t.unsqueeze(0)
        if len(list(t1d.size())) < 4:
            t1d = t1d.unsqueeze(0)
        if len(list(t0d.size())) < 3:
            t0d = t0d.unsqueeze(0)
        B, N, L = msa.shape
        xyz_t = xyz_t.float()
        t1d = t1d.float()
        t0d = t0d.float()
        t2d = xyz_to_t2d(xyz_t, t0d)

        idx_pdb = torch.arange(L).long().expand((B, L))
        msa = msa[:,:100]
        seq = msa[:,0]
        idx_pdb = idx_pdb
        t1d = t1d[:,:10]
        t2d = t2d[:,:10]

        msa, prob_s, logits, seq1hot, idx = self.model(msa, seq, idx_pdb, t1d=t1d, t2d=t2d)

        logits = list(logits)
        for i, v in enumerate(logits):
            logits[i] = v.permute(0,2,3,1)

        torch.cuda.empty_cache()

        return msa, prob_s, logits, seq1hot, idx



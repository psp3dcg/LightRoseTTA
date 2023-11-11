'''
LightRoseTTA Protein Structure Prediction Test File
'''
import os
import torch
import warnings
import numpy as np
from Bio import SeqIO
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

import confs.util as util
from model.tr_fold import TR_Fold
from data_pipeline import Protein_Dataset
from model.LightRoseTTA import Predict_Network
from utils.aa_info_util import init_res_name_map
from utils.aa_info_util import init_atom_num_map
from utils.aa_info_util import init_atom_name_map
from utils.generate_pdb import generate_pdb, write_pdb
warnings.filterwarnings("ignore")


def data_builder(args, data_path, test_mode):
    '''build data loader

    Input:
        - args(object):arguments
        - data_path(str):test data path
        - test_mode(str):test or train mode
    Output:
        - test_loader(torch.dataloader):test dataloader
    '''
    dataset = Protein_Dataset(data_path, test_mode)
    args.num_classes = dataset.data_num_classes
    args.num_features = dataset.data_num_features
    test_loader = DataLoader(dataset,batch_size=1,shuffle=False)

    return test_loader
   

def generate_residue_name(fasta_str, name_map, atom_num_map):
    '''get amino acid name

    Input:
        - fasta_str(list):fasta string list
        - name_map(dict):amino acid short name dict
        - atom_num_map(dict):amino acid atom number dict
    Output:
        - residue_name_list(list):amino acid name list
        - residue_index_list(list):amino acid index in sequence
    '''
    residue_name_list = []
    residue_index_list = []
    res_bb_atom_num = 3
    for i, res in enumerate(list(fasta_str)):
        for j in range(atom_num_map[res]):
            if j >= res_bb_atom_num:
                break
            residue_name_list.append(name_map[res])
            residue_index_list.append(i+1)
    return residue_name_list, residue_index_list

def generate_fasta_and_atom_name(data_path, protein_name):
    '''get fasta string and atom name

    Input:
        - data_path(str):fasta file path
        - protein_name(str):protein's name
    Output:
        - fasta_str(str):fasta string
        - atom_list(list):atoms in one protein
    '''
    def read_fasta(fasta_file_path):
        '''read fasta

        Input:
            - points_position(tensor):atom position
        Output:
            - ca_dist(tensor):distance of CA atom pairs
        '''
        result = ""
        for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
            for s in seq_record.seq:
                if s == 'X':
                    continue
                result += s
        return result
    atom_name_map = init_atom_name_map() 

    # fasta string
    fasta_str = read_fasta(os.path.join(data_path, 'raw', protein_name, protein_name+'.fasta'))

    atom_list = []
    res_bb_atom_num = 3
    for residue in fasta_str:
        res_atom_list = atom_name_map[residue]
        for i, atom_name in enumerate(res_atom_list):
            if i >= res_bb_atom_num:
                break
            atom_list.append(atom_name)

    return fasta_str, atom_list


def trans_and_rotate_coor(coord, logit_s):
    '''translate and rotate the atom coordinates through 
        the inter-residue distances (Cb) and orientations (omega, theta, phi)

    Input:
        - coord(tensor): predicted atom coordinates
        - logit_s(tensor): predicted inter-residue distances and orientations
    Output:
        - new_xyz(tensor): coordinates after translation and rotation
    '''
    active_fn = torch.nn.Softmax(dim=-1)
    L = coord.shape[0]
    
    prob_s = list()
    for logit in logit_s:
        prob = active_fn(logit.float()) # distogram
        prob_s.append(prob.squeeze())

    prob_trF = list()
    for prob in prob_s:
        prob = prob.permute(2,0,1)
        prob += 1e-8
        prob = prob / torch.sum(prob, dim=0)[None]
        prob_trF.append(prob)

    TRF = TR_Fold(prob_trF)

    new_xyz = TRF.fold(coord[:, 1, :]) 
    return new_xyz.reshape(-1, 3)


def compute_sub_out_coor(data, model, res_begin_idx, res_end_idx, 
                        N_atom_index, CA_atom_index, src_idx_list, 
                        dst_idx_list, atom_len, edge_len, 
                        end_res_flag, idx):
    '''compute split sequence and output atom positions

    Input:
        - data:test data
        - model(torch.nn.Module):trained model
        - res_begin_idx(int):residue begin index
        - res_end_idx(int):residue end index
        - N_atom_index(int):Nitrogen atom index
        - CA_atom_index(int):Carbon alpha atom index
        - src_idx_list(list):source index list
        - dst_idx_list(list):destination index list
        - atom_len(int):atom length
        - edge_len(int):edge length
        - end_res_flag(int):end residue flag
        - idx(int):atom index
    Output:
        - out_coor(tensor):output atom position
    '''
    sub_msa = data.msa[:,res_begin_idx:res_end_idx]
    sub_xyz_t = data.xyz_t[:,res_begin_idx:res_end_idx,:,:]
    sub_t1d = data.t1d[:,res_begin_idx:res_end_idx,:]
    atom_begin_index = N_atom_index[res_begin_idx]
    if end_res_flag:
        atom_end_index = atom_len
    else:
        atom_end_index = N_atom_index[res_end_idx]

    sub_CA_atom_index_list = CA_atom_index[res_begin_idx:res_end_idx]
    sub_CA_atom_index_list = torch.subtract(sub_CA_atom_index_list, (sub_CA_atom_index_list[0] - 1))

    sub_x = data.x[atom_begin_index:atom_end_index]

    edge_begin_index = src_idx_list.index(atom_begin_index)
    if end_res_flag:
        edge_end_index = edge_len
    else:
        edge_end_index = src_idx_list.index(atom_end_index)
    if idx > 0:
        edge_begin_index += 1
    

    if not end_res_flag: 
        dst_edge_end_idx = dst_idx_list.index(atom_end_index)
        sub_edge_index_front = data.edge_index[:,edge_begin_index:dst_edge_end_idx]
        sub_edge_index_rear  = data.edge_index[:,dst_edge_end_idx+1:edge_end_index]
        sub_edge_index = torch.cat([sub_edge_index_front, sub_edge_index_rear], dim=1)
    else:
        sub_edge_index = data.edge_index[:,edge_begin_index:edge_end_index]

    sub_edge_index = torch.subtract(sub_edge_index, atom_begin_index)

    sub_data = Data(x=sub_x, edge_index=sub_edge_index, CA_atom_index=sub_CA_atom_index_list,
                    msa = sub_msa, xyz_t = sub_xyz_t, t0d = data.t0d, t1d = sub_t1d, protein_name = data.protein_name[0])



    with torch.no_grad():
        out_coor, logit_s = model(sub_data)
    out_coor = trans_and_rotate_coor(out_coor.reshape(-1, 3, 3), logit_s)

    del sub_edge_index
    del sub_x
    del sub_msa
    del sub_xyz_t
    del sub_t1d
    del sub_data
    torch.cuda.empty_cache()

    return out_coor

def process_overlong_seq(data, model, standard_len = 300, rest_len = 70):
    '''split overlong sequence

    Input:
        - data(object): test data
        - model(torch.nn.Module): trained model
        - standard_len(int): unit length of segmented sequence 
        - rest_len(int): rest sequence length
    Output:
        - torch.cat(out_coor_list, dim=0)(tensor): atom position 
    '''
    seq_len = data.msa.shape[-1]
    num_of_times = seq_len // standard_len
    r_len = seq_len - num_of_times * standard_len

    CA_atom_index = data.CA_atom_index
    N_atom_index = torch.subtract(CA_atom_index, 1)
    atom_len = data.x.shape[0]
    edge_len = data.edge_index.shape[-1]

    res_begin_idx = 0
    res_end_idx = standard_len

    out_coor_list = []
    src_idx_list = data.edge_index[0].to('cpu').tolist()
    dst_idx_list = data.edge_index[1].to('cpu').tolist()

    end_res_flag = False
    sub_residue_count = 0
    for i in range(num_of_times):
        if i == num_of_times - 1:
            if r_len < rest_len:
                res_end_idx += r_len
                end_res_flag = True
        out_coor = compute_sub_out_coor(data, model, res_begin_idx, res_end_idx, 
                                        N_atom_index, CA_atom_index, src_idx_list, 
                                        dst_idx_list, atom_len, edge_len, 
                                        end_res_flag, i)
        out_coor_list.append(out_coor)
        res_begin_idx += standard_len
        res_end_idx += standard_len

        sub_residue_count += 1

    if r_len >= rest_len:
        res_end_idx = res_end_idx - standard_len + r_len
        out_coor = compute_sub_out_coor(data, model, res_begin_idx, res_end_idx, 
                                        N_atom_index, CA_atom_index, src_idx_list, 
                                        dst_idx_list, atom_len, edge_len, 
                                        True, sub_residue_count)
        out_coor_list.append(out_coor)

    return torch.cat(out_coor_list, dim=0)

def test(args, model, loader, file_path, write_path, standard_len=900, rest_len=70):
    '''test function

    Input:
        - args(object):argparse parameters
        - model(torch.nn.Module): trained model
        - loader(torch.dataloader): test dataloader
        - file_path(str): test dataset path
        - write_path(str): write generated pdb path
        - standard_len(int): unit length of segmented sequence 
        - rest_len(int): rest sequence length
    Output:
        - num_seqs(int): the number of sequences in test_dataset
    '''
    model.eval()
    atom_num_map = init_atom_num_map()
    name_map = init_res_name_map()
    print()
    print("----Test dataset name: %s----"%args.dataset)
    print()
    for i, data in enumerate(loader):
        protein_name = data.protein_name[0]
        print("test %d protein name:%s"%(i, protein_name))
        data = data.to(args.device)
        print('predicting protein structure...')

        if data.msa.shape[-1] > standard_len:
            out_coor = process_overlong_seq(data, model, standard_len, rest_len)
        else:
            with torch.no_grad():
                out_coor, logit_s = model(data)


            if args.dataset[-1] == '/':
                dataset_name = args.dataset.split("/")[-2] 
            else:
                dataset_name = args.dataset.split("/")[-1]

            # For antibody, the local CDR region is generally considered, 
            # so the global rotation and translation are not needed
            if dataset_name != "Antibody_data":
                out_coor = trans_and_rotate_coor(out_coor.reshape(-1, 3, 3), logit_s)

        fasta_str, atom_name_list = generate_fasta_and_atom_name(file_path, protein_name)
        residue_name_list, residue_index_list = generate_residue_name(fasta_str, name_map, atom_num_map)
        pred_coor_list = out_coor.tolist()
        pdb_file_list = generate_pdb(atom_name_list, pred_coor_list, residue_name_list, residue_index_list)

        new_write_path = os.path.join(write_path, protein_name+"_pred_result.pdb")
        write_pdb(pdb_file_list, new_write_path)


        print()
        del data
        torch.cuda.empty_cache()


def test_main_func():
    #parameter initialization
    parser = util.parser
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    #device selection
    args.device = 'cpu'
    test_mode = True

    if not os.path.exists(args.wdir):
        os.mkdir(args.wdir)
  
    model = Predict_Network(args).to(args.device)
    model.load_state_dict(torch.load(args.mdir))
    test_loader = data_builder(args, args.dataset, test_mode)
    test(args, model, test_loader, args.dataset, args.wdir)



if __name__ == "__main__":
    test_main_func()




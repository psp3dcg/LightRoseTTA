'''
process raw data
'''
import os
import copy
import torch
import numpy as np


from Bio import SeqIO
from msa_feat.ffindex import *
from msa_feat.parsers import *
from msa_feat.kinematics import xyz_to_t2d
from torch_geometric.data import Dataset
from torch_geometric.data import Data

rigid_group_bb_atom_positions = {
        0: [
            [-0.525, 1.363, 0.000],
            [0.000, 0.000, 0.000],
            [1.526, -0.000, -0.000],
        ],
        1: [
            [-0.524, 1.362, -0.000],
            [0.000, 0.000, 0.000],
            [1.525, -0.000, -0.000],
        ],
        2: [
            [-0.536, 1.357, 0.000],
            [0.000, 0.000, 0.000],
            [1.526, -0.000, -0.000],
        ],
        3: [
            [-0.525, 1.362, -0.000],
            [0.000, 0.000, 0.000],
            [1.527, 0.000, -0.000],
        ],
        4: [
            [-0.522, 1.362, -0.000],
            [0.000, 0.000, 0.000],
            [1.524, 0.000, 0.000],
        ],
        5: [
            [-0.526, 1.361, -0.000],
            [0.000, 0.000, 0.000],
            [1.526, 0.000, 0.000],
        ],
        6: [
            [-0.528, 1.361, 0.000],
            [0.000, 0.000, 0.000],
            [1.526, -0.000, -0.000],
        ],
        7: [
            [-0.572, 1.337, 0.000],
            [0.000, 0.000, 0.000],
            [1.517, -0.000, -0.000],
        ],
        8: [
            [-0.527, 1.360, 0.000],
            [0.000, 0.000, 0.000],
            [1.525, 0.000, 0.000],
        ],
        9: [
            [-0.493, 1.373, -0.000],
            [0.000, 0.000, 0.000],
            [1.527, -0.000, -0.000],
        ],
        10: [
            [-0.520, 1.363, 0.000],
            [0.000, 0.000, 0.000],
            [1.525, -0.000, -0.000],
        ],
        11: [
            [-0.526, 1.362, -0.000],
            [0.000, 0.000, 0.000],
            [1.526, 0.000, 0.000],
        ],
        12: [
            [-0.521, 1.364, -0.000],
            [0.000, 0.000, 0.000],
            [1.525, 0.000, 0.000],
        ],
        13: [
            [-0.518, 1.363, 0.000],
            [0.000, 0.000, 0.000],
            [1.524, 0.000, -0.000],
        ],
        14: [
            [-0.566, 1.351, -0.000],
            [0.000, 0.000, 0.000],
            [1.527, -0.000, 0.000],
        ],
        15: [
            [-0.529, 1.360, -0.000],
            [0.000, 0.000, 0.000],
            [1.525, -0.000, -0.000],
        ],
        16: [
            [-0.517, 1.364, 0.000],
            [0.000, 0.000, 0.000],
            [1.526, 0.000, -0.000],
        ],
        17: [
            [-0.521, 1.363, 0.000],
            [0.000, 0.000, 0.000],
            [1.525, -0.000, 0.000],
        ],
        18: [
            [-0.522, 1.362, 0.000],
            [0.000, 0.000, 0.000],
            [1.524, -0.000, -0.000],
        ],
        19: [
            [-0.494, 1.373, -0.000],
            [0.000, 0.000, 0.000],
            [1.527, -0.000, -0.000],
        ],
    }

class Protein_Dataset(Dataset):
    def __init__(self, root, FFDB, test_mode=True, pre_filter=None, pre_transform=None):
        '''init func

        Input:
            - root(str):test data path 
            - test_mode(str):train or test mode
            - pre_filter:filter process
            - pre_transform:transform process
        '''
        self.raw_file_path = os.path.join(root, 'raw')
        self.processed_file_path = os.path.join(root, 'processed')
        self.root = root
        self.FFDB = FFDB

        # for test
        self.test_mode = test_mode
        self.fasta_str_list = []
        self.all_atom_list = []
        self.protein_name_list = []

        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        super(Protein_Dataset, self).__init__(root, pre_filter, pre_transform)

        if not os.path.exists(self.processed_file_path) or len(os.listdir(self.processed_file_path)) < self.len():
            self.data_num_features, self.data_num_classes = self.process()
        else:
            if not self.test_mode:
                self.data_num_features = self.get(0).x.shape[1]
                self.data_num_classes = self.get(0).pos.shape[1]
            else:
                self.data_num_features = self.get(0).x.shape[1]
                self.data_num_classes = 3

    @property
    def raw_file_names(self):
        '''read raw file names

        Output:
            - file_path_list(list):raw file list
        '''
        file_name_list = os.listdir(self.raw_file_path)
        file_path_list = []
        for name in file_name_list:
            file_path_list.append(os.path.join(self.raw_file_path, name))

        return file_path_list

    @property
    def processed_file_names(self):
        '''generate processed file names

        Output:
            - data_pt_list(list):processed file list
        '''
        if not os.path.exists(self.processed_file_path):
            os.mkdir(self.processed_file_path)
        data_pt_list = []
        for i in range(len(os.listdir(self.raw_file_path))):
            data_pt_list.append('data_{}.pt'.format(i))
        return data_pt_list

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def read_fasta(self, fasta_file_path):
        '''read fasta file

        Input:
            - fasta_file_path(str):fasta file path
        Output:
            - result(str):fasta string
        '''
        result = ""
        for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
            for s in seq_record.seq:
                if s == 'X':
                    continue
                result += s
        return result
                
    def get_one_hot(self, index, total_num):
        ''' get one hot vector

        Input:
            - index(int):index
            - total_num(int):total number
        Output:
            - one_hot(list):one hot vector
        '''
        one_hot = []
        for i in range(total_num):
            if i == index:
                one_hot.append(1)
            else:
                one_hot.append(0)

        return one_hot

    def deal_with_edge_index(self, file_path, edge_file_name):
        '''read data and generate edge index

        Input:
            - file_path(str):edge file path
            - edge_file_name(str):edge file name
        Output:
            - new_edge_index(tensor):edge index
            - new_edge_attr(tensor):edge attribute
            - len(src)(int):source node length
        '''
        edge_index = []
        src = []
        dst = []
        edge_type = []
        file_e = open(os.path.join(file_path, edge_file_name), 'r')
        edges = file_e.readlines()
        for edge in edges:
            e_list = edge.replace('\n', '').split(' ')
            src.append(int(e_list[0]))
            dst.append(int(e_list[1]))

            edge_type.append([int(e_list[2]), int(e_list[3])])

        edge_index.append(src)
        edge_index.append(dst)

        new_edge_index = torch.tensor(edge_index, dtype=torch.long)
        new_edge_attr = torch.tensor(edge_type, dtype=torch.long)
        file_e.close()
        return new_edge_index, new_edge_attr, len(src)

    def deal_with_node_feature(self,file_path, node_file_name, atom_one_hot, relative_atomic_mass):
        '''process with node(atom) feature

        Input:
            - file_path(str):node file path
            - node_file_name(str):node file name
            - atom_one_hot(str):atom one hot vector
            - relative_atomic_mass(float):relative atomic mass
        Output:
            - new_x(tensor):new node feature
            - nodes(list):node attribute
            - len(x)(int):length of nodes
            - N_atom_index_list(list):Nitrogen atom index list
            - Ca_atom_index_list(list):Alpha Carbon atom index list
        '''
        x = []
        file_n = open(os.path.join(file_path, node_file_name), 'r')
        nodes = file_n.readlines()
        residue_num = 20
        aa_atom_num = 14
        backbone_atom_num = 2
        node_count = 0
        N_atom_index_list = []
        Ca_atom_index_list = []

        for j, node in enumerate(nodes):
            if node[0] == '\n':
                continue
            if node.split(' ')[0] == 'N':
                node_count = 0

                N_atom_index_list.append(j)
            else:
                node_count += 1
                if node.split(' ')[0] == 'CA':
                    Ca_atom_index_list.append(j)
            
            node_init_coor = node.split(' ')[1].split(',')
            node_residue_type = int(node.split(' ')[2])
            backbone_atom = int(node.split(' ')[3])

            node_index_one_hot = self.get_one_hot(node_count, aa_atom_num)
            node_residue_type_one_hot = self.get_one_hot(node_residue_type, residue_num)
            backbone_atom_one_hot = self.get_one_hot(backbone_atom, backbone_atom_num)
            
            n_feature = copy.deepcopy(atom_one_hot[node[0]])

            #node residue type
            for element in node_residue_type_one_hot:
                n_feature.append(element)

            #node backbone
            for element in backbone_atom_one_hot:
                n_feature.append(element)

            #node index
            for element in node_index_one_hot:
                n_feature.append(element)

            #init coor
            for coor in node_init_coor:
                n_feature.append(float(coor))

            x.append(n_feature)
        N_atom_index_list.append(len(x))
        new_x = torch.tensor(x, dtype=torch.float)
        file_n.close()

        return new_x, nodes, len(x), N_atom_index_list, Ca_atom_index_list


    # ---read msa info---
    def read_data_forsave(self, FFDB, seq_feat_path):
        '''read msa data

        Input:
            - FFDB(str):FF database name 
            - seq_feat_path(str):sequence feature path
        Output:
            - msa(np.array):msa information
            - xyz_t(tensor):template atom position
            - t0d(tensor):template 0 dim information
            - t1d(tensor):template 1 dim information
        '''
        FFindexDB = namedtuple("FFindexDB", "index, data")
        ffdb = None
        ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                        read_data(FFDB+'_pdb.ffdata'))
        data = []

        def check_file_ok(seq_feat_path):
            files = ["t000_.msa0.a3m", "t000_.hhr", "t000_.atab"]
            return all([os.path.exists(os.path.join(seq_feat_path, i)) for i in files])

        

        if not check_file_ok(seq_feat_path):
            print('t000_(.msa0.a3m, .hhr, .atab) files are partially missing!')
            return 
        # feature
        msa = parse_a3m(os.path.join(seq_feat_path, "t000_.msa0.a3m"))
        N, L = msa.shape
        msa = torch.tensor(msa, dtype=torch.long)


        xyz_t, t1d, t0d = read_templates(L, ffdb, os.path.join(seq_feat_path, "t000_.hhr"), \
            os.path.join(seq_feat_path, "t000_.atab"), n_templ=100)
        if xyz_t is None:
            seq = msa[0]
            defined_xyz_t = list()
            unfolded_bb_coord = list()
            for i in seq:
                unfolded_bb_coord.append(rigid_group_bb_atom_positions[i.tolist()])
            defined_xyz_t = [unfolded_bb_coord for i in range(10)]
            defined_xyz_t = torch.tensor(defined_xyz_t)
            
            defined_t0d = torch.randn(10, 3) + torch.ones(10, 3)
            defined_t1d = torch.randn(10, msa.shape[-1], 3) + torch.ones(10, msa.shape[-1], 3)
            return msa, defined_xyz_t, defined_t0d, defined_t1d
                    
        return msa, xyz_t, t0d, t1d


    def process(self):
        # Read data into huge `Data` list.
        '''process raw data

        Output:
            - num_features(int):atom feature dimension
            - num_classes(int):atom position dimension
        '''
        node_file_name = "node_label.txt"
        edge_file_name = "new_edge_1.txt"

        # FFDB="/opt/data/common/RoseTTAFold/pdb100_2021Mar03/pdb100_2021Mar03"
        FFDB=self.FFDB

        atom_one_hot = {"C":[1, 0, 0, 0],
                        "N":[0, 1, 0, 0],
                        "O":[0, 0, 1, 0],
                        "S":[0, 0, 0, 1]}

        # C:12,N:14,O:16,S:32 with norm
        relative_atomic_mass = {"C":0.1622,
                                "N":0.1892,
                                "O":0.2162,
                                "S":0.4324}


        data_list = []
        file_path_list = self.raw_file_names

        x_length = 0
        e_length = 0
        y_length = 0
        pos_length = 0

        node_dim = 0
        pos_dim = 3


        processed_file_num = len(os.listdir(self.processed_file_path))

        for i, file_path in enumerate(file_path_list):
            protein_path, protein_name = os.path.split(file_path)
            print("----process graph%d--name: %s"%(i, protein_name))

            if i < processed_file_num:
                print("----process graph%d--name: %s has been processed"%(i, protein_name))
                continue
            
            # ---edge_index---
            new_edge_index, new_edge_attr, len_e = self.deal_with_edge_index(file_path, edge_file_name)
            e_length += len_e


            # ---node_feature---
            new_x, nodes, len_x, N_atom_index_list, CA_atom_index_list = \
                self.deal_with_node_feature(file_path, node_file_name, atom_one_hot, relative_atomic_mass)
            node_dim = new_x.shape[1]
            x_length += len_x


            # ---msa info---
            try:
                new_msa, new_xyz_t, new_t0d, new_t1d = self.read_data_forsave(FFDB, file_path)
            except:
                with open('log.txt', 'a') as f:
                    f.write("msa with error:"+protein_name)
                    f.write('\n')
                continue


            new_CA_atom_index_list = torch.tensor(CA_atom_index_list, dtype = torch.long)
            data = Data(x=new_x, edge_index=new_edge_index, edge_attr = new_edge_attr, CA_atom_index=new_CA_atom_index_list,
                            msa = new_msa, xyz_t = new_xyz_t, t0d = new_t0d, t1d = new_t1d, protein_name = protein_name)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_file_path, 'data_{}.pt'.format(i)))


        data_num_features = node_dim
        data_num_classes  = pos_dim

        return data_num_features, data_num_classes

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_file_path, 'data_{}.pt'.format(idx)))
        return data





def data_builder(dataset_name, FFDB):

    dataset = Protein_Dataset(dataset_name, FFDB)

    # remove useless files
    filter_path = os.path.join(dataset_name, 'processed', 'pre_filter.pt')
    transform_path = os.path.join(dataset_name, 'processed', 'pre_transform.pt')

    if os.path.exists(filter_path):
        os.remove(filter_path)
    if os.path.exists(transform_path):
        os.remove(transform_path)



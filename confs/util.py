import argparse
# Parameter Configuration
# for training configure
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int,
                    help='DDP machine local rank')
parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument("-dataset", type=str, required=True, 
                    help="the benchmark datasets for test(CASP13/CASP14/Orphan/Denovo/Orphan25/Design55/Antibody")
parser.add_argument('-mdir', type=str, default='weights/LightRoseTTA.pth', 
                    help='folder with the trained model file')
parser.add_argument('-wdir', type=str, default='predict_pdb', 
                    help='folder of the output files')
parser.add_argument('--nhid', type=int, default=64,
                    help='hidden size')
parser.add_argument('--batch_size', type=int, default=1,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay')
parser.add_argument('--min_loss', type=float, default=1e10,
                    help='min loss value')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio') 
parser.add_argument('--epochs', type=int, default=100000,#default = 100000
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--training_times', type=int, default=50,
                    help='')


# for SE(3)-transformer
parser.add_argument('--model', type=str, default='SE3Transformer', 
        help="String name of model")
parser.add_argument('--num_layers', type=int, default=2,
        help="Number of equivariant layers")
parser.add_argument('--num_degrees', type=int, default=2,
        help="Number of irreps {0,1,...,num_degrees-1}")
parser.add_argument('--num_channels', type=int, default=8,
        help="Number of channels in middle layers")
parser.add_argument('--num_nlayers', type=int, default=0,
        help="Number of layers for nonlinearity")
parser.add_argument('--fully_connected', action='store_true',
        help="Include global node in graph")
parser.add_argument('--div', type=float, default=2,
        help="Low dimensional embedding fraction")
parser.add_argument('--pooling', type=str, default='none',
        help="Choose from avg or max")
parser.add_argument('--head', type=int, default=1,
            help="Number of attention heads")
parser.add_argument('--si_m', type=str, default='1x1',
            help="type of graph self-conv in SE(3) transformer")
parser.add_argument('--si_e', type=str, default='att',
            help="type of graph self-conv in SE(3) transformer")
parser.add_argument('--l0_in_feat', type=int, default=16, # (residue node dim:32) + (atom node integrate dim:32)
            help="residue node feat in")
parser.add_argument('--l0_out_feat', type=int, default=16,
            help="residue node feat out")
parser.add_argument('--l1_in_feat', type=int, default=3, # num of atoms
            help="backbone's offset to CA dim in")
parser.add_argument('--l1_out_feat', type=int, default=3,
            help="backbone's offset to CA dim out")
parser.add_argument('--edge_feat_dim', type=int, default=16, # dim of G.edata['w']
            help="residue graph edge feat dim")

# for build residue graph
parser.add_argument('--n_module', type=int, default=4,
            help="Number of modules")#
parser.add_argument('--n_module_str', type=int, default=1,
            help="Number of modules with structure")# 
parser.add_argument('--n_layer', type=int, default=1,
            help="Number of layers")# 
parser.add_argument('--d_msa', type=int, default=32,
            help="dimension of msa")
parser.add_argument('--d_pair', type=int, default=32,
            help="dimension of pair")
parser.add_argument('--d_templ', type=int, default=32,
            help="dimension of template")
parser.add_argument('--edge_d_pair', type=int, default=130,
            help="dimension of edge")
parser.add_argument('--n_head_msa', type=int, default=2,
            help="Number of attention heads of msa")
parser.add_argument('--n_head_pair', type=int, default=2,
            help="Number of attention heads of pair")
parser.add_argument('--n_head_templ', type=int, default=2,
            help="Number of attention heads of template")
parser.add_argument('--d_hidden', type=int, default=32,
            help="dimension of hidden layer")
parser.add_argument('--r_ff', type=int, default=4,
            help="Number of attention heads")
parser.add_argument('--n_resblock', type=int, default=1,
            help="Number of attention heads")
parser.add_argument('--p_drop', type=float, default=0.1,
            help="Number of attention heads")
parser.add_argument('--use_templ', type=bool, default=True,
            help="use template or not")
parser.add_argument('--performer_N_opts', type=dict, default={"nb_features": 8},
            help="Number of attention heads")
parser.add_argument('--performer_L_opts', type=dict, default={"nb_features": 8},
            help="Number of attention heads")




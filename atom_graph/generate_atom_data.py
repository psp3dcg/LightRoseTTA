import os
import shutil
from atom_graph.generate_protein_graph import read_fasta
from atom_graph.generate_protein_graph import build_graph
from atom_graph.generate_protein_graph import read_residue_graph
from atom_graph.generate_protein_graph import write_graph_to_file
from atom_graph.add_bond_type import add_edge_type

def build_test_data(fasta_path, fasta_write_path,
                    all_residue_dict, generate_mode):
    #------fasta
    try:
        fasta_str = read_fasta(fasta_path)#single
    except:
        print("fasta_str wrong")
        return

    if fasta_str == "":
        print("fasta_str none")
        return

    node_list, edge_list = build_graph(fasta_str, all_residue_dict, generate_mode)

    if len(node_list) == 0 or len(edge_list) == 0:
        print("build graph wrong")
        return


    node_length = len(node_list)
    edge_length = 0

    edge_connect_next_res = -1
    for i, edge in enumerate(edge_list):
        if int(edge.split(" ")[1]) == node_length:
            edge_connect_next_res = i
        if int(edge.split(" ")[0]) == (node_length-1):
            edge_length = i
            break
    if edge_connect_next_res > 0:
        edge_list.pop(edge_connect_next_res)
    else:
        edge_length += 1

    write_graph_to_file(fasta_write_path, node_list[:node_length], edge_list[:edge_length])
    copy_fasta_file(fasta_path, fasta_write_path)

def copy_fasta_file(srcfile, dstpath):
    if not os.path.isfile(srcfile):
        print("%s not exist"%(srcfile))
    else:
        f_path, f_name = os.path.split(srcfile)
        if not os.path.exists(dstpath):
            os.mkdir(dstpath)
        shutil.copy(srcfile, os.path.join(dstpath, f_name))




def generate_atom_graph_data(fasta_folder_path, write_folder_path):
    residue_graph_path = "atom_graph/amino acid/"
    all_residue_dict = read_residue_graph(residue_graph_path)
    generate_mode = 'all atom'

    fasta_list = os.listdir(fasta_folder_path)
    for fasta_name in fasta_list:
        print("----protein name:%s----"%fasta_name)
        fasta_path = os.path.join(fasta_folder_path, fasta_name)
        fasta_write_path = os.path.join(write_folder_path, fasta_name.split('.')[0])
        build_test_data(fasta_path, fasta_write_path
                         ,all_residue_dict, generate_mode)

    add_edge_type(write_folder_path)

import os
import math
import copy



def read_node_label_file(file_path):
    node_name_list = []
    with open(file_path, 'r') as f:
        file_list = f.readlines()
        for file_coor in file_list:
            node_name = file_coor.split(' ')[0]
            node_name_list.append(node_name)

    return node_name_list


def read_edge_file(file_path):
    edge_list = []
    with open(file_path, 'r') as f:
        file_list = f.readlines()
        for edge in file_list:
            if edge[0] == '\n':
                continue
            new_edge = []
            atom_idx_list = edge.split(' ')
            for atom_idx in atom_idx_list:
                new_edge.append(int(atom_idx))
            edge_list.append(new_edge)
    
    return edge_list





def edge_list_to_dict(edge_list):
    edge_dict = {}
    for edge in edge_list:
        if edge[0] not in edge_dict.keys():
            edge_dict[edge[0]] = []
        edge_dict[edge[0]].append(edge[1])
    
    return edge_dict

def bond_type_to_dict(edge_list):
    edge_dict = {}
    for edge in edge_list:
        if edge[0] not in edge_dict.keys():
            edge_dict[edge[0]] = {}
        edge_dict[edge[0]][edge[1]] = edge[2]
    
    return edge_dict



def edge_dict_to_list(edge_dict, edge_type_dict, bond_type_dict):
    edge_list = []
    for src_idx, dst_idx_list in edge_dict.items():
        for dst_idx in dst_idx_list:
            edge_type = edge_type_dict[src_idx][dst_idx] + 1
            # single bond
            if edge_type == 2:   
                bond_type = bond_type_dict[src_idx][dst_idx]
                # double bond
                if bond_type == 2:
                    edge_type = 0
                elif bond_type == 3:
                    # pi bond
                    edge_type = 1
            edge_list.append([src_idx, dst_idx, edge_type])


    return edge_list

def determine_bond_inner_or_outer_residue(node_label_list, edge_line_list):
    N_idx_list = []
    for i, node in enumerate(node_label_list):
        if node.split(' ')[0] == 'N':
            N_idx_list.append(i)
    N_idx_list.append(len(node_label_list))

    N_index = 0
    len_N_idx_list = len(N_idx_list)
    new_edge_list = []
    for i, edge_element in enumerate(edge_line_list):

        src_idx = edge_element[0]
        dst_idx = edge_element[1]
        if src_idx >= N_idx_list[N_index + 1] and N_index < len_N_idx_list - 1:
            N_index += 1
        
        if dst_idx >= N_idx_list[N_index] and dst_idx < N_idx_list[N_index + 1]:
            inner_bond_label = '1'
        else:
            inner_bond_label = '0'

        edge_element.insert(2, inner_bond_label)
        new_edge_list.append(edge_element)
    
    return new_edge_list


def add_hop_edge(edge_dict):

    add_idx_dict = {}
    edge_type_dict = {}


    for key in edge_dict.keys():
        add_idx_dict[key] = []
        edge_type_dict[key] = {}
        two_hop_edge = []
        three_hop_edge = []
        four_hop_edge = []
        for value in edge_dict[key]:
            edge_type_dict[key][value] = 1
    for key, idx_list in add_idx_dict.items():
        for idx in idx_list:
            edge_dict[key].append(idx)


    
    return edge_dict, edge_type_dict


def write_file(file_path, new_edge_list):
    f_path, f_name = os.path.split(file_path)
    if not os.path.exists(f_path):
        os.mkdir(f_path)
    f = open(file_path, 'w')
    for new_edge in new_edge_list:
        new_edge_str = ""
        for element in new_edge:
            new_edge_str += str(element)
            new_edge_str += ' '
        
        new_edge_str = new_edge_str[:-1]+'\n'
        f.write(new_edge_str)
    f.close()



def add_edge_type(file_path):
    file_list = os.listdir(file_path)
    write_path = file_path
    for file_name in file_list:

        node_name_list = read_node_label_file(os.path.join(file_path, file_name, "node_label.txt"))
        edge_list = read_edge_file(os.path.join(file_path, file_name, "A.txt"))
        
        edge_dict = edge_list_to_dict(edge_list)
        bond_type_dict = bond_type_to_dict(edge_list)

        new_edge_dict, edge_type_dict = add_hop_edge(edge_dict)
        edge_list = edge_dict_to_list(new_edge_dict, edge_type_dict, bond_type_dict)
        
        new_edge_list = determine_bond_inner_or_outer_residue(node_name_list, edge_list)

        write_file(os.path.join(write_path, file_name, "new_edge_1.txt"), new_edge_list)

    

        

    

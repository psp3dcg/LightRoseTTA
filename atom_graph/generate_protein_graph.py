import os
from Bio import SeqIO
def read_fasta(fasta_file_path):
    result = ""
    for seq_record in SeqIO.parse(fasta_file_path, "fasta"):
        for s in seq_record.seq:
            if s == 'X':
                continue
            result += s

    return result


def read_residue_graph(residue_graph_path):
    residue_list = os.listdir(residue_graph_path)
    node_file_name = "node_label.txt"
    edge_file_name = "A.txt"

    rigid_group_atom_positions = {
        'A': [
            ['N', 0, (-0.525, 1.363, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.526, -0.000, -0.000)],
            ['O', 3, (0.627, 1.062, 0.000)],
            ['CB', 0, (-0.529, -0.774, -1.205)],
        ],
        'R': [
            ['N', 0, (-0.524, 1.362, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.525, -0.000, -0.000)],
            ['O', 3, (0.626, 1.062, 0.000)],
            ['CB', 0, (-0.524, -0.778, -1.209)],
            ['CG', 4, (0.616, 1.390, -0.000)],
            ['CD', 5, (0.564, 1.414, 0.000)],
            ['NE', 6, (0.539, 1.357, -0.000)],
            ['NH1', 7, (0.206, 2.301, 0.000)],
            ['NH2', 7, (2.078, 0.978, -0.000)],
            ['CZ', 7, (0.758, 1.093, -0.000)],
        ],
        'N': [
            ['N', 0, (-0.536, 1.357, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.526, -0.000, -0.000)],
            ['O', 3, (0.625, 1.062, 0.000)],
            ['CB', 0, (-0.531, -0.787, -1.200)],
            ['CG', 4, (0.584, 1.399, 0.000)],
            ['ND2', 5, (0.593, -1.188, 0.001)],
            ['OD1', 5, (0.633, 1.059, 0.000)],
        ],
        'D': [
            ['N', 0, (-0.525, 1.362, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.527, 0.000, -0.000)],
            ['O', 3, (0.626, 1.062, -0.000)],
            ['CB', 0, (-0.526, -0.778, -1.208)],
            ['CG', 4, (0.593, 1.398, -0.000)],
            ['OD1', 5, (0.610, 1.091, 0.000)],
            ['OD2', 5, (0.592, -1.101, -0.003)],
        ],
        'C': [
            ['N', 0, (-0.522, 1.362, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.524, 0.000, 0.000)],
            ['O', 3, (0.625, 1.062, -0.000)],
            ['CB', 0, (-0.519, -0.773, -1.212)],
            ['SG', 4, (0.728, 1.653, 0.000)],
        ],
        'Q': [
            ['N', 0, (-0.526, 1.361, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.526, 0.000, 0.000)],
            ['O', 3, (0.626, 1.062, -0.000)],
            ['CB', 0, (-0.525, -0.779, -1.207)],
            ['CG', 4, (0.615, 1.393, 0.000)],
            ['CD', 5, (0.587, 1.399, -0.000)],
            ['NE2', 6, (0.593, -1.189, -0.001)],
            ['OE1', 6, (0.634, 1.060, 0.000)],
        ],
        'E': [
            ['N', 0, (-0.528, 1.361, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.526, -0.000, -0.000)],
            ['O', 3, (0.626, 1.062, 0.000)],
            ['CB', 0, (-0.526, -0.781, -1.207)],
            ['CG', 4, (0.615, 1.392, 0.000)],
            ['CD', 5, (0.600, 1.397, 0.000)],
            ['OE1', 6, (0.607, 1.095, -0.000)],
            ['OE2', 6, (0.589, -1.104, -0.001)],
        ],
        'G': [
            ['N', 0, (-0.572, 1.337, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.517, -0.000, -0.000)],
            ['O', 3, (0.626, 1.062, -0.000)],
        ],
        'H': [
            ['N', 0, (-0.527, 1.360, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.525, 0.000, 0.000)],
            ['O', 3, (0.625, 1.063, 0.000)],
            ['CB', 0, (-0.525, -0.778, -1.208)],
            ['CG', 4, (0.600, 1.370, -0.000)],
            ['CD2', 5, (0.889, -1.021, 0.003)],
            ['ND1', 5, (0.744, 1.160, -0.000)],
            ['CE1', 5, (2.030, 0.851, 0.002)],
            ['NE2', 5, (2.145, -0.466, 0.004)],
        ],
        'I': [
            ['N', 0, (-0.493, 1.373, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.527, -0.000, -0.000)],
            ['O', 3, (0.627, 1.062, -0.000)],
            ['CB', 0, (-0.536, -0.793, -1.213)],
            ['CG1', 4, (0.534, 1.437, -0.000)],
            ['CG2', 4, (0.540, -0.785, -1.199)],
            ['CD1', 5, (0.619, 1.391, 0.000)],
        ],
        'L': [
            ['N', 0, (-0.520, 1.363, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.525, -0.000, -0.000)],
            ['O', 3, (0.625, 1.063, -0.000)],
            ['CB', 0, (-0.522, -0.773, -1.214)],
            ['CG', 4, (0.678, 1.371, 0.000)],
            ['CD1', 5, (0.530, 1.430, -0.000)],
            ['CD2', 5, (0.535, -0.774, 1.200)],
        ],
        'K': [
            ['N', 0, (-0.526, 1.362, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.526, 0.000, 0.000)],
            ['O', 3, (0.626, 1.062, -0.000)],
            ['CB', 0, (-0.524, -0.778, -1.208)],
            ['CG', 4, (0.619, 1.390, 0.000)],
            ['CD', 5, (0.559, 1.417, 0.000)],
            ['CE', 6, (0.560, 1.416, 0.000)],
            ['NZ', 7, (0.554, 1.387, 0.000)],
        ],
        'M': [
            ['N', 0, (-0.521, 1.364, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.525, 0.000, 0.000)],
            ['O', 3, (0.625, 1.062, -0.000)],
            ['CB', 0, (-0.523, -0.776, -1.210)],
            ['CG', 4, (0.613, 1.391, -0.000)],
            ['SD', 5, (0.703, 1.695, 0.000)],
            ['CE', 6, (0.320, 1.786, -0.000)],
        ],
        'F': [
            ['N', 0, (-0.518, 1.363, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.524, 0.000, -0.000)],
            ['O', 3, (0.626, 1.062, -0.000)],
            ['CB', 0, (-0.525, -0.776, -1.212)],
            ['CG', 4, (0.607, 1.377, 0.000)],
            ['CD1', 5, (0.709, 1.195, -0.000)],
            ['CD2', 5, (0.706, -1.196, 0.000)],
            ['CE1', 5, (2.102, 1.198, -0.000)],
            ['CE2', 5, (2.098, -1.201, -0.000)],
            ['CZ', 5, (2.794, -0.003, -0.001)],
        ],
        'P': [
            ['N', 0, (-0.566, 1.351, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.527, -0.000, 0.000)],
            ['O', 3, (0.621, 1.066, 0.000)],
            ['CB', 0, (-0.546, -0.611, -1.293)],
            ['CG', 4, (0.382, 1.445, 0.0)],
            ['CD', 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
        ],
        'S': [
            ['N', 0, (-0.529, 1.360, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.525, -0.000, -0.000)],
            ['O', 3, (0.626, 1.062, -0.000)],
            ['CB', 0, (-0.518, -0.777, -1.211)],
            ['OG', 4, (0.503, 1.325, 0.000)],
        ],
        'T': [
            ['N', 0, (-0.517, 1.364, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.526, 0.000, -0.000)],
            ['O', 3, (0.626, 1.062, 0.000)],
            ['CB', 0, (-0.516, -0.793, -1.215)],
            ['CG2', 4, (0.550, -0.718, -1.228)],
            ['OG1', 4, (0.472, 1.353, 0.000)],
        ],
        'W': [
            ['N', 0, (-0.521, 1.363, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.525, -0.000, 0.000)],
            ['O', 3, (0.627, 1.062, 0.000)],
            ['CB', 0, (-0.523, -0.776, -1.212)],
            ['CG', 4, (0.609, 1.370, -0.000)],
            ['CD1', 5, (0.824, 1.091, 0.000)],
            ['CD2', 5, (0.854, -1.148, -0.005)],
            ['CE2', 5, (2.186, -0.678, -0.007)],
            ['CE3', 5, (0.622, -2.530, -0.007)],
            ['NE1', 5, (2.140, 0.690, -0.004)],
            ['CH2', 5, (3.028, -2.890, -0.013)],
            ['CZ2', 5, (3.283, -1.543, -0.011)],
            ['CZ3', 5, (1.715, -3.389, -0.011)],
        ],
        'Y': [
            ['N', 0, (-0.522, 1.362, 0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.524, -0.000, -0.000)],
            ['O', 3, (0.627, 1.062, -0.000)],
            ['CB', 0, (-0.522, -0.776, -1.213)],
            ['CG', 4, (0.607, 1.382, -0.000)],
            ['CD1', 5, (0.716, 1.195, -0.000)],
            ['CD2', 5, (0.713, -1.194, -0.001)],
            ['CE1', 5, (2.107, 1.200, -0.002)],
            ['CE2', 5, (2.104, -1.201, -0.003)],
            ['OH', 5, (4.168, -0.002, -0.005)],
            ['CZ', 5, (2.791, -0.001, -0.003)],
        ],
        'V': [
            ['N', 0, (-0.494, 1.373, -0.000)],
            ['CA', 0, (0.000, 0.000, 0.000)],
            ['C', 0, (1.527, -0.000, -0.000)],
            ['O', 3, (0.627, 1.062, -0.000)],
            ['CB', 0, (-0.533, -0.795, -1.213)],
            ['CG1', 4, (0.540, 1.429, -0.000)],
            ['CG2', 4, (0.533, -0.776, 1.203)],
        ],
    }

    all_residue_dict = {}

    for res in residue_list:
        node_list = []
        edge_list = []
        residue_dict = {}

        node_f = open(os.path.join(residue_graph_path, res, node_file_name))
        edge_f = open(os.path.join(residue_graph_path, res, edge_file_name))

        nodes = node_f.readlines()
        edges = edge_f.readlines()

        coor_list = rigid_group_atom_positions[res]
        for i, n in enumerate(nodes):
            if n !="\n":
                n = n[:-1]
                n += " "
                for element in coor_list[i][2]:
                    n += str(element)
                    n += ','
                n = n[:-1]+'\n'
                node_list.append(n)


        for e in edges:
            if e !="\n":
                edge_list.append(e)

        residue_dict["nodes"] = node_list
        residue_dict["edges"] = edge_list

        all_residue_dict[res] = residue_dict

        node_f.close()
        edge_f.close()


    return all_residue_dict

def build_graph(fasta_str, all_residue_dict, generate_mode):

    node_list = []
    edge_list = []

    total_num_nodes = 0
    last_total_num_nodes = 0

    residue_index_dict = {
        "A":"0",
        "C":"1",
        "D":"2",
        "E":"3",
        "F":"4",
        "G":"5",
        "H":"6",
        "I":"7",
        "K":"8",
        "L":"9",
        "M":"10",
        "N":"11",
        "P":"12",
        "Q":"13",
        "R":"14",
        "S":"15",
        "T":"16",
        "V":"17",
        "W":"18",
        "Y":"19"
    }

    backbone_atom_name = ("N", "CA", "C")
    for i, res in enumerate(fasta_str):

        num_nodes = 0
        try:
            nodes = all_residue_dict[res]["nodes"]
            edges = all_residue_dict[res]["edges"]
        except:
            print("unknown residue!")
            return node_list, edge_list

        for n in nodes:
            atom_name = n.split(" ")[0]
            if atom_name in backbone_atom_name:
                atom_type = '0'
            elif generate_mode == "all atom":
                atom_type = '1'
            elif generate_mode == "backbone":
                continue

            n = n[:-1]
            n = n + " " +residue_index_dict[res] + " " + atom_type + "\n"
            node_list.append(n)
            num_nodes += 1

        
        # add the bond connect with the previous residue
        if i > 0:
            link_bond_1 = str(total_num_nodes) + " " + str(last_total_num_nodes + 2) + " 1\n"
            edge_list.append(link_bond_1)
            last_total_num_nodes = total_num_nodes

        for e in edges:
            e_list = e.split(" ")

            #add bond connect the next residue
            if generate_mode == "backbone":
                if e_list[1] == '4':
                    continue
                if e_list[0] == '2' and e_list[1] == '3' and i != len(fasta_str) - 1:
                    link_bond_2 = str(2 + total_num_nodes)+" "+str(num_nodes + total_num_nodes)+" 1\n"
                    edge_list.append(link_bond_2)
                    break


            elif e_list[0] == '3' and e_list[1] == '2' and i != len(fasta_str) - 1:
                link_bond_2 = str(2 + total_num_nodes)+" "+str(num_nodes + total_num_nodes)+" 1\n"
                edge_list.append(link_bond_2)



            e_list[0] = str(int(e_list[0]) + total_num_nodes)
            e_list[1] = str(int(e_list[1]) + total_num_nodes)
            new_e = e_list[0]+" "+e_list[1]+" "+e_list[2]

            edge_list.append(new_e)

        total_num_nodes += num_nodes


    return node_list, edge_list

def write_graph_to_file(write_path, node_list, edge_list):
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    with open(os.path.join(write_path, "node_label.txt"), 'w') as node_f:
        for n in node_list:
            node_f.write(n)

    with open(os.path.join(write_path, "A.txt"), "w") as edge_f:
        for e in edge_list:
            if not e.endswith('\n'):
                edge_f.write(e+'\n')
            else:
                edge_f.write(e)

def write_fasta_name_to_file(write_path, name_list):
    with open(write_path, 'w') as f:
        for name in name_list:
            f.write(name)
            f.write(',')





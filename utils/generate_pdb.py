# generate pdb and write into file

def generate_pdb(atom_name_list, atom_coor_list, residue_name_list, residue_index_list, chain_label = "A"):
    '''generate predict pdb according to the provided information
    Input:
        - atom_name_list(list):all atom name in one protein
        - atom_coor_list(list):all atom 3d position in one protein
        - residue_name_list(list):all residue name in one protein
        - residue_index_list(list):all residue index in one protein
    Output:
        - pdb_lines_list(list):all lines to write in pdb file
    '''
    atom_label = "ATOM"
    atom_index = 1
    alt_loc = ''
    insertion_code = ''
    
    occupancy  = 1.00
    b_factor = 0.00
    charge = ''


    pdb_lines_list = []
    assert len(atom_name_list) == len(atom_coor_list)
    

    for i, name in enumerate(atom_name_list):
        # PDB is a columnar format, every space matters here!
        name = name.split(' ')[0]
        element = name[0]
        pos = atom_coor_list[i]
        pdb_line = (f'{atom_label:<6}{atom_index:>5}  {name:<3}{alt_loc:>1}'+
                    f'{residue_name_list[i]:>3} {chain_label:>1}'+
                    f'{residue_index_list[i]:>4}{insertion_code:>1}   '+
                    f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'+
                    f'{occupancy:>6.2f}{b_factor:>6.2f}          '+
                    f'{element:>2}{charge:>2}')
        atom_index += 1

        pdb_lines_list.append(pdb_line)

    chain_end = 'TER'
    chain_termination_line = (
        f'{chain_end:<6}{atom_index:>5}      {residue_name_list[-1]:>3} '
        f'{chain_label:>1}{residue_index_list[-1]:>4}')
    pdb_lines_list.append(chain_termination_line)
   
    end_line = "END"

    pdb_lines_list.append(end_line)

    return '\n'.join(pdb_lines_list)

def write_pdb(pdb_file_list, write_path):
    '''write pdb lines into file
    Input:
        - pdb_file_list(list):all lines to write in pdb file
        - write_path(str):write file path
    '''
    f = open(write_path, 'w')
    for pdb_line in pdb_file_list:
        f.write(pdb_line)
    f.close()
    

    

    






        




import os
import sys
from generate_pt_file import data_builder
from msa_feat.generate_msa import extract_msa_templ
from atom_graph.generate_atom_data import generate_atom_graph_data


def preprocess_data(fasta_path, write_base_path, 
                    seq_db_base_path, templ_db_base_path, big_seq_db_base_path):
    # param example
    # fasta_path = "./Orphan25_fasta"
    # write_base_path = "./Orphan25_data"
    # seq_db_base_path = "./uniref30_2020_06"
    # templ_db_base_path = "./pdb100_2021Mar03"
    # big_seq_db_base_path = "./bfd"
    data_name = write_base_path
    write_path = os.path.join(write_base_path, 'raw')
    if not os.path.exists(write_base_path):
        os.mkdir(write_base_path)
    
    if not os.path.exists(write_path):
        os.mkdir(write_path)
	    
    if os.path.exists(os.path.join(write_base_path, 'processed')):
	return
	    
    seq_search_db = os.path.join(seq_db_base_path, "UniRef30_2020_06")
    templ_search_db = os.path.join(templ_db_base_path, "pdb100_2021Mar03")
    big_seq_search_db = os.path.join(big_seq_db_base_path, "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt")
	
    print("--------1.generating atom graph files...")
    generate_atom_graph_data(fasta_path, write_path)
    print()
    
    print("--------2.extracting homologous files...")
    extract_msa_templ(fasta_path, write_path, seq_search_db, templ_search_db, big_seq_search_db)
    print()

    print("--------3.generating pt files...")
    data_builder(data_name, templ_search_db)
    print()

    print("--------generating process is finished...")

if __name__ == '__main__':

    fasta_path = sys.argv[1]
    write_base_path = sys.argv[2]
    seq_db_base_path = sys.argv[3]
    templ_db_base_path = sys.argv[4]
    big_seq_db_base_path = sys.argv[5]
    preprocess_data(fasta_path, write_base_path, seq_db_base_path, templ_db_base_path, big_seq_db_base_path)

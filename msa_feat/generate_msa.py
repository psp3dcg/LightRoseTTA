import sys
import os
import time
import codecs
import pdb
from multiprocessing import Pool
from msa_feat.delete_query_struct_in_templ import delete_struct_in_hhr
from msa_feat.delete_query_struct_in_templ import delete_struct_in_atab
def check_fasta(input_f):
    lines = [i.strip() for i in open(input_f)]
    return len(lines) > 1 and lines[1] != ''
def process(input_path, output_base_dir, fasta_name, seq_search_db, templ_search_db, big_seq_search_db, name_dict=None):

    if not os.path.exists(input_path):
        return
    if not check_fasta(input_path):
        return
    output_dir = os.path.join(output_base_dir, fasta_name)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    cmd = "bash msa_feat/run_generate_feat.sh %s %s %s %s %s" % (input_path, output_dir, seq_search_db, templ_search_db, big_seq_search_db)
    
    os.system(cmd)
    
    # delete query itself in .hhr and .atab
    hhr_path = os.path.join(output_base_dir, fasta_name, "t000_.hhr")
    atab_path = os.path.join(output_base_dir, fasta_name, "t000_.atab")

    if name_dict != None and fasta_name[:5] in name_dict.keys():
        protein_name = name_dict[fasta_name[:5]]
    else:
        protein_name = fasta_name
    new_hhr_path = hhr_path
    delete_struct_in_hhr(hhr_path, new_hhr_path, protein_name)
    new_atab_path = atab_path
    delete_struct_in_atab(atab_path, new_atab_path, protein_name)
    print("generate {:} done".format(fasta_name))
    


def extract_msa_templ(fasta_dir, output_dir, seq_search_db, templ_search_db, big_seq_search_db, name_dict=None):
    in_fa_dir = fasta_dir
    output_base_dir = output_dir
    for fasta_name in os.listdir(fasta_dir):
        print('extract name:',fasta_name)
        fa = os.path.join(in_fa_dir, fasta_name)
        process(fa, output_base_dir, fasta_name.split('.')[0], seq_search_db, templ_search_db, big_seq_search_db, name_dict)
        




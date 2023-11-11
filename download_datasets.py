import os

def download_homolog_datasets():
    print('Downloading the homologous datasets...')

    # download UniRef30 [46G]
    echo "downloading UniRef30..."
    seq_search_db_link = "http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz"
    seq_download_cmd = "wget "+seq_search_db_link
    make_folder_cmd = "mkdir -p UniRef30_2020_06"
    uncompress_seq_cmd = "tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06"
    os.system(seq_download_cmd)
    os.system(make_folder_cmd)
    os.system(uncompress_seq_cmd)

    # download bfd [272G]
    echo "downloading bfd..."
    bfd_seq_search_db_link = "https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz"
    bfd_seq_download_cmd = "wget "+seq_search_db_link
    make_folder_cmd = "mkdir -p bfd"
    uncompress_bfd_seq_cmd = "tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd"
    os.system(bfd_seq_download_cmd)
    os.system(make_folder_cmd)
    os.system(uncompress_bfd_seq_cmd)

    # download pdb100 [over 100G]
    echo "downloading pdb100..."
    templ_search_db_link = "https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz"
    templ_download_cmd = "wget "+templ_search_db_link
    uncompress_templ_cmd = "tar xfz pdb100_2021Mar03.tar.gz"
    os.system(templ_download_cmd)
    os.system(uncompress_templ_cmd)

    print('Downloading process is finished...')

if __name__ == '__main__':
    download_homolog_datasets()
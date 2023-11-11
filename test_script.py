import os
import sys
from generate_LightRoseTTA_data import preprocess_data


def test_LightRoseTTA_from_FASTA(fasta_path, write_base_path, 
                    seq_db_base_path, templ_db_base_path,
                    big_seq_db_base_path, model_path):

	
	# preprocess the FASTA data
	preprocess_data(fasta_path, write_base_path, 
                    seq_db_base_path, templ_db_base_path, big_seq_db_base_path)

	# run the LightRoseTTA test program
	data_path = write_base_path
	result_path = os.path.join(write_base_path, 'predict_pdb')
	test_cmd = "python LightRoseTTA_test.py -dataset %s -mdir %s -wdir %s"%(data_path, model_path, result_path)
	os.system(test_cmd)


if __name__ == '__main__':
	fasta_path = sys.argv[1]
	write_base_path = sys.argv[2]
	seq_db_base_path = sys.argv[3]
	templ_db_base_path = sys.argv[4]
	big_seq_db_base_path = sys.argv[5]
	model_path = sys.argv[6]
	
	test_LightRoseTTA_from_FASTA(fasta_path, write_base_path, seq_db_base_path, templ_db_base_path, big_seq_db_base_path, model_path)
	

